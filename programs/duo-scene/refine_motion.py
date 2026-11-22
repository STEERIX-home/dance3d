"""Give the captured bodies back the quick motion that the motion capture smooths away.

ExMR follows slow and medium-speed motion well, but a quick shake or hit comes out a quarter to
a third too small (measured against the 2D body keypoints of the same frames). The keypoints see
that motion directly in the picture. So, per person, in the camera frame:

  1. residual = detected keypoint - the body's own joint in the picture,
  2. only the QUICK part of the residual is used (quicker than --cut Hz). Its slow part holds the
     constant differences between a keypoint detector's joints and the body model's joints, and is
     left alone: the body's slow pose stays exactly as captured,
  3. the body is moved by the smallest 3D change (position sideways and up/down, turn, joint
     angles) whose picture motion matches the quick residual. Depth is not touched: it cannot be
     seen in one picture.
Keypoints that are uncertain or not on the person's own mask do not count, and frames in which the
person is not reliably seen are not changed. A foot that stands on the floor is not moved by its
keypoints (they tremble by a few pixels); it is held on its own smoothed path instead, so planted
feet stay still while the body above them shakes.

    refine_motion.py --mocap MOCAP_DIR --tracks TRACKS_DIR --output NEW_MOCAP_DIR      (needs the GPU)

The output folder has the same files as mocap_people.py's and is used in its place.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from paths import EXMR, SMPLX
from qa_fast_motion import load_masks

sys.path.insert(0, str(EXMR))

from exmr.body_models.smplx_model import SmplxModel  # noqa: E402
from exmr.geometry.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle  # noqa: E402

# COCO-23 keypoints (ExMR's order) -> SMPL-X joint names
KEYPOINT_JOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                   'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                   'right_knee', 'left_ankle', 'right_ankle', 'left_big_toe', 'left_small_toe', 'left_heel',
                   'right_big_toe', 'right_small_toe', 'right_heel']
HIPS = [11, 12]  # a detector's hip points sit elsewhere on the pelvis than the model's hip joints
FOOT_KEYPOINTS = ([15, 17, 18, 19], [16, 20, 21, 22])  # left, right: ankle, toes, heel
FOOT_JOINTS = (['left_ankle', 'left_foot', 'left_big_toe', 'left_small_toe', 'left_heel'],
               ['right_ankle', 'right_foot', 'right_big_toe', 'right_small_toe', 'right_heel'])
FOOT_WEIGHT = 3.0e5     # cost of a planted foot leaving its smoothed path, per square metre
PIXEL_SIGMA = 2.0       # expected keypoint noise (pixels)
MOVE_WEIGHT = 2.0e4     # cost of moving a joint, per square metre
ANGLE_WEIGHT = 4.0      # cost of changing a joint angle, per square radian


def gauss_weighted(values, weights, sigma):
    """Normalized Gaussian smoothing along time: values (L, N, D), weights (L, N)."""
    radius = max(1, int(np.ceil(sigma * 4)))
    kernel = torch.exp(-0.5 * (torch.arange(-radius, radius + 1, device=values.device, dtype=values.dtype) / sigma) ** 2)
    kernel = (kernel / kernel.sum())[None, None]

    def blur(x):  # (L, C) -> (L, C)
        padded = torch.nn.functional.pad(x.T[:, None], (radius, radius), mode='replicate')
        return torch.nn.functional.conv1d(padded, kernel)[:, 0].T

    L, N, D = values.shape
    total = blur((values * weights[..., None]).reshape(L, N * D)).reshape(L, N, D)
    return total / blur(weights).clamp_min(1e-6)[..., None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mocap', required=True)
    parser.add_argument('--tracks', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cut', type=float, default=1.5, help='only motion quicker than this (Hz) is corrected')
    parser.add_argument('--iterations', type=int, default=300)
    args = parser.parse_args()
    mocap, out = Path(args.mocap), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((mocap / 'people.json').read_text())
    fps, count, width, height = info['fps'], info['frames'], info['width'], info['height']
    masks = load_masks(Path(args.tracks) / 'masks.npz')
    device = torch.device('cuda')
    smplx = SmplxModel(SMPLX).to(device).eval()
    names = list(smplx.joint_names)
    joint_index = torch.tensor([names.index(name) for name in KEYPOINT_JOINTS], device=device)
    report = dict(cut_hz=args.cut, people=[])
    for person in info['people']:
        label = person['label']
        result = torch.load(mocap / f'person_{label}/results.pt', map_location='cpu', weights_only=False)
        keypoints = torch.load(mocap / f'person_{label}/keypoints.pt', map_location='cpu', weights_only=False)['value'][:, :23]
        base = {k: v.to(device).float() for k, v in result['smplx_incam'].items()}
        K = result['K_fullimg'][0].to(device).float()
        reliable = torch.tensor(person['reliable'], device=device)

        def project(joints):
            return joints[..., :2] / joints[..., 2:].clamp_min(0.05) * K[[0, 1], [0, 1]] + K[:2, 2]

        def body(delta_xy, delta_turn, delta_pose):
            params = dict(base)
            params['transl'] = base['transl'] + torch.cat([delta_xy, torch.zeros_like(delta_xy[:, :1])], dim=1)
            params['global_orient'] = matrix_to_axis_angle(axis_angle_to_matrix(delta_turn) @ axis_angle_to_matrix(base['global_orient']))
            params['body_pose'] = base['body_pose'] + delta_pose
            return params

        with torch.no_grad():
            joints_0 = smplx.joints(**base)
            pixels_0 = project(joints_0)
        # which keypoints count: confident, on the person's own mask, in a reliably seen frame
        own = masks[label]
        scale = torch.tensor([own.shape[2] / width, own.shape[1] / height])
        weight = ((keypoints[..., 2] - 0.3) / 0.4).clamp(0, 1)
        cells = (keypoints[..., :2] * scale).round().long()
        inside = torch.zeros(count, 23, dtype=torch.bool)
        for index in range(count):
            grown = torch.from_numpy(cv2.dilate(own[index].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
            column, row = cells[index, :, 0], cells[index, :, 1]
            valid = (column >= 0) & (column < own.shape[2]) & (row >= 0) & (row < own.shape[1])
            inside[index, valid] = grown[row[valid], column[valid]]
        weight = (weight * inside).to(device) * reliable[:, None]
        weight[:, HIPS] *= 0.3
        contact = result['foot_contact'].to(device) & reliable[:, None]  # (L, 2) left, right
        foot_index = torch.tensor([[names.index(name) for name in side] for side in FOOT_JOINTS], device=device)  # (2, 5)
        for side in range(2):
            weight[:, FOOT_KEYPOINTS[side]] *= (~contact[:, side:side + 1]).float()
        residual = keypoints[..., :2].to(device) - pixels_0[:, joint_index]
        slow = gauss_weighted(residual, weight, 0.1874 / args.cut * fps)
        quick = gauss_weighted(residual - slow, weight, 0.012 * fps) * (weight > 0)[..., None]
        quick = quick.clamp(-60, 60)

        # planted feet: their own path, smoothed over the time they stand (the camera moves slowly, so a
        # standing foot moves slowly in the camera frame)
        planted = contact[:, :, None].expand(-1, -1, foot_index.shape[1]).reshape(count, -1).float()  # (L, 10)
        foot_path = joints_0[:, foot_index.reshape(-1)]
        goal = foot_path.cpu().numpy().copy()
        standing = contact.cpu().numpy()
        for side in range(2):
            columns = slice(side * foot_index.shape[1], (side + 1) * foot_index.shape[1])
            edges = np.flatnonzero(np.diff(np.concatenate([[False], standing[:, side], [False]]).astype(int)))
            for first, stop in zip(edges[::2], edges[1::2]):  # one stance at a time: two stances are two places
                if stop - first >= 3:
                    goal[first:stop, columns] = gaussian_filter1d(goal[first:stop, columns], 0.1 * fps, axis=0, mode='nearest')
        foot_goal = torch.from_numpy(goal).to(device)

        delta_xy = torch.zeros(count, 2, device=device, requires_grad=True)
        delta_turn = torch.zeros(count, 3, device=device, requires_grad=True)
        delta_pose = torch.zeros(count, 63, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([delta_xy, delta_turn, delta_pose], lr=0.004)
        for step in range(args.iterations):
            joints = smplx.joints(**body(delta_xy, delta_turn, delta_pose))
            moved = project(joints)[:, joint_index] - pixels_0[:, joint_index]
            fit = (weight[..., None] * ((moved - quick) / PIXEL_SIGMA) ** 2).sum()
            keep = MOVE_WEIGHT * ((joints[:, :66] - joints_0[:, :66]) ** 2).sum() + ANGLE_WEIGHT * ((delta_pose ** 2).sum() + (delta_turn ** 2).sum())
            stand = FOOT_WEIGHT * (planted[..., None] * (joints[:, foot_index.reshape(-1)] - foot_goal) ** 2).sum()
            loss = fit + keep + stand
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == 0:
                first_fit = float(fit.detach())
        with torch.no_grad():
            params = body(delta_xy, delta_turn, delta_pose)
            # frames that were not reliably seen stay exactly as they were
            for key in ('transl', 'global_orient', 'body_pose'):
                params[key] = torch.where(reliable[:, None], params[key], base[key])
            joints = smplx.joints(**params)
            pixels = project(joints)
            moved = (pixels - pixels_0)[:, joint_index]
            seen = weight > 0
            entry = dict(label=label, keypoints_used=float(seen.float().mean()),
                         quick_residual_px=dict(before=float(quick[seen].pow(2).mean().sqrt()),
                                                after=float((quick - moved)[seen].pow(2).mean().sqrt())),
                         fit_cost=dict(before=first_fit, after=float(fit)),
                         planted_foot_off_its_path_mm=dict(
                             before=float(((foot_path - foot_goal).norm(dim=-1) * planted).sum() / planted.sum().clamp_min(1) * 1000),
                             after=float(((joints[:, foot_index.reshape(-1)] - foot_goal).norm(dim=-1) * planted).sum() / planted.sum().clamp_min(1) * 1000)),
                         joints_moved_mm=dict(median=float((joints[:, :22] - joints_0[:, :22]).norm(dim=-1)[reliable].median() * 1000),
                                              p99=float((joints[:, :22] - joints_0[:, :22]).norm(dim=-1)[reliable].quantile(0.99) * 1000)))
        report['people'].append(entry)
        print(json.dumps(entry), flush=True)
        work = out / f'person_{label}'
        work.mkdir(exist_ok=True)
        result['smplx_incam'] = {**result['smplx_incam'], **{k: params[k].cpu() for k in ('transl', 'global_orient', 'body_pose')}}
        result['joints_incam'] = joints.cpu()
        result['joints_2d'] = pixels.cpu()
        torch.save(result, work / 'results.pt')
        shutil.copy2(mocap / f'person_{label}/keypoints.pt', work / 'keypoints.pt')
    for name in ('people.json', 'camera.pt'):
        shutil.copy2(mocap / name, out / name)
    (out / 'refine.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
