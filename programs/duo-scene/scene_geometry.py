"""Camera and room geometry of the source video, measured on sampled frames with MoGe-2.

For each sampled frame: the field of view, the floor plane (its normal is the direction of
gravity, its distance the camera height) and a coloured metric point cloud of the room with the
dancers removed. These are estimates from single images; the summary keeps their spread so
later steps can judge how much to trust them.

Outputs in --output:
  geometry.json   per sampled frame: focal length, floor normal and height, pitch and roll
  points.npz      room points of a few frames (camera coordinates, metres) for inspection
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def load_people_masks(path, count):
    """(frames, h, w) bool: True where any tracked person is."""
    with np.load(path) as data:
        shape = tuple(data['shape'])
        union = np.zeros(shape, dtype=bool)
        for key in data.files:
            if key != 'shape':
                union |= np.unpackbits(data[key], axis=-1)[..., :shape[-1]].astype(bool)
    if len(union) != count:
        raise ValueError('Mask and video frame counts differ')
    return union


def fit_floor(points, normals, rng, tries=400, tolerance=0.03):
    """RANSAC plane through candidate floor points -> (unit normal towards the camera, distance,
    inlier share). Camera coordinates: x right, y down, z forward, so "up" has negative y."""
    best, best_count = None, 0
    for _ in range(tries):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        offset = -normal @ sample[0]
        if offset < 0:  # the camera centre (origin) must be on the positive side
            normal, offset = -normal, -offset
        if normal[1] > -0.3:  # floor normals point up: clearly negative y for any hand-held tilt
            continue
        count = int((np.abs(points @ normal + offset) < tolerance * np.maximum(1, points[:, 2] / 2)).sum())
        if count > best_count:
            best, best_count = (normal, offset), count
    if best is None:
        return None
    normal, offset = best
    inliers = np.abs(points @ normal + offset) < tolerance * np.maximum(1, points[:, 2] / 2)
    centre = points[inliers].mean(0)
    _, _, vt = np.linalg.svd(points[inliers] - centre, full_matrices=False)
    normal = vt[2]
    offset = -normal @ centre
    if offset < 0:
        normal, offset = -normal, -offset
    agree = float((normals[inliers] @ normal).mean()) if len(normals) else 0.0
    return normal, float(offset), best_count / len(points), agree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--masks', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--step', type=int, default=6, help='sample every n-th frame')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(args.model).cuda().eval()
    capture = cv2.VideoCapture(args.video)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    people = load_people_masks(args.masks, count)
    rng = np.random.default_rng(0)
    records, clouds = [], {}
    index = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if index % args.step:
            continue
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float() / 255
        with torch.inference_mode():
            result = model.infer(image)
        points = result['points'].float().cpu().numpy()
        valid = result['mask'].cpu().numpy() & np.isfinite(points).all(-1)
        normals = result['normal'].float().cpu().numpy() if 'normal' in result else np.zeros_like(points)
        intrinsics = result['intrinsics'].float().cpu().numpy()
        focal_x, focal_y = float(intrinsics[0, 0] * width), float(intrinsics[1, 1] * height)
        person = cv2.resize(people[index].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        person = cv2.dilate(person.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
        background = valid & ~person
        record = dict(frame=index, time=index / fps, focal_x=focal_x, focal_y=focal_y,
                      background_share=float(background.mean()))
        # floor candidates: surfaces whose normal points up in the picture
        candidates = background & (normals[..., 1] < -0.3)
        pts, nrm = points[candidates], normals[candidates]
        if len(pts) > 3000:
            pick = rng.choice(len(pts), min(len(pts), 30000), replace=False)
            fit = fit_floor(pts[pick], nrm[pick], rng)
            if fit is not None:
                normal, distance, share, agree = fit
                up = normal  # unit vector from the floor towards the camera side, camera coordinates
                record.update(floor_normal=[float(v) for v in up], camera_height=distance,
                              floor_inlier_share=float(share), floor_normal_agreement=agree,
                              # tilt of the optical axis below the horizon, and roll about it
                              pitch_deg=float(np.degrees(np.arcsin(np.clip(-up[2], -1, 1)))),
                              roll_deg=float(np.degrees(np.arctan2(up[0], -up[1]))))
        records.append(record)
        if index % (args.step * 10) == 0:
            sub = background[::4, ::4]
            clouds[f'points_{index:05d}'] = points[::4, ::4][sub].astype(np.float32)
            clouds[f'colors_{index:05d}'] = rgb[::4, ::4][sub]
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in record.items()
                          if k != 'floor_normal'}), flush=True)
    capture.release()

    focal = np.array([r['focal_x'] for r in records])
    heights = np.array([r['camera_height'] for r in records if 'camera_height' in r])
    summary = dict(
        source='MoGe-2 ViT-L (normal), single-image metric geometry; estimates, not a calibration',
        frames=count, fps=fps, width=width, height=height, step=args.step,
        focal_px_median=float(np.median(focal)), focal_px_p10=float(np.percentile(focal, 10)),
        focal_px_p90=float(np.percentile(focal, 90)),
        focal_full_frame_mm=float(np.median(focal) / np.hypot(width, height) * np.hypot(36, 24)),
        horizontal_fov_deg=float(np.degrees(2 * np.arctan(width / 2 / np.median(focal)))),
        vertical_fov_deg=float(np.degrees(2 * np.arctan(height / 2 / np.median(focal)))),
        camera_height_median=float(np.median(heights)) if len(heights) else None,
        floor_found_frames=int(len(heights)), sampled_frames=len(records), samples=records)
    (out / 'geometry.json').write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out / 'points.npz', **clouds)
    print('focal px median', round(summary['focal_px_median'], 1), 'p10-p90',
          round(summary['focal_px_p10'], 1), round(summary['focal_px_p90'], 1),
          '| full-frame mm', round(summary['focal_full_frame_mm'], 1),
          '| floor found in', len(heights), 'of', len(records), flush=True)


if __name__ == '__main__':
    main()
