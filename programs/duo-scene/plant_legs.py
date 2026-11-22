"""Legs that the camera has not seen yet: stand on the floor instead of following a guess.

At the start of a video a dancer may be framed from the waist up. The keypoint detector then puts
"knees" on the picture's bottom edge and "ankles" somewhere on the thighs, and the motion capture
folds the lower legs up to meet them (seen 11-13: half a second of standing on one leg during the
opening lunge, although the dance has both feet down). Nothing in the picture supports that.

For the opening span in which a leg has never been seen, this module replaces the leg's joint angles:
  - the foot starts where the first frame has it (standing) and ends where it is first really seen,
  - it stays planted on the floor at those two places and takes one quick, low step between them,
    at the moment the pelvis is half-way between its own two positions,
  - hip, knee and ankle angles follow from that (the pelvis itself is measured and not changed).
Only the opening span is treated: later, a leg hidden for a moment has been seen before and after,
and the motion capture's own continuation is kept there.
"""
import numpy as np
import torch

from geometry_utils import axis_angle_to_matrix

LEGS = dict(left=dict(joints=(1, 4, 7, 10), keypoints=(13, 15)), right=dict(joints=(2, 5, 8, 11), keypoints=(14, 16)))
EDGE = 14          # pixels: a keypoint this close to the picture border is a guess
STEP_SECONDS = 0.25
STEP_HEIGHT = 0.05
RAISED = 0.10      # a never-seen foot more than this above its standing height is treated as a wrong guess


def legs_seen(keypoints, width, height, fps, hold=0.2):
    """Per frame and leg: is the lower leg really in the picture? keypoints (L, 23, 3) in ExMR's
    COCO order. The knee and the ankle must be confident and inside the picture, and the ankle
    clearly below the knee. Returns {side: bool (L,)}, True only where that holds for `hold` seconds."""
    out = {}
    for side, leg in LEGS.items():
        knee, ankle = keypoints[:, leg['keypoints'][0]], keypoints[:, leg['keypoints'][1]]
        inside = lambda p: (p[:, 0] > EDGE) & (p[:, 0] < width - EDGE) & (p[:, 1] > EDGE) & (p[:, 1] < height - EDGE)
        seen = (knee[:, 2] > 0.5) & (ankle[:, 2] > 0.5) & inside(knee) & inside(ankle)
        hip = keypoints[:, 11 if side == 'left' else 12]
        thigh = np.linalg.norm(knee[:, :2] - hip[:, :2], axis=1)
        seen &= np.linalg.norm(ankle[:, :2] - knee[:, :2], axis=1) > 0.5 * thigh  # a shin of plausible length in the picture
        steady = np.ones(len(seen), dtype=bool)
        frames = max(1, int(round(hold * fps)))
        for shift in range(frames):
            steady[:len(seen) - shift] &= seen[shift:]
        steady[len(seen) - frames:] = False
        out[side] = steady
    return out


def feet_in_picture(keypoints, width, height):
    """Per frame: does the picture show both feet (both ankles found with confidence and not on the
    picture's border)? Where it does not, the lower legs are the motion capture's guess."""
    shown = np.ones(len(keypoints), dtype=bool)
    for leg in LEGS.values():
        ankle = keypoints[:, leg['keypoints'][1]]
        shown &= ((ankle[:, 2] > 0.5) & (ankle[:, 0] > EDGE) & (ankle[:, 0] < width - EDGE)
                  & (ankle[:, 1] > EDGE) & (ankle[:, 1] < height - EDGE))
    return shown


def rodrigues(vectors):
    angle = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = vectors / angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(vectors.shape[:-1] + (3, 3))
    angle = angle[..., None]
    return torch.eye(3, dtype=vectors.dtype) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def chain(pelvis_rotation, pelvis_position, offsets, hip, knee, ankle):
    """Forward kinematics of one leg: positions of knee, ankle and foot joint, and the ankle's rotation."""
    G_hip = pelvis_rotation @ rodrigues(hip)
    p_hip = pelvis_position + (pelvis_rotation @ offsets[0][:, None])[..., 0]
    p_knee = p_hip + (G_hip @ offsets[1][:, None])[..., 0]
    G_knee = G_hip @ rodrigues(knee)
    p_ankle = p_knee + (G_knee @ offsets[2][:, None])[..., 0]
    G_ankle = G_knee @ rodrigues(ankle)
    p_foot = p_ankle + (G_ankle @ offsets[3][:, None])[..., 0]
    return p_knee, p_ankle, p_foot, G_ankle


def plant_opening(params, joints, rest_joints, scale, seen, fps, report):
    """params: SMPL-X parameters (numpy, frames first) in the y-up world, changed in place for the
    opening span of each leg that has not been seen yet and was guessed off the floor.
    joints (L, 76, 3): world joints of these parameters (scaled). rest_joints (55+, 3): joints of the
    zero pose for this body shape (unscaled). seen: {side: bool (L,)} at the same frames."""
    count = len(joints)
    pelvis_rotation = torch.from_numpy(axis_angle_to_matrix(params['global_orient'])).double()
    pelvis_position = torch.from_numpy(joints[:, 0]).double()
    body = params['body_pose'].reshape(count, 21, 3)
    for side, leg in LEGS.items():
        hip_joint, knee_joint, ankle_joint, foot_joint = leg['joints']
        first = np.flatnonzero(seen[side])
        if not len(first) or first[0] < int(0.3 * fps):
            continue
        stop = int(first[0])  # first frame with the leg in the picture
        height = joints[:stop + 1, ankle_joint, 1]
        if height.max() - height[0] < RAISED:
            continue  # the guess stays on the floor: nothing to correct
        offsets = [torch.from_numpy((rest_joints[a] - rest_joints[b]) * scale).double() for a, b in
                   ((hip_joint, 0), (knee_joint, hip_joint), (ankle_joint, knee_joint), (foot_joint, ankle_joint))]
        local = {name: torch.from_numpy(body[:, joint - 1].copy()).double() for name, joint in
                 (('hip', hip_joint), ('knee', knee_joint), ('ankle', ankle_joint))}
        with torch.no_grad():
            _, ankle_now, _, ankle_rotation = chain(pelvis_rotation, pelvis_position, offsets, local['hip'], local['knee'], local['ankle'])
        check = float((ankle_now - torch.from_numpy(joints[:, ankle_joint])).norm(dim=-1).max())
        if check > 0.02:
            raise RuntimeError(f'Leg kinematics do not reproduce the body model ({check * 100:.1f} cm off)')
        # one step from the standing place to the first seen place, when the pelvis is half-way
        last = min(count - 1, stop + 2)  # two measured frames after the span keep the hand-over smooth
        frames = np.arange(last + 1)
        travel = (pelvis_position[stop] - pelvis_position[0]).numpy() * np.array([1.0, 0.0, 1.0])
        if np.linalg.norm(travel) > 0.1:
            progress = ((pelvis_position[:stop + 1].numpy() - pelvis_position[0].numpy()) @ travel) / (travel @ travel)
            crossing = np.flatnonzero(progress >= 0.5)
            middle = float(crossing[0]) if len(crossing) else stop / 2
        else:
            middle = stop / 2
        half = STEP_SECONDS * fps / 2
        middle = float(np.clip(middle, half, stop - half))
        phase = np.clip((frames - (middle - half)) / (2 * half), 0, 1)
        ease = phase * phase * (3 - 2 * phase)
        start, end = ankle_now[0].numpy(), ankle_now[stop].numpy()
        target = start + (end - start) * ease[:, None]
        target[:, 1] += STEP_HEIGHT * np.sin(np.pi * phase)
        target = torch.from_numpy(target)
        # the foot's direction turns from its first to its first-seen direction during the step
        from scipy.spatial.transform import Rotation, Slerp

        turn = Slerp([0, 1], Rotation.from_matrix(ankle_rotation[[0, stop]].numpy()))(ease).as_matrix()
        foot_target = target + (torch.from_numpy(turn) @ offsets[3][:, None])[..., 0]
        # start from the two known leg poses, blended over the step
        blend = torch.from_numpy(ease)[:, None]
        variables = {}
        for name, value in local.items():
            initial = value[0] * (1 - blend) + value[stop] * blend
            initial[stop:last + 1] = value[stop:last + 1]
            variables[name] = initial.clone().requires_grad_(True)
        initial = {name: value.detach().clone() for name, value in variables.items()}
        fixed = torch.zeros(last + 1, 1, dtype=torch.double)
        fixed[stop:] = 1.0  # measured frames are not changed
        optimizer = torch.optim.Adam(variables.values(), lr=0.02)
        for _ in range(800):
            current = {name: fixed * initial[name] + (1 - fixed) * variables[name] for name in variables}
            _, p_ankle, p_foot, _ = chain(pelvis_rotation[:last + 1], pelvis_position[:last + 1], offsets,
                                          current['hip'], current['knee'], current['ankle'])
            reach = ((p_ankle - target) ** 2).sum(-1) + ((p_foot - foot_target) ** 2).sum(-1)
            hinge = (current['knee'][:, 1:] ** 2).sum()          # a knee only bends
            near = sum(((current[name] - initial[name]) ** 2).sum() for name in ('hip', 'ankle'))
            smooth = sum(((value[2:] - 2 * value[1:-1] + value[:-2]) ** 2).sum() for value in current.values())
            loss = 1e4 * (reach * (1 - fixed[:, 0])).sum() + 1.0 * hinge + 0.05 * near + 20.0 * smooth
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            current = {name: fixed * initial[name] + (1 - fixed) * variables[name] for name in variables}
            _, p_ankle, _, _ = chain(pelvis_rotation[:last + 1], pelvis_position[:last + 1], offsets,
                                     current['hip'], current['knee'], current['ankle'])
            miss = (p_ankle - target).norm(dim=-1)[:stop]
        for name, joint in (('hip', hip_joint), ('knee', knee_joint), ('ankle', ankle_joint)):
            body[:stop, joint - 1] = current[name][:stop].numpy()
        report.append(dict(leg=side, seconds=[0.0, round(stop / fps, 3)], step_at=round(middle / fps, 3),
                           foot_was_up_to_m=round(float(height.max()), 3), standing_height_m=round(float(height[0]), 3),
                           foot_misses_its_place_cm=dict(median=round(float(miss.median()) * 100, 2), most=round(float(miss.max()) * 100, 2))))
    params['body_pose'] = body.reshape(count, 63)
    return params
