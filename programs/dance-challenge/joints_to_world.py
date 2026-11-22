"""A dance-joints result (results.npz) -> the files export_scene_inputs.py of the two-dancer program
reads, with this one dancer as dancer A.

The joints program keeps the SMPL-X motion in a y-up world with the floor at height 0, in metres,
which is the convention of align_people.py, so the motion is passed on as it is. That the stored
parameters really give the stored joints is checked here (one scale for the whole video, and what
is left over must be under 3 cm), so a file of another kind is refused instead of giving a wrong body.

Outputs in --output:
  people_world.npz, camera_world.npz   for export_scene_inputs.py
  motion_facts.json                    where the legs and the whole body were estimated, and which
                                       way the body faces, per source frame (for the cut plan); the
                                       mesh points that move with the knees and the top of the head
                                       (so the cameras can tell where the picture cuts the body)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from paths import EXMR, SMPLX

sys.path.insert(0, str(EXMR))

from exmr.body_models.smplx_model import SmplxModel  # noqa: E402

PARAMS = ('global_orient', 'transl', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'betas')
MEASURED = 0  # the joints program's code for a frame in which the dancer was seen (align_people.MEASURED)
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 16, 17, 1, 2
LEFT_KNEE, RIGHT_KNEE, NECK, HEAD = 4, 5, 12, 15


def spans(flags, fps, least):
    """[[start, end] seconds] of the runs of True that last at least `least` seconds."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(flags, dtype=int), [0]])))
    return [[round(a / fps, 3), round(b / fps, 3)] for a, b in zip(edges[::2].tolist(), edges[1::2].tolist()) if b - a >= least * fps]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--info', help='info.json of prepare_challenge.py (picture size of the source video)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.results)
    fps = float(data['fps'])
    count = len(data['smplx_global_transl'])
    params = {key: data[f'smplx_global_{key}'].astype(np.float32) for key in PARAMS}

    model = SmplxModel(SMPLX).eval()
    pick = np.unique(np.linspace(0, count - 1, min(count, 32)).astype(int))
    with torch.inference_mode():
        sampled = {key: torch.from_numpy(value[pick]) for key, value in params.items()}
        all_joints = model.joints(**sampled).numpy().astype(np.float64)
        vertices = model(**sampled).numpy().astype(np.float64)
    joints = all_joints[:, :22]
    given = data['joints_global'][pick, :22].astype(np.float64)
    scale = float((joints * given).sum() / (joints * joints).sum())
    left_over = float(np.linalg.norm(scale * joints - given, axis=-1).max())
    if not 0.5 < scale < 2.0 or left_over > 0.03:
        raise ValueError(f'the stored motion does not give the stored joints (scale {scale:.3f}, {left_over * 100:.1f} cm off)')

    # mesh points (vertex numbers, the same in the exported motion) that move with the knees: the 24 nearest
    # each knee over the sampled frames; and the top of the head: the point furthest along neck -> head
    body_points, knee_off = {}, {}
    for name, index in (('left_knee', LEFT_KNEE), ('right_knee', RIGHT_KNEE)):
        near = np.argsort(np.linalg.norm(vertices - all_joints[:, index:index + 1], axis=2).mean(0))[:24]
        body_points[name] = near.tolist()
        knee_off[name] = round(float(np.linalg.norm(vertices[:, near].mean(1) - all_joints[:, index], axis=1).max()) * scale * 100, 1)
    up = all_joints[:, HEAD] - all_joints[:, NECK]
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    body_points['head_top'] = int(np.argmax(np.einsum('fvk,fk->fv', vertices - all_joints[:, HEAD:HEAD + 1], up).mean(0)))

    reliable = data['reliable'].astype(bool)
    estimated = (data['estimated'] != MEASURED) if 'estimated' in data.files else ~reliable
    legs = data['legs_estimated'].astype(bool) if 'legs_estimated' in data.files else np.zeros(count, dtype=bool)
    np.savez(out / 'people_world.npz', labels=np.array(['A']), fps=np.float64(fps),
             **{f'A_{key}': value for key, value in params.items()},
             A_scale=np.float64(scale), A_reliable=reliable, A_estimated=estimated.astype(np.uint8))

    K = np.median(data['K_fullimg'], axis=0) if 'K_fullimg' in data.files else None
    info = json.loads(Path(args.info).read_text()) if args.info else {}
    width = int(info.get('width') or round(2 * K[0, 2]))
    height = int(info.get('height') or round(2 * K[1, 2]))
    np.savez(out / 'camera_world.npz', R_c2w=data['camera_R_c2w'].astype(np.float32), center=data['camera_center'].astype(np.float32),
             K=K.astype(np.float32), fps=np.float64(fps), width=np.int64(width), height=np.int64(height))

    # which way the body faces: shoulders and hips, across the body, turned about "up" (y)
    allj = data['joints_global'].astype(np.float64)
    across = (allj[:, LEFT_SHOULDER] - allj[:, RIGHT_SHOULDER]) + (allj[:, LEFT_HIP] - allj[:, RIGHT_HIP])
    forward = np.cross(across, np.array([0.0, 1.0, 0.0]))
    facing = np.degrees(np.arctan2(forward[:, 0], forward[:, 2]))  # 0 = +z of the world
    facts = dict(fps=fps, frames=count, seconds=round(count / fps, 3), scale=round(scale, 4), left_over_m=round(left_over, 4),
                 legs_estimated=legs.tolist(), legs_estimated_spans=spans(legs, fps, 0.0),
                 body_estimated_spans=spans(estimated, fps, 0.0), reliable_frames=int(reliable.sum()),
                 facing_deg_y_up=np.round(facing, 2).tolist(), body_points=body_points, knee_points_off_cm=knee_off,
                 note='facing: direction the chest points, angle about the y-up axis from +z towards +x, world of results.npz; '
                      'body_points: vertex numbers of the body mesh (knees: mean of the points; knee_points_off_cm: '
                      'how far that mean is from the knee joint at most)')
    (out / 'motion_facts.json').write_text(json.dumps(facts))
    print(f'{count} frames at {fps:g} fps; scale {scale:.4f}, joints off by at most {left_over * 100:.2f} cm; '
          f'legs estimated {facts["legs_estimated_spans"]}; body estimated {facts["body_estimated_spans"]}')


if __name__ == '__main__':
    main()
