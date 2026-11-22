"""How much of the dancers' quick motion (shakes, hits) do the reconstructed bodies keep?

The bodies are projected into the source camera and their band velocities (head, chest, hips) are
compared with the optical flow of the source video in the same bands (qa_fast_motion.py).

    qa_motion_fidelity.py --source traces_source.json --masks tracks/masks.npz --mocap MOCAP_DIR
    qa_motion_fidelity.py --source traces_source.json --masks tracks/masks.npz --world WORLD_DIR

--mocap checks the camera-frame result of the motion capture, --world the final common world seen
through the common camera (what the 3D scene is built from).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from paths import EXMR, SMPLX
from qa_fast_motion import compare, load_masks, mesh_traces, parse_windows

sys.path.insert(0, str(EXMR))

from exmr.body_models.smplx_model import SmplxModel  # noqa: E402

PARAMS = ('global_orient', 'transl', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'betas')


def body_vertices(smplx, params, step=48):
    count = len(params['transl'])
    with torch.inference_mode():
        return np.concatenate([smplx(**{k: torch.as_tensor(v[i:i + step]).float() for k, v in params.items()}).numpy()
                               for i in range(0, count, step)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='traces of the source video (qa_fast_motion.py video)')
    parser.add_argument('--masks', required=True)
    parser.add_argument('--mocap')
    parser.add_argument('--world')
    parser.add_argument('--windows', default='0:1.4,1.4:3.4,3.4:5.6,5.6:10,10:16,16:22.3')
    parser.add_argument('--cut', type=float, default=1.5)
    parser.add_argument('--same-camera', action='store_true',
                        help='--world: compare the bodies as the reconstructed source camera sees them with the source '
                             'picture as it is (default: the motion in the room on both sides)')
    parser.add_argument('--output')
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text())
    masks = load_masks(args.masks)
    smplx = SmplxModel(SMPLX).eval()
    faces = smplx.faces.numpy()
    vertices, following = {}, None
    if args.mocap:
        mocap = Path(args.mocap)
        info = json.loads((mocap / 'people.json').read_text())
        fps, size = info['fps'], (info['width'], info['height'])
        K = torch.load(mocap / 'camera.pt', map_location='cpu', weights_only=False)['K'].numpy().astype(np.float64)
        for person in info['people']:
            result = torch.load(mocap / f'person_{person["label"]}/results.pt', map_location='cpu', weights_only=False)
            vertices[person['label']] = body_vertices(smplx, result['smplx_incam']).astype(np.float64)
    else:
        world = Path(args.world)
        people, camera = np.load(world / 'people_world.npz'), np.load(world / 'camera_world.npz')
        fps, size, K = float(people['fps']), (int(camera['width']), int(camera['height'])), camera['K'].astype(np.float64)
        R, centre = camera['R_c2w'].astype(np.float64), camera['center'].astype(np.float64)
        following = {}
        for label in map(str, people['labels']):
            points = body_vertices(smplx, {k: people[f'{label}_{k}'] for k in PARAMS}) * float(people[f'{label}_scale'])
            vertices[label] = np.einsum('tji,tnj->tni', R, points - centre[:, None])
            # the next frame's body through this frame's camera: motion in the room only
            ahead = np.concatenate([points[1:], points[-1:]])
            following[label] = np.einsum('tji,tnj->tni', R, ahead - centre[:, None])
    if args.same_camera:
        following = None
    count = len(next(iter(vertices.values())))
    first, last = int(round(source['start'] * fps)), min(count, int(round(source['stop'] * fps)))

    def masks_of_frame(index):
        return {label: stack[min(first + index, len(stack) - 1)] for label, stack in masks.items()}

    traces = mesh_traces({label: points[first:last] for label, points in vertices.items()}, faces, K, masks_of_frame, size,
                         None if following is None else {label: points[first:last] for label, points in following.items()})
    test = dict(fps=fps, start=source['start'], traces={name: values.tolist() for name, values in traces.items()})
    # camera-frame result: same camera on both sides. World: motion in the room on both sides (the source's
    # background motion is taken away, the bodies are seen by a camera that stands still for the step)
    rows = compare(source, test, parse_windows(args.windows), args.cut, minus_camera=(following is not None, False))
    if args.output:
        Path(args.output).write_text(json.dumps(dict(rows=rows, test=test), allow_nan=True))


if __name__ == '__main__':
    main()
