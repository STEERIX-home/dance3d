"""Check video for the common world: [source | both bodies laid over the source | fixed outside view].

The middle panel re-projects the two bodies from the common world through the common camera. If
the common world is right, each body lies on its own dancer. The right panel looks at the same
world from a fixed virtual camera, where wrong distances or facing directions would show.
Frames in which a person is not reliably seen are drawn pale and marked EST (estimated).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from paths import EXMR, SMPLX

sys.path.insert(0, str(EXMR))

from exmr.body_models.smplx_model import SmplxModel  # noqa: E402
from exmr.vis.renderer import Renderer  # noqa: E402

COLORS = {'A': (0.25, 0.45, 1.0), 'B': (1.0, 0.3, 0.25)}
PALE = {'A': (0.75, 0.82, 1.0), 'B': (1.0, 0.8, 0.78)}
PARAMS = ('global_orient', 'transl', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'betas')


def look_at(position, target):
    """World-to-camera rotation and translation of a camera (x right, y down, z forward), world y up."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    return R, -R @ position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--world', required=True, help='folder written by align_people.py')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    world = Path(args.world)
    people = np.load(world / 'people_world.npz')
    camera = np.load(world / 'camera_world.npz')
    labels = [str(label) for label in people['labels']]
    count = len(people[f'{labels[0]}_transl'])
    width, height = int(camera['width']) // 2, int(camera['height']) // 2
    K = torch.from_numpy(camera['K'].copy())
    K[:2] *= 0.5

    smplx = SmplxModel(SMPLX).cuda().eval()
    vertices = {}
    with torch.inference_mode():
        for label in labels:
            params = {k: torch.from_numpy(people[f'{label}_{k}']).cuda() for k in PARAMS}
            chunks = [smplx(**{k: v[i:i + 64] for k, v in params.items()}) for i in range(0, count, 64)]
            vertices[label] = torch.cat(chunks) * float(people[f'{label}_scale'])
    faces = smplx.faces
    overlay_renderer = Renderer(width, height, K, faces)
    world_K = torch.tensor([[1.1 * width, 0, width / 2], [0, 1.1 * width, height / 2], [0, 0, 1]])
    world_renderer = Renderer(width, height, world_K, faces)

    R_c2w = torch.from_numpy(camera['R_c2w']).cuda()
    centre = torch.from_numpy(camera['center']).cuda()
    reliable = {label: people[f'{label}_reliable'] for label in labels}
    pelvis = np.stack([people[f'{label}_joints'][:, 0] for label in labels])  # (2, L, 3)
    both = reliable[labels[0]] & reliable[labels[1]]
    middle = pelvis[:, both].mean((0, 1))
    front = camera['center'][camera['seen']].mean(0) - middle
    front[1] = 0
    front /= np.linalg.norm(front)
    side = np.array([front[2], 0, -front[0]])
    position = middle + 4.2 * (0.82 * front + 0.57 * side) + np.array([0, 2.4, 0])
    target = np.array([middle[0], 0.85, middle[2]])
    R_view, T_view = look_at(position.astype(np.float64), target.astype(np.float64))
    R_view_t = torch.from_numpy(R_view).float().cuda()
    T_view_t = torch.from_numpy(T_view).float().cuda()

    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(Path(args.output).with_suffix('.raw.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (width * 3, height))
    for index in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        source = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).cuda()
        in_camera = {label: (vertices[label][index] - centre[index]) @ R_c2w[index] for label in labels}
        order = sorted(labels, key=lambda label: -float(in_camera[label][:, 2].mean()))  # far one first
        image = (source.float() * 0.55).to(torch.uint8)
        for label in order:
            color = COLORS[label] if reliable[label][index] else PALE[label]
            image = overlay_renderer.render_mesh(in_camera[label], image, color)
        image = (0.7 * image.float() + 0.3 * source.float()).to(torch.uint8)

        ground = world_renderer.render_ground(R_view_t, T_view_t, 8.0, float(middle[0]), float(middle[2]))
        view = ground
        in_view = {label: vertices[label][index] @ R_view_t.T + T_view_t for label in labels}
        for label in sorted(labels, key=lambda label: -float(in_view[label][:, 2].mean())):
            color = COLORS[label] if reliable[label][index] else PALE[label]
            view = world_renderer.render_mesh(in_view[label], view, color)

        panels = [small, cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2BGR),
                  cv2.cvtColor(view.cpu().numpy(), cv2.COLOR_RGB2BGR)]
        for label in labels:
            if not reliable[label][index]:
                x = 12 + 150 * labels.index(label)
                cv2.putText(panels[1], f'{label}: EST', (x, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(panels[1], f'{label}: EST', (x, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(panels[2], f'{index / fps:5.2f}s', (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        writer.write(np.concatenate(panels, axis=1))
    writer.release()
    capture.release()
    print('written', Path(args.output).with_suffix('.raw.mp4'))


if __name__ == '__main__':
    main()
