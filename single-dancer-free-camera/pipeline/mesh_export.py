"""ExMR's numeric reconstruction -> animated vertices, using the official SMPL-X body."""
import argparse
from pathlib import Path

import numpy as np
import torch
from exmr.body_models.smplx_model import SmplxModel
from exmr.export import resample_motion


def export(source, model, destination, frames):
    with np.load(source, allow_pickle=False) as data:
        keys = ('body_pose', 'betas', 'global_orient', 'transl', 'left_hand_pose', 'right_hand_pose')
        params = {key: torch.from_numpy(np.array(data['smplx_global_' + key], dtype=np.float32))
                  for key in keys}
        fps = float(data['fps'])
    count = len(params['transl'])
    dimensions = dict(body_pose=63, betas=10, global_orient=3, transl=3,
                      left_hand_pose=45, right_hand_pose=45)
    if not 2 <= count <= 1800 or not 1 <= fps <= 240:
        raise ValueError('지원하지 않는 모션 길이 또는 프레임 속도입니다.')
    for key, value in params.items():
        if value.shape != (count, dimensions[key]) or not torch.isfinite(value).all():
            raise ValueError(f'잘못된 ExMR 배열: {key}')
    if abs(fps - 24) > 0.001:
        params = resample_motion(params, fps, 24)
    if len(params['transl']) < frames:
        raise ValueError(f'3D 동작이 짧습니다: {len(params["transl"])} < {frames} 프레임')
    smplx = SmplxModel(model).eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, frames, 16):
            chunks.append(smplx(**{key: value[start:min(frames, start+16)]
                                   for key, value in params.items()}).numpy())
    vertices = np.concatenate(chunks)
    # ExMR: y up. Blender: z up, preserving handedness and meters.
    vertices = vertices[..., [0, 2, 1]]
    vertices[..., 1] *= -1
    if not np.isfinite(vertices).all():
        raise ValueError('유효하지 않은 3D 좌표입니다.')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, vertices=vertices, faces=smplx.faces.numpy(), fps=np.array(24),
             provenance=np.array('ExMR + official SMPL-X; z-up; meters'))
    print(f'Animated mesh: {frames} frames, {vertices.shape[1]} vertices -> {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', required=True, type=int)
    args = parser.parse_args()
    export(args.source, args.model, args.output, args.frames)
