"""Lay two single-dancer runs over the rendered 3D set (far dancer first, near dancer on top) and
add the original music.

Each run of `scail_run.py --only LABEL` replaced one dancer and left the other as the grey body of
the guide. Here every dancer's pixels are taken from that dancer's own run only, so the two
characters cannot be mixed up. Everything else is the sharp 3D render itself.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from composite import encoder
from scail_run import cover, identity_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--inputs', required=True, help='folder of export_scene_inputs.py (dancer positions)')
    parser.add_argument('--run', action='append', required=True, help='LABEL=output folder of scail_run.py --only LABEL')
    parser.add_argument('--output', required=True)
    parser.add_argument('--audio')
    parser.add_argument('--compare')
    parser.add_argument('--feather', type=float, default=1.5)
    args = parser.parse_args()
    scene = Path(args.scene)
    runs = {label: Path(folder) for label, folder in (entry.split('=', 1) for entry in args.run)}
    records = {label: json.loads((folder / 'run.json').read_text()) for label, folder in runs.items()}
    first = next(iter(records.values()))
    for record in records.values():
        if any(record[key] != first[key] for key in ('width', 'height', 'frames', 'first_frame', 'fps')):
            raise ValueError('The single-dancer runs differ in size or length')
    fps, total, start = first['fps'], first['frames'], first['first_frame']
    gen_w, gen_h = first['width'], first['height']
    positions = {person['label']: np.asarray(person['pelvis'])
                 for person in json.loads((Path(args.inputs) / 'scene_inputs.json').read_text())['people']}
    camera = [np.asarray(row['location']) for row in json.loads((scene / 'camera-path.json').read_text())]
    sample = cv2.imread(str(scene / 'guide' / f'frame_{start:04d}.jpg'))
    height, width = sample.shape[:2]
    scale = max(gen_w / width, gen_h / height)
    up_w, up_h = round(gen_w / scale), round(gen_h / scale)
    x0, y0 = (width - up_w) // 2, (height - up_h) // 2
    ys, xs = slice(max(0, y0), min(height, y0 + up_h)), slice(max(0, x0), min(width, x0 + up_w))
    sub = (slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0))
    duration = total / fps
    out = encoder(args.output, width, height, fps, args.audio, duration)
    compare = encoder(args.compare, width * 2, height, fps, args.audio, duration) if args.compare else None
    for index in range(total):
        frame = start + index
        guide = cv2.imread(str(scene / 'guide' / f'frame_{frame:04d}.jpg'))
        mask = cv2.imread(str(scene / 'mask' / f'frame_{frame:04d}.png'))
        small_guide = cover(guide, gen_w, gen_h, cv2.INTER_AREA).astype(np.float32)
        blurred_guide = cv2.GaussianBlur(small_guide, (0, 0), 1.0)
        result = guide.astype(np.float32)
        far_first = sorted(runs, key=lambda label: -np.linalg.norm(positions[label][frame - 1] - camera[frame - 1]))
        for label in far_first:
            picture = cv2.imread(str(runs[label] / 'generated' / f'frame_{index + 1:04d}.png')).astype(np.float32)
            grown = identity_mask(mask, gen_w, gen_h, records[label]['mask_grow_px'], only=label).min(-1) < 128
            body = identity_mask(mask, gen_w, gen_h, 0, only=label).min(-1) < 128
            if not grown.any():
                continue
            toned = picture - np.median((picture - small_guide)[~grown], axis=0) if (~grown).any() else picture
            difference = np.abs(cv2.GaussianBlur(toned, (0, 0), 1.0) - blurred_guide).max(-1)
            matte = np.where(body, 1.0, np.where(grown, np.clip((difference - 10) / 20, 0, 1), 0.0)).astype(np.float32)
            matte = cv2.GaussianBlur(matte, (0, 0), 0.8)
            matte = cv2.resize(matte, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
            matte = cv2.GaussianBlur(matte, (0, 0), args.feather)[..., None]
            enlarged = cv2.resize(picture, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
            result[ys, xs] = enlarged[sub] * matte[sub] + result[ys, xs] * (1 - matte[sub])
        result = np.clip(result, 0, 255).astype(np.uint8)
        out.stdin.write(result.tobytes())
        if compare:
            compare.stdin.write(np.concatenate([guide, result], axis=1).tobytes())
    for process in filter(None, (out, compare)):
        process.stdin.close()
        if process.wait():
            raise RuntimeError('ffmpeg failed')
    print(f'{total} frames, {duration:.2f} s -> {args.output}')


if __name__ == '__main__':
    main()
