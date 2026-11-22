"""Lay the generated characters over the rendered 3D set and add the original music.

Only the dancers' (grown) mask regions are taken from the generated frames. Everything else is the
rendered guide itself at full resolution, so the room is pixel-identical to the 3D render in every
frame and cannot drift. The characters are generated at a lower resolution and enlarged; the
written report keeps both numbers apart.

Outputs: --output MP4 (H.264 + AAC) and, with --compare, a side-by-side [3D guide | result] MP4.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from scail_run import cover, identity_mask


def encoder(path, width, height, fps, audio, duration):
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
               '-r', f'{fps:g}', '-i', '-']
    if audio:
        # the music goes on after the dance ends: fade it out instead of cutting it off
        command += ['-ss', '0', '-t', f'{duration:.4f}', '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
                    '-af', f'afade=t=out:st={max(0, duration - 0.5):.3f}:d=0.5', '-c:a', 'aac', '-b:a', '192k', '-shortest']
    command += ['-c:v', 'libx264', '-crf', '15', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--run', required=True, help='output folder of scail_run.py')
    parser.add_argument('--output', required=True)
    parser.add_argument('--audio', help='video or audio file whose sound starts with guide frame 1')
    parser.add_argument('--compare', help='also write a [guide | result] video here')
    parser.add_argument('--feather', type=float, default=1.5)
    args = parser.parse_args()
    scene, run = Path(args.scene), Path(args.run)
    record = json.loads((run / 'run.json').read_text())
    fps, total, first = record['fps'], record['frames'], record['first_frame']
    gen_w, gen_h = record['width'], record['height']
    files = sorted((run / 'generated').glob('frame_*.png'))
    if len(files) < total:
        total = len(files)
    sample = cv2.imread(str(scene / 'guide' / f'frame_{first:04d}.jpg'))
    height, width = sample.shape[:2]
    # the generated picture is the centre crop of the guide scaled to cover gen_w x gen_h
    scale = max(gen_w / width, gen_h / height)
    up_w, up_h = round(gen_w / scale), round(gen_h / scale)
    x0, y0 = (width - up_w) // 2, (height - up_h) // 2
    duration = total / fps
    audio_offset = (first - 1) / fps
    if args.audio and audio_offset > 0:
        raise NotImplementedError('audio for runs that do not start at frame 1')
    out = encoder(args.output, width, height, fps, args.audio, duration)
    compare = encoder(args.compare, width * 2, height, fps, args.audio, duration) if args.compare else None
    for index in range(total):
        frame = first + index
        guide = cv2.imread(str(scene / 'guide' / f'frame_{frame:04d}.jpg'))
        mask = cv2.imread(str(scene / 'mask' / f'frame_{frame:04d}.png'))
        generated = cv2.imread(str(files[index]))
        # Inside the grey bodies' own outline the generated picture is always used. In the band
        # around them (room for hair and clothes) it is used only where it really differs from the
        # rendered room, i.e. where the character is; where the model merely repainted the room,
        # the sharp 3D render stays. This avoids a soft halo around the characters.
        grown = identity_mask(mask, gen_w, gen_h, record['mask_grow_px']).min(-1) < 128
        body = identity_mask(mask, gen_w, gen_h, 0).min(-1) < 128
        small_guide = cover(guide, gen_w, gen_h, cv2.INTER_AREA).astype(np.float32)
        picture = generated.astype(np.float32)
        if (~grown).any():  # overall tone shift of the generated picture, measured on the untouched room
            picture = picture - np.median((picture - small_guide)[~grown], axis=0)
        difference = np.abs(cv2.GaussianBlur(picture, (0, 0), 1.0) - cv2.GaussianBlur(small_guide, (0, 0), 1.0)).max(-1)
        matte = np.where(body, 1.0, np.where(grown, np.clip((difference - 10) / 20, 0, 1), 0.0)).astype(np.float32)
        matte = cv2.GaussianBlur(matte, (0, 0), 0.8)
        matte = cv2.resize(matte, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        matte = cv2.GaussianBlur(matte, (0, 0), args.feather)[..., None]
        enlarged = cv2.resize(generated, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        result = guide.astype(np.float32)
        ys, xs = slice(max(0, y0), min(height, y0 + up_h)), slice(max(0, x0), min(width, x0 + up_w))
        sub = (slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0))
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
