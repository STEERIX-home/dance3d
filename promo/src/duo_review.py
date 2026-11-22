"""Sheets for looking at a finished two-dancer video over its whole length (nothing is judged here: they are looked at).

    duo_review.py JOB RESULT_FOLDER OUT_FOLDER [STEP_FRAMES]

Per camera of the run:
  pairs_<camera>_N.jpg   every STEP frames (default 12 = half a second): [3D guide | finished video]
  close_<camera>_N.jpg   the same moments, cut out around the two dancers (who wears what is seen best here)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

job, result, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
step = int(sys.argv[4]) if len(sys.argv) > 4 else 12
out.mkdir(parents=True, exist_ok=True)
names = {'original': '1_원본과_같은_카메라.mp4', 'orbit': '2_360도_회전.mp4', 'dynamic': '3_다이내믹_카메라.mp4', 'front': '4_정면.mp4'}
for camera in json.loads((job / 'outputs/options.json').read_text())['cameras']:
    scene = job / 'outputs/work/scene' / camera
    cap = cv2.VideoCapture(str(result / names[camera]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    pairs, closes, number = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        number += 1
        if (number - 1) % step:
            continue
        guide = cv2.imread(str(scene / 'pair' / f'frame_{number:04d}.jpg'))
        pair = np.concatenate([cv2.resize(guide, (162, 288)), cv2.resize(frame, (162, 288))], 1)
        cv2.putText(pair, f'{(number - 1) / fps:.1f}s', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        pairs.append(pair)
        mask = cv2.imread(str(scene / 'mask' / f'frame_{number:04d}.png'))
        ys, xs = np.nonzero(mask.min(2) < 128)
        if len(xs) < 100:
            close = cv2.resize(frame, (240, 400))
        else:
            cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
            half = int(max(xs.max() - xs.min(), (ys.max() - ys.min()) * 0.6) / 2 * 1.15) + 20
            tall = int(half / 0.6)
            part = frame[max(0, cy - tall):min(frame.shape[0], cy + tall), max(0, cx - half):min(frame.shape[1], cx + half)]
            close = cv2.resize(part, (240, 400))
        cv2.putText(close, f'{(number - 1) / fps:.1f}s', (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        closes.append(close)
    for kind, tiles, per, rows in (('pairs', pairs, 8, 3), ('close', closes, 12, 2)):
        for index in range(0, len(tiles), per * rows):
            chunk = tiles[index:index + per * rows]
            while len(chunk) % per:
                chunk.append(np.zeros_like(tiles[0]))
            sheet = np.concatenate([np.concatenate(chunk[j:j + per], 1) for j in range(0, len(chunk), per)], 0)
            cv2.imwrite(str(out / f'{kind}_{camera}_{index // (per * rows) + 1}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
    print(camera, number, 'frames,', len(pairs), 'moments')
