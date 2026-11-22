"""Try two new checks of the two-dancer program on finished runs (nothing is changed):
  A  do the characters stand where the guide has its mannequins? (outline agreement per frame)
  B  does each dancer wear its own body colours, or the other one's? (as the head check, on the body below the head)

    duo_checks_probe.py JOB CAMERA
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '/srv/training/projects/duo-scene')
from qa_fast_motion import band_regions  # noqa: E402
from qa_pair_identity import APART, fit, judge, make_up  # noqa: E402
from scail_run import cover  # noqa: E402

job, camera = Path(sys.argv[1]), sys.argv[2]
scene = job / 'outputs/work/scene' / camera
run = job / 'outputs/work/gen' / camera
record = json.loads((run / 'run.json').read_text())
first, total, fps = record['first_frame'], record['frames'], record['fps']
width, height = record['width'], record['height']
fits, bodies = [], {label: np.full((total, 28), np.nan) for label in 'AB'}
for index in range(total):
    frame = cv2.imread(str(run / 'generated' / f'frame_{index + 1:04d}.png'))
    number = first + index
    mask = cover(cv2.imread(str(scene / 'mask' / f'frame_{number:04d}.png')), width, height, cv2.INTER_NEAREST)
    plate_file = scene / 'plate' / f'frame_{number:04d}.jpg'
    if plate_file.exists():
        plate = cover(cv2.imread(str(plate_file)), width, height, cv2.INTER_AREA)
        differs = (np.abs(frame.astype(int) - plate.astype(int)).mean(-1) > 30).astype(np.uint8)
        differs = cv2.morphologyEx(differs, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        character = cv2.resize(differs, (width // 4, height // 4), interpolation=cv2.INTER_AREA) > 0
        outline = cv2.resize((mask.min(-1) < 200).astype(np.uint8), (width // 4, height // 4), interpolation=cv2.INTER_AREA) > 0
        fits.append(float((character & outline).sum() / max(1, (character | outline).sum())))
    blue, red = mask[..., 0].astype(int), mask[..., 2].astype(int)
    masks = {'A': (blue - red) > 128, 'B': (red - blue) > 128}
    for label in 'AB':
        regions = band_regions(masks[label], frame.shape[:2], erode=9)
        if not regions:
            continue
        body = regions['chest'] | regions['hips'] | ((cv2.erode(masks[label].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~(regions['head'] > 0)).astype(np.uint8)
        if body.sum() >= 1500:
            bodies[label][index] = make_up(frame[body > 0])
fits = np.asarray(fits)
usual, apart, wrong, bad, spans = judge(bodies, fps)
print(f'{job.name} {camera}: {total} frames; outline agreement median {np.median(fits):.2f}, p5 {np.percentile(fits, 5):.2f}, min {fits.min():.2f}' if len(fits) else 'no plate')
print(f'  body colours: the two differ by {apart:.2f} (told apart above {APART}); looks like the other: A {int(wrong["A"].sum())}, B {int(wrong["B"].sum())}; '
      f'counted wrong {int(bad.sum())}; spans {[(round(a / fps, 2), round(b / fps, 2)) for a, b in spans]}')
step = int(fps / 2)
line = []
for start in range(0, total, step):
    part = slice(start, start + step)
    f = f'{np.median(fits[part]):.2f}' if len(fits) else ' -- '
    line.append(f'{start / fps:4.1f}s fit {f} A{"x" if wrong["A"][part].mean() > 0.5 else "."}B{"x" if wrong["B"][part].mean() > 0.5 else "."}')
for k in range(0, len(line), 6):
    print('   ' + ' | '.join(line[k:k + 6]))
np.savez(Path(sys.argv[3]) if len(sys.argv) > 3 else Path('/dev/null'), fits=fits, A=bodies['A'], B=bodies['B']) if len(sys.argv) > 3 else None
