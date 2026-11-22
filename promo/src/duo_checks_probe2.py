"""More candidates for a check that catches "the wrong figure stands where a mannequin was" (nothing is changed):
per dancer and frame  filled = share of the dancer's own mask in which the picture differs from the empty room,
                      own    = how well the colours around the dancer's head fit that dancer's usual head colours.

    duo_checks_probe2.py JOB CAMERA [FIRST_BAD LAST_BAD]     (frame numbers of a stretch known to be wrong)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '/srv/training/projects/duo-scene')
from qa_pair_identity import frame_head_colours, judge  # noqa: E402
from scail_run import cover  # noqa: E402

job, camera = Path(sys.argv[1]), sys.argv[2]
known = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else None
scene = job / 'outputs/work/scene' / camera
run = job / 'outputs/work/gen' / camera
record = json.loads((run / 'run.json').read_text())
first, total, fps = record['first_frame'], record['frames'], record['fps']
width, height = record['width'], record['height']
filled = {label: np.full(total, np.nan) for label in 'AB'}
share = {label: np.full(total, np.nan) for label in 'AB'}
heads = {label: np.full((total, 28), np.nan) for label in 'AB'}
for index in range(total):
    frame = cv2.imread(str(run / 'generated' / f'frame_{index + 1:04d}.png'))
    number = first + index
    raw = cv2.imread(str(scene / 'mask' / f'frame_{number:04d}.png'))
    mask = cover(raw, width, height, cv2.INTER_NEAREST)
    plate = cover(cv2.imread(str(scene / 'plate' / f'frame_{number:04d}.jpg')), width, height, cv2.INTER_AREA)
    differs = np.abs(frame.astype(int) - plate.astype(int)).mean(-1) > 30
    blue, red = mask[..., 0].astype(int), mask[..., 2].astype(int)
    for label, region in (('A', (blue - red) > 128), ('B', (red - blue) > 128)):
        region = cv2.erode(region.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        share[label][index] = region.mean()
        if region.sum() >= 2000:
            filled[label][index] = differs[region].mean()
    for label, value in frame_head_colours(frame, raw).items():
        heads[label][index] = value
usual, apart, wrong, bad, spans = judge(heads, fps)
own = {label: np.minimum(heads[label], usual[label]).sum(axis=1) for label in 'AB'}
print(f'{job.name} {camera}: heads differ by {apart:.2f}')
for label in 'AB':
    f, o = filled[label], own[label]
    ok = np.isfinite(f)
    print(f'  {label}: filled  p1 {np.nanpercentile(f, 1):.2f}  p5 {np.nanpercentile(f, 5):.2f}  median {np.nanmedian(f):.2f}   |   head own-fit  p1 {np.nanpercentile(o, 1):.2f}  p5 {np.nanpercentile(o, 5):.2f}  median {np.nanmedian(o):.2f}')
    if known:
        part = slice(known[0] - first, known[1] - first + 1)
        print(f'     in the known bad stretch: filled median {np.nanmedian(f[part]):.2f} (max {np.nanmax(f[part]):.2f}), own-fit median {np.nanmedian(o[part]):.2f} (max {np.nanmax(o[part]):.2f}), mask share {np.nanmedian(share[label][part]):.3f}')
    # runs of low values: how long is the longest stretch below some levels?
    for name, values, levels in (('filled', f, (0.5, 0.6, 0.7)), ('own-fit', o, (0.3, 0.4, 0.5))):
        out = []
        for level in levels:
            low = np.where(np.isfinite(values), values < level, False)
            longest, current, at, best_at = 0, 0, 0, 0
            for i, flag in enumerate(list(low) + [False]):
                if flag:
                    if current == 0:
                        at = i
                    current += 1
                else:
                    if current > longest:
                        longest, best_at = current, at
                    current = 0
            out.append(f'<{level}: {int(low.sum())} frames, longest run {longest} from frame {best_at + first}')
        print(f'     {name}: ' + '; '.join(out))
