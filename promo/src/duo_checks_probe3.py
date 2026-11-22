"""Body colours of each dancer against its own usual ones and the other's (nothing is changed).

    duo_checks_probe3.py JOB CAMERA
margin = fit to own usual body colours minus fit to the other dancer's; clearly below 0: wears the other's colours.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '/srv/training/projects/duo-scene')
from qa_fast_motion import band_regions  # noqa: E402
from qa_pair_identity import make_up  # noqa: E402
from scail_run import cover  # noqa: E402

job, camera = Path(sys.argv[1]), sys.argv[2]
scene = job / 'outputs/work/scene' / camera
run = job / 'outputs/work/gen' / camera
record = json.loads((run / 'run.json').read_text())
first, total, fps = record['first_frame'], record['frames'], record['fps']
width, height = record['width'], record['height']
body = {label: np.full((total, 28), np.nan) for label in 'AB'}
area = {label: np.zeros(total) for label in 'AB'}
for index in range(total):
    frame = cv2.imread(str(run / 'generated' / f'frame_{index + 1:04d}.png'))
    mask = cover(cv2.imread(str(scene / 'mask' / f'frame_{first + index:04d}.png')), width, height, cv2.INTER_NEAREST)
    blue, red = mask[..., 0].astype(int), mask[..., 2].astype(int)
    for label, region in (('A', (blue - red) > 128), ('B', (red - blue) > 128)):
        area[label][index] = region.mean()
        bands = band_regions(region, frame.shape[:2], erode=9)
        if not bands:
            continue
        below = (cv2.erode(region.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~(bands['head'] > 0)
        rows = np.flatnonzero(region.any(1))
        below[:rows[0] + int(0.22 * (rows[-1] - rows[0] + 1))] = False
        if below.sum() >= 1500:
            body[label][index] = make_up(frame[below])
clear = {label: area[label] >= 0.3 * np.median(area[label][area[label] > 0]) for label in 'AB'}
both = clear['A'] & clear['B'] & np.isfinite(body['A'][:, 0]) & np.isfinite(body['B'][:, 0])
usual = {label: np.nanmedian(body[label][both], axis=0) for label in 'AB'}
usual = {label: v / max(v.sum(), 1e-9) for label, v in usual.items()}
apart = 1 - float(np.minimum(usual['A'], usual['B']).sum())
margin = {}
for label, other in (('A', 'B'), ('B', 'A')):
    margin[label] = np.minimum(body[label], usual[label]).sum(1) - np.minimum(body[label], usual[other]).sum(1)
print(f'{job.name} {camera}: usual body colours differ by {apart:.2f}; frames with both clearly seen {int(both.sum())}/{total}')
for label in 'AB':
    m = margin[label]
    print(f'  {label}: margin p1 {np.nanpercentile(m, 1):+.2f} p5 {np.nanpercentile(m, 5):+.2f} median {np.nanmedian(m):+.2f}; below -0.05: {int(np.nansum(m < -0.05))}, below -0.10: {int(np.nansum(m < -0.10))}, below -0.15: {int(np.nansum(m < -0.15))}')
step = 6
for label in 'AB':
    cells = []
    for start in range(0, total, step):
        m = margin[label][start:start + step]
        cells.append('  . ' if not np.isfinite(m).any() else f'{np.nanmedian(m):+.1f}'.replace('+0.', '+.').replace('-0.', '-.'))
    print(f'  {label} per quarter second: ' + ' '.join(cells))
np.savez(f'/tmp/claude-1000/-home-steerix-train-server/9e0c09cb-6e1d-4c41-b979-5746bf59c484/scratchpad/look/body_{job.name[-4:]}_{camera}.npz', A=margin['A'], B=margin['B'], clearA=clear['A'], clearB=clear['B'])
