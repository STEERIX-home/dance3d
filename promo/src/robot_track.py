"""Follow the robot's two eyes through the footage: where they are, how far apart, and how the line between them tilts.

    robot_track.py VIDEO OUT.json [--debug FOLDER]

The eyes are two round lenses in the white head; they mirror the room, so they are not simply dark. They are found as
the two round holes in the bright head: everything that is not bright but is enclosed by bright. The thin bar between
the lenses is removed by opening the hole with a disc wider than the bar.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('output')
parser.add_argument('--debug')
parser.add_argument('--left-edge', type=int, default=600, help='the robot is right of this column (the tablet is left of it)')
args = parser.parse_args()

times = [float(v.strip(',')) for v in subprocess.run(
    ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', args.video],
    capture_output=True, text=True, check=True).stdout.split()]
cap = cv2.VideoCapture(args.video)
rows, previous, number = [], None, 0
debug = Path(args.debug) if args.debug else None
if debug:
    debug.mkdir(parents=True, exist_ok=True)
bar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
speck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
while True:
    ok, frame = cap.read()
    if not ok:
        break
    number += 1
    part = frame[:, args.left_edge:]
    grey = cv2.GaussianBlur(cv2.cvtColor(part, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    bright = grey > 195
    holes = (binary_fill_holes(bright) & ~bright).astype(np.uint8)
    holes = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, speck)   # bright reflections inside a lens
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, bar)      # the bar between the lenses
    count, labels, stats, centres = cv2.connectedComponentsWithStats(holes, 8)
    blobs = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if 700 < area < 6000 and 0.4 < w / h < 1.6 and area > 0.6 * w * h:
            blobs.append((centres[index][0] + args.left_edge, centres[index][1], float(area), float(w), float(h)))
    best = None
    for a in blobs:
        for b in blobs:
            if a[0] >= b[0]:
                continue
            gap = np.hypot(b[0] - a[0], b[1] - a[1])
            size = max(a[4], b[4])  # a lens seen from the side gets narrower, not lower
            if not (1.0 * size < gap < 3.2 * size and 0.4 < a[2] / b[2] < 2.5 and abs(b[1] - a[1]) < 0.8 * gap):
                continue
            cost = 0.0 if previous is None else np.hypot((a[0] + b[0]) / 2 - previous[0], (a[1] + b[1]) / 2 - previous[1])
            if best is None or cost < best[0]:
                best = (cost, a, b)
    row = dict(frame=number, time=times[number - 1] if number <= len(times) else None, found=best is not None)
    if best:
        _, a, b = best
        previous = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        row.update(left=[round(a[0], 2), round(a[1], 2)], right=[round(b[0], 2), round(b[1], 2)],
                   centre=[round(previous[0], 2), round(previous[1], 2)], gap=round(float(np.hypot(b[0] - a[0], b[1] - a[1])), 2),
                   # tilt: positive = the lens on the picture's right is higher (picture y grows downwards)
                   tilt_deg=round(float(np.degrees(np.arctan2(-(b[1] - a[1]), b[0] - a[0]))), 3),
                   area=[a[2], b[2]], width=[a[3], b[3]], height=[a[4], b[4]])
    rows.append(row)
    if debug and number % 15 == 1:
        small = frame.copy()
        if best:
            for p in (best[1], best[2]):
                cv2.circle(small, (int(p[0]), int(p[1])), 6, (0, 255, 0), 2)
        cv2.imwrite(str(debug / f'{number:05d}.jpg'), cv2.resize(small, None, fx=0.5, fy=0.5))
Path(args.output).write_text(json.dumps(dict(frames=number, rows=rows)))
found = sum(r['found'] for r in rows)
print(f'{number} frames, eyes found in {found} ({found / number:.1%})')
