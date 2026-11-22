"""Follow the robot's two antennas: for every footage frame with the eyes found, the direction of each antenna.

    robot_antennas.py VIDEO EYES.json OUT.json [--debug FOLDER]

An antenna is a thin rod with a dark spring at its foot, on a top corner of the head. Thin dark things on a lighter
background are picked out (black-hat filter); beside each top corner the largest such blob is the spring, and its long
axis is the antenna's direction. Angles: degrees from the head's own "up", positive towards the outside of the head
(0 = straight up, 90 = level and pointing away, more = hanging down).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('eyes')
parser.add_argument('output')
parser.add_argument('--debug')
args = parser.parse_args()
rows = {r['frame']: r for r in json.loads(Path(args.eyes).read_text())['rows'] if r['found']}
cap = cv2.VideoCapture(args.video)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
out, number = [], 0
debug = Path(args.debug) if args.debug else None
if debug:
    debug.mkdir(parents=True, exist_ok=True)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    number += 1
    row = rows.get(number)
    if row is None:
        continue
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thin = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, kernel)
    dark = (thin > 45).astype(np.uint8)
    left, right = np.array(row['left']), np.array(row['right'])
    middle, gap = (left + right) / 2, np.linalg.norm(right - left)
    along = (right - left) / gap
    up = np.array([along[1], -along[0]])
    ys, xs = np.mgrid[0:grey.shape[0], 0:grey.shape[1]]
    lateral = ((xs - middle[0]) * along[0] + (ys - middle[1]) * along[1]) / gap     # in eye distances, + = picture right
    height = ((xs - middle[0]) * up[0] + (ys - middle[1]) * up[1]) / gap            # + = above the eye line
    record = dict(frame=number, time=row['time'])
    for side, sign in (('picture_left', -1.0), ('picture_right', 1.0)):
        # beside or above the head's top corner, outside the head itself (the head reaches 1.07 to the side and 0.72 up)
        zone = (sign * lateral > 0.55) & (sign * lateral < 2.1) & (height > -0.5) & (height < 1.9) & ((sign * lateral > 1.10) | (height > 0.76))
        count, labels, stats, centres = cv2.connectedComponentsWithStats((dark & zone).astype(np.uint8), 8)
        best = None
        for index in range(1, count):
            area = stats[index, cv2.CC_STAT_AREA]
            if area < 120:
                continue
            py, px = np.nonzero(labels == index)
            points = np.stack([px, py], 1).astype(np.float64)
            centre = points.mean(0)
            values, vectors = np.linalg.eigh(np.cov((points - centre).T))
            axis = vectors[:, 1]
            long_side, short_side = np.sqrt(values[1]) * 4, np.sqrt(values[0]) * 4
            corner = middle + along * sign * 0.95 * gap + up * 0.62 * gap
            reach = np.linalg.norm(centre - corner) / gap
            if reach > 0.75 or long_side < 14:
                continue
            score = area / (1 + 3 * reach)
            if best is None or score > best[0]:
                if np.dot(axis, centre - corner) < 0:
                    axis = -axis
                best = (score, centre, axis, long_side, short_side, area, reach)
        if best:
            _, centre, axis, long_side, short_side, area, reach = best
            angle = float(np.degrees(np.arctan2(np.dot(axis, along * sign), np.dot(axis, up))))
            record[side] = dict(angle_deg=angle, long=float(long_side), short=float(short_side), area=int(area), reach=float(reach),
                                centre=[float(centre[0]), float(centre[1])])
            if debug and number % 20 == 0:
                cv2.line(frame, tuple((centre - axis * 40).astype(int)), tuple((centre + axis * 70).astype(int)), (0, 0, 255), 3)
    out.append(record)
    if debug and number % 20 == 0:
        cv2.imwrite(str(debug / f'{number:05d}.jpg'), frame[40:520, 640:1280])
Path(args.output).write_text(json.dumps(out))
for side in ('picture_left', 'picture_right'):
    a = np.array([r[side]['angle_deg'] for r in out if side in r])
    print(side, 'found', len(a), 'of', len(out), 'angle p5/p50/p95', np.percentile(a, [5, 50, 95]).round(0))
