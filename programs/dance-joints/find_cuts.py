"""Where the video cuts to another shot: an edit to another camera, or a new picture put behind the
dancer (green-screen and game backgrounds that change).

Across a cut nothing can be measured about the camera: the background matcher finds a "turn" of
100 degrees and more, and the dancer would seem to spin round in the 3D world (11-14, a challenge
video with changing game backgrounds). Camera motion changes the picture around the dancer a little
from one frame to the next; a cut changes all of it at once. So, per pair of neighbouring frames:
the background (the dancer's mask, grown, left out) at low resolution, the smallest mean difference
over small shifts (camera motion), against the same measure of the frames around it.

    find_cuts.py --video source.mp4 --masks masks.npz --output cuts.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mocap_people import load_masks

WIDTH = 96          # pixels across, for the comparison
SHIFT = 3           # largest shift tried (pixels at that size): camera motion between two frames
CUT_LEAST = 18.0    # mean grey difference (0..255) a cut must reach at least
CUT_OVER = 4.0      # ... and this many times the larger difference of the frame pairs just before and after


def difference(previous, current, usable):
    """Smallest mean absolute difference of the usable pixels over shifts of the previous frame."""
    best = np.inf
    height, width = current.shape
    for dy in range(-SHIFT, SHIFT + 1):
        for dx in range(-SHIFT, SHIFT + 1):
            a = previous[max(0, dy):height + min(0, dy), max(0, dx):width + min(0, dx)]
            b = current[max(0, -dy):height + min(0, -dy), max(0, -dx):width + min(0, -dx)]
            m = usable[max(0, -dy):height + min(0, -dy), max(0, -dx):width + min(0, -dx)]
            if m.sum() > 0.05 * m.size:
                best = min(best, float(np.abs(a[m] - b[m]).mean()))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--masks', required=True, help='masks.npz of the tracked dancer')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    masks = np.zeros(1)
    for stack in load_masks(args.masks).values():
        masks = stack if masks.ndim == 1 else masks | stack
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames, background = [], []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height = int(round(frame.shape[0] * WIDTH / frame.shape[1]))
        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (WIDTH, height), interpolation=cv2.INTER_AREA),
                                              cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
        frames.append(small)
        person = masks[min(len(background), len(masks) - 1)].astype(np.uint8)
        person = cv2.resize(person, (WIDTH, height), interpolation=cv2.INTER_NEAREST)
        background.append(cv2.dilate(person, np.ones((7, 7), np.uint8)) == 0)
    capture.release()
    changes = np.zeros(len(frames))
    for index in range(1, len(frames)):
        usable = background[index] & background[index - 1]
        if usable.mean() < 0.1:  # hardly any background left: the whole picture
            usable = np.ones_like(usable)
        changes[index] = difference(frames[index - 1], frames[index], usable)
    changes[~np.isfinite(changes)] = 0.0
    cuts = []
    for index in range(1, len(frames)):
        around = max(changes[max(1, index - 2):index].max(initial=0.0), changes[index + 1:index + 3].max(initial=0.0))
        if changes[index] >= CUT_LEAST and changes[index] >= CUT_OVER * max(around, 1.0):
            cuts.append(index)  # a new shot starts at this frame
    Path(args.output).write_text(json.dumps(dict(
        video=str(args.video), frames=len(frames), fps=fps, cuts=cuts, seconds=[round(c / fps, 3) for c in cuts],
        change=dict(median=float(np.median(changes[1:])), p99=float(np.percentile(changes[1:], 99)), max=float(changes.max())),
        rule=f'background difference >= {CUT_LEAST} and >= {CUT_OVER} x the neighbouring frame pairs'), indent=1))
    print(f'{len(cuts)} cuts at', [f'{c / fps:.2f}s' for c in cuts], f'| change median {np.median(changes[1:]):.1f}, '
          f'p99 {np.percentile(changes[1:], 99):.1f}, max {changes.max():.1f}')
    for index in np.argsort(-changes)[:8]:
        print(f'  frame {index} ({index / fps:.2f}s): {changes[index]:.1f}')


if __name__ == '__main__':
    main()
