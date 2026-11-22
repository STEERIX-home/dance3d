"""Do the characters of a pair run (pair_run.py) move like the mannequins of their guide?

The quick motion (qa_fast_motion.py) of the guide video and of the generated video is measured in
the same bands of each dancer (head, chest, hips, taken from the 3D render's identity mask) and
compared. Both show the same camera, so the picture motion is compared as it is.

    qa_pair_motion.py --run RUN_DIR [--masks SCENE_DIR] [--windows a:b,c:d] [--cut 3.0]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paths import JOB
from qa_fast_motion import compare, parse_windows, video_traces
from scail_run import cover, write_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--masks', help='scene folder with the identity mask pass of the same camera path (default: the run\'s scene)')
    parser.add_argument('--windows', help='a:b,c:d seconds (default: the whole length in four-second pieces)')
    parser.add_argument('--cut', type=float, default=3.0)
    parser.add_argument('--output')
    args = parser.parse_args()
    run = Path(args.run)
    record = json.loads((run / 'run.json').read_text())
    scene, first, total, fps = Path(record['scene']), record['first_frame'], record['frames'], record['fps']
    masks_scene = Path(args.masks) if args.masks else scene
    size = (record['width'], record['height'])
    work = JOB / 'cache/tmp'
    work.mkdir(parents=True, exist_ok=True)
    guide = work / f'qa_{run.name}_guide.mp4'
    write_video([cover(cv2.imread(str(scene / record['guide'] / f'frame_{first + i:04d}.jpg')), *size, cv2.INTER_AREA)
                 for i in range(total)], guide, fps)
    made = work / f'qa_{run.name}_made.mp4'
    write_video([cv2.imread(str(run / 'generated' / f'frame_{i + 1:04d}.png')) for i in range(total)], made, fps)

    def mask_of_time(seconds):
        picture = cv2.imread(str(masks_scene / 'mask' / f'frame_{first + min(total - 1, int(round(seconds * fps))):04d}.png'))
        blue, red = picture[..., 0].astype(int), picture[..., 2].astype(int)
        return {'A': (blue - red) > 128, 'B': (red - blue) > 128}

    results = {}
    for name, video in (('guide', guide), ('made', made)):
        _, traces = video_traces(video, 0.0, total / fps, mask_of_time)
        results[name] = dict(fps=fps, start=(first - 1) / fps, traces={key: value.tolist() for key, value in traces.items()})
        video.unlink()
    print(f'generated characters against the mannequins of the guide, motion quicker than {args.cut:g} Hz')
    rows = compare(results['guide'], results['made'], parse_windows(args.windows, total / fps), args.cut, (False, False))
    # is the generated video ahead of or behind its guide? (frames; positive = generated is late)
    shifts = {}
    for name in sorted(results['guide']['traces']):
        if name == 'camera':
            continue
        a = np.nan_to_num(np.asarray(results['guide']['traces'][name], dtype=float))[:, 0]
        b = np.nan_to_num(np.asarray(results['made']['traces'][name], dtype=float))[:, 0]
        if a.std() < 0.2:
            continue
        best = max(np.linspace(-3, 3, 61), key=lambda lag: np.corrcoef(
            a, np.interp(np.arange(len(a)) + lag, np.arange(len(b)), b))[0, 1])
        shifts[name] = round(float(best), 2)
    print('generated is late by (frames):', shifts)
    if args.output:
        Path(args.output).write_text(json.dumps(dict(rows=rows, late_frames=shifts, **results), allow_nan=True))


if __name__ == '__main__':
    main()
