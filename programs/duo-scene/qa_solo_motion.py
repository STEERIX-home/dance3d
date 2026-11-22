"""Does the generated character of a solo run move like the grey dancer of its guide?

The quick motion (qa_fast_motion.py) of the guide window video and of the generated window video are
measured in the same bands and compared; the background motion of each (the window follows the
dancer, so the set moves in it) is taken away on both sides.

    qa_solo_motion.py --run RUN_DIR [--windows 1.4:2.0,2.0:3.4] [--cut 3.0]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paths import JOB
from qa_fast_motion import compare, parse_windows, video_traces
from scail_run import write_video
from solo_run import cut_window, own_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--windows', default='0:1.4,1.4:2.0,2.0:3.4,3.4:5.2')
    parser.add_argument('--cut', type=float, default=3.0)
    parser.add_argument('--output')
    args = parser.parse_args()
    run = Path(args.run)
    record = json.loads((run / 'run.json').read_text())
    windows = json.loads((run / 'windows.json').read_text())['windows']
    scene, label, first, total, fps = Path(record['scene']), record['dancer'], record['first_frame'], record['frames'], record['fps']
    size = (record['width'], record['height'])
    work = JOB / 'cache/tmp'
    work.mkdir(parents=True, exist_ok=True)
    expected = [cut_window(own_mask(scene, label, first + i).astype(np.uint8) * 255, windows[i], size, cv2.INTER_LINEAR) > 127
                for i in range(total)]
    guide = work / f'qa_{run.name}_guide.mp4'
    write_video([cut_window(cv2.imread(str(scene / f'solo_{label}' / f'frame_{first + i:04d}.jpg')), windows[i], size)
                 for i in range(total)], guide, fps)
    made = work / f'qa_{run.name}_made.mp4'
    write_video([cv2.imread(str(run / 'generated' / f'frame_{i + 1:04d}.png')) for i in range(total)], made, fps)
    alpha = [cv2.imread(str(run / 'alpha' / f'frame_{i + 1:04d}.png'), cv2.IMREAD_GRAYSCALE) > 127 for i in range(total)]
    results = {}
    for name, video, masks in (('guide', guide, expected), ('made', made, alpha)):
        _, traces = video_traces(video, 0.0, total / fps, lambda seconds, masks=masks: {label: masks[min(total - 1, int(round(seconds * fps)))]})
        results[name] = dict(fps=fps, start=(first - 1) / fps, traces={key: value.tolist() for key, value in traces.items()})
        video.unlink()
    print(f'dancer {label}: generated character against its guide, motion quicker than {args.cut:g} Hz')
    rows = compare(results['guide'], results['made'], parse_windows(args.windows), args.cut)
    if args.output:
        Path(args.output).write_text(json.dumps(dict(rows=rows, **results), allow_nan=True))


if __name__ == '__main__':
    main()
