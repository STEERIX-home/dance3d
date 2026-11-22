"""How well do neighbouring chunks of a pair run (pair_run.py) agree on the five frames they share?

Neighbours are cross-faded over those frames. Where they differ much, the fade shows as a ghost
or a jump for a fifth of a second. Per seam: mean absolute difference of the shared frames
(0-255) and the share of pixels that differ clearly (by more than 40).

    qa_pair_seams.py --run RUN_DIR
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paths import JOB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    record = json.loads((Path(args.run) / 'run.json').read_text())
    folder = JOB / 'outputs/work/comfy/output' / record['tag']
    overlap, count = record['overlap'], len(record['chunk_frames'])
    seams = []
    for index in range(count - 1):
        earlier = sorted((folder / f'chunk_{index:02d}').glob('frame_*.png'))[-overlap:]
        later = sorted((folder / f'chunk_{index + 1:02d}').glob('frame_*.png'))[:overlap]
        difference = [np.abs(cv2.imread(str(a)).astype(int) - cv2.imread(str(b)).astype(int)) for a, b in zip(earlier, later)]
        seams.append(dict(between=[index + 1, index + 2], at_seconds=round((sum(record['chunk_frames'][:index + 1]) - overlap * (index + 1)) / record['fps'], 2),
                          mean_difference=round(float(np.mean(difference)), 2),
                          clearly_different_share=round(float(np.mean(np.max(difference, axis=-1) > 40)), 4)))
        print(seams[-1])
    if args.output:
        Path(args.output).write_text(json.dumps(seams))


if __name__ == '__main__':
    main()
