"""Which chunks would the two-dancer program's new clothes check send back to be made again? The chunks of a finished
run are judged in the order they were made, with the usual colours learnt from the chunks before (as pair_run.py does);
nothing is changed.

    duo_chunk_probe.py JOB CAMERA
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '/srv/training/projects/duo-scene')
from qa_pair_identity import BODY_APART, clearly_seen, frame_body_colours, judge_bodies  # noqa: E402

job, camera = Path(sys.argv[1]), sys.argv[2]
scene = job / 'outputs/work/scene' / camera
record = json.loads((job / 'outputs/work/gen' / camera / 'run.json').read_text())
first, total, fps = record['first_frame'], record['frames'], record['fps']
clear = clearly_seen(scene, first, total)
so_far = {label: np.zeros((0, 28)) for label in 'AB'}
for entry in record['runs']:
    index = int(entry['chunk'].split()[1].split('/')[0]) - 1
    numbers = list(range(entry['guide_frames'][0], entry['guide_frames'][1] + 1))
    files = sorted((job / 'outputs/work/comfy/output' / camera / f'chunk_{index:02d}').glob('frame_*.png'))
    assert len(files) == len(numbers), (len(files), len(numbers))
    seen = {label: np.full((len(numbers), 28), np.nan) for label in 'AB'}
    for i, (file, number) in enumerate(zip(files, numbers)):
        frame = cv2.imread(str(file))
        picture = cv2.imread(str(scene / 'mask' / f'frame_{number:04d}.png'))
        for label, value in frame_body_colours(frame, picture).items():
            if clear[label][number - first]:
                seen[label][i] = value
    known = {label: np.concatenate([so_far[label], seen[label]]) for label in 'AB'}
    apart, wrong = judge_bodies(known, fps)
    flags = (wrong['A'] | wrong['B'])[len(so_far['A']):] if apart > BODY_APART else np.zeros(len(numbers), bool)
    bad = np.flatnonzero(flags)
    print(f'{entry["chunk"]:28s} guide frames {numbers[0]:3d}..{numbers[-1]:3d}  usual colours apart {apart:.2f}  '
          + (f'{len(bad)} frames in the other one\'s clothes ({numbers[bad[0]]}..{numbers[bad[-1]]}) -> made again' if len(bad) >= 3 else 'passes'))
    good = np.setdiff1d(np.arange(len(numbers)), bad)
    for label in 'AB':
        so_far[label] = np.concatenate([so_far[label], seen[label][good]])
