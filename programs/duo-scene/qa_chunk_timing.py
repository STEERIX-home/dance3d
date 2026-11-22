"""Is a generated character ahead of or behind its guide, chunk by chunk?

The character's outline (matte of matte_solo.py) of frame n is laid over the grey dancer's outline
of the frames n-2 .. n+2 of the guide, in the same window; frames in which the outline changes a
lot count most. Where the fit is best tells how many frames the character runs ahead (+) or
behind (-). Written to the run folder as timing.json; composite_solo.py can use it.

    qa_chunk_timing.py --run RUN_DIR [--apply 0.6]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from solo_run import OVERLAP, cut_window, own_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--apply', type=float, default=0.75,
                        help='a chunk that is at least this many frames ahead or behind is shown one frame later or '
                             'earlier. Smaller offsets cannot be corrected by whole frames and are left alone. '
                             '(Checked 11-13 on the original-view video against the source: its first chunk was '
                             '0.83 and 0.90 frames ahead; shown one frame later, the agreement of the head shake '
                             'with the source rose from 0.33 to 0.82.)')
    parser.add_argument('--gain', type=float, default=0.02,
                        help='... and only if the outlines then fit at least this much better (share of overlap): a small '
                             'gain means the measurement itself is unsure (small dancer in the window, little motion)')
    args = parser.parse_args()
    record = json.loads((args.run / 'run.json').read_text())
    windows = json.loads((args.run / 'windows.json').read_text())['windows']
    scene, label, first, total = Path(record['scene']), record['dancer'], record['first_frame'], record['frames']
    full, small = (record['width'], record['height']), (record['width'] // 4, record['height'] // 4)
    alpha = np.array([cv2.resize(cv2.imread(str(args.run / 'alpha' / f'frame_{i + 1:04d}.png'), 0), small, interpolation=cv2.INTER_AREA) > 127
                      for i in range(total)])
    # the guide outline of frame j seen in the window of frame i (j = i + k): what the character of frame i would
    # have to look like if it were k frames ahead
    loaded = {}

    def guide(j, i):
        if j not in loaded:
            for old in [key for key in loaded if key < j - 4]:
                del loaded[old]
            loaded[j] = own_mask(scene, label, first + j).astype(np.uint8) * 255
        return cv2.resize(cut_window(loaded[j], windows[i], full, cv2.INTER_LINEAR), small, interpolation=cv2.INTER_AREA) > 127

    shifts = (-2, -1, 0, 1, 2)
    fit = np.zeros((total, len(shifts)))
    change = np.zeros(total)
    previous = None
    for i in range(total):
        for column, k in enumerate(shifts):
            j = min(total - 1, max(0, i + k))
            other = guide(j, i)
            union = (alpha[i] | other).sum()
            fit[i, column] = (alpha[i] & other).sum() / max(1, union)
            if k == 0:
                change[i] = 0.0 if previous is None else float((other ^ previous).mean())
                previous = guide(i, min(total - 1, i + 1)) if i + 1 < total else other
    lengths = record['chunk_frames']
    starts = [sum(lengths[:i]) - OVERLAP * i for i in range(len(lengths))]
    chunks, source = [], np.arange(total)
    for index, (start, length) in enumerate(zip(starts, lengths)):
        own = np.arange(start + (OVERLAP if index else 0), start + length - (OVERLAP if index + 1 < len(lengths) else 0))
        weight = change[own] + 1e-9
        scores = (fit[own] * weight[:, None]).sum(0) / weight.sum()
        best = int(np.argmax(scores))
        ahead = float(shifts[best])
        if 0 < best < len(shifts) - 1:
            a, b, c = scores[best - 1], scores[best], scores[best + 1]
            ahead += float(0.5 * (a - c) / (a - 2 * b + c))
        gain = float(scores[best] - scores[shifts.index(0)])
        moved = int(np.sign(ahead)) if abs(ahead) >= args.apply and gain >= args.gain else 0
        if moved:
            # a character that is one frame ahead is shown one frame later (and the other way round)
            source[own] = np.clip(own - moved, own[0], own[-1])
        chunks.append(dict(chunk=index + 1, frames=[int(own[0]) + 1, int(own[-1]) + 1], ahead_frames=round(ahead, 2),
                           fit_same_frame=round(float(scores[shifts.index(0)]), 3), fit_best=round(float(scores[best]), 3),
                           shown_frames_later=moved))
    (args.run / 'timing.json').write_text(json.dumps(dict(chunks=chunks, source_frame=(source + 1).tolist())))
    print(label, record['camera'], ' '.join(f"{c['ahead_frames']:+.2f}{'*' if c['shown_frames_later'] else ''}" for c in chunks))


if __name__ == '__main__':
    main()
