"""Contact sheets of a finished challenge run, for looking through by eye: the finished frame of each shot
(before the edit's effects) beside the 3D guide frame the video model followed, and, with --reference,
beside the same moment of another run of the same joints (e.g. the other video model).

    contact_sheets.py --job RUN [--reference OTHER_RUN] --output DIR [--every 24] [--width 1600]

Moments: the first and the last frame of every shot and one every --every frames (24 = 1 s) in between.
Each tile is labelled with the time, the video frame and the shot. Sheets of at most 4 rows (3 pairs a row),
PNG, --width pixels wide: DIR/guide_NN.png, DIR/reference_NN.png, and DIR/moments.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PER_ROW, ROWS = 3, 4


def shot_frames(job):
    """{video frame number: (shot, file of the finished frame)} of a run, from its plan and finished/ folders."""
    work = Path(job) / 'outputs/work'
    plan = json.loads((work / 'plan.json').read_text())
    found = {}
    for shot in plan['shots']:
        start = shot['window'][0]
        for number in range(shot['first'], shot['last'] + 1):
            found[number] = (shot, work / 'finished' / shot['name'] / f'frame_{number - start + 1:04d}.png')
    return plan, found


def guide_file(job, shot, number):
    run = json.loads((Path(job) / 'outputs/work/gen' / shot['name'] / 'run.json').read_text())
    return Path(job) / 'outputs/work/scene' / shot['view'] / run.get('guide', 'pair') / f'frame_{number:04d}.jpg'


def tile(file, size):
    picture = cv2.imread(str(file)) if file is not None else None
    if picture is None:
        picture = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.putText(picture, 'missing', (10, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return picture
    return cv2.resize(picture, size, interpolation=cv2.INTER_AREA)


def sheets(pairs, labels, width, out, prefix):
    """Write the pairs (left file, right file) as sheets; returns the files written."""
    gap, label_height = 6, 26
    pair_width = (width - gap * (PER_ROW + 1)) // PER_ROW
    image_width = (pair_width - 2) // 2
    image_height = int(round(image_width * 16 / 9))
    written = []
    for page in range(0, len(pairs), PER_ROW * ROWS):
        chunk = pairs[page:page + PER_ROW * ROWS]
        rows = -(-len(chunk) // PER_ROW)
        sheet = np.full((gap + rows * (label_height + image_height + gap), width, 3), 255, np.uint8)
        for index, ((left, right), label) in enumerate(zip(chunk, labels[page:page + PER_ROW * ROWS])):
            row, column = divmod(index, PER_ROW)
            x = gap + column * (pair_width + gap)
            y = gap + row * (label_height + image_height + gap)
            cv2.putText(sheet, label, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            sheet[y + label_height:y + label_height + image_height, x:x + image_width] = tile(left, (image_width, image_height))
            sheet[y + label_height:y + label_height + image_height, x + image_width + 2:x + 2 * image_width + 2] = \
                tile(right, (image_width, image_height))
        file = out / f'{prefix}_{page // (PER_ROW * ROWS) + 1:02d}.png'
        cv2.imwrite(str(file), sheet)
        written.append(file.name)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, help='a finished run of challenge.py')
    parser.add_argument('--reference', help='another finished run of the same joints, shown beside the same moments')
    parser.add_argument('--output', required=True)
    parser.add_argument('--every', type=int, default=24, help='one moment every this many frames (24 = 1 s)')
    parser.add_argument('--width', type=int, default=1600)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plan, frames = shot_frames(args.job)
    moments = set(range(1, plan['frames'] + 1, args.every)) | {plan['frames']}
    for shot in plan['shots']:
        moments |= {shot['first'], shot['last']}
    moments = sorted(number for number in moments if number in frames)
    fps = plan['fps']
    labels = [f'{(number - 1) / fps:5.2f} s  #{number}  {frames[number][0]["view"]}' for number in moments]
    record = dict(job=str(args.job), moments=moments, fps=fps, left='3D guide', right='finished frame of the shot')
    record['guide_sheets'] = sheets([(guide_file(args.job, frames[n][0], n), frames[n][1]) for n in moments], labels, args.width, out,
                                    'guide')
    if args.reference:
        _, other = shot_frames(args.reference)
        record['reference'] = str(args.reference)
        record['reference_sheets'] = sheets([(frames[n][1], other.get(n, (None, None))[1]) for n in moments], labels, args.width, out,
                                            'reference')
        record['reference_layout'] = 'left: this run, right: the reference run, same video frame'
    (out / 'moments.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
