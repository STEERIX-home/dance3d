"""Lay the separately generated dancers (solo_run.py + matte_solo.py) over the rendered 3D set and
add the original music.

The background of every frame is the rendered plate itself (the set with the dancers' shadows),
at full resolution: it is pixel-identical to the 3D render and cannot drift. Each dancer's window
picture is put back where its window was. Where two dancers overlap, who is in front is decided
per pixel from the 3D bodies (the identity mask of the render shows which body is nearer), so an
arm in front of the other dancer stays in front even when the rest of that dancer is behind.

A generated chunk can run a frame ahead of its guide (measured by qa_chunk_timing.py, which
writes timing.json into the run folder). With --timing such a chunk is shown one frame later,
in the window of the frame it is shown in, so that the character is in step with the set, the
other dancer and the music.

Outputs: --output MP4 (H.264 + AAC) and, with --compare, a side-by-side [3D guide | result] MP4.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from composite import encoder
from solo_run import own_mask


def place(image, window, full_size, size, interpolation):
    """A window picture back at its place in the full picture."""
    left, upper, width, height = window
    matrix = np.array([[width / size[0], 0, left], [0, height / size[1], upper]])
    return cv2.warpAffine(image, matrix, full_size, flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def front_map(scene, labels, frame, default_front):
    """Per pixel: index (into labels) of the dancer who is in front there. Known where the two grey
    bodies overlap in the render; everywhere else the answer of the nearest such place, or, if the
    bodies do not overlap at all in this frame, the dancer nearer to the camera."""
    joint = cv2.imread(str(scene / 'mask' / f'frame_{frame:04d}.png'))
    solo = [own_mask(scene, label, frame) for label in labels]
    shown = [(joint[..., 0] > 200) & (joint[..., 2] < 60), (joint[..., 2] > 200) & (joint[..., 0] < 60)]  # A blue, B red
    shown = [shown['AB'.index(label)] for label in labels]
    both = solo[0] & solo[1]
    known = np.zeros(both.shape, np.uint8)  # 1: first dancer in front, 2: second
    known[both & shown[0]] = 1
    known[both & shown[1]] = 2
    if not known.any():
        return np.full(both.shape, default_front, np.uint8)
    _, nearest = cv2.distanceTransformWithLabels((known == 0).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    # label of every zero pixel of the transform input, in row order
    labels_at = np.zeros(int(nearest.max()) + 1, np.uint8)
    labels_at[nearest[known > 0]] = known[known > 0]
    return labels_at[nearest] - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--inputs', required=True, help='folder of export_scene_inputs.py (dancer positions)')
    parser.add_argument('--run', action='append', required=True, help='LABEL=output folder of solo_run.py (with alpha/)')
    parser.add_argument('--pictures', action='append', default=[],
                        help='LABEL=subfolder of that run with the pictures to use instead of generated/ (A=faces)')
    parser.add_argument('--timing', action='store_true', help='use each run\'s timing.json (qa_chunk_timing.py)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--audio')
    parser.add_argument('--compare')
    parser.add_argument('--frames-folder', help='also write the result frames here (PNG), e.g. for later corrections')
    args = parser.parse_args()
    scene = Path(args.scene)
    runs = {label: Path(folder) for label, folder in (entry.split('=', 1) for entry in args.run)}
    labels = list(runs)
    folders = {label: 'generated' for label in labels}
    folders.update(dict(entry.split('=', 1) for entry in args.pictures))
    records = {label: json.loads((folder / 'run.json').read_text()) for label, folder in runs.items()}
    windows = {label: json.loads((folder / 'windows.json').read_text())['windows'] for label, folder in runs.items()}
    shown = {label: json.loads((folder / 'timing.json').read_text())['source_frame'] if args.timing else None
             for label, folder in runs.items()}
    head = records[labels[0]]
    for record in records.values():
        if any(record[key] != head[key] for key in ('frames', 'first_frame', 'fps')):
            raise ValueError('The solo runs differ in length or start')
    fps, total, first = head['fps'], head['frames'], head['first_frame']
    if args.audio and first != 1:
        raise NotImplementedError('audio for runs that do not start at frame 1')
    positions = {person['label']: np.asarray(person['pelvis'])
                 for person in json.loads((Path(args.inputs) / 'scene_inputs.json').read_text())['people']}
    camera = [np.asarray(row['location']) for row in json.loads((scene / 'camera-path.json').read_text())]
    sample = cv2.imread(str(scene / 'plate' / f'frame_{first:04d}.jpg'))
    height, width = sample.shape[:2]
    duration = total / fps
    out = encoder(args.output, width, height, fps, args.audio, duration)
    compare = encoder(args.compare, width * 2, height, fps, args.audio, duration) if args.compare else None
    if args.frames_folder:
        Path(args.frames_folder).mkdir(parents=True, exist_ok=True)
    for index in range(total):
        frame = first + index
        plate = cv2.imread(str(scene / 'plate' / f'frame_{frame:04d}.jpg')).astype(np.float32)
        colour, alpha = [], []
        for label in labels:
            size = (records[label]['width'], records[label]['height'])
            made = shown[label][index] if shown[label] else index + 1  # the generated frame shown now
            picture = cv2.imread(str(runs[label] / folders[label] / f'frame_{made:04d}.png'))
            matte = cv2.imread(str(runs[label] / 'alpha' / f'frame_{made:04d}.png'), cv2.IMREAD_GRAYSCALE)
            if picture is None or matte is None:
                raise FileNotFoundError(f'frame {index + 1} of dancer {label} is missing in {runs[label]}')
            window = windows[label][index]
            colour.append(place(picture, window, (width, height), size, cv2.INTER_LANCZOS4).astype(np.float32))
            alpha.append(place(matte, window, (width, height), size, cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255)
        if len(labels) == 2:
            nearer = int(np.argmin([np.linalg.norm(positions[label][frame - 1] - camera[frame - 1]) for label in labels]))
            first_in_front = (front_map(scene, labels, frame, nearer) == 0)[..., None]
            behind_0 = colour[0] * alpha[0] + plate * (1 - alpha[0])   # dancer 0 over the plate
            behind_1 = colour[1] * alpha[1] + plate * (1 - alpha[1])
            result = np.where(first_in_front, colour[0] * alpha[0] + behind_1 * (1 - alpha[0]),
                              colour[1] * alpha[1] + behind_0 * (1 - alpha[1]))
        else:
            result = colour[0] * alpha[0] + plate * (1 - alpha[0])
        result = np.clip(result, 0, 255).astype(np.uint8)
        out.stdin.write(result.tobytes())
        if args.frames_folder:
            cv2.imwrite(str(Path(args.frames_folder) / f'frame_{index + 1:04d}.png'), result)
        if compare:
            guide = cv2.imread(str(scene / 'guide' / f'frame_{frame:04d}.jpg'))
            compare.stdin.write(np.concatenate([guide, result], axis=1).tobytes())
    for process in filter(None, (out, compare)):
        process.stdin.close()
        if process.wait():
            raise RuntimeError('ffmpeg failed')
    print(f'{total} frames, {duration:.2f} s -> {args.output}')


if __name__ == '__main__':
    main()
