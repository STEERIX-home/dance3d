"""Measure the quick motion of each dancer (shakes, hits, bounces) and compare it between the
source video, the reconstructed 3D bodies and a generated video.

How a dancer's picture moves is followed with optical flow inside the dancer's own mask, band by
band (head, chest, hips of the visible body). For a reconstructed body the same bands are filled
with the body's own surface points projected into the picture. Both give a 2D velocity per band
and frame; summed up over time and with the slow part taken away (slower than `cut` Hz) this is
what a shake looks like. Two measurements of the same dance are compared by how well their quick
parts agree (correlation) and how large one is against the other (amplitude ratio).

    qa_fast_motion.py video --video V --masks masks.npz --output source.json
    qa_fast_motion.py video --video V --scene SCENE_DIR --output result.json     (masks of the 3D render)
    qa_fast_motion.py compare --reference source.json --test result.json [--windows a:b,c:d]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BANDS = {'head': (0.0, 0.22), 'chest': (0.22, 0.5), 'hips': (0.5, 0.75)}


def load_masks(path):
    with np.load(path) as data:
        shape = tuple(data['shape'])
        return {key: np.unpackbits(data[key], axis=-1)[..., :shape[-1]].astype(bool)
                for key in data.files if key != 'shape'}


def band_regions(mask, shape, erode=5):
    """The dancer's mask (any size) cut into bands of its own visible height, at picture size
    `shape` (h, w). The top of the mask is 0, its bottom 1."""
    mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if erode:
        mask = cv2.erode(mask, np.ones((erode, erode), np.uint8))
    rows = np.flatnonzero(mask.any(1))
    if len(rows) < 20:
        return {}
    top, height = rows[0], rows[-1] - rows[0] + 1
    regions = {}
    for name, (start, stop) in BANDS.items():
        band = np.zeros_like(mask)
        band[top + int(start * height):top + int(stop * height)] = 1
        regions[name] = band & mask
    return regions


def median_flow(previous, current, region, count=400):
    points = cv2.goodFeaturesToTrack(previous, count, 0.01, 4, mask=region)
    if points is None or len(points) < 8:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(21, 21), maxLevel=3)
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None, winSize=(21, 21), maxLevel=3)
    good = (status[:, 0] == 1) & (status_back[:, 0] == 1) & (np.linalg.norm(back - points, axis=-1)[:, 0] < 1.0)
    if good.sum() < 8:
        return None
    return np.median((moved - points)[good, 0], axis=0)


def video_traces(video, start, stop, mask_of_time, scale=0.5):
    """Velocity (full-resolution pixels per frame) of every band and of the background.
    mask_of_time(seconds) -> {label: mask}. Returns fps, {name: (frames-1, 2)}; missing = nan."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    first = int(round(start * fps))
    last = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if stop is None else int(round(stop * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    previous, traces, missing = None, {}, np.full(2, np.nan)
    for index in range(first, last):
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        masks = mask_of_time(index / fps)
        if previous is not None:
            everyone = np.zeros(gray.shape, np.uint8)
            for mask in previous_masks.values():
                everyone |= cv2.resize(mask.astype(np.uint8), gray.shape[::-1], interpolation=cv2.INTER_NEAREST)
            background = (1 - cv2.dilate(everyone, np.ones((31, 31), np.uint8))).astype(np.uint8)
            camera = median_flow(previous, gray, background)
            traces.setdefault('camera', []).append(missing if camera is None else camera / scale)
            for label, mask in previous_masks.items():
                regions = band_regions(mask, gray.shape)
                for band in BANDS:
                    region = regions.get(band)
                    flow = median_flow(previous, gray, region) if region is not None and region.sum() > 200 else None
                    traces.setdefault(f'{label}/{band}', []).append(missing if flow is None else flow / scale)
        previous, previous_masks = gray, masks
    capture.release()
    return fps, {name: np.asarray(values) for name, values in traces.items()}


def mesh_traces(vertices, faces, K, masks_of_frame, size, following=None):
    """The same band velocities for reconstructed bodies: vertices {label: (L, V, 3)} in camera
    coordinates (x right, y down, z forward) of a camera with intrinsics K (3, 3) and picture
    `size` (w, h). masks_of_frame(index) -> {label: mask}: the bands are the ones of the picture.
    Only surface points that face the camera and lie on the dancer's own mask are used.
    following: {label: (L, V, 3)} the body of the NEXT frame seen by THIS frame's camera; the
    velocity is then the body's motion in the room, without the camera's own motion."""
    width, height = size
    count = len(next(iter(vertices.values())))
    traces = {f'{label}/{band}': np.full((count - 1, 2), np.nan) for label in vertices for band in BANDS}
    for label, points in vertices.items():
        pixels = points @ K.T
        pixels = pixels[..., :2] / np.maximum(pixels[..., 2:], 1e-6)
        if following is not None:
            ahead = following[label] @ K.T
            ahead = ahead[..., :2] / np.maximum(ahead[..., 2:], 1e-6)
        for index in range(count - 1):
            mask = masks_of_frame(index).get(label)
            if mask is None:
                continue
            regions = band_regions(mask, mask.shape, erode=3)
            if not regions:
                continue
            triangles = points[index][faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            facing = np.zeros(points.shape[1], dtype=bool)
            facing[faces[(normals * triangles[:, 0]).sum(-1) < 0].reshape(-1)] = True
            column = np.round(pixels[index, :, 0] * mask.shape[1] / width).astype(int)
            row = np.round(pixels[index, :, 1] * mask.shape[0] / height).astype(int)
            inside = facing & (column >= 0) & (column < mask.shape[1]) & (row >= 0) & (row < mask.shape[0]) & (points[index, :, 2] > 0.1)
            step = (pixels[index + 1] if following is None else ahead[index]) - pixels[index]
            for band, region in regions.items():
                chosen = inside.copy()
                chosen[inside] = region[row[inside], column[inside]] > 0
                if chosen.sum() >= 30:
                    traces[f'{label}/{band}'][index] = np.median(step[chosen], axis=0)
    return traces


def quick_part(velocity, fps, cut, noise=0.0):
    """Position (sum of the velocities) without its slow part: (frames, 2). Gaps (nan) count as
    no motion. `noise`: optional smoothing (seconds) against frame-to-frame measuring noise."""
    velocity = np.where(np.isfinite(velocity), velocity, 0.0)
    position = np.concatenate([np.zeros((1, 2)), np.cumsum(velocity, axis=0)])

    def gauss(values, sigma):
        radius = max(1, int(np.ceil(sigma * 4)))
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(values, ((radius, radius), (0, 0)), mode='edge')
        return np.stack([np.convolve(padded[:, axis], kernel, mode='valid') for axis in range(values.shape[1])], axis=1)

    # a Gaussian low-pass has half amplitude at 0.1874 / sigma Hz
    quick = position - gauss(position, 0.1874 / cut * fps)
    if noise:
        quick = gauss(quick, noise * fps)
    return quick


def resample(series, fps, target_fps, length):
    time_in = np.arange(len(series)) / fps
    time_out = np.arange(length) / target_fps
    return np.stack([np.interp(time_out, time_in, series[:, axis]) for axis in range(series.shape[1])], axis=1)


def compare(reference, test, windows, cut=1.5, minus_camera=(True, True), report=print):
    """reference/test: dicts as written by `video` (fps, start, traces). Returns rows per band and window."""
    rows = []
    fps = max(reference['fps'], test['fps'])
    for name in sorted(reference['traces']):
        if name == 'camera' or name not in test['traces']:
            continue
        series = []
        for data, subtract in ((reference, minus_camera[0]), (test, minus_camera[1])):
            velocity = np.asarray(data['traces'][name], dtype=float)
            if subtract and 'camera' in data['traces']:
                velocity = velocity - np.nan_to_num(np.asarray(data['traces']['camera'], dtype=float))
            seen = np.isfinite(velocity).all(1)
            quick = quick_part(velocity, data['fps'], cut, noise=0.012)
            length = int(round((len(quick) - 1) / data['fps'] * fps)) + 1
            series.append((resample(quick, data['fps'], fps, length),
                           resample(np.concatenate([seen[:1], seen]).astype(float)[:, None], data['fps'], fps, length)[:, 0] > 0.5))
        length = min(len(series[0][0]), len(series[1][0]))
        a, b = series[0][0][:length], series[1][0][:length]
        seen = series[0][1][:length] & series[1][1][:length]
        time = reference['start'] + np.arange(length) / fps
        for first, last in windows:
            keep = seen & (time >= first) & (time < last)
            if keep.sum() < fps * 0.4:
                continue
            row = dict(band=name, window=[first, last], seen=float(keep.sum() / max(1, ((time >= first) & (time < last)).sum())))
            for axis, key in enumerate('xy'):
                x, y = a[keep, axis], b[keep, axis]
                row[key] = dict(reference_rms_px=float(x.std()), test_rms_px=float(y.std()),
                                ratio=float(y.std() / max(x.std(), 1e-6)),
                                correlation=float(np.corrcoef(x, y)[0, 1]) if x.std() > 0.3 and y.std() > 0.05 else None,
                                error_rms_px=float((y - x).std()))
            rows.append(row)
            correlation = row['x']['correlation']
            report(f"{name:9s} {first:5.1f}-{last:4.1f}s  x: source {row['x']['reference_rms_px']:5.1f} px, test {row['x']['test_rms_px']:5.1f} px,"
                   f" ratio {row['x']['ratio']:4.2f}, agreement {'  - ' if correlation is None else format(correlation, '4.2f')}"
                   f"   y: ratio {row['y']['ratio']:4.2f}, agreement "
                   f"{'  - ' if row['y']['correlation'] is None else format(row['y']['correlation'], '4.2f')}")
    return rows


def parse_windows(text, duration=None):
    """'a:b,c:d' -> [(a, b), (c, d)]; without a text: the whole length in four-second pieces."""
    if text:
        return [tuple(float(v) for v in part.split(':')) for part in text.split(',')]
    return [(float(start), float(min(start + 4, duration))) for start in range(0, max(1, int(duration - 1.5)), 4)]


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    video = commands.add_parser('video')
    video.add_argument('--video', required=True)
    video.add_argument('--masks', help='masks.npz of track_people.py (source video)')
    video.add_argument('--mask-fps', type=float, default=60.0)
    video.add_argument('--scene', help='Blender scene folder: its identity mask pass is used (3D guide, generated video)')
    video.add_argument('--scene-fps', type=float, default=24.0)
    video.add_argument('--start', type=float, default=0.0)
    video.add_argument('--stop', type=float, help='last second to measure (default: to the end)')
    video.add_argument('--output', required=True)
    check = commands.add_parser('compare')
    check.add_argument('--reference', required=True)
    check.add_argument('--test', required=True)
    check.add_argument('--windows', help='a:b,c:d seconds (default: the whole length in four-second pieces)')
    check.add_argument('--cut', type=float, default=1.5, help='slower motion than this (Hz) is not compared')
    check.add_argument('--raw', action='store_true', help='do not take the background motion away (same camera on both sides)')
    check.add_argument('--output')
    args = parser.parse_args()
    if args.command == 'video':
        if args.masks:
            masks = load_masks(args.masks)

            def mask_of_time(seconds):
                index = int(round(seconds * args.mask_fps))
                return {label: stack[min(index, len(stack) - 1)] for label, stack in masks.items()}
        else:
            scene = Path(args.scene)

            def mask_of_time(seconds):
                picture = cv2.imread(str(scene / 'mask' / f'frame_{int(round(seconds * args.scene_fps)) + 1:04d}.png'))
                blue, red = picture[..., 0].astype(int), picture[..., 2].astype(int)
                return {'A': (blue - red) > 128, 'B': (red - blue) > 128}
        fps, traces = video_traces(args.video, args.start, args.stop, mask_of_time)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(dict(
            video=str(args.video), fps=fps, start=args.start, stop=args.stop,
            traces={name: np.round(values, 3).tolist() for name, values in traces.items()}), allow_nan=True))
        for name, values in traces.items():
            print(f'{name:9s} measured in {np.isfinite(values).all(1).mean() * 100:5.1f} % of the frames')
    else:
        reference, test = (json.loads(Path(path).read_text()) for path in (args.reference, args.test))
        length = min(data['start'] + len(next(iter(data['traces'].values()))) / data['fps'] for data in (reference, test))
        rows = compare(reference, test, parse_windows(args.windows, length), args.cut, (not args.raw, not args.raw))
        if args.output:
            Path(args.output).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
