"""Common-world motion -> files the Blender scene reads: one point cache per dancer + scene data.

The motion is resampled from the source frame rate to the output rate over the same time
(rotations on the sphere, positions linearly), so the dance keeps its original timing.
Coordinates are converted from the reconstruction's y-up world to Blender's z-up world; metres.

Two physical corrections are made here, on the bodies as they will be shown (after --size):
  - floor: a body that would sink into the floor is lifted just enough, smoothly,
  - keep apart: where the two bodies would pass through each other (torso, head or legs inside
    the other body; an arm deeper than a touch), both are moved apart sideways, half each.
Both are small, slow shifts of the whole body; the body's own motion is not changed.
With --mocap, legs that the camera has not seen yet at the start (a dancer framed from the waist
up) are put on the floor instead of following the motion capture's guess (plant_legs.py). The two
corrections above are computed before that, so the other dancer is not moved by it.

The source camera is exported for the "original" view. Where a character is shown smaller or
larger than the dancer (--size, --height), the camera goes with it: it is moved by the same factor
about the same point as that body, so the character is framed exactly as the dancer was (a close-up
of the dancer's face stays a close-up of a child's face instead of showing the air above it). With
two different factors the camera goes with the largest character the source shows at that moment,
so that nobody is cut off who was not cut off in the source (follow_weights). With --mocap, how
closely the exported bodies in that camera repeat the source picture is measured and reported.

Outputs in --output: <label>.pc2 per dancer, faces.npy, scene_inputs.json
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation, Slerp

from paths import EXMR, SMPLX

sys.path.insert(0, str(EXMR))

from exmr.body_models.smplx_model import SmplxModel  # noqa: E402
from exmr.export import resample_motion  # noqa: E402

from plant_legs import feet_in_picture, legs_seen, plant_opening  # noqa: E402

PARAMS = ('global_orient', 'transl', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'betas')
Y_UP_TO_Z_UP = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # (x, y, z) -> (x, -z, y)
# body parts as rounded sticks between two joints: (joint, joint, radius in m of an adult body)
CORE = [(0, 3, 0.125), (3, 6, 0.125), (6, 9, 0.135), (9, 12, 0.09), (12, 15, 0.10),
        (1, 4, 0.08), (2, 5, 0.08), (4, 7, 0.055), (5, 8, 0.055)]
ARMS = [(16, 18, 0.045), (17, 19, 0.045), (18, 20, 0.04), (19, 21, 0.04)]
TOUCH = 0.03  # an arm may press this far into the other body (a hand on a shoulder)


def write_pc2(path, vertices):
    """Blender/3ds Max point cache: frames x points x 3 float32, one sample per frame from frame 1."""
    frames, points = vertices.shape[:2]
    with open(path, 'wb') as file:
        file.write(struct.pack('<12siiffi', b'POINTCACHE2\0', 1, points, 1.0, 1.0, frames))
        file.write(np.ascontiguousarray(vertices, dtype='<f4').tobytes())


def segment_distance(p1, q1, p2, q2):
    """Shortest distance between the segments p1-q1 and p2-q2, arrays (..., 3)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    b, c = (d1 * d2).sum(-1), (d1 * r).sum(-1)
    denominator = np.maximum(a * e - b * b, 1e-12)
    s = np.clip((b * f - c * e) / denominator, 0, 1)
    t = np.clip((b * s + f) / np.maximum(e, 1e-12), 0, 1)
    s = np.clip((b * t - c) / np.maximum(a, 1e-12), 0, 1)
    return np.linalg.norm((p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None]), axis=-1)


def overlap(joints_a, joints_b, size_a, size_b):
    """How deep the two bodies are inside each other, per frame (m; 0 = not at all)."""
    deepest = np.zeros(len(joints_a))
    for i, j, radius in CORE + ARMS:
        for k, m, other_radius in CORE + ARMS:
            arms = (i >= 16) + (k >= 16)
            if arms == 2:
                continue  # arms may cross arms
            distance = segment_distance(joints_a[:, i], joints_a[:, j], joints_b[:, k], joints_b[:, m])
            allowed = radius * size_a + other_radius * size_b - (TOUCH if arms else 0.0)
            deepest = np.maximum(deepest, allowed - distance)
    return np.maximum(deepest, 0.0)


def follow_weights(sizes, shown, fps):
    """Whose size the source camera goes with, per frame: {label: weight (frames,)}, summing to 1.
    sizes {label: factor}; shown {label: (frames,)}: how much of the source picture each dancer
    takes up (any unit). A dancer is in the shot who takes up at least half as much as the one who
    takes up most (fading out down to a quarter: a shoulder behind a close-up is not in the shot).
    The largest character in the shot has the camera; a smaller one only while no larger one is in
    it. Changes are spread over about a second."""
    labels = list(sizes)
    area = {label: gaussian_filter1d(np.asarray(shown[label], dtype=np.float64), 0.2 * fps, mode='nearest') for label in labels}
    most = np.maximum(np.maximum.reduce([area[label] for label in labels]), 1e-9)
    present = {label: np.clip((area[label] / most - 0.25) / 0.25, 0.0, 1.0) for label in labels}
    left = np.ones(len(most))
    weights = {label: np.zeros(len(most)) for label in labels}
    for size in sorted(set(sizes.values()), reverse=True):
        group = [label for label in labels if sizes[label] == size]
        there = np.maximum.reduce([present[label] for label in group])
        total = np.maximum(sum(present[label] for label in group), 1e-9)
        for label in group:
            weights[label] += left * there * present[label] / total
        left = left * (1 - there)
    group = [label for label in labels if sizes[label] == max(sizes.values())]
    for label in group:  # nobody is in the picture: as for the largest
        weights[label] += left / len(group)
    return {label: gaussian_filter1d(weight, 0.2 * fps, mode='nearest') for label, weight in weights.items()}


def project(points, K):
    """Camera coordinates (..., 3) -> pixels (..., 2)."""
    return np.stack([K[0, 0] * points[..., 0] / points[..., 2] + K[0, 2], K[1, 1] * points[..., 1] / points[..., 2] + K[1, 2]], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', required=True, help='folder written by align_people.py')
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=float, default=24)
    parser.add_argument('--end', type=float, help='last second to keep (default: while both dancers are reliably seen)')
    parser.add_argument('--frames', type=int, help='keep exactly this many output frames (e.g. a whole number of chunks)')
    parser.add_argument('--size', action='append', default=[],
                        help='LABEL=factor: show this dancer smaller or larger than reconstructed, e.g. a child '
                             'character on an adult dancer\'s motion. Scaled about the dancer\'s own average floor '
                             'position, so planted feet stay planted and the place in the room is kept.')
    parser.add_argument('--height', action='append', default=[],
                        help='LABEL=centimetres: show this dancer as tall as that (standing straight); the factor of '
                             '--size is worked out from the reconstructed body')
    parser.add_argument('--mocap', help='folder of mocap_people.py / refine_motion.py: its keypoints tell which legs '
                        'the camera has not seen yet at the start; those legs are planted on the floor')
    parser.add_argument('--quick-gain', type=float, default=1.0,
                        help='make the quick part of the motion this much larger in the exported bodies. The video model '
                             'that follows them shows quick motion smaller than it is given (measured 11-13 on four '
                             'videos: 0.98 below 1.5 Hz, 0.96 at 1.5-3 Hz, 0.83 at 3-5 Hz, 0.81 at 5-8 Hz), so 1.25 '
                             'gives it back in the finished video. 1 = the bodies as reconstructed')
    parser.add_argument('--quick-from', type=float, default=3.0, help='Hz at which half of --quick-gain is reached')
    args = parser.parse_args()
    world, out = Path(args.world), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    people = np.load(world / 'people_world.npz')
    camera = np.load(world / 'camera_world.npz')
    labels = [str(label) for label in people['labels']]
    source_fps = float(people['fps'])
    count = len(people[f'{labels[0]}_transl'])
    both = np.logical_and.reduce([people[f'{label}_reliable'] for label in labels])
    last = int(np.nonzero(both)[0].max())
    if args.end is not None:
        last = min(last, int(round(args.end * source_fps)))
    keep = last + 1
    frames = int(np.floor((keep - 1) / source_fps * args.fps + 1e-6)) + 1
    if args.frames:
        if args.frames > frames:
            raise ValueError(f'only {frames} frames are available')
        frames = args.frames
    time_out = np.minimum(np.arange(frames) / args.fps, (keep - 1) / source_fps)
    sizes = {label: float(factor) for label, factor in (entry.split('=', 1) for entry in args.size)}
    nearest = np.clip(np.round(time_out * source_fps).astype(int), 0, keep - 1)

    smplx = SmplxModel(SMPLX).eval()
    standing = {}
    for label in labels:  # how tall each reconstructed body is when it stands straight (metres)
        with torch.inference_mode():
            zero = {k: torch.zeros(1, *people[f'{label}_{k}'].shape[1:]) for k in PARAMS}
            zero['betas'] = torch.from_numpy(people[f'{label}_betas'][:1]).float()
            rest = smplx(**zero)[0].numpy()
        standing[label] = float((rest[:, 1].max() - rest[:, 1].min()) * float(people[f'{label}_scale']))
    for label, centimetres in (entry.split('=', 1) for entry in args.height):
        sizes[label] = float(centimetres) / 100 / standing[label]
    summary = dict(frames=frames, fps=args.fps, source_fps=source_fps, source_frames_used=keep,
                   duration_seconds=frames / args.fps, units='metres', up_axis='z',
                   conversion='reconstruction (x, y, z) -> Blender (x, -z, y)', people=[])
    bodies = {}
    for label in labels:
        params = {k: torch.from_numpy(people[f'{label}_{k}'][:keep]) for k in PARAMS}
        params = resample_motion(params, source_fps, args.fps)
        params = {k: v[:frames].float() for k, v in params.items()}
        scale, size = float(people[f'{label}_scale']), sizes.get(label, 1.0)
        def shape(values):
            with torch.inference_mode():
                made = [np.concatenate([function(**{k: v[i:i + 32] for k, v in values.items()}).numpy()
                                        for i in range(0, frames, 32)]).astype(np.float64) * scale
                        for function in (smplx, smplx.joints)]
            return made

        vertices, joints = shape(params)
        joints_raw = joints
        pivot = joints[people[f'{label}_reliable'][nearest], 0].mean(0) * np.array([1.0, 0.0, 1.0])  # on the floor
        if size != 1.0:
            vertices, joints = pivot + size * (vertices - pivot), pivot + size * (joints - pivot)
        if not np.isfinite(vertices).all():
            raise ValueError(f'Invalid vertices for person {label}')
        estimated = people[f'{label}_estimated'][nearest] if f'{label}_estimated' in people else ~people[f'{label}_reliable'][nearest]
        bodies[label] = dict(vertices=vertices, joints=joints, scale=scale, size=size, params=params, shape=shape,
                             joints_raw=joints_raw, pivot=pivot, shift=np.zeros((frames, 3)),
                             reliable=people[f'{label}_reliable'][nearest], estimated=np.asarray(estimated).astype(int))

    # floor: never below it (y is up here)
    wide, soft = max(3, int(round(0.3 * args.fps)) | 1), 0.1 * args.fps
    for body in bodies.values():
        sink = np.maximum(-body['vertices'][..., 1].min(axis=1), 0.0)
        lift = gaussian_filter1d(maximum_filter1d(sink, wide, mode='nearest'), soft, mode='nearest')
        lift = np.maximum(lift, sink)
        body['vertices'][..., 1] += lift[:, None]
        body['joints'][..., 1] += lift[:, None]
        body['shift'][:, 1] += lift
        body['lifted_m'] = float(lift.max())

    # keep apart: two bodies cannot be in the same place
    moved = np.zeros(frames)
    if len(labels) == 2:
        a, b = (bodies[label] for label in labels)
        for _ in range(4):
            deep = overlap(a['joints'], b['joints'], a['size'], b['size'])
            if deep.max() < 0.005:
                break
            push = gaussian_filter1d(maximum_filter1d(deep + 0.01 * (deep > 0), wide, mode='nearest'), soft, mode='nearest')
            push = np.maximum(push, deep)
            away = (a['joints'][:, 0] - b['joints'][:, 0]) * np.array([1.0, 0.0, 1.0])
            length = np.linalg.norm(away, axis=1, keepdims=True)
            away = np.where(length > 1e-3, away / np.maximum(length, 1e-3), np.array([1.0, 0.0, 0.0]))
            for body, sign in ((a, 0.5), (b, -0.5)):
                shift = sign * push[:, None] * away
                body['vertices'] += shift[:, None]
                body['joints'] += shift[:, None]
                body['shift'] += shift
            moved += push
        left = overlap(a['joints'], b['joints'], a['size'], b['size'])
        summary['kept_apart'] = dict(frames_moved=int((moved > 0.005).sum()), most_m=float(moved.max()),
                                     overlap_left_m=float(left.max()),
                                     seconds=sorted({round(float(t), 1) for t in time_out[moved > 0.005]}))

    # legs never seen yet at the start: on the floor (after the corrections above, which stay as they are)
    if args.mocap:
        info = json.loads((Path(args.mocap) / 'people.json').read_text())
        summary['legs_planted'], summary['legs_out_of_the_picture'] = {}, {}
        for label in labels:
            body = bodies[label]
            keypoints = torch.load(Path(args.mocap) / f'person_{label}/keypoints.pt', map_location='cpu',
                                   weights_only=False)['value'].numpy()[:, :23]
            seen = {side: flags[nearest] for side, flags in legs_seen(keypoints, info['width'], info['height'], source_fps).items()}
            # where the source shows this dancer but not both feet, the lower legs are the motion capture's guess:
            # spans of half a second or more, for the result page
            hidden = body['reliable'] & ~feet_in_picture(keypoints, info['width'], info['height'])[nearest]
            edges = np.flatnonzero(np.diff(np.concatenate([[0], hidden.astype(int), [0]])))
            summary['legs_out_of_the_picture'][label] = [
                [round(first / args.fps, 1), round(stop / args.fps, 1)] for first, stop in zip(edges[::2].tolist(), edges[1::2].tolist())
                if stop - first >= 0.5 * args.fps]
            if body['estimated'][0]:
                continue  # the whole body is an estimate at the start: it already stands
            values = {k: v.numpy().astype(np.float64) for k, v in body['params'].items()}
            with torch.inference_mode():
                zero = {k: torch.zeros_like(v[:1]) for k, v in body['params'].items()}
                zero['betas'] = body['params']['betas'][:1]
                rest = smplx.joints(**zero)[0].numpy().astype(np.float64)
            done = []
            values = plant_opening(values, body['joints_raw'], rest, body['scale'], seen, args.fps, done)
            if not done:
                continue
            vertices, joints = body['shape']({k: torch.from_numpy(v).float() for k, v in values.items()})
            if body['size'] != 1.0:
                vertices = body['pivot'] + body['size'] * (vertices - body['pivot'])
                joints = body['pivot'] + body['size'] * (joints - body['pivot'])
            body['vertices'], body['joints'] = vertices + body['shift'][:, None], joints + body['shift'][:, None]
            summary['legs_planted'][label] = done
        if len(labels) == 2:
            summary['kept_apart']['overlap_after_planting_m'] = float(
                overlap(bodies[labels[0]]['joints'], bodies[labels[1]]['joints'], bodies[labels[0]]['size'], bodies[labels[1]]['size']).max())

    # quick motion a little larger, to make up for what the video model takes off it (see --quick-gain).
    # Done on the finished surface points: a planted foot has no quick motion, so it stays planted.
    if args.quick_gain != 1.0:
        sigma = 0.1874 / args.quick_from * args.fps  # Gaussian low-pass with half amplitude at --quick-from
        for body in bodies.values():
            for key in ('vertices', 'joints'):
                body[key] = body[key] + (args.quick_gain - 1.0) * (body[key] - gaussian_filter1d(body[key], sigma, axis=0, mode='nearest'))
        summary['quick_motion'] = dict(gain=args.quick_gain, half_reached_at_hz=args.quick_from,
                                       reason='the video model reproduces motion quicker than 3 Hz at about 0.82 of its size')

    for label in labels:
        body = bodies[label]
        vertices = body['vertices'] @ Y_UP_TO_Z_UP.T
        write_pc2(out / f'{label}.pc2', vertices)
        summary['people'].append(dict(
            label=label, cache=f'{label}.pc2', scale=body['scale'], shown_size=body['size'],
            standing_height_m=round(standing[label] * body['size'], 3),
            cache_sha256=hashlib.sha256((out / f'{label}.pc2').read_bytes()).hexdigest(),
            reliable=body['reliable'].tolist(), estimated=body['estimated'].tolist(),
            pelvis=(body['joints'][:, 0] @ Y_UP_TO_Z_UP.T).round(4).tolist(),
            head=(body['joints'][:, 15] @ Y_UP_TO_Z_UP.T).round(4).tolist(),
            lifted_off_the_floor_m=body['lifted_m'],
            lowest_z=float(vertices[..., 2].min()), highest_z=float(vertices[..., 2].max()),
            bounds_min=vertices.min((0, 1)).round(3).tolist(), bounds_max=vertices.max((0, 1)).round(3).tolist()))
    np.save(out / 'faces.npy', smplx.faces.numpy().astype(np.int32))

    # the reconstructed source camera, resampled the same way
    time_in = np.arange(keep) / source_fps
    rotations = Slerp(time_in, Rotation.from_matrix(camera['R_c2w'][:keep]))(time_out).as_matrix()
    centres = np.stack([np.interp(time_out, time_in, camera['center'][:keep, d]) for d in range(3)], -1)
    # camera axes: reconstruction uses x right, y down, z forward; Blender cameras x right, y up, z back
    blender_rotations = Y_UP_TO_Z_UP @ rotations @ np.diag([1.0, -1.0, -1.0])
    K = camera['K'].astype(np.float64)
    # the camera goes with the size of the characters it films (see the top of this file)
    shown = {label: bodies[label]['reliable'] for label in labels}
    if args.mocap:  # how much of the source picture each dancer takes up (track_people.py)
        areas = {entry['label']: entry.get('area') for entry in json.loads((Path(args.mocap) / 'people.json').read_text())['people']}
        if all(areas.get(label) for label in labels):
            shown = {label: np.array([value or 0.0 for value in areas[label]])[nearest] for label in labels}
    weights = follow_weights({label: bodies[label]['size'] for label in labels}, shown, args.fps)
    moved = sum(weights[label][:, None] * (bodies[label]['pivot'] + bodies[label]['size'] * (centres - bodies[label]['pivot']))
                for label in labels)
    up = np.einsum('tji,j->ti', rotations, np.array([0.0, 1.0, 0.0]))  # the room's "up" in camera coordinates
    roll = np.degrees(np.arctan2(up[:, 0], -up[:, 1]))
    summary['source_camera'] = dict(
        note='pose implied by the two reconstructed bodies, "up" from the floor seen in the picture; an estimate, not a '
             'calibration. Moved with the size of the characters: "follows" is whose size it goes with, per frame',
        width=int(camera['width']), height=int(camera['height']), focal_px=float(K[0, 0]),
        rotation=blender_rotations.round(6).tolist(), location=(moved @ Y_UP_TO_Z_UP.T).round(4).tolist(),
        roll_deg=dict(min=round(float(roll.min()), 1), max=round(float(roll.max()), 1)),
        moved_for_size_m=round(float(np.linalg.norm(moved - centres, axis=1).max()), 3),
        follows={label: weights[label].round(3).tolist() for label in labels})
    if args.mocap:
        # how closely the exported bodies, seen by this camera, repeat the source picture: pelvis and head of
        # each dancer against where the source shows them, while the camera goes with that dancer's size
        resized = any(body['size'] != 1.0 for body in bodies.values())
        match = {}
        for label in labels:
            body = bodies[label]
            source = torch.load(Path(args.mocap) / f'person_{label}/results.pt', map_location='cpu',
                                weights_only=False)['joints_incam'].numpy().astype(np.float64)[nearest][:, [0, 15]]
            ours = np.einsum('tji,tnj->tni', rotations, body['joints'][:, [0, 15]] - moved[:, None])
            follows = weights[label] >= 0.9 if resized else np.ones(frames, dtype=bool)
            use = body['reliable'] & follows & (ours[..., 2] > 0.05).all(1) & (source[..., 2] > 0.05).all(1)
            entry = dict(frames_compared=int(use.sum()), frames_camera_with_another_size=int((body['reliable'] & ~follows).sum()))
            if use.any():
                ours_px, source_px = project(ours[use], K), project(source[use], K)
                away = np.linalg.norm(ours_px - source_px, axis=-1).mean(1) / float(camera['height']) * 100
                ratio = np.linalg.norm(ours_px[:, 1] - ours_px[:, 0], axis=-1) / np.maximum(
                    np.linalg.norm(source_px[:, 1] - source_px[:, 0], axis=-1), 1e-6)
                entry.update(away_percent_of_picture_height=dict(median=round(float(np.median(away)), 2),
                                                                 p90=round(float(np.percentile(away, 90)), 2)),
                             size_ratio=dict(median=round(float(np.median(ratio)), 3), p10=round(float(np.percentile(ratio, 10)), 3),
                                             p90=round(float(np.percentile(ratio, 90)), 3)))
            match[label] = entry
        summary['source_camera']['picture_match'] = match
    (out / 'scene_inputs.json').write_text(json.dumps(summary))
    print(f'{frames} frames at {args.fps:g} fps ({frames / args.fps:.2f} s) from {keep} source frames -> {out}')
    for person in summary['people']:
        print(person['label'], 'z range', round(person['lowest_z'], 3), round(person['highest_z'], 3),
              'lifted at most', round(person['lifted_off_the_floor_m'], 3), 'm; bounds', person['bounds_min'], person['bounds_max'])
    if 'kept_apart' in summary:
        print('kept apart:', summary['kept_apart'])
    for label, legs in summary.get('legs_planted', {}).items():
        print('legs planted,', label, legs)
    for label, spans in summary.get('legs_out_of_the_picture', {}).items():
        print('legs out of the picture (seconds),', label, spans)
    print('source camera: roll', summary['source_camera']['roll_deg'], 'moved for size by at most',
          summary['source_camera']['moved_for_size_m'], 'm')
    for label, entry in summary['source_camera'].get('picture_match', {}).items():
        print('picture match,', label, entry)


if __name__ == '__main__':
    main()
