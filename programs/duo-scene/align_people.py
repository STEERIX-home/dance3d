"""Put every dancer's motion into one common world, seen by one camera.

ExMR reconstructs each person in a world of their own (gravity up, floor y = 0, start at the
origin), so two separately reconstructed dancers cannot simply be shown together: their distance
and facing would be arbitrary and would drift apart. What IS shared is the camera: in every
frame both people were seen by the same camera, and each person's result tells where that camera
is in their own world. Requiring those two camera poses to be the same camera gives the
transform between the two worlds, frame by frame.

  1. per person and frame: camera pose in that person's world (rigid fit of the camera-frame body
     to the world-frame body),
  2. quick motion: ExMR's world-frame motion is smoothed and loses shakes and hits. The
     camera-frame result follows the picture frame by frame, so the body is taken from there and
     carried into the world by the person's camera pose smoothed over --camera-seconds: what
     is slower than that still comes from ExMR's world, what is quicker from the picture.
     Depth along the viewing ray, the least certain direction, is smoothed over --depth-seconds,
  3. world-to-world transform G(t) (turn about the vertical axis + shift on the floor) from the two
     camera poses; smoothed over time, using only frames in which both people are reliably seen,
  4. the correction is split evenly: both people are moved half-way, so nobody carries all of it,
  5. frames in which a person is not reliably seen are never taken from the motion capture: gaps
     are bridged between the reliably seen frames on both sides; a person who is hidden behind
     another at the start stands still behind them until they first appear. Both are marked as
     estimated in the output,
  6. one camera path in the common world (for re-rendering the original view and for checks).

Which way is up (--geometry): ExMR finds gravity from the bodies alone and takes a rolled or tilted
hand-held camera for nearly level, so the camera's roll ends up in the dancers, who then lean in a
level room (measured 11-14 on a dance short filmed with a rolling phone: the floor in the picture
rolls +-18 degrees, 13 % of that reached the camera path, the rest was body lean). The floor that is
seen in the picture does not have that problem. Where scene_geometry.py found it, each dancer's
world is turned about the point where they stand until its "up" is the floor's: the dancers stand
upright again and the roll and tilt are the camera's, as in the source video.

Outputs in --output: people_world.npz, camera_world.npz, report.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

from geometry_utils import axis_angle_to_matrix, kabsch, matrix_to_axis_angle, rot_y, smooth_weighted, yaw_of
from paths import EXMR, SMPLX

sys.path.insert(0, str(EXMR))

from exmr.body_models.smplx_model import SmplxModel  # noqa: E402

FEET = slice(60, 66)  # toe and heel surface points, left foot first
PARAMS = ('global_orient', 'transl', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'betas')
ROTATIONS = ('orient', 'body', 'left_hand', 'right_hand')  # rotation matrices (L, joints, 3, 3)
MEASURED, BRIDGED, STAND_IN = 0, 1, 2
UP = np.array([0.0, 1.0, 0.0])
# the floor seen in the picture as the direction of gravity (floor_up)
FLOOR_MAX_TILT = 45.0     # degrees from the motion capture's own "up": more than that is a wall or a slope, not the floor
FLOOR_OUTLIER = 8.0       # degrees from the measurements just before and after: a wrong plane
FLOOR_SECONDS = 0.15      # smoothing over time (a measurement scatters by about 1 degree)
FLOOR_NEAR, FLOOR_FAR = 0.5, 1.5  # seconds from a measurement: full trust within NEAR, none beyond FAR


def se2_half(angle, shift):
    """Square root of floor-plane motions (turn `angle` about y, then shift (x, z)): the motion
    that applied twice gives the input. angle (L,), shift (L, 2) as (x, z)."""
    half = angle / 2

    def v_matrix(theta):
        theta = np.where(np.abs(theta) < 1e-9, 1e-9, theta)
        a, b = np.sin(theta) / theta, (1 - np.cos(theta)) / theta
        # rotation about +y maps (x, z) -> (c x + s z, -s x + c z)
        return np.stack([np.stack([a, b], -1), np.stack([-b, a], -1)], -2)

    log_shift = np.linalg.solve(v_matrix(angle), shift[..., None])[..., 0]
    return half, (v_matrix(half) @ (log_shift / 2)[..., None])[..., 0]


def apply_plane(angle, shift, points):
    """Turn points (L, N, 3) about y by angle (L,) and shift them on the floor by (L, 2)."""
    out = np.einsum('tij,tnj->tni', rot_y(angle), points)
    out[..., 0] += shift[:, None, 0]
    out[..., 2] += shift[:, None, 1]
    return out


def project_rotations(matrices):
    """Nearest rotation matrices (averaged matrices are not exactly rotations)."""
    U, _, Vt = np.linalg.svd(matrices)
    flip = np.linalg.det(U @ Vt) < 0
    U[flip, :, -1] *= -1
    return U @ Vt


def smooth_rotations(R, weights, sigma):
    return project_rotations(smooth_weighted(R, weights, sigma))


def low_pass(values, fps, cut_hz):
    """Zero-phase second-order Butterworth low-pass along time (applied forward and backward)."""
    b, a = butter(2, cut_hz / (fps / 2))
    return filtfilt(b, a, values, axis=0)


def foot_report(joints, contact, reliable, fps):
    """Planted feet: how far they slide in a quarter second, how much they tremble, how high they are."""
    feet = joints[:, FEET]
    planted = np.repeat(contact, 3, axis=1) & reliable[:, None]
    lag = max(1, int(round(0.25 * fps)))
    held = planted[:-lag] & planted[lag:]
    slide = np.linalg.norm(feet[lag:] - feet[:-lag], axis=-1)[held]
    window = max(3, int(round(0.1 * fps)) | 1)
    padded = np.pad(feet, ((window // 2, window // 2), (0, 0), (0, 0)), mode='edge')
    average = np.stack([padded[i:i + len(feet)] for i in range(window)]).mean(0)
    tremble = np.linalg.norm(feet - average, axis=-1)[planted]
    height = feet[..., 1][planted]
    return dict(slide_in_quarter_second_m=dict(median=float(np.median(slide)), p90=float(np.percentile(slide, 90))),
                tremble_mm=dict(median=float(np.median(tremble) * 1000), p90=float(np.percentile(tremble, 90) * 1000)),
                height_m=dict(median=float(np.median(height)), p5=float(np.percentile(height, 5)), p95=float(np.percentile(height, 95))))


def spans(flags):
    """[first, stop) index pairs of the runs where flags is True."""
    padded = np.concatenate([[False], np.asarray(flags, dtype=bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def smootherstep(u):
    return u * u * u * (u * (u * 6 - 15) + 10)


def interpolate(motion, first, stop, fps, ease):
    """Fill motion[first:stop] between the frames first - 1 and stop (both kept). Rotations turn
    at a constant rate (ease: slow start and arrival), the position follows a curve that leaves
    and arrives with the body's own velocity (ease: with none)."""
    a, b = first - 1, stop
    u = (np.arange(first, stop) - a) / (b - a)
    if ease:
        u = smootherstep(u)
    for key in ROTATIONS:
        for joint in range(motion[key].shape[1]):
            pair = Rotation.from_matrix(motion[key][[a, b], joint])
            motion[key][first:stop, joint] = Slerp([0, 1], pair)(u).as_matrix()
    p0, p1 = motion['pelvis'][a], motion['pelvis'][b]
    if ease:
        motion['pelvis'][first:stop] = p0 + (p1 - p0) * u[:, None]
        return
    duration = (b - a) / fps
    reach = max(1, int(round(0.1 * fps)))
    tangents = []
    for edge, other in ((a, max(0, a - reach)), (b, min(len(motion['pelvis']) - 1, b + reach))):
        velocity = (motion['pelvis'][edge] - motion['pelvis'][other]) / ((edge - other) / fps) if edge != other else np.zeros(3)
        tangent = velocity * duration
        limit = 2.0 * np.linalg.norm(p1 - p0) + 0.05  # a long gap must not throw the body off its way
        size = np.linalg.norm(tangent)
        tangents.append(tangent * min(1.0, limit / max(size, 1e-9)))
    u = u[:, None]
    motion['pelvis'][first:stop] = ((2 * u ** 3 - 3 * u ** 2 + 1) * p0 + (u ** 3 - 2 * u ** 2 + u) * tangents[0]
                                    + (-2 * u ** 3 + 3 * u ** 2) * p1 + (u ** 3 - u ** 2) * tangents[1])


def to_params(motion, rest_pelvis, betas, scale):
    """Rotation matrices and pelvis path -> SMPL-X parameters. The mesh is scaled about the world
    origin afterwards, so the translation is stored unscaled."""
    count = len(motion['pelvis'])
    return dict(global_orient=matrix_to_axis_angle(motion['orient'][:, 0]),
                transl=motion['pelvis'] / scale - rest_pelvis,
                body_pose=matrix_to_axis_angle(motion['body']).reshape(count, -1),
                left_hand_pose=matrix_to_axis_angle(motion['left_hand']).reshape(count, -1),
                right_hand_pose=matrix_to_axis_angle(motion['right_hand']).reshape(count, -1),
                betas=np.repeat(betas[None], count, axis=0))


def body_points(smplx, params, scale, vertices=False, step=64):
    count = len(params['transl'])
    function = smplx if vertices else smplx.joints
    with torch.inference_mode():
        chunks = [function(**{k: torch.as_tensor(v[i:i + step]).float() for k, v in params.items()}).numpy()
                  for i in range(0, count, step)]
    return np.concatenate(chunks).astype(np.float64) * scale


def floor_up(geometry, count, fps, focal, reference):
    """"Up" in camera coordinates (x right, y down, z forward) for every frame, from the floor plane
    that scene_geometry.py fitted on sampled frames, and how far to trust it (0..1 per frame).
    reference (count, 3): "up" as the motion capture has it, to tell a floor from something else.
    A measurement counts by how well the plane fits; one that contradicts the measurements around
    it is left out; the rest is smoothed over time. Far from any measurement (no floor in the
    picture for a while) the trust goes to 0 and the motion capture's own gravity is kept."""
    rows = []
    for sample in geometry['samples']:
        if 'floor_normal' not in sample or sample['frame'] >= count:
            continue
        normal = np.array(sample['floor_normal'], dtype=np.float64)
        normal[2] *= sample['focal_x'] / focal  # the same horizon line in the picture at the focal length used here
        normal /= np.linalg.norm(normal)
        quality = sample['floor_inlier_share'] * float(np.clip((sample['floor_normal_agreement'] - 0.9) / 0.08, 0.05, 1.0))
        rows.append((sample['frame'], normal, quality))
    report = dict(source='floor plane seen in the picture (scene_geometry.py)', measurements=len(rows))
    steep = [np.degrees(np.arccos(np.clip(normal @ reference[frame], -1, 1))) > FLOOR_MAX_TILT for frame, normal, _ in rows]
    rows = [row for row, far in zip(rows, steep) if not far]
    report['left_out_not_the_floor'] = int(sum(steep))
    # against the measurements within 0.4 s on both sides
    times = np.array([frame for frame, _, _ in rows]) / fps
    normals = np.array([normal for _, normal, _ in rows]).reshape(-1, 3)
    keep = np.ones(len(rows), dtype=bool)
    for index in range(len(rows)):
        near = (np.abs(times - times[index]) <= 0.4) & (np.arange(len(rows)) != index)
        if near.sum() >= 4:
            middle = np.median(normals[near], axis=0)
            middle /= np.linalg.norm(middle)
            keep[index] = np.degrees(np.arccos(np.clip(normals[index] @ middle, -1, 1))) <= FLOOR_OUTLIER
    report['left_out_against_neighbours'] = int((~keep).sum())
    rows = [row for row, good in zip(rows, keep) if good]
    report['used'] = len(rows)
    up, weight, good = np.zeros((count, 3)), np.zeros(count), np.zeros(count, dtype=bool)
    for frame, normal, quality in rows:
        up[frame], weight[frame] = normal, quality
        good[frame] = quality >= 0.25
    if not good.any():
        report['source'] = "the motion capture's own gravity (no floor found in the picture)"
        return reference, np.zeros(count), report
    up = smooth_weighted(up, weight, FLOOR_SECONDS * fps)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    measured = np.flatnonzero(good)
    frames = np.arange(count)
    after = measured[np.clip(np.searchsorted(measured, frames), 0, len(measured) - 1)]
    before = measured[np.clip(np.searchsorted(measured, frames, side='right') - 1, 0, len(measured) - 1)]
    away = np.minimum(np.abs(frames - after), np.abs(frames - before)) / fps
    trust = np.clip((FLOOR_FAR - away) / (FLOOR_FAR - FLOOR_NEAR), 0.0, 1.0)
    report.update(frames_fully_trusted=int((trust >= 1).sum()), frames_not_trusted=int((trust <= 0).sum()),
                  camera_roll_deg=dict(min=float(np.degrees(np.arctan2(up[:, 0], -up[:, 1])).min()),
                                       max=float(np.degrees(np.arctan2(up[:, 0], -up[:, 1])).max())),
                  camera_pitch_deg=dict(min=float(np.degrees(np.arcsin(np.clip(-up[:, 2], -1, 1))).min()),
                                        max=float(np.degrees(np.arcsin(np.clip(-up[:, 2], -1, 1))).max())))
    return up, trust, report


def stand_upright(person, up, trust, fps):
    """Turn this person's world, with its camera, until its vertical is the floor's "up" (by the
    trusted share of the angle). The turn is about the point of the floor where the person stands
    (the planted foot; between the feet when both or none are planted), so that foot stays where
    it is and the body above it comes upright. Returns the angle turned, in degrees per frame."""
    count = len(up)
    tilt = np.einsum('tij,tj->ti', person['R'], up)  # the floor's "up" in this world: (0, 1, 0) if gravity was right
    axis = np.stack([-tilt[:, 2], np.zeros(count), tilt[:, 0]], axis=-1)  # tilt x UP: horizontal
    size = np.linalg.norm(axis, axis=1)
    angle = np.arctan2(size, tilt[:, 1]) * trust
    level = axis_angle_to_matrix(axis / np.maximum(size, 1e-12)[:, None] * angle[:, None])
    feet = person['joints_world'][:, FEET]
    planted = person['contact'].astype(np.float64)
    planted = np.where(planted.sum(1, keepdims=True) > 0, planted, 0.5)
    pivot = (feet[:, :3].mean(1) * planted[:, :1] + feet[:, 3:].mean(1) * planted[:, 1:]) / planted.sum(1, keepdims=True)
    pivot[:, 1] = 0.0
    pivot = smooth_weighted(pivot, person['reliable'] + 1e-3, FLOOR_SECONDS * fps)
    person['joints_world'] = pivot[:, None] + np.einsum('tij,tnj->tni', level, person['joints_world'] - pivot[:, None])
    person['c'] = pivot + np.einsum('tij,tj->ti', level, person['c'] - pivot)
    person['R'] = level @ person['R']
    person['orient'] = level[:, None] @ person['orient']
    return np.degrees(angle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mocap', required=True, help='folder written by mocap_people.py')
    parser.add_argument('--output', required=True)
    parser.add_argument('--smooth', type=float, default=0.5, help='time constant of the world alignment (s)')
    parser.add_argument('--camera-seconds', type=float, default=0.1,
                        help='motion quicker than this comes from the camera-frame result (0: ExMR world motion as it is)')
    parser.add_argument('--depth-seconds', type=float, default=0.08, help='smoothing of the distance from the camera')
    parser.add_argument('--camera-flow', help='traces of the source video (qa_fast_motion.py video): the quick part of '
                        'the background motion is the hand-held camera trembling, not the dancers, and is taken out')
    parser.add_argument('--final-hz', type=float, default=9.0,
                        help='last filter against frame-to-frame noise: motion quicker than this is removed '
                             '(a 4.5 Hz shake keeps 94 %% of its size at 9 Hz); 0: none')
    parser.add_argument('--behind', type=float, default=None,
                        help='distance (m) at which a dancer hidden at the start stands behind the other one '
                             '(default: their distance along the view when first reliably seen)')
    parser.add_argument('--geometry', help='geometry.json of scene_geometry.py: where it found the floor, "up" is taken '
                        "from the floor instead of from ExMR's own gravity, which takes a rolled camera for level")
    args = parser.parse_args()
    mocap, out = Path(args.mocap), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((mocap / 'people.json').read_text())
    fps, count = info['fps'], info['frames']
    if len(info['people']) != 2:
        raise NotImplementedError('The alignment is written for two people')
    smplx = SmplxModel(SMPLX).eval()
    K = torch.load(mocap / 'camera.pt', map_location='cpu', weights_only=False)['K'].numpy().astype(np.float64)

    # quick camera motion, seen as the background moving in the picture: undone by turning the
    # camera-frame bodies about the camera (the slow camera motion is in the smoothed camera pose)
    steady = np.repeat(np.eye(3)[None], count, axis=0)
    if args.camera_flow and args.camera_seconds > 0:
        flow = json.loads(Path(args.camera_flow).read_text())
        if abs(flow['fps'] - fps) > 1e-3:
            raise ValueError('The background motion was measured at another frame rate than the motion capture')
        start = int(round(flow['start'] * fps))
        path = np.concatenate([np.zeros((1, 2)), np.cumsum(np.nan_to_num(np.asarray(flow['traces']['camera'], dtype=np.float64)), axis=0)])
        shift = np.zeros((count, 2))
        stop = min(count, start + len(path))
        shift[start:stop] = path[:stop - start]
        shift[stop:] = shift[stop - 1]
        quick = shift - smooth_weighted(shift, np.ones(count), args.camera_seconds * fps)
        yaw, pitch = -quick[:, 0] / K[0, 0], quick[:, 1] / K[1, 1]
        turn_y, turn_x = steady.copy(), steady.copy()
        turn_y[:, 0, 0], turn_y[:, 0, 2], turn_y[:, 2, 0], turn_y[:, 2, 2] = np.cos(yaw), np.sin(yaw), -np.sin(yaw), np.cos(yaw)
        turn_x[:, 1, 1], turn_x[:, 1, 2], turn_x[:, 2, 1], turn_x[:, 2, 2] = np.cos(pitch), -np.sin(pitch), np.sin(pitch), np.cos(pitch)
        steady = turn_y @ turn_x
        camera_tremble = dict(rms_px=[float(v) for v in quick.std(0)], max_px=[float(v) for v in np.abs(quick).max(0)])
    else:
        camera_tremble = None

    people = []
    for entry in info['people']:
        result = torch.load(mocap / f'person_{entry["label"]}/results.pt', map_location='cpu', weights_only=False)
        joints_cam = result['joints_incam'].numpy().astype(np.float64)
        joints_world = result['joints_global'].numpy().astype(np.float64)
        reliable = np.array(entry['reliable'])
        R, c, rms = np.zeros((count, 3, 3)), np.zeros((count, 3)), np.zeros(count)
        for t in range(count):
            R[t], c[t], rms[t] = kabsch(joints_cam[t, :22], joints_world[t, :22])
        source = result['smplx_incam' if args.camera_seconds > 0 else 'smplx_global']
        source = {k: source[k].numpy().astype(np.float64) for k in PARAMS}
        orient = axis_angle_to_matrix(source['global_orient'])
        if args.camera_seconds > 0:
            # the body as the camera saw it, carried into the world by the smoothed camera pose
            weight = reliable.astype(np.float64)
            R = smooth_rotations(R, weight, args.camera_seconds * fps)
            c = smooth_weighted(c, weight, args.camera_seconds * fps)
            depth = smooth_weighted(joints_cam[:, 0, 2:], weight, max(args.depth_seconds * fps, 1e-3))[:, 0]
            slide = joints_cam[:, 0] * (depth / joints_cam[:, 0, 2] - 1)[:, None]  # along the pelvis' viewing ray
            joints_cam_used = np.einsum('tij,tnj->tni', steady, joints_cam + slide[:, None])
            joints_world = np.einsum('tij,tnj->tni', R, joints_cam_used) + c[:, None]
            orient = R @ steady @ orient
        else:
            joints_cam_used = joints_cam
        people.append(dict(
            label=entry['label'], reliable=reliable, observed=np.array(entry['observed']),
            joints_cam=joints_cam_used, joints_world=joints_world, R=R, c=c, fit_rms=rms,
            contact=result['foot_contact'].numpy().astype(bool), betas=source['betas'][0],
            rest_pelvis=(result['joints_incam'][:, 0].numpy() - result['smplx_incam']['transl'].numpy()).astype(np.float64).mean(0),
            orient=orient[:, None], body=axis_angle_to_matrix(source['body_pose'].reshape(count, -1, 3)),
            left_hand=axis_angle_to_matrix(source['left_hand_pose'].reshape(count, -1, 3)),
            right_hand=axis_angle_to_matrix(source['right_hand_pose'].reshape(count, -1, 3))))

    # which way is up: the floor seen in the picture, where there is one (see the top of this file)
    gravity = dict(source="the motion capture's own gravity (no floor measurement given)")
    if args.geometry:
        own = [np.einsum('tji,j->ti', person['R'], UP) for person in people]  # each world's "up" in camera coordinates
        reference = sum(up * (person['reliable'][:, None] + 1e-3) for up, person in zip(own, people))
        reference /= np.linalg.norm(reference, axis=1, keepdims=True)
        up, trust, gravity = floor_up(json.loads(Path(args.geometry).read_text()), count, fps, K[0, 0], reference)
        if trust.any():
            gravity['turned_deg'] = {}
            for person in people:
                turned = stand_upright(person, up, trust, fps)[person['reliable']]
                gravity['turned_deg'][person['label']] = dict(median=float(np.median(turned)), p90=float(np.percentile(turned, 90)),
                                                              max=float(turned.max()))
    A, B = people
    both = (A['reliable'] & B['reliable']).astype(np.float64)
    if both.sum() < fps:
        raise RuntimeError('The two people are reliably visible together for less than a second')

    # relative size: one camera above one floor -> both must imply the same camera height
    usable = (both > 0) & A['contact'].any(1) & B['contact'].any(1)
    log_ratio = np.log(A['c'][usable, 1] / B['c'][usable, 1])
    height_ratio = float(np.exp(np.median(log_ratio)))
    height_spread = float(1.4826 * np.median(np.abs(log_ratio - np.median(log_ratio))))
    # a person reconstructed too large is also placed too far and "sees" the camera too high
    scale_A, scale_B = (1 / np.sqrt(height_ratio), np.sqrt(height_ratio)) if abs(np.log(height_ratio)) > 0.02 else (1.0, 1.0)
    for person, scale in ((A, scale_A), (B, scale_B)):
        person['scale'] = float(scale)
        person['c'] = person['c'] * scale
        person['joints_world'] = person['joints_world'] * scale

    # world B -> world A, per frame, from "it is the same camera"
    sigma = args.smooth * fps
    angle_raw = yaw_of(A['R'] @ np.swapaxes(B['R'], 1, 2))
    # angles are averaged as directions, so that a full turn in an unreliable frame cannot leak in
    direction = smooth_weighted(np.stack([np.cos(angle_raw), np.sin(angle_raw)], -1), both, sigma)
    angle = np.unwrap(np.arctan2(direction[:, 1], direction[:, 0]))
    # the shift is measured where it matters, at B's pelvis: where A's camera sees it, against where
    # B's own world puts it (measuring at the far-away world origin would turn small angle noise
    # into large position noise)
    pelvis_B_seen = np.einsum('tij,tj->ti', A['R'], B['joints_cam'][:, 0] * B['scale']) + A['c']
    shift_raw = (pelvis_B_seen - np.einsum('tij,tj->ti', rot_y(angle), B['joints_world'][:, 0]))[:, [0, 2]]
    shift = smooth_weighted(shift_raw, both, sigma)

    # split around the mean: E(t) = G(t) o G0^-1;  A moves by E^-1/2, B by E^1/2 o G0
    angle0 = float(np.average(angle, weights=both))
    shift0 = np.average(shift, axis=0, weights=both)
    e_angle = angle - angle0
    rot_e = rot_y(e_angle)[:, [0, 2]][:, :, [0, 2]]
    e_shift = shift - np.einsum('tij,j->ti', rot_e, shift0)
    half_angle, half_shift = se2_half(e_angle, e_shift)
    rot_half = rot_y(half_angle)[:, [0, 2]][:, :, [0, 2]]
    # B: first G0, then half of E
    B_angle = half_angle + angle0
    B_shift = np.einsum('tij,j->ti', rot_half, shift0) + half_shift
    # A: inverse of half of E
    A_angle = -half_angle
    A_shift = -np.einsum('tji,tj->ti', rot_half, half_shift)

    cameras = []
    for person, (plane_angle, plane_shift) in ((A, (A_angle, A_shift)), (B, (B_angle, B_shift))):
        person['plane_angle'], person['plane_shift'] = plane_angle, plane_shift
        person['joints_measured'] = apply_plane(plane_angle, plane_shift, person['joints_world'])
        cameras.append((rot_y(plane_angle) @ person['R'], apply_plane(plane_angle, plane_shift, person['c'][:, None])[:, 0]))
        person['motion'] = dict(orient=rot_y(plane_angle)[:, None] @ person['orient'], body=person['body'].copy(),
                                left_hand=person['left_hand'].copy(), right_hand=person['right_hand'].copy(),
                                pelvis=person['joints_measured'][:, 0].copy())

    # one camera: average of the two estimates, weighted by who is reliably seen
    weight_A = smooth_weighted(A['reliable'].astype(float), np.ones(count), 3)
    weight_B = smooth_weighted(B['reliable'].astype(float), np.ones(count), 3)
    total = np.maximum(weight_A + weight_B, 1e-6)
    seen = ((A['reliable'] | B['reliable'])).astype(float)
    camera_sigma = max(2.0, args.camera_seconds * fps)
    R_mix = (cameras[0][0] * weight_A[:, None, None] + cameras[1][0] * weight_B[:, None, None]) / total[:, None, None]
    c_mix = (cameras[0][1] * weight_A[:, None] + cameras[1][1] * weight_B[:, None]) / total[:, None]
    R_cam = smooth_rotations(R_mix, seen, camera_sigma)
    c_cam = smooth_weighted(c_mix, seen, camera_sigma)

    # frames that were not reliably seen: never the motion capture's guess
    estimates = []
    for person, other in ((A, B), (B, A)):
        motion, good = person['motion'], person['reliable'].copy()
        kind = np.full(count, MEASURED)
        for first, stop in spans(~good):
            if first == 0 and stop < count:
                # hidden at the start: stand still behind the other dancer (their own standing pose at
                # the first frame), in line with the camera, until the first sight; then come out
                appear = np.flatnonzero(person['observed'][:stop])
                appear = int(appear[0]) if len(appear) else stop
                appear = max(1, min(appear, stop - max(2, int(0.2 * fps))))
                view = other['motion']['pelvis'][:stop] - c_cam[:stop]
                view[:, 1] = 0
                view /= np.linalg.norm(view, axis=1, keepdims=True)
                if args.behind is not None:
                    behind = args.behind
                else:
                    ahead = slice(stop, stop + max(2, int(0.2 * fps)))
                    gap = (motion['pelvis'][ahead] - other['motion']['pelvis'][ahead])[:, [0, 2]]
                    look = other['motion']['pelvis'][ahead] - c_cam[ahead]
                    look = look[:, [0, 2]] / np.linalg.norm(look[:, [0, 2]], axis=1, keepdims=True)
                    behind = float(np.clip(np.median((gap * look).sum(1)), 0.35, 0.8))
                for key in ROTATIONS:
                    motion[key][:appear] = other['motion'][key][0]
                # slowly following, so that the hidden dancer does not copy the front dancer's small moves
                hidden = smooth_weighted(other['motion']['pelvis'][:stop] + behind * view, np.ones(stop), 0.5 * fps)
                motion['pelvis'][:appear] = hidden[:appear]
                # feet on the floor for this body's own size
                standing = to_params({k: v[:1] for k, v in motion.items()}, person['rest_pelvis'], person['betas'], person['scale'])
                lowest = body_points(smplx, standing, person['scale'], vertices=True)[0, :, 1].min()
                motion['pelvis'][:appear, 1] -= lowest
                if appear < stop:
                    interpolate(motion, appear, stop, fps, ease=True)
                kind[:stop] = STAND_IN
                estimates.append(dict(label=person['label'], kind='stands behind the other dancer, then comes out',
                                      seconds=[0.0, round(stop / fps, 3)], comes_out_from=round(appear / fps, 3),
                                      behind_m=round(behind, 3)))
            elif stop == count:
                for key in (*ROTATIONS, 'pelvis'):
                    motion[key][first:] = motion[key][first - 1]
                kind[first:] = BRIDGED
                estimates.append(dict(label=person['label'], kind='last seen pose held', seconds=[round(first / fps, 3), round(stop / fps, 3)]))
            else:
                interpolate(motion, first, stop, fps, ease=False)
                kind[first:stop] = BRIDGED
                estimates.append(dict(label=person['label'], kind='bridged between the seen frames',
                                      seconds=[round(first / fps, 3), round(stop / fps, 3)]))
        if args.final_hz > 0:
            for key in ROTATIONS:
                motion[key] = project_rotations(low_pass(motion[key], fps, args.final_hz))
            motion['pelvis'] = low_pass(motion['pelvis'], fps, args.final_hz)
        person['kind'] = kind
        person['params_common'] = to_params(motion, person['rest_pelvis'], person['betas'], person['scale'])
        person['joints_common'] = body_points(smplx, person['params_common'], person['scale'])

    # checks
    report = dict(frames=count, fps=fps, smooth_seconds=args.smooth, camera_seconds=args.camera_seconds,
                  depth_seconds=args.depth_seconds, final_hz=args.final_hz,
                  both_reliable_frames=int(both.sum()),
                  camera_height_ratio_A_over_B=height_ratio, camera_height_ratio_spread=height_spread,
                  world_turn_deg=dict(mean=float(np.degrees(angle0)), min=float(np.degrees(angle.min())),
                                      max=float(np.degrees(angle.max()))),
                  world_shift_range_m=dict(x=[float(shift[:, 0].min()), float(shift[:, 0].max())],
                                           z=[float(shift[:, 1].min()), float(shift[:, 1].max())]),
                  camera_tremble_removed=camera_tremble, gravity=gravity, estimated=estimates, people=[])
    for person in (A, B):
        # re-projection of the common-world body through the common camera vs the camera-frame result
        cam_points = np.einsum('tji,tnj->tni', R_cam, person['joints_common'][:, :22] - c_cam[:, None])
        pixels = np.einsum('ij,tnj->tni', K, cam_points)
        pixels = pixels[..., :2] / pixels[..., 2:]
        reference = np.einsum('ij,tnj->tni', K, person['joints_cam'][:, :22])
        reference = reference[..., :2] / reference[..., 2:]
        error = np.linalg.norm(pixels - reference, axis=-1).mean(1)[person['reliable']]
        report['people'].append(dict(
            label=person['label'], scale=person['scale'], reliable_frames=int(person['reliable'].sum()),
            observed_frames=int(person['observed'].sum()), estimated_frames=int((person['kind'] != MEASURED).sum()),
            camera_fit_rms_m=float(np.median(person['fit_rms'])),
            reprojection_px=dict(median=float(np.median(error)), p90=float(np.percentile(error, 90))),
            planted_feet=foot_report(person['joints_common'], person['contact'], person['reliable'], fps),
            body_height_m=float((person['joints_common'][:, 15, 1]).max())))
    distance = np.linalg.norm((A['joints_common'][:, 0] - B['joints_common'][:, 0])[:, [0, 2]], axis=1)
    report['pelvis_distance_m'] = dict(median=float(np.median(distance[both > 0])), min=float(distance[both > 0].min()),
                                       max=float(distance[both > 0].max()), min_all_frames=float(distance.min()))
    # agreement of the common world with what the camera saw: B's pelvis relative to A's, both ways
    seen_B = np.einsum('tij,tj->ti', R_cam, B['joints_cam'][:, 0] * B['scale']) + c_cam
    seen_A = np.einsum('tij,tj->ti', R_cam, A['joints_cam'][:, 0] * A['scale']) + c_cam
    relative_error = np.linalg.norm(((B['joints_common'][:, 0] - A['joints_common'][:, 0]) - (seen_B - seen_A))[:, [0, 2]], axis=1)
    report['relative_position_error_m'] = dict(median=float(np.median(relative_error[both > 0])),
                                               p90=float(np.percentile(relative_error[both > 0], 90)))
    up_seen = np.einsum('tji,j->ti', R_cam, UP)[seen > 0]  # the common world's "up" in camera coordinates
    roll, pitch = np.degrees(np.arctan2(up_seen[:, 0], -up_seen[:, 1])), np.degrees(np.arcsin(np.clip(-up_seen[:, 2], -1, 1)))
    report['camera'] = dict(height_m=dict(median=float(np.median(c_cam[seen > 0, 1])), min=float(c_cam[seen > 0, 1].min()),
                                          max=float(c_cam[seen > 0, 1].max())),
                            roll_deg=dict(min=float(roll.min()), max=float(roll.max())),
                            pitch_deg=dict(min=float(pitch.min()), max=float(pitch.max())),
                            source='average of the camera poses implied by the two bodies; not an independent measurement')
    (out / 'report.json').write_text(json.dumps(report, indent=1))

    save = dict(fps=fps, labels=np.array([A['label'], B['label']]))
    for person in (A, B):
        label = person['label']
        for key, value in person['params_common'].items():
            save[f'{label}_{key}'] = value.astype(np.float32)
        save[f'{label}_scale'] = np.float32(person['scale'])
        save[f'{label}_joints'] = person['joints_common'].astype(np.float32)
        save[f'{label}_reliable'] = person['reliable']
        save[f'{label}_observed'] = person['observed']
        save[f'{label}_estimated'] = person['kind'].astype(np.uint8)
        save[f'{label}_contact'] = person['contact']
        save[f'{label}_plane_angle'] = person['plane_angle'].astype(np.float32)
        save[f'{label}_plane_shift'] = person['plane_shift'].astype(np.float32)
    np.savez(out / 'people_world.npz', **save)
    np.savez(out / 'camera_world.npz', R_c2w=R_cam.astype(np.float32), center=c_cam.astype(np.float32),
             K=K.astype(np.float32), fps=fps, width=info['width'], height=info['height'], seen=seen > 0)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
