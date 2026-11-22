"""The one dancer's motion in a world whose "up" is the floor seen in the picture, written in the
form of the one-person program's results.npz (ExMR's own file), as a BVH and as an AMASS file.

The steps are those of the two-dancer alignment (duo-scene/align_people.py) for a single person:
  1. per frame: where the camera is in ExMR's world (rigid fit of the camera-frame body to the
     world-frame body),
  2. quick motion: ExMR's world motion is smoothed and loses shakes and hits. The camera-frame
     result follows the picture frame by frame, so the body is taken from there and carried into the
     world by the camera pose smoothed over --camera-seconds (never across a cut). Before that the
     quick part of the background's motion in the picture (a trembling hand-held camera, not the
     dancer) is taken out. Across a cut to another shot (--cuts) the camera may change, but the
     dancer does not jump: the world after the cut is turned and shifted so that the dancer's facing
     and position go on,
  3. which way is up: ExMR finds gravity from the body alone and takes a rolled or tilted hand-held
     camera for level, so the camera's roll would end up in the dancer as a lean. Where
     scene_geometry.py found the floor in the picture, the world is turned about the point where the
     dancer stands until its "up" is the floor's. Only if the dancer stands on that floor: the
     planted feet in the picture must meet the measured plane at their own depth (feet_on_floor);
     a dancer put in front of a picture of a room (green screen, game backgrounds) does not, and
     then ExMR's own gravity is kept. The floor is measured shot by shot (a cut changes the camera,
     so the floor of one shot says nothing about the next), and a plane more than FLOOR_TILT from
     the body's own "up" is a wall or a slope, not the floor,
  4. legs that are not in the picture (a close-up, a selfie part) are ExMR's guess, often a crouch
     that would read as a deep up-and-down move. In those frames the leg joints and the pelvis
     height come from the frames on both sides where the legs are seen (held before the first and
     after the last one), the rest of the body stays as measured; marked in legs_estimated,
  5. frames in which the dancer is not reliably seen at all are not taken from the motion capture:
     they are bridged between the seen frames on both sides and marked as estimated,
  6. a last low-pass filter against frame-to-frame noise.

The floor is y = 0 and y points up; lengths in metres. results.npz has the keys of ExMR's file
(exmr-h3/docs/OUTPUTS.md), the world ("global") ones with the values of the steps above, and these
besides: reliable, estimated (0 seen, 1 bridged or held), legs_estimated, floor_trust (0..1),
turned_to_floor_deg, camera_R_c2w, camera_center (the camera in that world), flipped_left_right.

    world_one.py --mocap MOCAP_DIR [--geometry geometry.json] [--camera-flow flow.json] [--cuts cuts.json]
                 [--camera-held camera_held.json] --output DIR

Outputs in --output: results.npz, motion.bvh, motion_amass.npz, camera_world.npz, report.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from align_people import (BRIDGED, MEASURED, PARAMS, ROTATIONS, UP, body_points, floor_up, foot_report, interpolate,
                          low_pass, project_rotations, smooth_rotations, smootherstep, spans, stand_upright, to_params)
from geometry_utils import axis_angle_to_matrix, kabsch, rot_y, smooth_weighted, yaw_of
from paths import EXMR, SMPLX
from plant_legs import LEGS, feet_in_picture
from refine_motion import HIPS, KEYPOINT_JOINTS

sys.path.insert(0, str(EXMR))

from exmr.body_models.smplx_model import SmplxModel  # noqa: E402
from exmr.export import save_amass, save_bvh  # noqa: E402

LEG_ROWS = [joint - 1 for leg in LEGS.values() for joint in leg['joints']]  # body_pose rows of hips, knees, ankles, feet
# the floor seen in the picture must be the floor the dancer stands on (feet_on_floor)
FEET_CHECK_LEAST = 10   # planted feet in the picture needed for a verdict
FEET_ON_FLOOR = 0.15    # floor depth / foot depth within this of the video's median: the foot is on that floor
FEET_FLOOR_SHARE = 0.7  # below this share the floor seen is not the dancer's (11-14: real videos 0.88-1.00, a composite 0.42)
# degrees between a measured plane's "up" and the body's: more is not the floor. Hand-held dance videos
# rolled up to 18 degrees (11-14); after a cut to a close-up the planes found were walls at 36-72
FLOOR_TILT = 30.0


def shots_of(cuts, count):
    """[first, stop) of every shot between the cuts."""
    edges = [0, *[cut for cut in cuts if 0 < cut < count], count]
    return [(first, stop) for first, stop in zip(edges[:-1], edges[1:]) if stop > first]


def by_shot(function, values, weights, sigma, shots):
    """function(values, weights, sigma) on every shot on its own: nothing is smoothed across a cut."""
    out = np.empty(np.shape(values))
    for first, stop in shots:
        part = weights[first:stop]
        out[first:stop] = function(values[first:stop], part if part.any() else np.ones_like(part), sigma)
    return out


def join_shots(person, cuts, reliable):
    """Across a cut the camera may change (another camera, another angle) but the dancer does not
    jump: the world after each cut is turned about the vertical and shifted on the floor so that the
    dancer's facing and position go on from the last reliable frame before it. Returns the turns
    (degrees)."""
    turns = []
    seen = np.flatnonzero(reliable)
    for cut in cuts:
        before, after = seen[seen < cut], seen[seen >= cut]
        if not len(before) or not len(after):
            continue
        a, b = before[-1], after[0]
        angle = yaw_of(person['orient'][a, 0]) - yaw_of(person['orient'][b, 0])
        turn = rot_y(angle)
        pivot = person['joints_world'][b, 0] * [1, 0, 1]
        later = slice(cut, None)
        person['joints_world'][later] = pivot + np.einsum('ij,tnj->tni', turn, person['joints_world'][later] - pivot)
        person['c'][later] = pivot + (person['c'][later] - pivot) @ turn.T
        person['R'][later] = turn @ person['R'][later]
        person['orient'][later] = turn @ person['orient'][later]
        shift = (person['joints_world'][a, 0] - person['joints_world'][b, 0]) * [1, 0, 1]
        person['joints_world'][later] += shift
        person['c'][later] += shift
        turns.append(round(float(np.degrees(np.arctan2(np.sin(angle), np.cos(angle)))), 1))
    return turns


def camera_steady(flow_path, camera_seconds, fps, count, K, cuts, width):
    """Turns of the camera-frame body that undo the quick part of the background's motion in the
    picture, as in duo-scene/align_people.py (identity without a flow file). A jump of the background
    at a cut, or a background that slides by in an edit (more than 5 % of the picture's width in one
    frame), is not a trembling hand and is left out."""
    steady = np.repeat(np.eye(3)[None], count, axis=0)
    if not flow_path:
        return steady, None
    flow = json.loads(Path(flow_path).read_text())
    if abs(flow['fps'] - fps) > 1e-3:
        raise ValueError('The background motion was measured at another frame rate than the motion capture')
    start = int(round(flow['start'] * fps))
    velocity = np.nan_to_num(np.asarray(flow['traces']['camera'], dtype=np.float64))
    velocity[np.linalg.norm(velocity, axis=1) > 0.05 * width] = 0
    for cut in cuts:
        if 0 <= cut - 1 - start < len(velocity):
            velocity[cut - 1 - start] = 0
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(velocity, axis=0)])
    shift = np.zeros((count, 2))
    stop = min(count, start + len(path))
    shift[start:stop] = path[:stop - start]
    shift[stop:] = shift[stop - 1]
    quick = shift - smooth_weighted(shift, np.ones(count), camera_seconds * fps)
    yaw, pitch = -quick[:, 0] / K[0, 0], quick[:, 1] / K[1, 1]
    turn_y, turn_x = steady.copy(), steady.copy()
    turn_y[:, 0, 0], turn_y[:, 0, 2], turn_y[:, 2, 0], turn_y[:, 2, 2] = np.cos(yaw), np.sin(yaw), -np.sin(yaw), np.cos(yaw)
    turn_x[:, 1, 1], turn_x[:, 1, 2], turn_x[:, 2, 1], turn_x[:, 2, 2] = np.cos(pitch), -np.sin(pitch), np.sin(pitch), np.cos(pitch)
    return turn_y @ turn_x, dict(rms_px=[float(v) for v in quick.std(0)], max_px=[float(v) for v in np.abs(quick).max(0)])


def feet_on_floor(geometry, joints_cam, contact, usable, K, width, height):
    """Is the floor that scene_geometry.py measured the floor the dancer stands on? For each sampled
    frame with a planted foot in the picture: the depth at which the foot's viewing ray meets the
    measured plane, against the motion capture's own depth of that foot. On the dancer's floor the
    two agree up to one scale for the whole video (11-14: 88-100 % of the samples within 15 % of the
    median in three real videos, 42 % for a dancer put onto changing game backgrounds, whose "floor"
    is part of a picture behind them).
    Returns (report, use the floor, sampled frames whose plane contradicts the feet)."""
    offsets = {}
    for sample in geometry['samples']:
        frame = sample['frame']
        if 'floor_normal' not in sample or frame >= len(joints_cam) or not usable[frame]:
            continue
        normal = np.array(sample['floor_normal'])
        for side, feet in enumerate((slice(60, 63), slice(63, 66))):  # toes and heel of the left, then the right foot
            if contact[frame, side]:
                point = joints_cam[frame, feet].mean(0)
                pixel = K[:2, :2] @ (point[:2] / point[2]) + K[:2, 2]
                ray = np.array([*((pixel - [width / 2, height / 2]) / sample['focal_x']), 1.0])
                along = normal @ ray
                ratio = -sample['camera_height'] / along / point[2] if along < 0 else 0.0
                offsets.setdefault(frame, []).append(np.log(ratio) if ratio > 0 else np.inf)
    report = dict(checked_feet=sum(len(v) for v in offsets.values()))
    if report['checked_feet'] < FEET_CHECK_LEAST:
        report['verdict'] = 'too few planted feet in the picture to check'
        return report, True, set()
    values = np.concatenate([v for v in offsets.values()])
    middle = float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else 0.0
    share = float(np.mean(np.abs(values - middle) < np.log(1 + FEET_ON_FLOOR)))
    report.update(floor_over_foot_depth=float(np.exp(middle)), share_on_floor=share)
    if share < FEET_FLOOR_SHARE:
        report['verdict'] = 'the floor seen in the picture is not the one the dancer stands on'
        return report, False, set()
    wrong = {frame for frame, value in offsets.items() if np.median(np.abs(np.array(value) - middle)) > np.log(1 + 2 * FEET_ON_FLOOR)}
    report.update(verdict='the dancer stands on the floor seen in the picture', samples_left_out=len(wrong))
    return report, True, wrong


def floor_by_shot(geometry, shots, fps, focal, reference):
    """floor_up (duo-scene/align_people.py) on every shot on its own, leaving out planes more than
    FLOOR_TILT from the body's "up". Returns up, trust and a report per shot."""
    count = len(reference)
    up, trust, reports = reference.copy(), np.zeros(count), []
    for first, stop in shots:
        samples = []
        for sample in geometry['samples']:
            if not first <= sample['frame'] < stop:
                continue
            sample = dict(sample, frame=sample['frame'] - first)
            if 'floor_normal' in sample:
                normal = np.array(sample['floor_normal'], dtype=np.float64)
                normal[2] *= sample['focal_x'] / focal  # as floor_up sees it
                normal /= np.linalg.norm(normal)
                if np.degrees(np.arccos(np.clip(normal @ reference[first + sample['frame']], -1, 1))) > FLOOR_TILT:
                    sample.pop('floor_normal')
            samples.append(sample)
        part_up, part_trust, report = floor_up(dict(geometry, samples=samples), stop - first, fps, focal, reference[first:stop])
        up[first:stop], trust[first:stop] = part_up, part_trust
        reports.append(dict(seconds=[round(first / fps, 3), round(stop / fps, 3)], used=report.get('used', 0),
                            frames_fully_trusted=report.get('frames_fully_trusted', 0)))
    return up, trust, reports


def bridge_legs(motion, good):
    """The leg rotations and the pelvis height in the frames where the legs are not seen: from the
    frames on both sides where they are (held before the first and after the last one); the rest of
    the body stays as it is. Returns the filled spans [first, stop)."""
    count = len(good)
    if not good.any():
        return []
    filled = spans(~good)
    for first, stop in filled:
        if first == 0 or stop == count:
            edge = stop if first == 0 else first - 1
            motion['body'][first:stop, LEG_ROWS] = motion['body'][edge, LEG_ROWS]
            motion['pelvis'][first:stop, 1] = motion['pelvis'][edge, 1]
            continue
        a, b = first - 1, stop
        u = smootherstep((np.arange(first, stop) - a) / (b - a))
        for row in LEG_ROWS:
            motion['body'][first:stop, row] = Slerp([0, 1], Rotation.from_matrix(motion['body'][[a, b], row]))(u).as_matrix()
        motion['pelvis'][first:stop, 1] = motion['pelvis'][a, 1] + (motion['pelvis'][b, 1] - motion['pelvis'][a, 1]) * u
    return filled


def picture_error(joints, R_cam, c_cam, K, keypoints, boxes, reliable, names):
    """How far the final world joints, seen through the camera path, are from the 2D keypoints
    detected in the picture: per frame the median over confident keypoints, in percent of the
    person's box height (the hips left out: a detector's hip points sit elsewhere on the pelvis)."""
    used = [i for i in range(len(KEYPOINT_JOINTS)) if i not in HIPS]
    index = [names.index(KEYPOINT_JOINTS[i]) for i in used]
    points = np.einsum('tji,tnj->tni', R_cam, joints[:, index] - c_cam[:, None])
    pixels = points @ K.T
    pixels = pixels[..., :2] / np.maximum(pixels[..., 2:], 1e-6)
    detected, confidence = keypoints[:, used, :2], keypoints[:, used, 2]
    height = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    distance = np.linalg.norm(pixels - detected, axis=-1) / height[:, None] * 100
    distance = np.where(confidence > 0.5, distance, np.nan)
    good = reliable & (np.isfinite(distance).sum(1) >= 6)
    per_frame = np.full(len(joints), np.nan)
    per_frame[good] = np.nanmedian(distance[good], axis=1)
    return per_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mocap', required=True, help='folder written by refine_motion.py (or mocap_people.py), one person')
    parser.add_argument('--output', required=True)
    parser.add_argument('--geometry', help='geometry.json of scene_geometry.py: "up" from the floor seen in the picture')
    parser.add_argument('--camera-flow', help='traces of the video (qa_fast_motion.py video): the quick part of the '
                        'background motion is a trembling camera and is taken out')
    parser.add_argument('--camera-seconds', type=float, default=0.1,
                        help='motion quicker than this comes from the camera-frame result (seconds, above 0)')
    parser.add_argument('--depth-seconds', type=float, default=0.08, help='smoothing of the distance from the camera')
    parser.add_argument('--final-hz', type=float, default=9.0,
                        help='motion quicker than this is removed at the end (a 4.5 Hz shake keeps 94 %% of its size); 0: none')
    parser.add_argument('--cuts', help='cuts.json of find_cuts.py: first frames of new shots')
    parser.add_argument('--camera-held', help='camera_held.json of mocap_people.py: steps where no camera turn was measured')
    parser.add_argument('--flipped', action='store_true', help='the video was turned left-right before the motion capture')
    args = parser.parse_args()
    if args.camera_seconds <= 0:
        parser.error('--camera-seconds must be above 0')
    mocap, out = Path(args.mocap), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((mocap / 'people.json').read_text())
    fps, count = info['fps'], info['frames']
    if len(info['people']) != 1:
        raise ValueError('world_one.py is written for one person')
    entry = info['people'][0]
    smplx = SmplxModel(SMPLX).eval()
    K = torch.load(mocap / 'camera.pt', map_location='cpu', weights_only=False)['K'].numpy().astype(np.float64)
    cuts = json.loads(Path(args.cuts).read_text())['cuts'] if args.cuts else []
    shots = shots_of(cuts, count)
    steady, tremble = camera_steady(args.camera_flow, args.camera_seconds, fps, count, K, cuts, info['width'])

    result = torch.load(mocap / f'person_{entry["label"]}/results.pt', map_location='cpu', weights_only=False)
    names = [str(name) for name in (result['joint_names'] if 'joint_names' in result else smplx.joint_names)]
    reliable = np.array(entry['reliable'])
    if reliable.sum() < fps:
        raise RuntimeError('The dancer is reliably seen for less than a second')
    joints_cam = result['joints_incam'].numpy().astype(np.float64)
    joints_world = result['joints_global'].numpy().astype(np.float64)
    keypoints = torch.load(mocap / f'person_{entry["label"]}/keypoints.pt', map_location='cpu',
                           weights_only=False)['value'][:, :23].numpy().astype(np.float64)
    contact = result['foot_contact'].numpy().astype(bool)
    legs_seen = reliable & feet_in_picture(keypoints, info['width'], info['height'])
    R, c, fit_rms = np.zeros((count, 3, 3)), np.zeros((count, 3)), np.zeros(count)
    for t in range(count):
        R[t], c[t], fit_rms[t] = kabsch(joints_cam[t, :22], joints_world[t, :22])
    source = {k: result['smplx_incam'][k].numpy().astype(np.float64) for k in PARAMS}
    rest_pelvis = (result['joints_incam'][:, 0].numpy() - result['smplx_incam']['transl'].numpy()).astype(np.float64).mean(0)

    # the body as the camera saw it, carried into the world by the smoothed camera pose
    weight = reliable.astype(np.float64)
    R = by_shot(smooth_rotations, R, weight, args.camera_seconds * fps, shots)
    c = by_shot(smooth_weighted, c, weight, args.camera_seconds * fps, shots)
    depth = by_shot(smooth_weighted, joints_cam[:, 0, 2:], weight, max(args.depth_seconds * fps, 1e-3), shots)[:, 0]
    slide = joints_cam[:, 0] * (depth / joints_cam[:, 0, 2] - 1)[:, None]  # along the pelvis' viewing ray
    joints_cam = np.einsum('tij,tnj->tni', steady, joints_cam + slide[:, None])
    person = dict(reliable=reliable, R=R, c=c, contact=contact,
                  joints_world=np.einsum('tij,tnj->tni', R, joints_cam) + c[:, None],
                  orient=(R @ steady @ axis_angle_to_matrix(source['global_orient']))[:, None])
    joined = join_shots(person, cuts, reliable)

    # which way is up: the floor seen in the picture, where there is one (see the top of this file)
    gravity = dict(source="ExMR's own gravity (no floor measurement given)")
    trust, turned = np.zeros(count), np.zeros(count)
    if args.geometry:
        geometry = json.loads(Path(args.geometry).read_text())
        check, use_floor, wrong = feet_on_floor(geometry, result['joints_incam'].numpy().astype(np.float64), contact, legs_seen, K,
                                                info['width'], info['height'])
        if not use_floor:
            gravity = dict(source="ExMR's own gravity: the floor seen in the picture is not the one the dancer stands on")
        else:
            geometry['samples'] = [{k: v for k, v in sample.items() if k != 'floor_normal' or sample['frame'] not in wrong}
                                   for sample in geometry['samples']]
            reference = np.einsum('tji,j->ti', person['R'], UP)  # the world's "up" in camera coordinates
            up, trust, shots_floor = floor_by_shot(geometry, shots, fps, K[0, 0], reference)
            gravity = dict(source='floor plane seen in the picture (scene_geometry.py), shot by shot', shots=shots_floor)
            if trust.any():
                turned = stand_upright(person, up, trust, fps)
                gravity['turned_deg'] = dict(median=float(np.median(turned[reliable])), p90=float(np.percentile(turned[reliable], 90)),
                                             max=float(turned[reliable].max()))
                firm = trust >= 0.5  # the camera's roll and tilt where the floor was measured
                roll = np.degrees(np.arctan2(up[firm, 0], -up[firm, 1]))
                pitch = np.degrees(np.arcsin(np.clip(-up[firm, 2], -1, 1)))
                gravity.update(camera_roll_deg=dict(min=float(roll.min()), max=float(roll.max())),
                               camera_pitch_deg=dict(min=float(pitch.min()), max=float(pitch.max())))
            else:
                gravity['source'] = "ExMR's own gravity (no floor found in the picture)"
        gravity['feet_check'] = check

    # legs that are not in the picture: from the frames where they are (see the top of this file). The
    # camera goes up and down with the pelvis, so that it still sees the upper body where it was seen
    motion = dict(orient=person['orient'], body=axis_angle_to_matrix(source['body_pose'].reshape(count, -1, 3)),
                  left_hand=axis_angle_to_matrix(source['left_hand_pose'].reshape(count, -1, 3)),
                  right_hand=axis_angle_to_matrix(source['right_hand_pose'].reshape(count, -1, 3)),
                  pelvis=person['joints_world'][:, 0].copy())
    height = motion['pelvis'][:, 1].copy()
    legs = [dict(seconds=[round(first / fps, 3), round(stop / fps, 3)]) for first, stop in bridge_legs(motion, legs_seen)]
    person['c'][:, 1] += motion['pelvis'][:, 1] - height

    # the camera path in that world
    camera_sigma = max(2.0, args.camera_seconds * fps)
    R_cam = by_shot(smooth_rotations, person['R'], weight, camera_sigma, shots)
    c_cam = by_shot(smooth_weighted, person['c'], weight, camera_sigma, shots)

    # frames that were not reliably seen: never the motion capture's guess
    kind, estimates = np.full(count, MEASURED), []
    for first, stop in spans(~reliable):
        if first == 0:
            for key in (*ROTATIONS, 'pelvis'):
                motion[key][:stop] = motion[key][stop]
            what = 'first seen pose held'
        elif stop == count:
            for key in (*ROTATIONS, 'pelvis'):
                motion[key][first:] = motion[key][first - 1]
            what = 'last seen pose held'
        else:
            interpolate(motion, first, stop, fps, ease=False)
            what = 'bridged between the seen frames'
        kind[first:stop] = BRIDGED
        estimates.append(dict(kind=what, seconds=[round(first / fps, 3), round(stop / fps, 3)]))
    if args.final_hz > 0:
        cut = min(args.final_hz, 0.45 * fps)
        for key in ROTATIONS:
            motion[key] = project_rotations(low_pass(motion[key], fps, cut))
        motion['pelvis'] = low_pass(motion['pelvis'], fps, cut)
    params = to_params(motion, rest_pelvis, source['betas'][0], 1.0)
    joints = body_points(smplx, params, 1.0)

    # checks
    error = picture_error(joints, R_cam, c_cam, K, keypoints, result['bbx_xyxy'].numpy(), reliable, names)
    measured = error[np.isfinite(error)]
    pelvis = joints[:, 0]
    forward = R_cam[reliable, :, 2]  # the camera's viewing direction in the world, where the dancer is seen
    yaw = np.degrees(np.unwrap(np.arctan2(forward[:, 0], forward[:, 2])))
    up_seen = np.einsum('tji,j->ti', R_cam[reliable], UP)  # the world's "up" in camera coordinates
    roll = np.degrees(np.arctan2(up_seen[:, 0], -up_seen[:, 1]))
    pitch = np.degrees(np.arcsin(np.clip(-up_seen[:, 2], -1, 1)))
    report = dict(
        frames=count, fps=fps, seconds=count / fps, flipped_left_right=args.flipped,
        reliable_frames=int(reliable.sum()), estimated_frames=int((kind != MEASURED).sum()), estimated=estimates,
        legs=dict(seen_frames=int(legs_seen.sum()), never_seen=not legs_seen.any(), filled=legs),
        camera_seconds=args.camera_seconds, depth_seconds=args.depth_seconds, final_hz=args.final_hz,
        camera_tremble_removed=tremble, gravity=gravity,
        cuts=dict(seconds=[round(cut / fps, 3) for cut in cuts], dancer_turned_back_deg=joined),
        camera_held=[dict(seconds=[round(first / fps, 3), round(last / fps, 3)], why=why)
                     for first, last, why in (json.loads(Path(args.camera_held).read_text())['steps']
                                              if args.camera_held and Path(args.camera_held).exists() else [])],
        floor_trusted_share=float((trust >= 1).mean()),
        picture_error_percent_of_body=dict(median=float(np.median(measured)) if len(measured) else None,
                                           p90=float(np.percentile(measured, 90)) if len(measured) else None,
                                           frames=int(len(measured))),
        camera=dict(turn_range_deg=float(yaw.max() - yaw.min()), roll_deg=[float(roll.min()), float(roll.max())],
                    pitch_deg=[float(pitch.min()), float(pitch.max())],
                    path_m=float(np.linalg.norm(np.diff(c_cam[reliable], axis=0), axis=1).sum()),
                    height_m=dict(median=float(np.median(c_cam[reliable, 1])), min=float(c_cam[reliable, 1].min()),
                                  max=float(c_cam[reliable, 1].max()))),
        pelvis_height_m=dict(median=float(np.median(pelvis[:, 1])), p5=float(np.percentile(pelvis[:, 1], 5)),
                             p95=float(np.percentile(pelvis[:, 1], 95)), min=float(pelvis[:, 1].min()), max=float(pelvis[:, 1].max())),
        travel_m=dict(x=[float(pelvis[:, 0].min()), float(pelvis[:, 0].max())], z=[float(pelvis[:, 2].min()), float(pelvis[:, 2].max())]),
        head_joint_height_m=float(np.percentile(joints[:, names.index('head'), 1], 95)),
        planted_feet=foot_report(joints, contact, legs_seen, fps) if (contact & legs_seen[:, None]).any() else None,
        camera_fit_rms_m=float(np.median(fit_rms)), joint_names=names)
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    np.save(out / 'picture_error.npy', error.astype(np.float32))

    save = {f'smplx_incam_{key}': result['smplx_incam'][key].numpy().astype(np.float32) for key in PARAMS}
    save |= {f'smplx_global_{key}': params[key].astype(np.float32) for key in PARAMS}
    save |= {key: result[key].numpy() for key in ('static_conf_logits', 'foot_contact', 'K_fullimg', 'bbx_xyxy')}
    save |= dict(fps=np.float64(fps), joints_incam=result['joints_incam'].numpy().astype(np.float32),
                 joints_global=joints.astype(np.float32), joints_2d=result['joints_2d'].numpy().astype(np.float32),
                 joint_names=np.array(names), reliable=reliable, estimated=kind.astype(np.uint8), legs_estimated=~legs_seen,
                 floor_trust=trust.astype(np.float32), turned_to_floor_deg=turned.astype(np.float32),
                 camera_R_c2w=R_cam.astype(np.float32), camera_center=c_cam.astype(np.float32),
                 flipped_left_right=np.bool_(args.flipped))
    np.savez(out / 'results.npz', **save)
    exported = {key: torch.from_numpy(params[key]).float() for key in PARAMS}
    save_amass(exported, fps, out / 'motion_amass.npz', smplx)
    save_bvh(exported, fps, out / 'motion.bvh', smplx)
    np.savez(out / 'camera_world.npz', R_c2w=R_cam.astype(np.float32), center=c_cam.astype(np.float32), K=K.astype(np.float32),
             fps=fps, width=info['width'], height=info['height'])
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
