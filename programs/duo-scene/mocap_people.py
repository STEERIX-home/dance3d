"""Reconstruct the 3D motion of every tracked dancer with ExMR, all in one shared camera.

ExMR itself follows a single person. Here it runs once per tracked identity, with
  - that person's own boxes (from the fixed-identity tracks, never the "largest person"),
  - frames in which the other dancers are painted out, so that their limbs cannot be mistaken
    for this person's where they overlap,
  - one camera for everybody: the same intrinsics and the same camera rotation, estimated from
    the static background only (dancers and screen overlays are masked out).

Each person's camera-frame result (smplx_incam) therefore lives in the same camera coordinates;
`align_people.py` turns them into one common world. Frames in which too little of a person is
visible are flagged as not reliable and are never presented as observed motion.

Outputs in --output:  person_A/results.pt, person_B/results.pt, ... , people.json, camera.pt
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from paths import EXMR

sys.path.insert(0, str(EXMR))

from exmr.assets import Assets  # noqa: E402
from exmr.body_models.smplx_model import SmplxModel  # noqa: E402
from exmr.geometry.camera import estimate_K  # noqa: E402
from exmr.geometry.smoothing import moving_average_smooth  # noqa: E402
from exmr.pipeline import (Options, _cached, _free_gpu_memory, _run_model, compute_joints,  # noqa: E402
                           hand_boxes, model_inputs, save_results)
from exmr.preprocessing import video as V  # noqa: E402
from exmr.preprocessing.keypoints import VISIBILITY_THRESHOLD, detect_keypoints  # noqa: E402
from exmr.preprocessing.tracker import _interpolate_missing  # noqa: E402
from exmr.preprocessing.visual_odometry import SimpleVO, _interpolate_poses, _match, _reset_random_state  # noqa: E402

log = logging.getLogger('duo')
BODY_JOINTS = slice(0, 17)  # COCO body keypoints within the 63
MIN_VISIBLE_JOINTS = 6      # fewer joints inside the person's own mask: not reliable


def load_masks(path):
    with np.load(path) as data:
        shape = tuple(data['shape'])
        return {key: np.unpackbits(data[key], axis=-1)[..., :shape[-1]].astype(bool)
                for key in data.files if key != 'shape'}


def screen_overlays(paths, step=12):
    """Pixels that never change while the picture moves: logos and captions burned into the video.
    They are static in the image, not in the room, and would bias the camera estimate."""
    frames = [cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2GRAY).astype(np.float32) for frame in V.read_frames(paths[::step])]
    deviation = np.std(np.stack(frames), axis=0)
    if np.median(deviation) < 6:  # the whole picture is static: nothing can be told apart
        return np.zeros(deviation.shape, dtype=bool)
    overlay = (deviation < 2.5).astype(np.uint8)
    overlay = cv2.morphologyEx(overlay, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(overlay, np.ones((21, 21), np.uint8)) > 0


class BackgroundVO(SimpleVO):
    """SimpleVO matching image features of the static background only. A frame in which hardly
    any background is left (people fill the picture, a title card, a black frame) cannot crash it:
    the camera is held as it was until the background is seen again.
    More holds, only when asked for (the two-dancer program does not): across a cut to another
    shot (cuts: first frames of new shots); where the matched "turn" is quicker than a camera
    filming a dancer turns (max_turn, degrees per sampled step), as a background that slides or
    changes behind the dancer in an edit; and where the turn does not fit how the background moved
    in the picture (flow: its position summed frame to frame, full-size pixels, nan where it was not
    measured). A camera that turns moves the whole background; a "turn" matched across a change of
    the light while the background stays put does not (11-14: 30-45 degrees in a dance short filmed
    from a tripod)."""
    MIN_FEATURES = 16
    FLOW_CHECKED = 0.25  # turns that move the middle of the picture by more than this share of its width are checked
    FLOW_LEAST = 0.35    # ... and held when the background moved along by less than this share of it

    def __init__(self, ignore, cuts=(), max_turn=None, flow=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore = ignore  # (L, h, w) bool: dancers and overlays
        self.cuts, self.max_turn = sorted(cuts), max_turn
        self.flow = flow  # (path (L, 2), measured (L,)) or None
        self._order, self._calls = [], 0
        self.held = 0  # sampled frames whose camera could not be measured
        self.held_steps = []  # [first, last] frames of the steps held for a cut or a turn that cannot be the camera, and why

    def _fits_flow(self, R, first, last, focal, width):
        """Did the background move in the picture from frame first to last as far as the turn R
        moves it? Only a clear lack counts: a turn that would move the middle of the picture a good
        way, while the background moved along that way by much less. A walking camera moves the
        background without turning (parallax), so more motion than the turn explains is fine."""
        path, measured = self.flow
        if measured[first + 1:last + 1].mean() < 0.6:
            return True
        middle = R @ np.array([0.0, 0.0, 1.0])
        if middle[2] < 0.2:  # turned so far that the middle of the picture left it
            return False
        expected = focal * middle[:2] / middle[2]
        size = np.linalg.norm(expected)
        if size < self.FLOW_CHECKED * width:
            return True
        return (path[last] - path[first]) @ expected / size >= self.FLOW_LEAST * size

    def poses(self, frame_dir):
        import pycolmap

        _reset_random_state()
        paths = V.frame_paths(frame_dir)
        count = len(paths)
        self._order, self._calls, self.held = sorted({*range(0, count, self.step), count - 1}), 0, 0
        frames = [cv2.cvtColor(cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                  for img in V.read_frames([paths[i] for i in self._order])]
        H, W = frames[0].shape[:2]
        focal_length = (W ** 2 + H ** 2) ** 0.5 / (24 ** 2 + 36 ** 2) ** 0.5 * self.f_mm
        camera = pycolmap.Camera(camera_id=0, model='SIMPLE_PINHOLE', width=W, height=H, params=[focal_length, W / 2, H / 2])
        options = pycolmap.TwoViewGeometryOptions(min_num_inliers=10, min_E_F_inlier_ratio=0.8, max_H_inlier_ratio=0.9,
                                                  compute_relative_pose=True)

        def usable(features):
            return features[1] is not None and len(features[0]) >= self.MIN_FEATURES

        T_w2c = [np.eye(4)]
        seen = self._features(frames[0])  # the last frame with enough background
        seen_frame = self._order[0]
        for step, frame in enumerate(frames[1:], 1):
            features = self._features(frame)
            T_delta = np.eye(4)  # last measured -> current camera
            measured = False
            first, last = self._order[step - 1], self._order[step]
            if any(first < cut <= last for cut in self.cuts):
                self.held_steps.append([first, last, 'cut'])
            elif usable(seen) and usable(features):
                pts0, pts1 = _match(seen, features)
                if len(pts0) >= 10:
                    matches = np.stack([np.arange(len(pts0)), np.arange(len(pts0))], axis=-1)
                    answer = pycolmap.estimate_calibrated_two_view_geometry(
                        camera, pts0.astype(np.float64), camera, pts1.astype(np.float64), matches=matches, options=options)
                    T_delta[:3] = answer.cam2_from_cam1.matrix().astype(np.float32)
                    measured = True
                    turn = np.degrees(np.arccos(np.clip((np.trace(T_delta[:3, :3]) - 1) / 2, -1, 1)))
                    if self.max_turn is not None and turn > self.max_turn:
                        T_delta, measured = np.eye(4), False
                        self.held_steps.append([first, last, f'turn of {turn:.0f} degrees'])
                    elif self.flow is not None and not self._fits_flow(T_delta[:3, :3], seen_frame, last, focal_length / self.scale,
                                                                       W / self.scale):
                        T_delta, measured = np.eye(4), False
                        self.held_steps.append([first, last, f'turn of {turn:.0f} degrees without the background moving'])
            if usable(features) or not usable(seen):
                seen, seen_frame = features, last  # also after a failed match (a cut): go on from the new view
            self.held += not measured
            T_w2c.append(T_delta @ T_w2c[-1] if measured else T_w2c[-1].copy())
        return _interpolate_poses(T_w2c, self._order)

    def _features(self, img):
        index = self._order[self._calls]
        self._calls += 1
        ignore = cv2.resize(self.ignore[index].astype(np.uint8), (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        ignore = cv2.dilate(ignore, np.ones((13, 13), np.uint8))
        return self.sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (1 - ignore) * 255)


def isolate(paths, own, others, out_dir):
    """Copies of the frames with the other dancers painted out (filled from their surroundings),
    except where this person is in front of them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / '.done'
    if marker.exists():
        return
    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        height, width = frame.shape[:2]
        hide = np.zeros(own.shape[1:], dtype=bool)
        for other in others:
            hide |= other[index]
        if hide.any():
            hide = cv2.dilate(hide.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            hide &= ~own[index]
            small = cv2.resize(frame, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
            small_hide = cv2.resize(hide.astype(np.uint8), (width // 4, height // 4), interpolation=cv2.INTER_NEAREST)
            filled = cv2.inpaint(small, small_hide, 5, cv2.INPAINT_TELEA)
            filled = cv2.GaussianBlur(cv2.resize(filled, (width, height), interpolation=cv2.INTER_CUBIC), (0, 0), 6)
            full_hide = cv2.resize(hide.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            soft = cv2.GaussianBlur(full_hide.astype(np.float32), (0, 0), 3)[..., None]
            frame = (frame * (1 - soft) + filled * soft).astype(np.uint8)
        cv2.imwrite(str(out_dir / path.name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    marker.write_text(str(len(paths)))


def person_boxes(boxes, count):
    """Track boxes -> (L, 4) tensor: frames without a box interpolated, then smoothed like ExMR's
    own tracker."""
    present = torch.tensor([box is not None for box in boxes])
    if not present.any():
        raise RuntimeError('A tracked person has no box at all')
    full = torch.zeros((count, 4), dtype=torch.float32)
    full[present] = torch.tensor([box for box in boxes if box is not None], dtype=torch.float32)
    if not present.all():
        full = _interpolate_missing(full, present)
    full = moving_average_smooth(full, window_size=5, dim=0)
    return moving_average_smooth(full, window_size=5, dim=0).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--tracks', required=True, help='folder written by track_people.py')
    parser.add_argument('--geometry', help='geometry.json of scene_geometry.py (focal length)')
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--f-mm', type=float, help='full-frame focal length; overrides --geometry')
    parser.add_argument('--resolution', default='448x336', choices=['448x336', '576x432', '1024x768'])
    parser.add_argument('--keep-frames', action='store_true')
    parser.add_argument('--cuts', help='cuts.json (find_cuts.py of dance-joints): no camera turn is measured across a cut')
    parser.add_argument('--max-turn-rate', type=float,
                        help='degrees per second: a quicker matched camera turn is an edit, not the camera (held)')
    parser.add_argument('--camera-flow', help='traces of the video (qa_fast_motion.py video): a matched camera turn that '
                        'the background did not move with is not the camera (held)')
    parser.add_argument('--smoothing', default='none', choices=['none', 'global', 'both'],
                        help="ExMR's own smoothing of the result (one frame at 30 fps); it takes a third off a quick "
                             'shake, so the default leaves the result as it is and align_people.py filters it')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    for name in ('duo', 'exmr'):
        logging.getLogger(name).setLevel(logging.INFO)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tracks = json.loads((Path(args.tracks) / 'tracks.json').read_text())
    masks = load_masks(Path(args.tracks) / 'masks.npz')
    labels = [person['label'] for person in tracks['people']]
    assets = Assets(Path(args.checkpoints))
    options = Options(pose_backend='sapiens2', sapiens2_resolution=args.resolution, static_cam=False,
                      snap_to_ground=True, smoothing=args.smoothing, render=False)
    assets.check(options.pose_backend, options.detector, False)
    torch.backends.cuda.matmul.allow_fp16_accumulation = 'GeForce' in torch.cuda.get_device_name()

    frame_dir = out / 'frames'
    marker = frame_dir / '.extracted'
    if not marker.exists():
        count = V.extract_frames(Path(args.video), frame_dir)
        marker.write_text(f'{args.video}\n{count}')
    paths = V.frame_paths(frame_dir)
    count = len(paths)
    width, height = V.frame_size(frame_dir)
    fps = V.video_fps(Path(args.video))
    if count != tracks['frames']:
        raise RuntimeError(f'{count} frames extracted but the tracks cover {tracks["frames"]}')

    f_mm, focal_source = args.f_mm, 'given'
    if f_mm is None and args.geometry:
        geometry = json.loads(Path(args.geometry).read_text())
        f_mm, focal_source = geometry['focal_full_frame_mm'], 'MoGe-2 median over sampled frames'
    K = estimate_K(width, height, f_mm)[None].repeat(count, 1, 1)
    log.info(f'{count} frames {width}x{height} {fps:.2f} fps; focal {K[0, 0, 0]:.0f} px '
             f'({f_mm if f_mm else 43.3:.1f} mm full-frame, {focal_source if f_mm else "ExMR default"})')

    # one camera rotation for everybody, from the static background
    ignore = np.zeros_like(next(iter(masks.values())))
    for mask in masks.values():
        ignore |= mask
    ignore |= screen_overlays(paths)[None]
    vo_settings = dict(video=str(args.video), frames=count, f_mm=f_mm, kind='background-simplevo-v1')
    holds = {}
    if args.cuts:
        holds['cuts'] = json.loads(Path(args.cuts).read_text())['cuts']
        vo_settings['cuts'] = holds['cuts']
    if args.max_turn_rate:
        holds['max_turn'] = args.max_turn_rate * BackgroundVO(ignore, progress=False).step / fps  # degrees per sampled step
        vo_settings['max_turn_rate'] = args.max_turn_rate
    if args.camera_flow:
        flow = json.loads(Path(args.camera_flow).read_text())
        velocity = np.asarray(flow['traces']['camera'], dtype=np.float64)[:count - 1]
        measured = np.concatenate([[False], np.isfinite(velocity).all(1)])
        measured = np.pad(measured, (0, count - len(measured)))
        path = np.concatenate([np.zeros((1, 2)), np.cumsum(np.nan_to_num(velocity), axis=0)])
        holds['flow'] = (np.pad(path, ((0, count - len(path)), (0, 0)), mode='edge'), measured)
        vo_settings['camera_flow'] = 'background-turn-check-v1'

    def camera_rotations():
        vo = BackgroundVO(ignore, progress=False, **holds) if f_mm is None else BackgroundVO(ignore, f_mm=f_mm, progress=False, **holds)
        rotations = vo(frame_dir).float()
        if vo.held:
            log.info(f'camera held in {vo.held} of {len(vo._order) - 1} sampled steps (too little background, a cut or a too quick turn)')
        if holds:
            (out / 'camera_held.json').write_text(json.dumps(dict(steps=vo.held_steps, fps=fps)))
        return rotations

    R_w2c = _cached(out / 'camera_rotations.pt', vo_settings, camera_rotations)
    torch.save(dict(K=K[0], R_w2c=R_w2c, fps=fps, width=width, height=height, f_mm=f_mm,
                    focal_source=focal_source), out / 'camera.pt')

    smplx = SmplxModel(assets.smplx)
    summary = dict(video=str(args.video), frames=count, fps=fps, width=width, height=height, smoothing=args.smoothing,
                   focal_px=float(K[0, 0, 0]), focal_full_frame_mm=f_mm, focal_source=focal_source,
                   camera_rotation='SimpleVO on the static background (dancers and overlays masked)',
                   anchor_frame=tracks['anchor_frame'], people=[])
    for person in tracks['people']:
        label = person['label']
        work = out / f'person_{label}'
        work.mkdir(exist_ok=True)
        log.info(f'--- person {label}')
        own_frames = work / 'frames'
        isolate(paths, masks[label], [masks[other] for other in labels if other != label], own_frames)
        boxes = person_boxes(person['boxes'], count)
        settings = dict(video=str(args.video), frames=count, person=label, isolate='inpaint-v1',
                        boxes=float(boxes.sum()))
        kp_settings = {**settings, 'resolution': args.resolution}
        kp2d = _cached(work / 'keypoints.pt', kp_settings, lambda: detect_keypoints(
            'sapiens2', assets.sapiens2, own_frames, boxes, resolution=args.resolution))

        def body_features():
            from exmr.preprocessing.body_features import SamBodyFeatures

            fp16 = torch.backends.cuda.matmul.allow_fp16_accumulation
            return SamBodyFeatures(assets.sam_3d_body, fp16=fp16)(own_frames, boxes, K)

        body_feat = _cached(work / 'body_features.pt', {**settings, 'f_mm': f_mm}, body_features)

        def hand_features():
            from exmr.preprocessing.hand_features import HamerHandFeatures

            return HamerHandFeatures(assets.hamer)(own_frames, hand_boxes(kp2d, 'sapiens2'))

        hand_feat = _cached(work / 'hand_features.pt', kp_settings, hand_features)
        data = {**model_inputs(kp2d, boxes, K, R_w2c, body_feat, hand_feat, 'sapiens2'), 'bbx_xyxy': boxes}
        results = _run_model(data, options, fps, assets)
        results['fps'] = fps
        results |= compute_joints(results, smplx)
        save_results(results, work)
        _free_gpu_memory()

        # how much of the body is really seen: confident body joints inside the person's own mask
        own = masks[label]
        scale_x, scale_y = own.shape[2] / width, own.shape[1] / height
        visible = np.zeros(count, dtype=int)
        for index in range(count):
            grown = cv2.dilate(own[index].astype(np.uint8), np.ones((9, 9), np.uint8))
            for x, y, score in kp2d[index, BODY_JOINTS].tolist():
                column, row = int(x * scale_x), int(y * scale_y)
                if score > VISIBILITY_THRESHOLD['sapiens2'] and 0 <= row < grown.shape[0] \
                        and 0 <= column < grown.shape[1] and grown[row, column]:
                    visible[index] += 1
        observed = np.array(person['observed'])
        reliable = observed & (visible >= MIN_VISIBLE_JOINTS)
        summary['people'].append(dict(label=label, observed=observed.tolist(), visible_joints=visible.tolist(),
                                      reliable=reliable.tolist(), reliable_frames=int(reliable.sum()),
                                      area=person['area']))
        log.info(f'person {label}: reliable in {int(reliable.sum())}/{count} frames '
                 f'(observed {int(observed.sum())})')
        if not args.keep_frames:
            shutil.rmtree(own_frames)
    (out / 'people.json').write_text(json.dumps(summary))
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    log.info(f'done: {out}')


if __name__ == '__main__':
    main()
