"""Track every dancer of a video with one fixed identity each (SAM 3 video tracking).

The identities are defined once, on an automatically chosen anchor frame where all people are
clearly visible and apart, and are then propagated forward to the end and backward to the start.
Nothing is re-assigned by screen position afterwards, so a person keeps the same label when the
dancers cross or hide each other. Frames in which a person is not visible are recorded as not
observed; they are never filled in here.

Outputs in --output:
  tracks.json   anchor frame, per person and frame: box, visible area, observed flag
  masks.npz     per person bit-packed masks at half resolution
  preview.mp4   the video with the labelled masks, for checking the identities by eye
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

LABELS = 'ABCDEF'
COLORS_BGR = [(255, 80, 40), (40, 40, 255), (40, 220, 40), (255, 40, 255), (255, 255, 40), (40, 255, 255)]
MIN_AREA = 0.002  # of the frame: smaller masks count as not observed


def read_frames(video):
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 2:
        raise RuntimeError(f'Cannot decode {video}')
    return frames, fps


def write_clip(frames, path, fps):
    """Near-lossless H.264 clip of the given frames (BGR), written through ffmpeg."""
    height, width = frames[0].shape[:2]
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
               '-r', f'{fps:.6f}', '-i', '-', '-c:v', 'libx264', '-crf', '10', '-preset', 'veryfast',
               '-pix_fmt', 'yuv420p', str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f'ffmpeg failed for {path}')


def choose_anchor(frames, detector_path, people):
    """The frame where `people` persons are all large, apart from each other and steadily detected."""
    from ultralytics import YOLO

    detector = YOLO(str(detector_path))
    height, width = frames[0].shape[:2]
    scores = np.zeros(len(frames))
    for start in range(0, len(frames), 16):
        batch = frames[start:start + 16]
        for offset, result in enumerate(detector.predict(batch, classes=0, conf=0.5, verbose=False, half=True)):
            boxes = result.boxes.xyxy.cpu().numpy()
            if len(boxes) != people:
                continue
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (width * height)
            overlap = 0.0
            for i in range(people):
                for j in range(i + 1, people):
                    x1, y1 = np.maximum(boxes[i, :2], boxes[j, :2])
                    x2, y2 = np.minimum(boxes[i, 2:], boxes[j, 2:])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    overlap = max(overlap, inter / min((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]),
                                                       (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])))
            # whole bodies inside the picture score higher than close-ups cut by the border
            inside = np.all((boxes[:, 0] > 2) & (boxes[:, 1] > 2) & (boxes[:, 2] < width - 2) & (boxes[:, 3] < height - 2))
            scores[start + offset] = areas.min() * (1 - overlap) * (1.0 if inside else 0.4)
    del detector
    torch.cuda.empty_cache()
    # steady: the worst score within +-7 frames
    steady = np.array([scores[max(0, i - 7):i + 8].min() for i in range(len(scores))])
    if steady.max() <= 0:
        raise RuntimeError(f'No frame shows exactly {people} separate people')
    return int(steady.argmax()), scores


def keep_only(predictor, drop_ids):
    """Remove tracked objects from the predictor's state (objects other than the dancers)."""
    if not drop_ids:
        return
    metadata = predictor.inference_state.get('tracker_metadata', {})
    ids = metadata.get('obj_ids_all_gpu', np.array([]))
    if len(ids):
        metadata['obj_ids_all_gpu'] = ids[np.array([int(i) not in drop_ids for i in ids])]
    for key in ('obj_id_to_score', 'obj_id_to_cls', 'obj_id_to_tracker_score'):
        if isinstance(metadata.get(key), dict):
            metadata[key] = {k: v for k, v in metadata[key].items() if int(k) not in drop_ids}
    for state in predictor.inference_state.get('tracker_inference_states', []):
        if getattr(state, 'obj_ids', None) is not None:
            state.obj_ids = state.obj_ids[np.array([int(i) not in drop_ids for i in state.obj_ids])]


def propagate(predictor, clip, count, people, half_size):
    """Track the `people` largest persons of the clip's first frame through the clip.
    Returns (track ids ordered left to right in the first frame, masks (people, count, h, w) bool)."""
    predictor.inference_state = {}
    if hasattr(predictor, 'dataset'):
        predictor.dataset = None
    width, height = half_size
    masks = np.zeros((people, count, height, width), dtype=bool)
    order = None
    for index, result in enumerate(predictor(source=str(clip), text=['person'], stream=True)):
        if index >= count:
            break
        if result.masks is None or result.boxes.id is None:
            if index == 0:
                raise RuntimeError('No person found in the anchor frame')
            continue
        data = result.masks.data
        ids = [int(i) for i in result.boxes.id.cpu().numpy()]
        small = torch.nn.functional.interpolate(data[None].float(), size=(height, width), mode='area')[0] > 0.5
        small = small.cpu().numpy()
        if index == 0:
            areas = small.reshape(len(ids), -1).sum(1)
            ranked = [ids[i] for i in np.argsort(-areas)]
            if len(ranked) < people:
                raise RuntimeError(f'Anchor frame shows {len(ranked)} people, expected {people}')
            keep, drop = ranked[:people], set(ranked[people:])
            keep_only(predictor, drop)
            centre = {}
            for track_id, mask in zip(ids, small):
                if track_id in keep:
                    centre[track_id] = np.nonzero(mask.any(0))[0].mean()
            order = sorted(keep, key=centre.get)
        for track_id, mask in zip(ids, small):
            if track_id in order:
                masks[order.index(track_id), index] = mask
    return order, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sam3', required=True)
    parser.add_argument('--detector', required=True)
    parser.add_argument('--people', type=int, default=2)
    parser.add_argument('--anchor', type=int, default=None, help='anchor frame (default: chosen automatically)')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    frames, fps = read_frames(args.video)
    count = len(frames)
    height, width = frames[0].shape[:2]
    half = (width // 2, height // 2)
    if args.anchor is None:
        anchor, scores = choose_anchor(frames, args.detector, args.people)
    else:
        anchor, scores = args.anchor, None
    print(f'{count} frames, {width}x{height}, {fps:.3f} fps; anchor frame {anchor} ({anchor / fps:.2f} s)', flush=True)

    forward_clip, backward_clip = out / 'anchor_forward.mp4', out / 'anchor_backward.mp4'
    write_clip(frames[anchor:], forward_clip, fps)
    write_clip(frames[anchor::-1], backward_clip, fps)

    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=640, model=str(args.sam3), half=True,
                     save=False, verbose=False)
    # new_det_thresh=1: only the anchor frame defines objects; nobody is added or renamed later
    predictor = SAM3VideoSemanticPredictor(overrides=overrides, new_det_thresh=1.0)
    masks = np.zeros((args.people, count, half[1], half[0]), dtype=bool)
    _, forward = propagate(predictor, forward_clip, count - anchor, args.people, half)
    masks[:, anchor:] = forward
    if anchor > 0:
        _, backward = propagate(predictor, backward_clip, anchor + 1, args.people, half)
        # both runs start on the same frame: match the backward objects to the forward ones by overlap
        match = []
        for person in range(args.people):
            overlaps = [(forward[person, 0] & backward[other, 0]).sum() / max(1, (forward[person, 0] | backward[other, 0]).sum())
                        for other in range(args.people)]
            match.append(int(np.argmax(overlaps)))
        if sorted(match) != list(range(args.people)):
            raise RuntimeError(f'Forward/backward identities do not match at the anchor: {match}')
        for person in range(args.people):
            masks[person, :anchor] = backward[match[person], 1:][::-1]
    del predictor
    torch.cuda.empty_cache()
    forward_clip.unlink()
    backward_clip.unlink()

    people = []
    for person in range(args.people):
        boxes, areas, observed = [], [], []
        for index in range(count):
            mask = masks[person, index]
            area = float(mask.mean())
            seen = area >= MIN_AREA
            if seen:
                ys, xs = np.nonzero(mask)
                box = [float(xs.min() * 2), float(ys.min() * 2), float((xs.max() + 1) * 2), float((ys.max() + 1) * 2)]
            else:
                box = None
            boxes.append(box)
            areas.append(round(area, 5))
            observed.append(bool(seen))
        people.append(dict(label=LABELS[person], boxes=boxes, area=areas, observed=observed,
                           observed_frames=int(np.sum(observed))))
        print(f'person {LABELS[person]}: observed in {int(np.sum(observed))}/{count} frames', flush=True)
    (out / 'tracks.json').write_text(json.dumps(dict(
        video=str(args.video), frames=count, fps=fps, width=width, height=height, anchor_frame=anchor,
        identity_rule='left to right on the anchor frame; fixed for every other frame',
        tracker='SAM 3 video (text prompt "person"), propagated forward and backward from the anchor',
        mask_scale=0.5, people=people)))
    np.savez_compressed(out / 'masks.npz', **{LABELS[p]: np.packbits(masks[p], axis=-1) for p in range(args.people)},
                        shape=np.array(masks.shape[1:]))

    preview = []
    for index, frame in enumerate(frames):
        small = cv2.resize(frame, half, interpolation=cv2.INTER_AREA)
        for person in range(args.people):
            mask = masks[person, index]
            if mask.any():
                small[mask] = (0.55 * small[mask] + 0.45 * np.array(COLORS_BGR[person])).astype(np.uint8)
                ys, xs = np.nonzero(mask)
                cv2.putText(small, LABELS[person], (int(xs.mean()) - 12, max(30, int(ys.min()) + 34)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(small, LABELS[person], (int(xs.mean()) - 12, max(30, int(ys.min()) + 34)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLORS_BGR[person], 2, cv2.LINE_AA)
        cv2.putText(small, f'{index:04d} {index / fps:5.2f}s' + (' ANCHOR' if index == anchor else ''), (8, half[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        preview.append(small)
    write_clip(preview, out / 'preview.mp4', fps)
    print('tracks written to', out, flush=True)


if __name__ == '__main__':
    main()
