"""Track the one dancer of a video with a fixed identity (SAM 3 video tracking).

The dancer is the person who is clearly the largest in the picture. The anchor frame is the one
where that person is large, steadily detected, inside the picture and much larger than anybody
else: a bystander, or the dancer's own reflection in a studio mirror, does not make a frame
unusable, but it has to be clearly smaller. From the anchor the person is followed forward to the
end and backward to the start (duo-scene/track_people.py), so somebody who comes closer later is
never taken for the dancer.

Outputs in --output: tracks.json, masks.npz, preview.mp4 (as duo-scene/track_people.py writes them)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from track_people import COLORS_BGR, LABELS, MIN_AREA, read_frames, write_clip


def choose_anchor(frames, detector_path):
    """The frame where the largest person is large, inside the picture and clearly larger than the
    next one (a second person half as large or more makes a frame unusable). If no frame is like
    that (a reflection or a second dancer is always there), the largest person of the frames where
    they are at least larger than everybody else."""
    from ultralytics import YOLO

    detector = YOLO(str(detector_path))
    height, width = frames[0].shape[:2]
    scores, fallback = np.zeros(len(frames)), np.zeros(len(frames))
    for start in range(0, len(frames), 16):
        batch = frames[start:start + 16]
        for offset, result in enumerate(detector.predict(batch, classes=0, conf=0.5, verbose=False, half=True)):
            boxes = result.boxes.xyxy.cpu().numpy()
            if not len(boxes):
                continue
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (width * height)
            order = np.argsort(-areas)
            largest = boxes[order[0]]
            second = areas[order[1]] if len(boxes) > 1 else 0.0
            # a whole body inside the picture scores higher than a close-up cut by the border
            inside = largest[0] > 2 and largest[1] > 2 and largest[2] < width - 2 and largest[3] < height - 2
            scores[start + offset] = areas[order[0]] * max(0.0, 1 - 2 * second / areas[order[0]]) * (1.0 if inside else 0.4)
            fallback[start + offset] = areas[order[0]] * max(0.0, 1 - second / areas[order[0]]) * (1.0 if inside else 0.4)
    del detector
    torch.cuda.empty_cache()
    for values in (scores, fallback):
        # steady: the worst score within +-7 frames
        steady = np.array([values[max(0, i - 7):i + 8].min() for i in range(len(values))])
        if steady.max() > 0:
            return int(steady.argmax())
    raise RuntimeError('No frame shows one person clearly larger than everybody else')


def follow(predictor, clip, count, half_size, reference=None):
    """Track the largest person of the clip's first frame (or the one on the reference mask) through
    the clip -> masks (count, h, w).
    Like duo-scene/track_people.py's propagate for one person, but the other people of the first
    frame are not removed from the tracker's state: they are tracked along and left out here.
    (Removing them made the tracker fail on a person it still held, 11-14: KeyError in postprocess.)"""
    predictor.inference_state = {}
    if hasattr(predictor, 'dataset'):
        predictor.dataset = None
    width, height = half_size
    masks = np.zeros((count, height, width), dtype=bool)
    dancer = None
    for index, result in enumerate(predictor(source=str(clip), text=['person'], stream=True)):
        if index >= count:
            break
        if result.masks is None or result.boxes.id is None:
            if index == 0:
                raise RuntimeError('No person found in the anchor frame')
            continue
        ids = [int(i) for i in result.boxes.id.cpu().numpy()]
        small = torch.nn.functional.interpolate(result.masks.data[None].float(), size=(height, width), mode='area')[0] > 0.5
        small = small.cpu().numpy()
        if index == 0:
            flat = small.reshape(len(ids), -1)
            if reference is None:
                dancer = ids[int(np.argmax(flat.sum(1)))]
            else:  # the backward run starts on the same frame: the same person as the forward run
                both = reference.reshape(1, -1)
                dancer = ids[int(np.argmax((flat & both).sum(1) / np.maximum((flat | both).sum(1), 1)))]
        if dancer in ids:
            masks[index] = small[ids.index(dancer)]
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sam3', required=True)
    parser.add_argument('--detector', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    frames, fps = read_frames(args.video)
    count = len(frames)
    height, width = frames[0].shape[:2]
    half = (width // 2, height // 2)
    anchor = choose_anchor(frames, args.detector)
    print(f'{count} frames, {width}x{height}, {fps:.3f} fps; anchor frame {anchor} ({anchor / fps:.2f} s)', flush=True)

    forward_clip, backward_clip = out / 'anchor_forward.mp4', out / 'anchor_backward.mp4'
    write_clip(frames[anchor:], forward_clip, fps)
    write_clip(frames[anchor::-1], backward_clip, fps)

    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=640, model=str(args.sam3), half=True,
                     save=False, verbose=False)
    # new_det_thresh=1: only the anchor frame defines the dancer; nobody is added or renamed later
    predictor = SAM3VideoSemanticPredictor(overrides=overrides, new_det_thresh=1.0)
    masks = np.zeros((count, half[1], half[0]), dtype=bool)
    masks[anchor:] = follow(predictor, forward_clip, count - anchor, half)
    if anchor > 0:
        masks[:anchor] = follow(predictor, backward_clip, anchor + 1, half, reference=masks[anchor])[1:][::-1]
    del predictor
    torch.cuda.empty_cache()
    forward_clip.unlink()
    backward_clip.unlink()

    boxes, areas, observed = [], [], []
    for mask in masks:
        area = float(mask.mean())
        seen = area >= MIN_AREA
        if seen:
            ys, xs = np.nonzero(mask)
            boxes.append([float(xs.min() * 2), float(ys.min() * 2), float((xs.max() + 1) * 2), float((ys.max() + 1) * 2)])
        else:
            boxes.append(None)
        areas.append(round(area, 5))
        observed.append(bool(seen))
    print(f'person {LABELS[0]}: observed in {int(np.sum(observed))}/{count} frames', flush=True)
    (out / 'tracks.json').write_text(json.dumps(dict(
        video=str(args.video), frames=count, fps=fps, width=width, height=height, anchor_frame=anchor,
        identity_rule='the largest person, clearly larger than anybody else on the anchor frame; fixed for every other frame',
        tracker='SAM 3 video (text prompt "person"), propagated forward and backward from the anchor',
        mask_scale=0.5, people=[dict(label=LABELS[0], boxes=boxes, area=areas, observed=observed,
                                     observed_frames=int(np.sum(observed)))])))
    np.savez_compressed(out / 'masks.npz', **{LABELS[0]: np.packbits(masks, axis=-1)}, shape=np.array(masks.shape))

    preview = []
    even = (half[0] - half[0] % 2, half[1] - half[1] % 2)  # the video encoder takes even sizes only (1080x1918 -> 540x959)
    for index, frame in enumerate(frames):
        small = cv2.resize(frame, half, interpolation=cv2.INTER_AREA)[:even[1], :even[0]]
        mask = masks[index, :even[1], :even[0]]
        if mask.any():
            small[mask] = (0.55 * small[mask] + 0.45 * np.array(COLORS_BGR[0])).astype(np.uint8)
        cv2.putText(small, f'{index:04d} {index / fps:5.2f}s' + (' ANCHOR' if index == anchor else ''), (8, even[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        preview.append(small)
    write_clip(preview, out / 'preview.mp4', fps)
    print('tracks written to', out, flush=True)


if __name__ == '__main__':
    main()
