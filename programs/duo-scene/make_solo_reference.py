"""Reference pictures of ONE character for solo_run.py (H3 backend).

    make_solo_reference.py --image photo.png --sam3 sam3.pt --output DIR [--keep-background]

  picture_full.png      the whole character at the source picture's own resolution. A photograph's
                        surroundings are removed (plain grey behind the cut-out), because the video
                        model otherwise brings the photo's room into the video. A character on a plain
                        background is only cropped (--keep-background), so that its outlines stay whole.
                        --background auto decides this from the picture itself.
  picture_closeup.png   head and shoulders, enlarged: the face as large as the model can read it
  cutout.png            the character with transparency, for other tools
The source picture itself is never modified.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from make_reference import cut_out

GREY = (196, 196, 196)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--sam3', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--keep-background', action='store_true')
    parser.add_argument('--background', default=None, choices=['auto', 'keep', 'remove'],
                        help='auto: keep the surroundings only if they are plain (one colour), remove them otherwise')
    parser.add_argument('--pad', type=float, default=0.06, help='free border around the character, share of its height')
    parser.add_argument('--closeup-height', type=float, default=0.3, help='share of the character\'s height shown in the close-up')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    from ultralytics.models.sam import SAM3SemanticPredictor

    predictor = SAM3SemanticPredictor(overrides=dict(conf=0.25, task='segment', mode='predict', imgsz=1036, model=str(args.sam3),
                                                     half=True, save=False, verbose=False))
    crop, mask, box = cut_out(args.image, predictor)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    height, width = mask.shape
    pad = int(round(args.pad * height))
    # plain surroundings: nearly all pixels around the character are one colour
    around = crop[~cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)].astype(int)
    plain_share = float((np.abs(around - np.median(around, axis=0)).max(axis=1) < 24).mean()) if len(around) > 200 else 0.0
    if args.background == 'auto':
        args.keep_background = plain_share > 0.9
    elif args.background:
        args.keep_background = args.background == 'keep'
    if args.keep_background:
        x0, y0 = max(0, box[0] - pad), max(0, box[1] - pad)
        x1, y1 = min(image.shape[1], box[2] + pad), min(image.shape[0], box[3] + pad)
        full = image[y0:y1, x0:x1].copy()
        offset = (box[0] - x0, box[1] - y0)
    else:
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)[..., None]
        full = np.full((height + 2 * pad, width + 2 * pad, 3), GREY, np.uint8)
        full[pad:pad + height, pad:pad + width] = (crop * soft + np.array(GREY) * (1 - soft)).astype(np.uint8)
        offset = (pad, pad)
    cv2.imwrite(str(out / 'picture_full.png'), full)
    cv2.imwrite(str(out / 'cutout.png'), np.dstack([crop, mask.astype(np.uint8) * 255]))

    # head and shoulders: the top part of the outline, centred on the outline there
    rows = int(round(args.closeup_height * height))
    columns = np.flatnonzero(mask[:max(8, int(rows * 0.5))].any(0))
    centre = offset[0] + (columns[0] + columns[-1]) / 2
    side = int(round(rows * 1.15))
    top = max(0, offset[1] - int(0.08 * rows))
    left = int(np.clip(centre - side / 2, 0, max(0, full.shape[1] - side)))
    closeup = full[top:top + side, left:left + side]
    zoom = max(1.0, 1024 / closeup.shape[0])
    closeup = cv2.resize(closeup, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(out / 'picture_closeup.png'), closeup)
    (out / 'reference.json').write_text(json.dumps(dict(
        source=str(args.image), box=box, keep_background=args.keep_background, plain_surroundings_share=round(plain_share, 3),
        full_size=[full.shape[1], full.shape[0]],
        closeup=dict(left=left, top=top, side=side, zoom=round(zoom, 3)))))
    print('reference written:', out, full.shape[1], 'x', full.shape[0], '; close-up zoom', round(zoom, 2))


if __name__ == '__main__':
    main()
