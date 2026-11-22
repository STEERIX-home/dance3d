"""Build the appearance reference for the two characters: one picture with both cut-outs side by
side on black, plus the matching identity mask (one flat colour per character, same colours as
the masks of the 3D guide). The source pictures themselves are never modified.

    make_reference.py --character A=photo.png --character B=drawing.png --output DIR [--closeups]

--closeups adds two enlarged views of each character as extra references (reference_extra_N.png + mask):
the head and shoulders (top 30 % of the figure: face and hairstyle) and the figure down to mid-thigh (top 60 %,
the part of the body a close shot shows, so that the model sees where shorts or a skirt end: with only the first
one SCAIL-2 dressed reze.webp in long trousers in the close shots, 11-22; a 50 % crop would cut her shorts above
the hem and look like trousers too). It also writes closeup_<LABEL>.png, the head crop of the picture itself with
its own background, for describe_characters.py.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COLORS_BGR = {'A': (255, 0, 0), 'B': (0, 0, 255)}  # A blue, B red
CLOSEUPS = (('head and shoulders', 0.30), ('down to mid-thigh', 0.60))  # kind, share of the figure's height from its top


def cut_out(path, predictor):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    result = predictor(source=str(path), text=['person', 'character'])[0]
    if result.masks is None or len(result.masks) == 0:
        raise RuntimeError(f'No person found in {path}')
    masks = result.masks.data.float().cpu().numpy()
    mask = masks[masks.reshape(len(masks), -1).sum(1).argmax()]
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = mask > 0.5
    # keep the largest connected piece and close pinholes
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count > 2:
        mask = labels == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    ys, xs = np.nonzero(mask)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return image[box[1]:box[3], box[0]:box[2]], mask[box[1]:box[3], box[0]:box[2]], [int(v) for v in box]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--character', action='append', required=True, help='LABEL=path, left to right')
    parser.add_argument('--sam3', help='SAM 3 weights; not needed with --from-cutouts')
    parser.add_argument('--from-cutouts', help='folder with cutout_<LABEL>.png of an earlier run (no GPU needed)')
    parser.add_argument('--shrink', type=int, default=0,
                        help='pixels to shave off each cut-out edge: removes the rim of the old background '
                             '(a white rim on a drawing comes back as a white outline in the video)')
    parser.add_argument('--closeups', action='store_true',
                        help='also write one head-and-shoulders reference per character (reference_extra_N.png)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1792)
    parser.add_argument('--relative-height', action='append', default=[],
                        help='LABEL=fraction of the tallest character (default: each as large as fits)')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    predictor = None
    if not args.from_cutouts:
        from ultralytics.models.sam import SAM3SemanticPredictor

        predictor = SAM3SemanticPredictor(overrides=dict(conf=0.25, task='segment', mode='predict', imgsz=1036,
                                                         model=args.sam3, half=True, save=False, verbose=False))
    characters = [entry.split('=', 1) for entry in args.character]
    extras = []
    relative = dict(entry.split('=', 1) for entry in args.relative_height)
    canvas = np.zeros((args.height, args.width, 3), np.uint8)
    identity = np.zeros((args.height, args.width, 3), np.uint8)
    slot = args.width / len(characters)
    record = dict(width=args.width, height=args.height, characters=[])
    for index, (label, path) in enumerate(characters):
        if args.from_cutouts:
            saved = cv2.imread(str(Path(args.from_cutouts) / f'cutout_{label}.png'), cv2.IMREAD_UNCHANGED)
            image, mask, box = saved[..., :3].copy(), saved[..., 3] > 127, None
        else:
            image, mask, box = cut_out(path, predictor)
        if args.shrink:
            size = 2 * args.shrink + 1
            mask = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
        cutout = np.dstack([image, mask.astype(np.uint8) * 255])
        if args.closeups:
            # the top of the figure, enlarged to fill the whole reference picture
            rows = np.nonzero(mask.any(1))[0]
            for kind, share in CLOSEUPS:
                top, bottom = rows[0], rows[0] + int(share * (rows[-1] - rows[0]))
                columns = np.nonzero(mask[top:bottom].any(0))[0]
                crop_image, crop_mask = image[top:bottom, columns[0]:columns[-1] + 1], mask[top:bottom, columns[0]:columns[-1] + 1]
                if kind == 'head and shoulders':  # the picture's own pixels around the head, for the describing step
                    cv2.imwrite(str(out / f'closeup_{label}.png'), crop_image)
                zoom = min(args.width * 0.94 / crop_image.shape[1], args.height * 0.94 / crop_image.shape[0])
                size = (max(1, round(crop_image.shape[1] * zoom)), max(1, round(crop_image.shape[0] * zoom)))
                crop_image = cv2.resize(crop_image, size, interpolation=cv2.INTER_LANCZOS4 if zoom > 1 else cv2.INTER_AREA)
                crop_mask = cv2.resize(crop_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
                close, close_identity = np.zeros_like(canvas), np.zeros_like(canvas)
                x, y = (args.width - size[0]) // 2, (args.height - size[1]) // 2
                close[y:y + size[1], x:x + size[0]][crop_mask] = crop_image[crop_mask]
                close_identity[y:y + size[1], x:x + size[0]][crop_mask] = COLORS_BGR[label]
                extras.append((label, kind, close, close_identity, round(float(zoom), 3)))
        scale = min(slot * 0.92 / image.shape[1], args.height * 0.90 / image.shape[0]) * float(relative.get(label, 1.0))
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        left = round(slot * (index + 0.5) - size[0] / 2)
        top = round(args.height * 0.96) - size[1]
        region = (slice(top, top + size[1]), slice(left, left + size[0]))
        canvas[region][mask] = image[mask]
        identity[region][mask] = COLORS_BGR[label]
        record['characters'].append(dict(label=label, source=str(path), source_box=box, scale=round(scale, 4),
                                         placed=[left, top, size[0], size[1]]))
        cv2.imwrite(str(out / f'cutout_{label}.png'), cutout)
    record['extra_references'] = []
    for number, (label, kind, close, close_identity, zoom) in enumerate(extras, 1):
        cv2.imwrite(str(out / f'reference_extra_{number}.png'), close)
        cv2.imwrite(str(out / f'reference_extra_{number}_mask.png'), close_identity)
        record['extra_references'].append(dict(file=f'reference_extra_{number}.png', label=label, kind=kind, zoom=zoom))
    cv2.imwrite(str(out / 'reference.png'), canvas)
    cv2.imwrite(str(out / 'reference_mask.png'), identity)
    (out / 'reference.json').write_text(json.dumps(record, indent=1))
    print('reference written:', out)


if __name__ == '__main__':
    main()
