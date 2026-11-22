"""Does the generated character still look like its picture? For one dancer: the share of the dancer's body (the
3D guide's mask, shrunk to stay inside the body) painted in colours that the character picture does not have.

    palette = picture_palette(picture_bgr, mask)            the picture's colours (k-means in Lab)
    share = foreign_share(frame_bgr, region, palette)       0..1 for one generated frame

A colour counts as foreign when it is farther than FOREIGN from every colour of the picture (Lab, lightness weighted
by LIGHT_WEIGHT: the studio light makes a colour lighter or darker, it does not change it). Clothes of another colour
than in the picture, a different hair colour or bare skin where the picture shows stockings raise the share; a
different cut of the same colours does not.

    qa_solo_look.py --run RUN_FOLDER [--reference DIR | --cutout CUTOUT.png]     (a challenge run: one line per shot)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LIGHT_WEIGHT = 0.5
FOREIGN = 22.0  # Lab distance
COLOURS = 16


def _lab(pixels_bgr):
    return cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32) \
        * np.float32([100 / 255, 1, 1]) - np.float32([0, 128, 128])


def picture_palette(picture, mask, colours=COLOURS):
    """The main colours of the character picture (Lab, lightness weighted), from the pixels inside `mask`."""
    pixels = _lab(picture[mask > 0])
    if len(pixels) > 40000:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), 40000, replace=False)]
    pixels = pixels * np.float32([LIGHT_WEIGHT, 1, 1])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, _, centres = cv2.kmeans(pixels, colours, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return centres


def body_region(mask_bgr, height):
    """The dancer's part of a guide mask (blue), shrunk by 1.5 % of the picture height to stay inside the body."""
    blue = (mask_bgr[..., 0] > 200) & (mask_bgr[..., 2] < 60)
    size = max(3, int(round(0.015 * height)) * 2 + 1)
    return cv2.erode(blue.astype(np.uint8), np.ones((size, size), np.uint8)) > 0


def foreign_share(frame, region, palette):
    """Share of the region's pixels farther than FOREIGN from every colour of the palette (None for a tiny region)."""
    if region.sum() < 400:
        return None
    pixels = _lab(frame[region]) * np.float32([LIGHT_WEIGHT, 1, 1])
    if len(pixels) > 20000:
        pixels = pixels[np.random.default_rng(1).choice(len(pixels), 20000, replace=False)]
    nearest = np.sqrt(((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(-1)).min(1)
    return float((nearest > FOREIGN).mean())


def cover(picture, width, height, interpolation):
    """Scale to cover width x height and cut the middle (as pair_run.py does with the guide)."""
    h, w = picture.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(picture, (int(round(w * scale)), int(round(h * scale))), interpolation=interpolation)
    y, x = (resized.shape[0] - height) // 2, (resized.shape[1] - width) // 2
    return resized[y:y + height, x:x + width]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='a challenge run folder')
    parser.add_argument('--reference', help='folder of make_reference.py (reference.png, reference_mask.png); default: the run\'s')
    parser.add_argument('--cutout', help='a cut-out picture with transparency instead (e.g. ref_A/cutout.png of an H3 run)')
    parser.add_argument('--output', help='JSON file for the numbers')
    args = parser.parse_args()
    work = Path(args.run) / 'outputs/work'
    if args.cutout:
        picture = cv2.imread(args.cutout, cv2.IMREAD_UNCHANGED)
        mask, picture = picture[..., 3] > 128, picture[..., :3]
    else:
        folder = Path(args.reference) if args.reference else work / 'ref_scail'
        picture = cv2.imread(str(folder / 'reference.png'))
        mask = body_region(cv2.imread(str(folder / 'reference_mask.png')), 0)
    palette = picture_palette(picture, mask)
    plan = json.loads((work / 'plan.json').read_text())
    result = {}
    for shot in plan['shots']:
        run = json.loads((work / 'gen' / shot['name'] / 'run.json').read_text())
        scene, first, width, height = Path(run['scene']), run['first_frame'], run['width'], run['height']
        shares = []
        for number in range(shot['first'], shot['last'] + 1):
            frame = cv2.imread(str(work / 'gen' / shot['name'] / 'generated' / f'frame_{number - first + 1:04d}.png'))
            mask_picture = cv2.imread(str(scene / 'mask' / f'frame_{number:04d}.png'))
            if frame is None or mask_picture is None:
                continue
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) if frame.shape[:2] != (height, width) else frame
            share = foreign_share(frame, body_region(cover(mask_picture, width, height, cv2.INTER_NEAREST), height), palette)
            if share is not None:
                shares.append((number, share))
        values = np.asarray([s for _, s in shares])
        result[shot['name']] = dict(frames=len(values), median=round(float(np.median(values)), 3) if len(values) else None,
                                    p90=round(float(np.percentile(values, 90)), 3) if len(values) else None,
                                    worst_frame=max(shares, key=lambda item: item[1])[0] if shares else None,
                                    shares=[[n, round(s, 3)] for n, s in shares])
        print(f'{shot["name"]:24s} frames {len(values):3d} median {result[shot["name"]]["median"]} p90 {result[shot["name"]]["p90"]}')
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
