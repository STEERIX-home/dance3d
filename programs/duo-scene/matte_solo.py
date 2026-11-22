"""Cut the generated character out of the window pictures of a solo run (solo_run.py).

The video model repaints the whole window, not only the dancer, so its background is not used:
the character is found in every generated frame (SAM 3 video segmentation, text prompt) and only
the character is laid over the rendered set afterwards. Only objects that lie where the grey
dancer was in the guide are kept, so that nothing else the model may have painted (a second
person, a mirror image) can get into the final video.

Outputs in the run folder: alpha/frame_####.png (0 = background ... 255 = character), matte.json
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from paths import JOB
from scail_run import write_video
from solo_run import cut_window, own_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='output folder of solo_run.py')
    parser.add_argument('--sam3', required=True)
    parser.add_argument('--text', default='person,character', help='comma-separated prompts for the character')
    parser.add_argument('--reach', type=float, default=0.10, help='how far (share of the window height) the character '
                        'may extend beyond the grey dancer of the guide: hair, loose clothes')
    parser.add_argument('--soften', type=float, default=1.1, help='edge softening in pixels')
    args = parser.parse_args()
    run = Path(args.run)
    record = json.loads((run / 'run.json').read_text())
    windows = json.loads((run / 'windows.json').read_text())
    scene, label, first, total = Path(record['scene']), record['dancer'], record['first_frame'], record['frames']
    size = (record['width'], record['height'])
    files = [run / 'generated' / f'frame_{index + 1:04d}.png' for index in range(total)]
    clip = JOB / 'cache/tmp' / f'matte_{run.name}.mp4'
    clip.parent.mkdir(parents=True, exist_ok=True)
    write_video([cv2.imread(str(file)) for file in files], clip, record['fps'])

    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    predictor = SAM3VideoSemanticPredictor(overrides=dict(conf=0.25, task='segment', mode='predict', imgsz=1036,
                                                          model=str(args.sam3), half=True, save=False, verbose=False))
    out = run / 'alpha'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    reach = max(3, int(round(args.reach * size[1])))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    previous, missing, coverage = None, 0, []
    for index, result in enumerate(predictor(source=str(clip), text=args.text.split(','), stream=True)):
        if index >= total:
            break
        expected = cut_window(own_mask(scene, label, first + index).astype(np.uint8) * 255, windows['windows'][index], size,
                              cv2.INTER_LINEAR) > 127
        allowed = cv2.dilate(expected.astype(np.uint8), kernel) > 0
        chosen = np.zeros((size[1], size[0]), dtype=bool)
        if result.masks is not None and len(result.masks.data):
            masks = torch.nn.functional.interpolate(result.masks.data[None].float(), size=(size[1], size[0]),
                                                    mode='bilinear', align_corners=False)[0] > 0.5
            for mask in masks.cpu().numpy():
                area = mask.sum()
                # the character: mostly inside the reach of the guide dancer and covering part of it
                if area > 50 and (mask & allowed).sum() > 0.7 * area and (mask & expected).sum() > 0.15 * max(1, expected.sum()):
                    chosen |= mask
        if not chosen.any() and expected.any():
            missing += 1
            chosen = previous if previous is not None else allowed
        chosen &= allowed
        previous = chosen
        coverage.append(float((chosen & expected).sum() / max(1, expected.sum())))
        # close pinholes, then soften the edge a little inside the outline (no bright rim from the generated background)
        solid = cv2.morphologyEx(chosen.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        solid = cv2.erode(solid, np.ones((3, 3), np.uint8))
        alpha = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), args.soften)
        cv2.imwrite(str(out / f'frame_{index + 1:04d}.png'), np.clip(alpha * 255, 0, 255).astype(np.uint8))
    del predictor
    torch.cuda.empty_cache()
    clip.unlink()
    written = len(list(out.glob('frame_*.png')))
    if written != total:
        raise RuntimeError(f'{written} mattes written, expected {total}')
    summary = dict(frames=total, frames_without_character=missing, text=args.text,
                   guide_dancer_covered=dict(median=float(np.median(coverage)), p5=float(np.percentile(coverage, 5))))
    (run / 'matte.json').write_text(json.dumps(summary))
    print('matte:', summary)


if __name__ == '__main__':
    main()
