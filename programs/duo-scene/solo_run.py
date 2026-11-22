"""Give ONE grey dancer of the rendered 3D guide the appearance of a reference character.

Every dancer is generated alone, in a window of the picture that follows that dancer:
  - the dancer fills the generated picture, so the character gets several times more pixels than
    when all dancers share one small picture,
  - the guide shows this dancer only (solo pass), never hidden by anybody, so a dancer who
    starts behind another one still gets the right appearance,
  - two characters can never be mixed up, because a run knows one character only.
`composite_solo.py` then lays the dancers over the rendered set in the right front/back order.

Backends (both through the private ComfyUI of comfy_serve.py):
  h3     MiniMax H3 reference-to-video + Character Swap LoRA: chunks on H3's 17k+5 frame grid.
         One or more start chunks are generated on their own (--anchor); every other chunk is tied
         to its already generated neighbours: it starts on the last frames of the chunk before it
         and/or ends on the first frames of the chunk after it (no reversed time). With the first
         and the last chunk as start chunks, no chunk is more than half the clip away from one.
  scail  SCAIL-2 replacement mode with the dancer's own mask: 81-frame chunks, 5 anchor frames,
         chunks before the anchor are generated on the time-reversed guide

Outputs in --output: generated/frame_####.png (window pictures), windows.json, run.json
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from paths import JOB
from scail_run import MODELS as SCAIL_MODELS
from scail_run import request, write_video

COMFY_INPUT = JOB / 'outputs/work/comfy/input'
COMFY_OUTPUT = JOB / 'outputs/work/comfy/output'
H3_MODELS = dict(unet='minimax_h3_ref2va_pruned_int8_convrot.safetensors',
                 turbo='minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors',
                 swap='h3_character_swap_v1.5.safetensors', text='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
                 vae='minimax_h3_video_vae_int8_convrot.safetensors', audio_vae='minimax_h3_audio_vae_fp32.safetensors')
OVERLAP = 5


def own_mask(scene, label, frame):
    """The dancer's whole outline in the full picture (solo mask pass): bool (h, w)."""
    picture = cv2.imread(str(scene / f'mask_{label}' / f'frame_{frame:04d}.png'))
    if picture is None:
        raise FileNotFoundError(f'solo mask of dancer {label}, frame {frame}, is missing in {scene}')
    return picture.min(-1) < 128


def follow_windows(scene, label, frames, aspect, fps, top=0.16, bottom=0.07, side=0.16, least=0.42):
    """One window (left, top, width, height in full-picture pixels) per frame that follows the
    dancer: large enough for the dancer plus room for hair and clothes, changing slowly like a
    camera that pans and zooms, and never leaving the picture."""
    boxes, shape = [], None
    for frame in frames:
        mask = own_mask(scene, label, frame)
        shape = mask.shape
        rows, columns = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        boxes.append([columns[0], rows[0], columns[-1] + 1, rows[-1] + 1] if len(rows) > 8 else [np.nan] * 4)
    boxes = np.asarray(boxes, dtype=np.float64)
    seen = np.isfinite(boxes[:, 0])
    if not seen.any():
        raise RuntimeError(f'Dancer {label} is never in the picture of {scene}')
    index = np.arange(len(boxes))
    for column in range(4):
        boxes[~seen, column] = np.interp(index[~seen], index[seen], boxes[seen, column])
    full_h, full_w = shape
    box_w, box_h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    need = np.maximum(box_h * (1 + top + bottom), box_w * (1 + 2 * side) / aspect)
    most = min(full_h, full_w / aspect)
    wide = max(3, int(round(1.5 * fps)) | 1)
    height = gaussian_filter1d(maximum_filter1d(need, wide, mode='nearest'), 0.35 * fps, mode='nearest')
    height = np.clip(np.maximum(height, need), least * full_h, most)
    width = height * aspect
    centre_x = gaussian_filter1d((boxes[:, 0] + boxes[:, 2]) / 2, 0.3 * fps, mode='nearest')
    centre_y = gaussian_filter1d((boxes[:, 1] - top * box_h + boxes[:, 3] + bottom * box_h) / 2, 0.3 * fps, mode='nearest')
    left = np.clip(centre_x - width / 2, 0, full_w - width)
    upper = np.clip(centre_y - height / 2, 0, full_h - height)
    return np.stack([left, upper, width, height], axis=1), seen


def cut_window(image, window, size, interpolation=cv2.INTER_CUBIC):
    """The window of a full picture, scaled to size (w, h)."""
    left, upper, width, height = window
    scale_x, scale_y = size[0] / width, size[1] / height
    matrix = np.array([[scale_x, 0, -left * scale_x], [0, scale_y, -upper * scale_y]])
    if scale_x < 0.8:  # shrinking: average first so that thin lines do not flicker
        image = cv2.GaussianBlur(image, (0, 0), 0.4 / scale_x)
    return cv2.warpAffine(image, matrix, size, flags=interpolation, borderMode=cv2.BORDER_REPLICATE)


def h3_plan(total, longest):
    """Chunk lengths on H3's 17k+5 grid that cover `total` frames exactly with 5 shared frames
    between neighbours: total = 17 * (sum of k) + 5."""
    units, rest = divmod(total - OVERLAP, 17)
    if rest:
        raise ValueError(f'{total} frames do not fit H3: use {17 * units + OVERLAP} or {17 * (units + 1) + OVERLAP}')
    most = (longest - OVERLAP) // 17
    count = -(-units // most)
    sizes = [units // count + (1 if i < units % count else 0) for i in range(count)]
    return [17 * size + OVERLAP for size in sizes]


def h3_graph(tag, index, width, height, length, prompt, seed, steps, pictures, anchors, detail):
    """anchors: list of ('start' | 'end', file): five frames that this chunk must begin or end with."""
    graph = {
        '1': ('UNETLoader', dict(unet_name=H3_MODELS['unet'], weight_dtype='default')),
        '2': ('LoraLoaderModelOnly', dict(model=['1', 0], lora_name=H3_MODELS['turbo'], strength_model=1.0)),
        '3': ('LoraLoaderModelOnly', dict(model=['2', 0], lora_name=H3_MODELS['swap'], strength_model=1.0)),
        '5': ('CLIPLoader', dict(clip_name=H3_MODELS['text'], type='minimax', device='default')),
        '8': ('VAELoader', dict(vae_name=H3_MODELS['vae'])),
        '9': ('VAELoader', dict(vae_name=H3_MODELS['audio_vae'])),
        '13': ('LoadVideo', dict(file=f'{tag}/guide_{index:02d}.mp4')),
        '14': ('GetVideoComponents', dict(video=['13', 0])),
        '17': ('MiniMaxH3ReferenceToVideo', dict(
            clip=['5', 0], vae=['8', 0], audio_vae=['9', 0], prompt=prompt, width=width, height=height, length=length,
            ref_image_size='max' if detail else 'match', **{'ref_videos.ref_video_0': ['14', 0]})),
        '20': ('KSamplerSelect', dict(sampler_name='er_sde')),
        '21': ('BasicScheduler', dict(model=['3', 0], scheduler='normal', steps=steps, denoise=1.0)),
        '22': ('RandomNoise', dict(noise_seed=seed)),
        '23': ('BasicGuider', dict(model=['3', 0], conditioning=['17', 0])),
        '24': ('SamplerCustomAdvanced', dict(noise=['22', 0], guider=['23', 0], sampler=['20', 0], sigmas=['21', 0],
                                             latent_image=['17', 1])),
        '25': ('VAEDecode', dict(samples=['24', 0], vae=['8', 0])),
        '26': ('SaveImage', dict(images=['25', 0], filename_prefix=f'{tag}/chunk_{index:02d}/frame')),
    }
    for number, name in enumerate(pictures):
        graph[f'p{number}'] = ('LoadImage', dict(image=f'{tag}/{name}'))
        graph['17'][1][f'ref_images.ref_image_{number}'] = [f'p{number}', 0]
    conditioning = ['17', 0]
    for where, name in anchors:
        graph[f'{where}v'] = ('LoadVideo', dict(file=f'{tag}/{name}'))
        graph[f'{where}c'] = ('GetVideoComponents', dict(video=[f'{where}v', 0]))
        graph[f'{where}g'] = ('MiniMaxH3AddGuide', dict(positive=conditioning, latent=['17', 1], vae=['8', 0],
                                                        image=[f'{where}c', 0], frame_idx=0 if where == 'start' else -OVERLAP))
        conditioning = [f'{where}g', 0]
    graph['23'][1]['conditioning'] = conditioning
    return {key: dict(class_type=kind, inputs=inputs) for key, (kind, inputs) in graph.items()}


def scail_graph(tag, index, width, height, length, prompt, seed, steps, has_previous):
    from scail_run import workflow

    return workflow(tag, index, width, height, length, prompt, seed, steps, 0.0, has_previous, 0)


def run_graph(graph):
    prompt_id = request('/prompt', dict(prompt=graph))['prompt_id']
    started = time.time()
    while True:
        time.sleep(3)
        history = request(f'/history/{prompt_id}')
        if prompt_id in history:
            status = history[prompt_id].get('status', {})
            if status.get('status_str') == 'error' or not status.get('completed', False):
                raise RuntimeError('ComfyUI run failed: ' + json.dumps(status)[:3000])
            return time.time() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='camera-path folder with solo_<label>/ and mask_<label>/')
    parser.add_argument('--dancer', required=True, help='label of the dancer to replace')
    parser.add_argument('--backend', default='h3', choices=['h3', 'scail'])
    parser.add_argument('--picture', action='append', required=True,
                        help='reference picture of the character; h3: repeat for <Picture 2> (close-up) ...; '
                             'scail: the picture and then its mask')
    parser.add_argument('--prompt', required=True, help='text file')
    parser.add_argument('--output', required=True)
    parser.add_argument('--tag', required=True, help='short name of this run (used for ComfyUI file names)')
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=1344)
    parser.add_argument('--first', type=int, default=1, help='first guide frame (1-based)')
    parser.add_argument('--frames', type=int, help='number of frames (default: as many as fit the chunk grid)')
    parser.add_argument('--chunk', type=int, default=124, help='h3: longest chunk (17k+5 frames)')
    parser.add_argument('--anchor', default='auto', help='start chunk(s): "auto" (one, chosen by view and place in the '
                        'clip), "ends" (first and last), or indexes like 0 or 0,4')
    parser.add_argument('--steps', type=int, help='default: 8 (h3), 6 (scail)')
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--detail', action='store_true', help='h3: reference pictures at full size (slower)')
    parser.add_argument('--unet', help='h3: another base model file instead of the official ref2va one (e.g. a '
                        'community fine-tune to compare); everything else stays the same')
    parser.add_argument('--turbo', help='h3: another step-distillation LoRA file')
    parser.add_argument('--grow', type=float, default=0.03, help='scail: mask growth as a share of the window height')
    parser.add_argument('--redo', help='h3: chunk indexes (e.g. 0 or 0,2) to generate again in an existing run, after '
                        'the guide changed there. Each is tied to its kept neighbours on both sides; all other chunks '
                        'and their windows stay as they are.')
    args = parser.parse_args()
    scene, out = Path(args.scene), Path(args.output)
    (out / 'generated').mkdir(parents=True, exist_ok=True)
    if args.unet:
        H3_MODELS['unet'] = args.unet
    if args.turbo:
        H3_MODELS['turbo'] = args.turbo
    manifest = json.loads((scene / 'manifest.json').read_text())
    fps = manifest['fps']
    available = manifest['frames'] - args.first + 1
    if args.backend == 'h3':
        total = args.frames or 17 * ((available - OVERLAP) // 17) + OVERLAP
        lengths = h3_plan(total, args.chunk)
        steps = args.steps or 8
    else:
        total = args.frames or 81 + 76 * ((available - 81) // 76)
        if (total - 81) % 76:
            raise ValueError('SCAIL-2 needs 81 + 76 n frames')
        lengths = [81] * (1 + (total - 81) // 76)
        steps = args.steps or 6
    if total > available:
        raise ValueError(f'{total} frames wanted, the guide has {available} from frame {args.first}')
    starts = [args.first + sum(lengths[:i]) - OVERLAP * i for i in range(len(lengths))]  # first guide frame of each chunk
    numbers = list(range(args.first, args.first + total))
    size = (args.width, args.height)
    windows, seen = follow_windows(scene, args.dancer, numbers, args.width / args.height, fps)
    redo = sorted({int(value) for value in args.redo.split(',')}) if args.redo else []
    if redo:
        if args.backend != 'h3':
            raise ValueError('--redo needs the h3 backend')
        # the kept chunks were made in the old windows: keep those, and let the new windows of a redone
        # chunk run into them over one second before each tie
        old = np.asarray(json.loads((out / 'windows.json').read_text())['windows'])
        share = np.zeros(total)
        ramp = int(round(fps))
        for index in redo:
            first, stop = starts[index] - args.first, starts[index] - args.first + lengths[index]
            inside = np.ones(stop - first)
            if index > 0:
                inside[:OVERLAP + ramp] = np.minimum(inside[:OVERLAP + ramp], np.clip((np.arange(OVERLAP + ramp) - OVERLAP + 1) / ramp, 0, 1))
            if index + 1 < len(lengths):
                inside[-(OVERLAP + ramp):] = np.minimum(inside[-(OVERLAP + ramp):],
                                                        np.clip((np.arange(OVERLAP + ramp)[::-1] - OVERLAP + 1) / ramp, 0, 1))
            share[first:stop] = np.maximum(share[first:stop], inside)
        windows = old + (windows - old) * share[:, None]
    (out / 'windows.json').write_text(json.dumps(dict(
        dancer=args.dancer, first_frame=args.first, size=list(size), full_size=manifest['resolution'],
        windows=np.round(windows, 3).tolist(), in_picture=seen.tolist())))
    prompt = Path(args.prompt).read_text().strip()
    stage = COMFY_INPUT / args.tag
    stage.mkdir(parents=True, exist_ok=True)
    pictures = []
    for number, picture in enumerate(args.picture):
        name = f'picture_{number + 1}.png' if args.backend == 'h3' else ('reference.png', 'reference_mask.png')[number]
        if args.backend == 'scail' and number == 1:
            mask = cv2.imread(str(picture))
            mask[mask.max(-1) > 40] = (255, 0, 0)  # a single character always uses the first identity colour
            cv2.imwrite(str(stage / name), mask)
        else:
            shutil.copy2(picture, stage / name)
        pictures.append(name)
    request('/system_stats')

    # start where the camera looks at the dancers from the front and the dancer is large, near the
    # middle of the clip: the look drifts a little with every chunk it is carried over
    path = json.loads((scene / 'camera-path.json').read_text())
    scores = []
    for index, (start, length) in enumerate(zip(starts, lengths)):
        yaws = [path[frame - 1].get('yaw') for frame in range(start, start + length, 6) if frame - 1 < len(path)]
        frontal = 1.0 if not yaws or yaws[0] is None else float(np.mean([0.25 + 0.75 * max(0.0, np.cos(np.radians(yaw))) for yaw in yaws]))
        inside = float(seen[start - args.first:start - args.first + length].mean())
        middle = (len(lengths) - 1) / 2
        central = 1.0 - 0.5 * abs(index - middle) / max(1.0, middle)
        scores.append(round(frontal * central * inside, 4))
    if args.anchor == 'auto':
        anchors = [int(np.argmax(scores))]
    elif args.anchor == 'ends':
        anchors = [0, len(lengths) - 1] if len(lengths) >= 3 else [0]
    else:
        anchors = sorted({int(value) for value in args.anchor.split(',')})
    if any(b - a < 2 for a, b in zip(anchors, anchors[1:])):
        raise ValueError('Two start chunks need at least one chunk between them')
    if len(anchors) > 1 and args.backend != 'h3':
        raise ValueError('Several start chunks need the h3 backend')
    earlier = json.loads((out / 'run.json').read_text()) if redo else None
    record = dict(scene=str(scene), camera=manifest['camera'], dancer=args.dancer, backend=args.backend, tag=args.tag,
                  width=args.width, height=args.height, fps=fps, first_frame=args.first, frames=total,
                  chunk_frames=lengths, overlap=OVERLAP, steps=steps, seed=args.seed, start_chunks=anchors,
                  anchor_scores=scores, models=H3_MODELS if args.backend == 'h3' else SCAIL_MODELS,
                  pictures=[str(p) for p in args.picture], detail=args.detail, prompt=prompt, runs=[])
    if redo:
        if earlier['chunk_frames'] != lengths or earlier['first_frame'] != args.first:
            raise ValueError('The existing run has other chunks')
        record['runs'], record['start_chunks'], record['redone_chunks'] = earlier['runs'], earlier['start_chunks'], redo
    print(f'{len(lengths)} chunks {lengths}; start chunks {[a + 1 for a in anchors]} (scores {scores})'
          + (f'; again: {[i + 1 for i in redo]}' if redo else ''), flush=True)

    def guide_frames(frame_numbers):
        frames = []
        for frame in frame_numbers:
            guide = cv2.imread(str(scene / f'solo_{args.dancer}' / f'frame_{frame:04d}.jpg'))
            if guide is None:
                raise FileNotFoundError(f'solo guide frame {frame} of dancer {args.dancer} is missing in {scene}')
            frames.append(cut_window(guide, windows[frame - args.first], size))
        return frames

    def mask_frames(frame_numbers):
        frames, grow = [], args.grow * args.height
        for frame in frame_numbers:
            own = cut_window(own_mask(scene, args.dancer, frame).astype(np.uint8) * 255, windows[frame - args.first], size,
                             cv2.INTER_LINEAR) > 127
            if grow > 0 and own.any():
                own = cv2.distanceTransform((~own).astype(np.uint8), cv2.DIST_L2, 5) <= grow
            picture = np.full((args.height, args.width, 3), 255, np.uint8)
            picture[own] = (255, 0, 0)  # the first identity colour
            frames.append(picture)
        return frames

    def generate(name, index, frame_numbers, start_frames, end_frames):
        """One chunk for the given guide frames (in generation order). start_frames / end_frames: five
        already generated frames the chunk must begin / end with. Returns its frames."""
        run = len(record['runs'])
        length = len(frame_numbers)
        write_video(guide_frames(frame_numbers), stage / f'guide_{run:02d}.mp4', fps)
        ties = []
        for where, frames in (('start', start_frames), ('end', end_frames)):
            if frames is not None:
                file = f'{"previous" if where == "start" else "following"}_{run:02d}.mp4'
                write_video(frames, stage / file, fps, lossless_rgb=True)
                ties.append((where, file))
        if args.backend == 'h3':
            graph = h3_graph(args.tag, run, args.width, args.height, length, prompt, args.seed + index, steps, pictures,
                             ties, args.detail)
        else:
            write_video(mask_frames(frame_numbers), stage / f'mask_{run:02d}.mp4', fps, lossless_rgb=True)
            graph = scail_graph(args.tag, run, args.width, args.height, length, prompt, args.seed + index, steps,
                                start_frames is not None)
        target = COMFY_OUTPUT / args.tag / f'chunk_{run:02d}'
        if target.exists():
            shutil.rmtree(target)
        seconds = run_graph(graph)
        files = sorted(target.glob('frame_*.png'))
        if len(files) != length:
            raise RuntimeError(f'{name}: expected {length} frames, got {len(files)}')
        frames = [cv2.imread(str(file)) for file in files]
        record['runs'].append(dict(chunk=name, guide_frames=[frame_numbers[0], frame_numbers[-1]], frames=length,
                                   tied=[where for where, _ in ties], seconds=round(seconds, 1)))
        (out / 'run.json').write_text(json.dumps(record, indent=1))
        print(f'{name}: guide frames {frame_numbers[0]}..{frame_numbers[-1]}, {seconds:.0f} s', flush=True)
        return frames

    chunk_numbers = [list(range(start, start + length)) for start, length in zip(starts, lengths)]
    count = len(lengths)
    # generation order: (chunk, chunk it starts on, chunk it ends on)
    order = [(a, None, None) for a in anchors]
    order += [(index, None, index + 1) for index in range(anchors[0] - 1, -1, -1)]
    order += [(index, index - 1, None) for index in range(anchors[-1] + 1, count)]
    for a, b in zip(anchors, anchors[1:]):
        low, high = a + 1, b - 1
        while low < high:
            order += [(low, low - 1, None), (high, None, high + 1)]
            low, high = low + 1, high - 1
        if low == high:
            order.append((low, low - 1, low + 1))
    made = {}
    if redo:
        # the kept chunks as they were generated (the newest run of each chunk)
        for number, run in enumerate(record['runs']):
            index = starts.index(run['guide_frames'][0])
            if index not in redo:
                files = sorted((COMFY_OUTPUT / args.tag / f'chunk_{number:02d}').glob('frame_*.png'))
                if len(files) != lengths[index]:
                    raise RuntimeError(f'The kept frames of chunk {index + 1} are missing in {COMFY_OUTPUT / args.tag}')
                made[index] = [cv2.imread(str(file)) for file in files]
        order = [(index, index - 1 if index > 0 else None, index + 1 if index + 1 < count else None) for index in redo]
        if any(neighbour in redo for _, before, after in order for neighbour in (before, after) if neighbour is not None):
            raise ValueError('Chunks to redo must not be neighbours')
    for index, before, after in order:
        name = f'chunk {index + 1}/{count}' + ('' if before is None and after is None else
                                              ' tied to ' + ' and '.join(str(v + 1) for v in (before, after) if v is not None))
        if args.backend == 'h3' or after is None:
            made[index] = generate(name, index, chunk_numbers[index],
                                   None if before is None else made[before][-OVERLAP:],
                                   None if after is None else made[after][:OVERLAP])
        else:
            # SCAIL-2 can only continue: reversed time, starting on the following chunk's first frames (newest first)
            made[index] = generate(name, index, chunk_numbers[index][::-1], made[after][:OVERLAP][::-1], None)[::-1]
    # neighbours share five frames: fade from the earlier chunk to the later one over them
    for index in range(count):
        frames = [frame.astype(np.float32) for frame in made[index]]
        first_own = 0 if index == 0 else OVERLAP
        for offset in range(first_own, len(frames)):
            image = frames[offset]
            ahead = offset - (len(frames) - OVERLAP)
            if index + 1 < count and ahead >= 0:
                share = (ahead + 1) / (OVERLAP + 1)
                image = image * (1 - share) + made[index + 1][ahead].astype(np.float32) * share
            cv2.imwrite(str(out / 'generated' / f'frame_{chunk_numbers[index][offset] - args.first + 1:04d}.png'),
                        np.clip(image + 0.5, 0, 255).astype(np.uint8))
    written = len(list((out / 'generated').glob('frame_*.png')))
    if written != total:
        raise RuntimeError(f'{written} frames written, expected {total}')
    print('generated', total, 'frames ->', out / 'generated', flush=True)


if __name__ == '__main__':
    main()
