"""Give the grey dancers of a rendered 3D guide the appearance of the reference characters (SCAIL-2).

Input: a camera-path folder of blender_duo_scene.py (guide/ and mask/ frames) and the reference
picture + identity mask of make_reference.py. The identity colours tie each dancer of the guide to
one character of the reference, so who becomes whom is fixed by the masks, not by a text prompt.

The clip is generated in overlapping chunks of 81 frames (the last 5 frames of a chunk anchor the
next one) by the private ComfyUI started with comfy_serve.py. Replacement mode keeps everything
outside the dancers' masks from the guide, i.e. the rendered 3D set.

Generation starts at an anchor chunk in which every dancer is clearly visible, runs forward from
there to the end, and then backward (on the time-reversed guide) to the beginning. A run that
starts while one dancer is hidden behind the other gives that dancer a wrong appearance (seen
11-13: no hair, wrong clothes), and the wrong look is then carried from chunk to chunk.

Outputs in --output: generated/frame_####.png (generation resolution), run.json
"""
import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from paths import JOB
from scail_tiers import COMMON

URL = f'http://127.0.0.1:{os.environ.get("DANCE3D_COMFY_PORT", "8189")}'
COMFY_INPUT = JOB / 'outputs/work/comfy/input'
COMFY_OUTPUT = JOB / 'outputs/work/comfy/output'
CHUNK, OVERLAP = 81, 5
# the full-size model (the 24 GB tier of scail_tiers.py); a run of another tier hands its own row to workflow()
MODELS = dict(COMMON, unet='wan2.1_14B_SCAIL_2_fp8_scaled.safetensors', loader='native', text_device='default')


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(URL + path, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'ComfyUI {path}: {error.code} {error.read().decode()[:2000]}') from None


def cover(image, width, height, interpolation):
    """Scale to cover width x height, then crop the centre (what ComfyUI's "center" resize does)."""
    scale = max(width / image.shape[1], height / image.shape[0])
    size = (max(width, round(image.shape[1] * scale)), max(height, round(image.shape[0] * scale)))
    image = cv2.resize(image, size, interpolation=interpolation)
    x, y = (size[0] - width) // 2, (size[1] - height) // 2
    return image[y:y + height, x:x + width]


def identity_mask(mask_bgr, width, height, grow, only=None):
    """Crisp identity mask at generation size; each dancer's region grown by `grow` pixels so that
    hair and clothes of the new character have room. Where grown regions meet, the nearer dancer wins.
    only='A' or 'B': just that dancer, drawn in the first identity colour (blue); the other dancer
    stays part of the kept picture."""
    small = cover(mask_bgr, width, height, cv2.INTER_NEAREST)
    blue = (small[..., 0] > 200) & (small[..., 2] < 60)   # A
    red = (small[..., 2] > 200) & (small[..., 0] < 60)    # B
    out = np.full((height, width, 3), 255, np.uint8)
    if only:
        own, other = (blue, red) if only == 'A' else (red, blue)
        if grow > 0 and own.any():
            own = (cv2.distanceTransform((~own).astype(np.uint8), cv2.DIST_L2, 5) <= grow) & ~other | own
        out[own] = (255, 0, 0)
        return out
    if grow > 0:
        far = 1e6
        distance_blue = cv2.distanceTransform((~blue).astype(np.uint8), cv2.DIST_L2, 5) if blue.any() else np.full(blue.shape, far)
        distance_red = cv2.distanceTransform((~red).astype(np.uint8), cv2.DIST_L2, 5) if red.any() else np.full(red.shape, far)
        blue = (distance_blue <= grow) & (distance_blue <= distance_red) & ~red | blue
        red = (distance_red <= grow) & (distance_red < distance_blue) & ~blue | red
    out[blue] = (255, 0, 0)
    out[red] = (0, 0, 255)
    return out


def write_video(frames, path, fps, lossless_rgb=False):
    height, width = frames[0].shape[:2]
    codec = ['-c:v', 'libx264rgb', '-crf', '0', '-preset', 'veryfast', '-pix_fmt', 'rgb24'] if lossless_rgb else \
            ['-c:v', 'libx264', '-crf', '6', '-preset', 'medium', '-pix_fmt', 'yuv420p']
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
                                '-r', f'{fps:g}', '-i', '-', *codec, str(path)], stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f'ffmpeg failed for {path}')


def workflow(tag, index, width, height, length, prompt, seed, steps, relight, has_previous, extras=0, models=None, prefix=None):
    """The ComfyUI graph of one chunk. models: a tier row of scail_tiers.py (default: the full-size model).
    prefix: where SaveImage writes (default {tag}/chunk_{index})."""
    models = models or MODELS
    if models.get('loader') == 'gguf':  # ComfyUI-GGUF (city96): the quantised SCAIL-2 file in models/unet
        loader = ('UnetLoaderGGUF', dict(unet_name=models['unet']))
    else:
        loader = ('UNETLoader', dict(unet_name=models['unet'], weight_dtype='default'))
    graph = {
        '1': loader,
        '2': ('LoraLoaderModelOnly', dict(model=['1', 0], lora_name=models['dpo'], strength_model=1.0)),
        '3': ('LoraLoaderModelOnly', dict(model=['2', 0], lora_name=models['distill'], strength_model=0.8)),
        '4': ('ModelSamplingSD3', dict(model=['3', 0], shift=5.0)),
        '5': ('CLIPLoader', dict(clip_name=models['text'], type='wan', device=models.get('text_device', 'default'))),
        '6': ('CLIPTextEncode', dict(clip=['5', 0], text=prompt)),
        '7': ('CLIPTextEncode', dict(clip=['5', 0], text='')),
        '8': ('VAELoader', dict(vae_name=models['vae'])),
        '9': ('CLIPVisionLoader', dict(clip_name=models['vision'])),
        '10': ('LoadImage', dict(image=f'{tag}/reference.png')),
        '11': ('LoadImage', dict(image=f'{tag}/reference_mask.png')),
        '12': ('CLIPVisionEncode', dict(clip_vision=['9', 0], image=['10', 0], crop='none')),
        '13': ('LoadVideo', dict(file=f'{tag}/guide_{index:02d}.mp4')),
        '14': ('GetVideoComponents', dict(video=['13', 0])),
        '15': ('LoadVideo', dict(file=f'{tag}/mask_{index:02d}.mp4')),
        '16': ('GetVideoComponents', dict(video=['15', 0])),
        '17': ('WanSCAILToVideo', dict(
            positive=['6', 0], negative=['7', 0], vae=['8', 0], width=width, height=height, length=length,
            batch_size=1, pose_video=['14', 0], pose_video_mask=['16', 0], replacement_mode=True, pose_strength=1.0,
            pose_start=0.0, pose_end=1.0, reference_image=['10', 0], reference_image_mask=['11', 0],
            clip_vision_output=['12', 0], video_frame_offset=0, previous_frame_count=OVERLAP)),
        '20': ('KSamplerSelect', dict(sampler_name='euler')),
        '21': ('BasicScheduler', dict(model=['4', 0], scheduler='simple', steps=steps, denoise=1.0)),
        '22': ('SamplerCustom', dict(model=['4', 0], add_noise=True, noise_seed=seed, cfg=1.0, positive=['17', 0],
                                     negative=['17', 1], sampler=['20', 0], sigmas=['21', 0], latent_image=['17', 2])),
        '23': ('VAEDecode', dict(samples=['22', 0], vae=['8', 0])),
        '24': ('SaveImage', dict(images=['23', 0], filename_prefix=prefix or f'{tag}/chunk_{index:02d}/frame')),
    }
    # additional views (close-ups) are appended to the primary reference; each has its own identity mask
    image_node, mask_node = ['10', 0], ['11', 0]
    for number in range(1, extras + 1):
        graph[f'e{number}i'] = ('LoadImage', dict(image=f'{tag}/reference_extra_{number}.png'))
        graph[f'e{number}m'] = ('LoadImage', dict(image=f'{tag}/reference_extra_{number}_mask.png'))
        graph[f'e{number}bi'] = ('ImageBatch', dict(image1=image_node, image2=[f'e{number}i', 0]))
        graph[f'e{number}bm'] = ('ImageBatch', dict(image1=mask_node, image2=[f'e{number}m', 0]))
        image_node, mask_node = [f'e{number}bi', 0], [f'e{number}bm', 0]
    graph['17'][1]['reference_image'], graph['17'][1]['reference_image_mask'] = image_node, mask_node
    if relight:
        graph['3b'] = ('LoraLoaderModelOnly', dict(model=['3', 0], lora_name=models['relight'], strength_model=float(relight)))
        graph['4'][1]['model'] = ['3b', 0]
    if has_previous:
        graph['18'] = ('LoadVideo', dict(file=f'{tag}/previous_{index:02d}.mp4'))
        graph['19'] = ('GetVideoComponents', dict(video=['18', 0]))
        graph['17'][1]['previous_frames'] = ['19', 0]
    return {key: dict(class_type=kind, inputs=inputs) for key, (kind, inputs) in graph.items()}


def run_chunk(graph):
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


def visible_share(scene, frame, width, height, only=None):
    """How well each dancer can be seen in one guide frame: the share of the picture their own mask
    covers, reduced when the other dancer stands in front of them (a dancer of whom only slivers
    show beside the other one counts as hidden, however large the slivers are)."""
    mask = cover(cv2.imread(str(scene / 'mask' / f'frame_{frame:04d}.png')), width // 4, height // 4, cv2.INTER_NEAREST)
    regions = dict(A=(mask[..., 0] > 200) & (mask[..., 2] < 60), B=(mask[..., 2] > 200) & (mask[..., 0] < 60))
    scores = {}
    for label, own in regions.items():
        other = regions['B' if label == 'A' else 'A']
        if not own.any():
            scores[label] = 0.0
            continue
        rows, columns = np.nonzero(own.any(1))[0], np.nonzero(own.any(0))[0]
        box = (slice(rows[0], rows[-1] + 1), slice(columns[0], columns[-1] + 1))
        covered = float(other[box].mean())  # how much of this dancer's outline box the other one fills
        scores[label] = float(own.mean()) * (1 - covered) ** 2
    return (scores[only],) if only else (scores['A'], scores['B'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='camera-path folder with guide/ and mask/')
    parser.add_argument('--reference', required=True, help='folder of make_reference.py')
    parser.add_argument('--prompt', required=True, help='text file describing the finished video')
    parser.add_argument('--output', required=True)
    parser.add_argument('--tag', required=True, help='short name of this run (used for ComfyUI file names)')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=896)
    parser.add_argument('--first', type=int, default=1, help='first guide frame (1-based)')
    parser.add_argument('--chunks', type=int, help='number of chunks (default: as many as the guide allows)')
    parser.add_argument('--anchor', default='auto', help='chunk index to start from, or "auto": the chunk whose '
                        'first second shows every dancer best')
    parser.add_argument('--grow', type=float, default=0.02, help='mask growth as a share of the picture height')
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--relight', type=float, default=0.0, help='strength of the replacement relighting LoRA')
    parser.add_argument('--extra-references', type=int, default=0, help='use the first N reference_extra_*.png')
    parser.add_argument('--only', choices=['A', 'B'], help='replace just this dancer (the reference then shows '
                        'only that character); the other dancer is left as in the guide. Two such runs, one per '
                        'dancer, cannot mix up the characters and can run on two GPUs at once.')
    args = parser.parse_args()
    scene, out = Path(args.scene), Path(args.output)
    (out / 'generated').mkdir(parents=True, exist_ok=True)
    manifest = json.loads((scene / 'manifest.json').read_text())
    fps = manifest['fps']
    step = CHUNK - OVERLAP
    available = manifest['frames'] - args.first + 1
    chunks = args.chunks or 1 + max(0, (available - CHUNK) // step)
    total = CHUNK + (chunks - 1) * step
    if total > available:
        raise ValueError(f'{chunks} chunks need {total} frames, the guide has {available} from frame {args.first}')
    last = args.first + total - 1
    prompt = Path(args.prompt).read_text().strip()
    stage = COMFY_INPUT / args.tag
    stage.mkdir(parents=True, exist_ok=True)
    names = ['reference.png', 'reference_mask.png']
    for number in range(1, args.extra_references + 1):
        names += [f'reference_extra_{number}.png', f'reference_extra_{number}_mask.png']
    for name in names:
        if args.only and name.endswith('_mask.png'):  # a single character always uses the first identity colour
            mask = cv2.imread(str(Path(args.reference) / name))
            mask[mask.max(-1) > 40] = (255, 0, 0)
            cv2.imwrite(str(stage / name), mask)
        else:
            shutil.copy2(Path(args.reference) / name, stage / name)
    grow = args.grow * args.height
    request('/system_stats')

    # anchor: the chunk start at which the least visible dancer is most visible during the first
    # second AND the camera looks at the dancers from the front, like the reference picture does.
    # A run started on a view from behind swapped and mixed the two characters (seen 11-13).
    path = json.loads((scene / 'camera-path.json').read_text())
    scores = []
    for index in range(chunks):
        begin = args.first + index * step
        shares = [visible_share(scene, frame, args.width, args.height, args.only) for frame in range(begin, begin + 24, 3)]
        yaw = path[begin - 1].get('yaw') if begin - 1 < len(path) else None
        frontal = 1.0 if yaw is None else 0.25 + 0.75 * max(0.0, float(np.cos(np.radians(yaw))))
        # the look drifts a little with every chunk it is carried over, so a start near the middle
        # of the clip (few steps to either end) is preferred over one at an end
        middle = (chunks - 1) / 2
        central = 1.0 - 0.7 * abs(index - middle) / max(1.0, middle)
        scores.append(round(min(min(share) for share in shares) * frontal * central, 5))
    anchor = int(np.argmax(scores)) if args.anchor == 'auto' else int(args.anchor)
    anchor_frame = args.first + anchor * step
    record = dict(scene=str(scene), camera=manifest['camera'], tag=args.tag, width=args.width, height=args.height, fps=fps,
                  first_frame=args.first, frames=total, chunks=chunks, chunk_frames=CHUNK, overlap=OVERLAP,
                  mask_grow_px=round(grow, 1), steps=args.steps, seed=args.seed, relight=args.relight, models=MODELS,
                  extra_references=args.extra_references, only=args.only, anchor_chunk=anchor, anchor_frame=anchor_frame,
                  anchor_scores=scores, order='anchor chunk, forward to the end, then backward to the start',
                  prompt=prompt, mode='SCAIL-2 replacement, ' + (f'dancer {args.only} only' if args.only else
                                                                 'identity colours A blue / B red'), runs=[])
    print(f'anchor: chunk {anchor + 1}/{chunks} (guide frame {anchor_frame}); visibility scores {scores}', flush=True)

    def generate(name, frame_numbers, previous, seed):
        """One chunk for the given guide frames (in generation order). Returns its 81 frames."""
        guides, masks = [], []
        for frame in frame_numbers:
            guide = cv2.imread(str(scene / 'guide' / f'frame_{frame:04d}.jpg'))
            mask = cv2.imread(str(scene / 'mask' / f'frame_{frame:04d}.png'))
            if guide is None or mask is None:
                raise FileNotFoundError(f'guide or mask frame {frame} is missing in {scene}')
            guides.append(cover(guide, args.width, args.height, cv2.INTER_AREA))
            masks.append(identity_mask(mask, args.width, args.height, grow, args.only))
        index = len(record['runs'])
        write_video(guides, stage / f'guide_{index:02d}.mp4', fps)
        write_video(masks, stage / f'mask_{index:02d}.mp4', fps, lossless_rgb=True)
        if previous is not None:
            write_video(previous, stage / f'previous_{index:02d}.mp4', fps, lossless_rgb=True)
        target = COMFY_OUTPUT / args.tag / f'chunk_{index:02d}'
        if target.exists():
            shutil.rmtree(target)
        seconds = run_chunk(workflow(args.tag, index, args.width, args.height, CHUNK, prompt, seed, args.steps,
                                     args.relight, previous is not None, args.extra_references))
        files = sorted(target.glob('frame_*.png'))
        if len(files) != CHUNK:
            raise RuntimeError(f'{name}: expected {CHUNK} frames, got {len(files)}')
        frames = [cv2.imread(str(file)) for file in files]
        record['runs'].append(dict(chunk=name, guide_frames=[frame_numbers[0], frame_numbers[-1]], seconds=round(seconds, 1)))
        (out / 'run.json').write_text(json.dumps(record, indent=1))
        print(f'{name}: guide frames {frame_numbers[0]}..{frame_numbers[-1]}, {seconds:.0f} s', flush=True)
        return frames

    def save(frame, image):
        cv2.imwrite(str(out / 'generated' / f'frame_{frame - args.first + 1:04d}.png'), image)

    # forward from the anchor
    previous, anchor_frames = None, None
    for index in range(anchor, chunks):
        begin = args.first + index * step
        numbers = list(range(begin, begin + CHUNK))
        frames = generate(f'forward {index + 1}/{chunks}', numbers, None if previous is None else previous[-OVERLAP:],
                          args.seed + index)
        for offset in range(0 if previous is None else OVERLAP, CHUNK):
            save(numbers[offset], frames[offset])
        if previous is None:
            anchor_frames = frames
        previous = frames
    # backward from the anchor: the guide runs in reverse and starts on the anchor chunk's first frames
    previous = anchor_frames[OVERLAP - 1::-1] if anchor > 0 else None  # frames anchor+4 ... anchor, newest first
    for back in range(anchor):
        begin = anchor_frame + OVERLAP - 1 - back * step
        numbers = list(range(begin, begin - CHUNK, -1))
        frames = generate(f'backward {back + 1}/{anchor}', numbers, previous[-OVERLAP:] if back else previous,
                          args.seed + 1000 + back)
        for offset in range(OVERLAP, CHUNK):
            save(numbers[offset], frames[offset])
        previous = frames
    written = len(list((out / 'generated').glob('frame_*.png')))
    if written != total:
        raise RuntimeError(f'{written} frames written, expected {total} (guide frames {args.first}..{last})')
    print('generated', total, 'frames ->', out / 'generated', flush=True)


if __name__ == '__main__':
    main()
