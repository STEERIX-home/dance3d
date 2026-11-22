"""Give ALL dancers of the rendered 3D guide their characters in one run, in the whole picture.

Two video models can do it (--backend):

  h3     MiniMax H3 (reference-to-video + Character Swap LoRA). The guide shows the set and every dancer
         as a shaded mannequin in its own colour (pass `pair` of blender_duo_scene.py: A blue, B red); H3
         gets one reference picture per character and a prompt that says which colour becomes which
         character, and paints the dancers, their shadows and the set together, so the characters belong
         to the room. Licence: not for use in Korea, the US, the EU or the UK.
  scail  SCAIL-2 (zai-org, MIT) in replacement mode: the guide video (grey mannequins, pass `guide`) and the
         identity mask video (pass `mask`, A blue, B red on white, a little grown so that hair and clothes
         have room) say where each character goes; everything outside the masks stays the guide's room, the
         mannequins' floor shadows become the characters' shadows. The references are make_reference.py's
         cut-outs on black with their identity masks (and head close-ups as extra views). Model files and
         memory settings come from the tier of scail_tiers.py (--tier).

Nothing is cut out and laid over a separate background afterwards (the cut-out method cut characters off
wherever the model's figure left the place of the 3D body, and the characters did not look part of the room).

Chunks. H3: short (39 frames, 1.6 s) on its 17k+5 grid: in long chunks (107-124 frames) H3 often fell one of
its time units (15-16 frames, 0.65 s) behind the guide somewhere after the first second of a chunk and stayed
behind for seconds (measured 11-14 on a quick dance, 8 of 20 chunks); the same stretch in 39-frame chunks was
in step at the first try. A chunk is made longer where a dancer is hidden: the model must see every
mannequin clearly in a third of the chunk, or it invents the missing one. SCAIL-2: 4k+1 frames, at most 81
(what it was trained with), 5 frames shared with the chunk before; it can only go on from frames before a
chunk, so a chunk before the start chunk is made on the time-reversed guide, going on from the first frames
of the chunk after it.

A chunk that was already made with exactly the same inputs (guide frames, pictures, prompt, seed, models,
neighbours it is tied to) is read back instead of being made again, so a run that was stopped goes on from
where it was. Every chunk is checked: if the room of the guide is gone in some frames (the model sometimes
shows a reference picture as the whole background for a moment), or a dancer wears the other character's head
for half a second or more, or the other character's clothes for a third of a second or more, or a mannequin was
left unreplaced (wholly, or as a large unchanged part of the body) or a mannequin stands in the picture where
the guide has none, or the characters dance behind their guide (after a quick turn the model sometimes holds a
pose and then goes on more than half a second late), the chunk is made again from other noise.

Outputs in --output: generated/frame_####.png (generation resolution), run.json
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from paths import JOB
from qa_pair_identity import APART, BODY_APART, frame_body_colours, frame_head_colours, judge, judge_bodies
from scail_run import cover, identity_mask, request, write_video
from solo_run import H3_MODELS, OVERLAP, h3_graph, h3_plan, run_graph

COMFY_INPUT = JOB / 'outputs/work/comfy/input'
COMFY_OUTPUT = JOB / 'outputs/work/comfy/output'
SCAIL_LONGEST = 81  # SCAIL-2 was trained on 81-frame chunks that go on 76 frames at a time
FACE_FROM_BEHIND = 0.5  # detector score of a face where the camera sees the back of the head (on the H3 runs a drawn face
                        # seen from the front scored 0.62-0.84, from the side 0.54-0.66, and no face was found from behind)


def mannequin_region(guide, mask_picture=None):
    """Where the guide shows a mannequin: from the identity mask pass if there is one (any colour of
    mannequin), else from the guide's own colours (the blue and red mannequins of the `pair` pass)."""
    if mask_picture is not None:
        return mask_picture.min(-1) < 200
    blue, green, red = guide[..., 0].astype(int), guide[..., 1].astype(int), guide[..., 2].astype(int)
    return ((blue - np.maximum(green, red)) > 40) | ((red - np.maximum(green, blue)) > 40)


def room_departure(frames, guides, masks=None):
    """Per frame: how far the picture outside the mannequins is from the guide (mean absolute
    difference, 0-255). The model keeps the room of the guide, so a frame far off it has failed
    (seen 11-13: for 18 frames the close-up reference picture filled the whole background).
    masks: the identity mask pictures of the frames at generation size (None: told by the guide's colours)."""
    values = []
    for i, (frame, guide) in enumerate(zip(frames, guides)):
        small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(int)
        wanted = cv2.resize(guide, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA).astype(int)
        mask = None if masks is None else cv2.resize(masks[i], (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        mannequin = mannequin_region(wanted, mask)
        room = cv2.dilate(mannequin.astype(np.uint8), np.ones((21, 21), np.uint8)) == 0
        values.append(float(np.abs(small - wanted).mean(-1)[room].mean()) if room.any() else 0.0)
    return np.asarray(values)


def failed_frames(departure):
    """Indexes of the frames whose room is clearly not the room of the other frames of the chunk.
    The usual level differs between base models (one that repaints the room is far from the guide
    in every frame), so a frame is judged against the chunk's own middle value and spread."""
    middle = float(np.median(departure))
    spread = 1.4826 * float(np.median(np.abs(departure - middle)))
    return np.flatnonzero(departure > middle + max(25.0, 8 * spread))


def moment_shown(frames, plates, outlines, numbers, behind=24, ahead=12):
    """Which moment of the guide does each generated frame show? Per frame: how many frames the
    characters are behind (-) or ahead (+) of their own guide frame; 0 where they are in step or
    where that cannot be told. Second value: can the frames be judged at all? Not if the picture differs
    from the empty room nearly everywhere (a base model that repaints the room) or if the characters
    fit no moment of the guide.
    The characters (where the frame differs from the empty room, `plates`) are laid over the
    mannequins' outline of the guide frames from `behind` frames before to `ahead` frames after
    (`outlines`: {guide frame number: outline at a quarter of the size}). A chunk made on the
    time-reversed guide falls behind towards later frames: there the window is turned round.
    Seen 11-14 on a quick turn: the characters held their pose for a third of a second, went on
    0.65 s late for 2.5 s and then jumped back in step (16 frames behind, 62 frames in a row)."""
    lags, fitting, covered = np.zeros(len(frames), dtype=int), [], []
    for i, (frame, plate, number) in enumerate(zip(frames, plates, numbers)):
        differs = (np.abs(frame.astype(int) - plate.astype(int)).mean(-1) > 30).astype(np.uint8)
        differs = cv2.morphologyEx(differs, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        character = cv2.resize(differs, (frame.shape[1] // 4, frame.shape[0] // 4), interpolation=cv2.INTER_AREA) > 0
        fits = {lag: float((character & outlines[number + lag]).sum() / max(1, (character | outlines[number + lag]).sum()))
                for lag in range(-behind, ahead + 1) if number + lag in outlines}
        best = max(fits, key=fits.get)
        fitting.append(fits[best])
        covered.append(character.mean())
        if abs(best) >= 5 and fits[best] >= fits[0] + 0.05:
            lags[i] = best
    return lags, bool(np.median(fitting) >= 0.15 and np.median(covered) <= 0.65)


def mannequin_shares(frames, plates):
    """Per frame: how much of the characters (where the frame differs from the empty room) has the
    colouring of the guide's mannequins, clearly blue or clearly red (0..1). Measured 11-14: 0.00-0.02
    with a photographed child and a drawn character in black and white, 0.35 in frames in which the
    model had painted a mannequin instead of a character."""
    shares = []
    for frame, plate in zip(frames, plates):
        small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(int)
        room = cv2.resize(plate, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA).astype(int)
        character = np.abs(small - room).mean(-1) > 30
        blue, green, red = small[..., 0], small[..., 1], small[..., 2]
        mannequin = ((blue - np.maximum(green, red)) > 60) | ((red - np.maximum(green, blue)) > 60)
        shares.append(float((mannequin & character).sum() / character.sum()) if character.mean() > 0.01 else 0.0)
    return np.asarray(shares)


def runs(flags, least):
    """Indexes of the frames that are flagged for `least` frames or more in a row."""
    bad, run = [], []
    for i, flag in enumerate(list(flags) + [False]):
        if flag:
            run.append(i)
        else:
            if len(run) >= least:
                bad += run
            run = []
    return np.asarray(bad, dtype=int)


def plan_chunks(total, longest, seen):
    """Chunk lengths (17k+5 frames, 5 shared between neighbours) covering `total` frames: `longest`
    as a rule, longer (up to 124) where a dancer is hidden, until every dancer is clearly seen in a
    third of the chunk. seen: per dancer a bool array over the frames, True where the guide shows
    the dancer clearly; None: no such information, all chunks as h3_plan cuts them."""
    if not seen:
        return h3_plan(total, longest)
    units, rest = divmod(total - OVERLAP, 17)
    if rest:
        raise ValueError(f'{total} frames do not fit H3: use {17 * units + OVERLAP} or {17 * (units + 1) + OVERLAP}')
    usual, most = max(1, (longest - OVERLAP) // 17), (124 - OVERLAP) // 17

    def clear(first_unit, size):
        part = slice(17 * first_unit, 17 * (first_unit + size) + OVERLAP)
        return all(flags[part].sum() >= max(12, (17 * size + OVERLAP) / 3) for flags in seen)

    sizes, at = [], 0
    while at < units:
        size = min(usual, units - at)
        while not clear(at, size) and size < most and at + size < units:
            size += 1
        sizes.append(size)
        at += size
    if len(sizes) > 1 and not clear(units - sizes[-1], sizes[-1]) and sizes[-2] + sizes[-1] <= most:
        sizes[-2:] = [sizes[-2] + sizes[-1]]  # the end of the video: joined to the chunk before it
    return [17 * size + OVERLAP for size in sizes]


def scail_plan(total, longest=SCAIL_LONGEST):
    """Chunk lengths for SCAIL-2 covering `total` frames: every chunk 4k+1 frames and at most `longest`,
    neighbours sharing 5 frames (total = sum of the lengths - 5 per tie), as evenly long as possible."""
    if total < 5 or (total - 1) % 4:
        low = max(5, total - (total - 1) % 4)
        raise ValueError(f'{total} frames do not fit SCAIL-2 (4k+1): use {low} or {low + 4}')
    units = (total - 1) // 4  # first chunk 4 u + 1 frames, every later one 4 u + 5 (its first 5 are shared)
    first_most, later_most = (longest - 1) // 4, (longest - OVERLAP) // 4
    count = 1 if units <= first_most else 1 + -(-(units - first_most) // later_most)
    sizes = [units // count + (1 if i < units % count else 0) for i in range(count)]
    return [4 * sizes[0] + 1] + [4 * size + OVERLAP for size in sizes[1:]]


def out_of_step(lags, least=8):
    """Indexes of the frames that are out of step for `least` frames or more in a row."""
    return runs(np.asarray(lags) != 0, least)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='camera-path folder with the guide pass, mask/ and plate/')
    parser.add_argument('--backend', default='h3', choices=['h3', 'scail'])
    parser.add_argument('--guide', help='name of the guide pass folder (default: pair for h3, guide for scail)')
    parser.add_argument('--picture', action='append', default=[], help='h3: reference pictures, in the order <Picture 1>, <Picture 2>, ...')
    parser.add_argument('--reference', help='scail: folder of make_reference.py (reference.png, reference_mask.png, '
                        'reference_extra_N.png and their masks)')
    parser.add_argument('--tier', help='scail: JSON file with the tier row of scail_tiers.py (model files); default: full size')
    parser.add_argument('--prompt', required=True, help='text file')
    parser.add_argument('--output', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=1344)
    parser.add_argument('--first', type=int, default=1)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--chunk', type=int, help='longest chunk in frames; default 39 (h3, 17k+5; see the top of this file), '
                        f'{SCAIL_LONGEST} (scail, 4k+1)')
    parser.add_argument('--anchor', default='best',
                        help='the chunks made first, from nothing; the others are tied to a neighbour. "best": the one '
                             'chunk in which all dancers are clearly seen longest (the middle one of several); "ends": the '
                             'first and the last chunk (h3 only); or indexes like 0 or 1,3 (several: h3 only)')
    parser.add_argument('--steps', type=int, help='default 8 (h3), 6 (scail)')
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--detail', action='store_true', help='h3: reference pictures at full size (slower)')
    parser.add_argument('--unet', help='h3: another base model file instead of the official ref2va one (e.g. a community '
                        'fine-tune to compare); everything else stays the same')
    parser.add_argument('--turbo', help='h3: another step-distillation LoRA file')
    parser.add_argument('--grow', type=float, default=0.03, help='scail: mask growth as a share of the picture height')
    parser.add_argument('--extra-references', type=int, default=-1, help='scail: use the first N reference_extra_*.png '
                        '(default: all there are)')
    parser.add_argument('--relight', type=float, default=0.0, help='scail: strength of the replacement relighting LoRA')
    parser.add_argument('--faces-away', default='', help='guide frames (first-last,...) in which the camera sees the dancer\'s back: '
                        'a face found there (InsightFace, config python.face and models.face) is a failed frame; with '
                        '--inputs (an SMPL-X body) the frames come from where the 3D face points instead')
    parser.add_argument('--inputs', help='the scene inputs folder (export_scene_inputs.py: body motion cache): with it every '
                        'try is measured for hair the model added around dancer A\'s head (qa_hair.py), the face check '
                        'looks where the 3D face is turned away and the mannequin check also judges the head')
    parser.add_argument('--hair-from', action='append', default=[], help='with --inputs: run.json of an earlier shot of the '
                        'same run that faces the character (front); from their added hair qa_hair.limit_from() sets the most '
                        'this shot may show (median of a chunk); above it the chunk\'s frames over the limit are failed '
                        'frames (a side or back shot given a ponytail or braids the picture does not have)')
    parser.add_argument('--hair-limit', type=float, help='that limit as a number instead (head areas)')
    parser.add_argument('--hair-retries', type=int, default=1, help='how often a chunk whose only failed frames are hair over '
                        'the limit is made again (hair that the front does not show can be real, a braid down the back: '
                        'then every try has it)')
    parser.add_argument('--retries', type=int, default=3, help='how often a chunk with failed frames (the room of the guide '
                        'is gone, a wrong head, the other one\'s clothes, a mannequin left, out of step with the guide, hair '
                        'over the limit) is made again from other noise')
    args = parser.parse_args()
    scail = args.backend == 'scail'
    guide_pass = args.guide or ('guide' if scail else 'pair')
    steps = args.steps or (6 if scail else 8)
    longest = args.chunk or (SCAIL_LONGEST if scail else 39)
    scene, out = Path(args.scene), Path(args.output)
    (out / 'generated').mkdir(parents=True, exist_ok=True)
    if args.unet:
        H3_MODELS['unet'] = args.unet
    if args.turbo:
        H3_MODELS['turbo'] = args.turbo
    if scail:
        from scail_run import MODELS as SCAIL_MODELS
        from scail_run import workflow as scail_graph
        models = json.loads(Path(args.tier).read_text()) if args.tier else dict(SCAIL_MODELS)
        if not args.reference:
            parser.error('--backend scail needs --reference (a folder of make_reference.py)')
        reference = Path(args.reference)
        extras = sorted(reference.glob('reference_extra_[0-9]*.png'))
        extras = [file for file in extras if not file.stem.endswith('_mask')]
        extra_count = len(extras) if args.extra_references < 0 else min(args.extra_references, len(extras))
    else:
        models = H3_MODELS
        if not args.picture:
            parser.error('--backend h3 needs --picture')
    manifest = json.loads((scene / 'manifest.json').read_text())
    fps = manifest['fps']
    available = manifest['frames'] - args.first + 1
    if scail:
        total = args.frames or 4 * ((available - 1) // 4) + 1
    else:
        total = args.frames or 17 * ((available - OVERLAP) // 17) + OVERLAP
    if total > available:
        raise ValueError(f'{total} frames wanted, the guide has {available} from frame {args.first}')
    # where the guide shows each dancer clearly: at least 30 % of the dancer's usual size in the picture
    areas = []
    for number in range(args.first, args.first + total):
        picture = cv2.imread(str(scene / 'mask' / f'frame_{number:04d}.png'))
        if picture is None:
            areas = None
            break
        picture = cv2.resize(picture, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_NEAREST).astype(int)
        areas.append([((picture[..., 0] - picture[..., 2]) > 128).sum(), ((picture[..., 2] - picture[..., 0]) > 128).sum()])
    clearly = None
    if areas:
        areas = np.asarray(areas, dtype=float)
        clearly = [areas[:, d] >= 0.3 * np.median(areas[:, d][areas[:, d] > 0]) for d in range(areas.shape[1]) if (areas[:, d] > 0).any()]
    seen_clearly = dict(zip('AB', clearly)) if clearly and len(clearly) == 2 else None  # per dancer: flags over the frames
    lengths = scail_plan(total, longest) if scail else plan_chunks(total, longest, clearly)
    count = len(lengths)
    starts = [args.first + sum(lengths[:i]) - OVERLAP * i for i in range(count)]
    if args.anchor == 'ends':
        anchors = [0, count - 1] if count >= 3 else [0]
    elif args.anchor == 'best':
        # the head check learns each dancer's usual head colours from the chunks made so far, so the
        # first one made must show everybody: a video may open with one dancer hidden behind the other.
        # The chunk in which all dancers are clearly seen in the largest share of its frames; of several, the middle one
        share = [float(np.logical_and.reduce([flags[start - args.first:start - args.first + length] for flags in clearly]).mean())
                 if clearly else 1.0 for start, length in zip(starts, lengths)]
        anchors = [min((index for index in range(count) if share[index] >= max(share) - 1e-9), key=lambda index: abs(index - (count - 1) / 2))]
    else:
        anchors = sorted({int(value) for value in args.anchor.split(',')})
    if any(b - a < 2 for a, b in zip(anchors, anchors[1:])):
        raise ValueError('Two start chunks need at least one chunk between them')
    if scail and len(anchors) > 1:
        raise ValueError('SCAIL-2 can only go on from frames before a chunk: one start chunk (--anchor best or one index)')
    prompt = Path(args.prompt).read_text().strip()
    stage = COMFY_INPUT / args.tag
    stage.mkdir(parents=True, exist_ok=True)
    pictures, inputs = [], []
    if scail:
        names = ['reference.png', 'reference_mask.png']
        for number in range(1, extra_count + 1):
            names += [f'reference_extra_{number}.png', f'reference_extra_{number}_mask.png']
        for name in names:
            if not (reference / name).is_file():
                raise FileNotFoundError(f'{name} is missing in {reference}')
            shutil.copy2(reference / name, stage / name)
            inputs.append(reference / name)
    else:
        for number, picture in enumerate(args.picture):
            name = f'picture_{number + 1}.png'
            shutil.copy2(picture, stage / name)
            pictures.append(name)
            inputs.append(Path(picture))
    request('/system_stats')
    record = dict(scene=str(scene), camera=manifest['camera'], guide=guide_pass,
                  backend='scail-2 replacement mode, identity colours' if scail else 'h3, all dancers in one run',
                  tag=args.tag, width=args.width, height=args.height, fps=fps, first_frame=args.first, frames=total,
                  chunk_frames=lengths, overlap=OVERLAP, steps=steps, seed=args.seed, start_chunks=anchors,
                  models=models, pictures=[str(p) for p in inputs], detail=args.detail, prompt=prompt, runs=[])
    if scail:
        record.update(mask_grow_px=round(args.grow * args.height, 1), extra_references=extra_count, relight=args.relight)
    print(f'{count} chunks {lengths}; start chunks {[a + 1 for a in anchors]}', flush=True)

    heads_so_far = {label: np.zeros((0, 28)) for label in 'AB'}  # head colours in the frames of the chunks kept so far

    def head_colours(frames, frame_numbers):
        """{label: (frames, 28)} colours around each dancer's head, nan where it is not seen; None without masks."""
        seen = {label: np.full((len(frames), 28), np.nan) for label in 'AB'}
        for i, (frame, number) in enumerate(zip(frames, frame_numbers)):
            picture = cv2.imread(str(scene / 'mask' / f'frame_{number:04d}.png'))
            if picture is None:
                return None
            for label, value in frame_head_colours(frame, picture).items():
                seen[label][i] = value
        return seen

    def wrong_heads(seen):
        """Indexes of the frames in which a dancer shows the other character's head for half a second
        or more (qa_pair_identity.py; needs the identity mask pass of the scene and two dancers). Each
        dancer's usual head colours are taken from this chunk together with the chunks kept so far: a
        short chunk alone cannot tell (one dancer may be hidden, or wrong, in all of it)."""
        if seen is None:
            return np.array([], dtype=int)
        known = {label: np.concatenate([heads_so_far[label], seen[label]]) for label in 'AB'}
        if not all(np.isfinite(known[label][:, 0]).any() for label in 'AB'):
            return np.array([], dtype=int)
        _, apart, _, bad, _ = judge(known, fps)
        if apart <= APART:
            return np.array([], dtype=int)
        bad, wrong, run = bad[len(heads_so_far['A']):], [], []
        for i, flag in enumerate(list(bad) + [False]):
            if flag:
                run.append(i)
            else:
                if len(run) >= 0.5 * fps:
                    wrong += run
                run = []
        return np.asarray(wrong, dtype=int)

    bodies_so_far = {label: np.zeros((0, 28)) for label in 'AB'}  # body colours in the frames of the chunks kept so far

    def body_colours(frames, frame_numbers):
        """{label: (frames, 28)} colours of each dancer's body below the head, nan where the guide does not show
        the dancer clearly; None without masks or with another number of dancers than two."""
        if seen_clearly is None:
            return None
        seen = {label: np.full((len(frames), 28), np.nan) for label in 'AB'}
        for i, (frame, number) in enumerate(zip(frames, frame_numbers)):
            picture = cv2.imread(str(scene / 'mask' / f'frame_{number:04d}.png'))
            if picture is None:
                return None
            for label, value in frame_body_colours(frame, picture).items():
                if seen_clearly[label][number - args.first]:
                    seen[label][i] = value
        return seen

    def wrong_bodies(seen):
        """Indexes of the frames in which a dancer wears the other character's colours below the head for a third
        of a second or more (qa_pair_identity.judge_bodies). As with the heads, each dancer's usual colours are
        taken from this chunk together with the chunks kept so far."""
        if seen is None:
            return np.array([], dtype=int)
        known = {label: np.concatenate([bodies_so_far[label], seen[label]]) for label in 'AB'}
        apart, wrong = judge_bodies(known, fps)
        if apart <= BODY_APART:
            return np.array([], dtype=int)
        return np.flatnonzero((wrong['A'] | wrong['B'])[len(bodies_so_far['A']):])

    def masks_at_size(frame_numbers):
        """The identity mask pictures of these guide frames at the generation size; None if the pass is missing."""
        found = []
        for number in frame_numbers:
            picture = cv2.imread(str(scene / 'mask' / f'frame_{number:04d}.png'))
            if picture is None:
                return None
            found.append(cover(picture, args.width, args.height, cv2.INTER_NEAREST))
        return found

    def left_mannequins(frames, guides, masks, frame_numbers):
        """Indexes of the frames in which a dancer is still the mannequin of the guide (the model replaced only
        the other one, or none): the whole dancer as the guide drew it (a replaced dancer differs from its
        mannequin by 45 and more (0-255), a mannequin that was left by about 10), or, with the 3D head (--inputs),
        dancer A's head still the mannequin's bald head (qa_hair.head_likeness). Neither looks at the colour of the
        clothes. Not judged: a part of the body left grey. SCAIL-2 shades the character as the guide shades the
        mannequin, so clothes near the mannequin's grey (dark grey trousers, black shorts in the light) repeat the
        guide pixel for pixel: the patch rule used until 11-22 failed every frame of makima's back shot and caught
        no left part in five runs."""
        if masks is None:
            return np.array([], dtype=int)
        bad = []
        for i, (frame, guide, picture, number) in enumerate(zip(frames, guides, masks, frame_numbers)):
            blue, red = picture[..., 0].astype(int), picture[..., 2].astype(int)
            difference = np.abs(frame.astype(int) - guide.astype(int)).mean(-1)
            for label, mask in (('A', (blue - red) > 128), ('B', (red - blue) > 128)):
                region = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                if region.sum() < 600:
                    continue
                if difference[region].mean() < 30:
                    bad.append(i)
                    break
                if label == 'A' and heads is not None and qa_hair.head_left(
                        qa_hair.head_likeness(frame, guide, heads.mask(number, frame.shape[1], frame.shape[0]))):
                    bad.append(i)
                    break
        return np.asarray(bad, dtype=int)

    def empty_room(frame_numbers):
        """The empty room (pass plate) of these guide frames at the generation size; None if it was not rendered."""
        files = [scene / 'plate' / f'frame_{number:04d}.jpg' for number in frame_numbers]
        if not all(file.exists() for file in files):
            return None
        return [cover(cv2.imread(str(file)), args.width, args.height, cv2.INTER_AREA) for file in files]

    shares_so_far = []  # mannequin_shares of the frames of the chunks kept so far

    def stray_mannequins(shares):
        """Indexes of the frames in which, for half a second or more, far more of the characters is coloured
        like a mannequin than is usual in this video (a character may wear clear red or blue itself)."""
        if shares is None:
            return np.array([], dtype=int)
        usual = float(np.median(np.concatenate([shares_so_far, shares])))
        return runs(shares > max(0.12, usual + 0.12), 0.5 * fps)

    def out_of_step_frames(frames, plates, frame_numbers, reversed_time=False):
        """Indexes of the frames in which the characters are behind or ahead of the guide (moment_shown), and
        how many frames they are off there; needs the scene's passes plate (the empty room) and mask."""
        if plates is None:
            return np.array([], dtype=int), []
        outlines = {}
        for number in range(frame_numbers[0] - 24, frame_numbers[-1] + 25):
            file = scene / 'mask' / f'frame_{number:04d}.png'
            if file.exists():
                picture = cover(cv2.imread(str(file)), args.width, args.height, cv2.INTER_NEAREST)
                outlines[number] = cv2.resize((picture.min(-1) < 200).astype(np.uint8), (args.width // 4, args.height // 4),
                                              interpolation=cv2.INTER_AREA) > 0
        if any(number not in outlines for number in frame_numbers):
            return np.array([], dtype=int), []
        lags, judged = moment_shown(frames, plates, outlines, frame_numbers, *((12, 24) if reversed_time else (24, 12)))
        late = out_of_step(lags) if judged else np.array([], dtype=int)
        return late, [int(lags[i]) for i in late]

    faces_away = set()
    for part in filter(None, args.faces_away.split(',')):
        low, high = (int(value) for value in part.split('-'))
        faces_away.update(range(low, high + 1))

    def back_of_head(number):
        """Does the camera see the back of the dancer's head in this guide frame? From where the 3D face points
        (--inputs with an SMPL-X body), else from --faces-away (where the body points: a body seen from behind or
        from the side often turns its face to the camera, and a drawn profile scores 0.54-0.66, over the limit)."""
        facing = None if heads is None else heads.facing(number)
        return number in faces_away if facing is None else facing < qa_hair.AWAY

    def faces_from_behind(frame_numbers, folder):
        """Indexes of the frames in which the camera sees the back of the dancer's head (back_of_head) and the face
        detector finds a face anyway: a face painted on the back of the head (qa_back_faces.py on the CPU,
        python.face)."""
        wanted = [i for i, number in enumerate(frame_numbers) if back_of_head(number)]
        files = sorted(folder.glob('frame_*.png'))
        if not wanted or len(files) != len(frame_numbers):
            return np.array([], dtype=int)
        import paths
        command = [str(paths.FACE_PYTHON), '-B', str(Path(__file__).resolve().parent / 'qa_back_faces.py'), '--models',
                   str(paths.FACE_MODELS), '--files', *[str(files[i]) for i in wanted]]
        done = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        if done.returncode:
            raise RuntimeError('the face check failed: ' + done.stderr[-1500:])
        scores = json.loads(done.stdout.strip().splitlines()[-1])
        return np.asarray([wanted[k] for k, score in enumerate(scores) if score is not None and score >= FACE_FROM_BEHIND], dtype=int)

    heads = None
    if args.inputs:
        import qa_hair
        heads = qa_hair.Heads(scene, args.inputs)
        record['back_of_head_from'] = 'the 3D face' if heads.smplx else '--faces-away (the body)'
        if args.hair_limit is None and args.hair_from:
            facing = [value for file in args.hair_from for entry in json.loads(Path(file).read_text()).get('runs', [])
                      for _, value in entry.get('added_hair', [])]
            args.hair_limit = qa_hair.limit_from(facing)
        record['hair_limit'] = None if args.hair_limit is None else round(args.hair_limit, 3)

    def added_hair(frames, guides, masks, frame_numbers):
        """The hair the model added around the head ([guide frame, head areas] every 2nd frame) and, with --hair-limit,
        the indexes of the frames over the limit when the chunk's median is over it."""
        if heads is None or masks is None:
            return [], np.array([], dtype=int)
        values = qa_hair.chunk_values(heads, frames, guides, masks, frame_numbers)
        if args.hair_limit is None or not values or np.median([v for _, v in values]) <= args.hair_limit:
            return values, np.array([], dtype=int)
        return values, np.asarray([frame_numbers.index(n) for n, v in values if v > args.hair_limit], dtype=int)

    def make(index, frame_numbers, start_frames, end_frames, attempt, guides, masks):
        """One try of one chunk with the video model. Returns its frames in the order of frame_numbers."""
        seed = args.seed + index + 7919 * attempt  # another try of the same chunk only changes the noise it starts from
        target = COMFY_OUTPUT / args.tag / f'chunk_{index:02d}'
        if target.exists():
            shutil.rmtree(target)
        if scail:
            backward = end_frames is not None
            if backward and start_frames is not None:
                raise ValueError('SCAIL-2 cannot tie a chunk at both ends')
            order = list(range(len(frame_numbers)))[::-1] if backward else list(range(len(frame_numbers)))
            write_video([guides[i] for i in order], stage / f'guide_{index:02d}.mp4', fps)
            grow = args.grow * args.height
            write_video([identity_mask(masks[i], args.width, args.height, grow) for i in order], stage / f'mask_{index:02d}.mp4',
                        fps, lossless_rgb=True)
            tie = end_frames[::-1] if backward else start_frames  # the frames the chunk goes on from, oldest first
            if tie is not None:
                write_video(tie, stage / f'previous_{index:02d}.mp4', fps, lossless_rgb=True)
            graph = scail_graph(args.tag, index, args.width, args.height, len(frame_numbers), prompt, seed, steps, args.relight,
                                tie is not None, extra_count, models=models)
        else:
            backward, order = False, None
            write_video(guides, stage / f'guide_{index:02d}.mp4', fps)
            ties = []
            for where, tie in (('start', start_frames), ('end', end_frames)):
                if tie is not None:
                    file = f'{"previous" if where == "start" else "following"}_{index:02d}.mp4'
                    write_video(tie, stage / file, fps, lossless_rgb=True)
                    ties.append((where, file))
            graph = h3_graph(args.tag, index, args.width, args.height, len(frame_numbers), prompt, seed, steps, pictures, ties,
                             args.detail)
        spent = run_graph(graph)
        files = sorted(target.glob('frame_*.png'))
        if len(files) != len(frame_numbers):
            raise RuntimeError(f'chunk {index + 1}: expected {len(frame_numbers)} frames, got {len(files)}')
        frames = [cv2.imread(str(file)) for file in files]
        if backward:  # made on the reversed guide: back into the order of the video (and on disk too)
            frames = frames[::-1]
            for file, frame in zip(files, frames):
                cv2.imwrite(str(file), frame)
        return frames, spent

    def generate(name, index, frame_numbers, start_frames, end_frames):
        guides, digest = [], hashlib.sha256()
        digest.update(json.dumps([frame_numbers, args.width, args.height, prompt, args.seed + index, steps, models,
                                  args.detail, args.backend, guide_pass, args.grow if scail else None,
                                  args.relight if scail else None], sort_keys=True).encode())
        for frame in frame_numbers:
            file = scene / guide_pass / f'frame_{frame:04d}.jpg'
            if not file.exists():
                raise FileNotFoundError(f'guide frame {frame} is missing in {scene / guide_pass}')
            digest.update(file.read_bytes())
            guides.append(cover(cv2.imread(str(file)), args.width, args.height, cv2.INTER_AREA))
        masks = masks_at_size(frame_numbers)
        if scail:
            if masks is None:
                raise FileNotFoundError(f'the identity mask pass is missing in {scene / "mask"} (needed by SCAIL-2)')
            for picture in masks:
                digest.update(np.ascontiguousarray(picture).tobytes())
        for picture in inputs:
            digest.update(Path(picture).read_bytes())
        for frames in (start_frames, end_frames):
            digest.update(b'tie' if frames is None else np.ascontiguousarray(np.stack(frames)).tobytes())
        signature = digest.hexdigest()
        target = COMFY_OUTPUT / args.tag / f'chunk_{index:02d}'
        note = COMFY_OUTPUT / args.tag / f'chunk_{index:02d}.json'
        tied = [where for where, frames in (('start', start_frames), ('end', end_frames)) if frames is not None]
        plates = empty_room(frame_numbers)
        kept = json.loads(note.read_text()) if note.exists() else {}
        frames, seconds, reused, attempt, tries, best = None, 0.0, False, 0, [], None
        if kept.get('signature') == signature and len(list(target.glob('frame_*.png'))) == len(frame_numbers):
            frames = [cv2.imread(str(file)) for file in sorted(target.glob('frame_*.png'))]
            seconds, reused, attempt = kept['seconds'], True, kept.get('attempt', 0)
            tries = [entry for entry in kept.get('tries', []) if entry['attempt'] != attempt]  # earlier tries of this chunk
        while True:
            if frames is None:
                note.unlink(missing_ok=True)
                frames, spent = make(index, frame_numbers, start_frames, end_frames, attempt, guides, masks)
                seconds, reused = seconds + spent, False
            colours, bodies = head_colours(frames, frame_numbers), body_colours(frames, frame_numbers)
            shares = None if plates is None else mannequin_shares(frames, plates)
            lost = failed_frames(room_departure(frames, guides, masks))
            mixed, dressed = wrong_heads(colours), wrong_bodies(bodies)
            left = np.union1d(left_mannequins(frames, guides, masks, frame_numbers), stray_mannequins(shares)).astype(int)
            late, off = out_of_step_frames(frames, plates, frame_numbers, reversed_time=scail and end_frames is not None)
            faced = faces_from_behind(frame_numbers, target)
            hair, haired = added_hair(frames, guides, masks, frame_numbers)
            bad = np.union1d(np.union1d(np.union1d(np.union1d(np.union1d(np.union1d(lost, mixed), dressed), left), late), faced),
                             haired).astype(int)
            tries.append(dict(attempt=attempt, frames_off_the_room=[frame_numbers[i] for i in lost],
                              frames_with_the_wrong_head=[frame_numbers[i] for i in mixed],
                              frames_in_the_other_ones_colours=[frame_numbers[i] for i in dressed],
                              frames_with_a_mannequin_left=[frame_numbers[i] for i in left],
                              frames_out_of_step=[frame_numbers[i] for i in late],
                              frames_with_a_face_from_behind=[frame_numbers[i] for i in faced],
                              frames_with_added_hair=[frame_numbers[i] for i in haired],
                              added_hair=round(float(np.median([v for _, v in hair])), 3) if hair else None,
                              out_of_step_by_frames=int(np.median(off)) if off else 0, seconds=round(seconds, 1)))
            note.write_text(json.dumps(dict(signature=signature, seconds=round(seconds, 1), attempt=attempt, tries=tries)))
            if best is None or len(bad) < len(best[0]):
                best = (bad, frames, attempt, colours, shares, bodies, hair)
            if len(bad) < 3 or attempt >= args.retries or (len(np.setdiff1d(bad, haired)) < 3 and attempt >= args.hair_retries):
                break
            print(f'{name}: {len(lost)} frames without the room of the guide, {len(mixed)} with the wrong head, {len(dressed)} in the '
                  f'other one\'s clothes, {len(left)} with a mannequin left, {len(late)} out of step with the guide, {len(faced)} with a face '
                  f'on the back of the head, {len(haired)} with added hair over the limit (guide frames {frame_numbers[bad[0]]}..'
                  f'{frame_numbers[bad[-1]]}): trying again', flush=True)
            attempt, frames = attempt + 1, None
        if best[2] != attempt:  # no try was clean: keep the one with the fewest failed frames
            bad, frames, attempt, colours, shares, bodies, hair = best
            for file, frame in zip(sorted(target.glob('frame_*.png')), frames):
                cv2.imwrite(str(file), frame)
            note.write_text(json.dumps(dict(signature=signature, seconds=round(seconds, 1), attempt=attempt, tries=tries)))
        good = np.setdiff1d(np.arange(len(frames)), best[0])  # what the following chunks are judged against
        if colours is not None:
            for label in 'AB':
                heads_so_far[label] = np.concatenate([heads_so_far[label], colours[label][good]])
        if shares is not None:
            shares_so_far.extend(shares[good].tolist())
        if bodies is not None:
            for label in 'AB':
                bodies_so_far[label] = np.concatenate([bodies_so_far[label], bodies[label][good]])
        record['runs'].append(dict(chunk=name, guide_frames=[frame_numbers[0], frame_numbers[-1]], frames=len(frames),
                                   tied=tied, seconds=round(seconds, 1), read_back=reused, tries=tries, kept_try=attempt,
                                   added_hair=[[int(n), v] for n, v in hair]))
        (out / 'run.json').write_text(json.dumps(record, indent=1))
        print(f'{name}: guide frames {frame_numbers[0]}..{frame_numbers[-1]}, {seconds:.0f} s' + (' (read back)' if reused else '')
              + (f', try {attempt + 1}' if attempt else '') + (f', {len(best[0])} FAILED FRAMES LEFT' if len(best[0]) >= 3 else ''),
              flush=True)
        return frames

    chunk_numbers = [list(range(start, start + length)) for start, length in zip(starts, lengths)]
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
    for index, before, after in order:
        name = f'chunk {index + 1}/{count}' + ('' if before is None and after is None else
                                              ' tied to ' + ' and '.join(str(v + 1) for v in (before, after) if v is not None))
        made[index] = generate(name, index, chunk_numbers[index],
                               None if before is None else made[before][-OVERLAP:],
                               None if after is None else made[after][:OVERLAP])
    # neighbours share five frames: fade from the earlier chunk to the later one over them
    for index in range(count):
        frames = [frame.astype(np.float32) for frame in made[index]]
        for offset in range(0 if index == 0 else OVERLAP, len(frames)):
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
