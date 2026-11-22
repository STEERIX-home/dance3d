"""Did every dancer of a pair run (pair_run.py) keep the right character in every frame?

A tripwire, not a proof, and it needs nothing typed in for a new character. In every frame the
colours of the generated picture around each dancer's head (the head band of the 3D render's
identity mask, grown a little for hair) are measured. Each dancer's usual head colours (over the
frames in which both heads are seen) are taken as that dancer's own. A frame is wrong when BOTH
dancers show the other one's colours better than their own (a swap), or when one does while the
other one is not to be seen (the hidden one's head on the wrong body). Runs of half a second and
more are listed; shorter ones are poses in which a head is hidden or turned away. If the two
characters' head colours are too much alike to tell them apart, the result says so instead of
guessing.

Clothes are checked the same way, per dancer: the colours of the body below the head (inside the
dancer's own mask) against that dancer's usual ones and the other dancer's. A dancer that wears the
other one's colours for a third of a second or more is listed. Seen 11-16 with two drawn characters
(one in shorts and long socks, one in a shirt and trousers): for the first seconds, while one dancer
stood hidden behind the other, the visible one had the right head on the other one's clothes; the
head check cannot see that.

Whether the pair as a whole was given the right way round cannot be told from colours: a
comparison with the reference pictures was tried and dropped (the room's light makes dark purple
hair match black hair better than its own flat-coloured drawing, and other clothes or a hood
change the colours around a head; it gave a false alarm on a correct video).

    qa_pair_identity.py --run RUN_DIR [--masks SCENE_DIR]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from qa_fast_motion import band_regions
from scail_run import cover

APART = 0.15   # the two dancers' usual head colours must differ by this much (0-1) to be told apart
WRONG = -0.02  # a frame's margin (fit to own minus fit to other) below this: looks like the other character
# clothes: measured 11-16 on five videos. Right clothes: margin +0.2..+0.6, in doubtful poses (far away, half hidden) down
# to -0.1; the other one's clothes: -0.2..-0.3. Usual body colours of the two: 0.37-0.65 apart, 0.11 for two in alike clothes
BODY_APART = 0.25   # the two dancers' usual body colours must differ by this much to tell whose clothes are worn
BODY_WRONG = -0.15  # a frame's margin (fit to own usual body colours minus fit to the other's) below this: the other's clothes


def make_up(pixels):
    """Colour make-up of some pixels (n, 3) BGR: shares of 12 hues x 2 strengths for coloured
    pixels and of 4 brightness steps for grey ones; sums to 1."""
    hsv = cv2.cvtColor(np.ascontiguousarray(pixels, dtype=np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    coloured = (hsv[:, 1] > 45) & (hsv[:, 2] > 45)
    bins = np.where(coloured, (hsv[:, 0] // 15) * 2 + (hsv[:, 1] > 115), 24 + hsv[:, 2] // 64)
    return np.bincount(bins, minlength=28) / max(1, len(bins))


def fit(a, b):
    return float(np.minimum(a, b).sum())


def frame_head_colours(frame, picture, exchange=False):
    """Colour make-up around each dancer's head in one generated frame: {label: (28,)}; a dancer
    whose head is not seen is left out. picture: the identity mask of that frame (A blue, B red)."""
    picture = cover(picture, frame.shape[1], frame.shape[0], cv2.INTER_NEAREST)  # as the guide was fitted to the generation size
    blue, red = picture[..., 0].astype(int), picture[..., 2].astype(int)
    masks = {'A': (blue - red) > 128, 'B': (red - blue) > 128}
    if exchange:
        masks = {'A': masks['B'], 'B': masks['A']}
    out = {}
    for label, other in (('A', 'B'), ('B', 'A')):
        region = band_regions(masks[label], frame.shape[:2], erode=0).get('head')
        if region is None or region.sum() < 150:
            continue
        body = masks[other].astype(np.uint8)
        around = (cv2.dilate(region, np.ones((25, 25), np.uint8)) > 0) & (cv2.dilate(body, np.ones((9, 9), np.uint8)) == 0)
        if around.sum() >= 400:
            out[label] = make_up(frame[around])
    return out


def dancer_masks(picture):
    """{label: mask} of an identity mask picture (A blue, B red)."""
    blue, red = picture[..., 0].astype(int), picture[..., 2].astype(int)
    return {'A': (blue - red) > 128, 'B': (red - blue) > 128}


def clearly_seen(scene, first, total):
    """{label: flags over the frames} where the guide shows the dancer clearly: at least 30 % of the dancer's usual
    size in the picture (the same rule pair_run.py cuts its chunks by); None if a mask picture is missing."""
    areas = []
    for number in range(first, first + total):
        picture = cv2.imread(str(scene / 'mask' / f'frame_{number:04d}.png'))
        if picture is None:
            return None
        masks = dancer_masks(cv2.resize(picture, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_NEAREST))
        areas.append([masks['A'].sum(), masks['B'].sum()])
    areas = np.asarray(areas, dtype=float)
    if not all((areas[:, d] > 0).any() for d in (0, 1)):
        return None
    return {label: areas[:, d] >= 0.3 * np.median(areas[:, d][areas[:, d] > 0]) for d, label in enumerate('AB')}


def frame_body_colours(frame, picture):
    """Colour make-up of each dancer's body below the head in one generated frame: {label: (28,)}; a dancer of
    whom too little is seen is left out. picture: the identity mask of that frame (A blue, B red). The region is
    the dancer's own mask, a little inside its edge, without the top 22 % of its height (the head band)."""
    picture = cover(picture, frame.shape[1], frame.shape[0], cv2.INTER_NEAREST)
    out = {}
    for label, mask in dancer_masks(picture).items():
        rows = np.flatnonzero(mask.any(1))
        if len(rows) < 20:
            continue
        below = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        below[:rows[0] + int(0.22 * (rows[-1] - rows[0] + 1))] = False
        if below.sum() >= 1500:
            out[label] = make_up(frame[below])
    return out


def judge_bodies(seen, fps):
    """seen: {label: (frames, 28)} body colours, nan where the dancer is not clearly seen.
    Returns (how different the two dancers' usual body colours are, {label: flags "wears the other one's colours"
    for a third of a second or more}). The usual colours come from the frames in which both are clearly seen."""
    usable = {label: np.isfinite(seen[label][:, 0]) for label in 'AB'}
    both = usable['A'] & usable['B']
    none = {label: np.zeros(len(both), dtype=bool) for label in 'AB'}
    if both.sum() < 12:
        return 0.0, none
    usual = {label: np.nanmedian(seen[label][both], axis=0) for label in 'AB'}
    usual = {label: value / max(value.sum(), 1e-9) for label, value in usual.items()}
    apart = 1.0 - fit(usual['A'], usual['B'])
    wrong = {}
    for label, other in (('A', 'B'), ('B', 'A')):
        margin = np.minimum(seen[label], usual[label]).sum(axis=1) - np.minimum(seen[label], usual[other]).sum(axis=1)
        low = np.where(usable[label], margin < BODY_WRONG, False)
        flags, start = none[label].copy(), None
        for index, flag in enumerate(list(low) + [False]):
            if flag and start is None:
                start = index
            if not flag and start is not None:
                if index - start >= fps / 3:
                    flags[start:index] = True
                start = None
        wrong[label] = flags
    return apart, wrong


def body_colours(run, scene, first, total):
    """Per dancer and frame the colour make-up of the body below the head: {label: (frames, 28)}, nan where the
    dancer is not clearly seen; None if the scene has no identity masks."""
    clear = clearly_seen(scene, first, total)
    if clear is None:
        return None
    seen = {label: np.full((total, 28), np.nan) for label in 'AB'}
    for index in range(total):
        frame = cv2.imread(str(run / 'generated' / f'frame_{index + 1:04d}.png'))
        picture = cv2.imread(str(scene / 'mask' / f'frame_{first + index:04d}.png'))
        for label, value in frame_body_colours(frame, picture).items():
            if clear[label][index]:
                seen[label][index] = value
    return seen


def head_colours(run, scene, first, total, exchange=()):
    """Per dancer and frame the colour make-up around the head: {label: (frames, 28)}, nan where
    the head is not seen. exchange: frames in which the two dancers are exchanged (self-test)."""
    seen = {label: np.full((total, 28), np.nan) for label in 'AB'}
    for index in range(total):
        frame = cv2.imread(str(run / 'generated' / f'frame_{index + 1:04d}.png'))
        picture = cv2.imread(str(scene / 'mask' / f'frame_{first + index:04d}.png'))
        for label, value in frame_head_colours(frame, picture, index in exchange).items():
            seen[label][index] = value
    return seen


def judge(seen, fps):
    """seen: {label: (frames, 28)} head colours (nan where the head is not seen).
    Returns (usual colours per dancer, how different the two are, per-dancer flags "looks like the
    other one", flags of the frames that count as wrong, their runs of half a second and more)."""
    both_seen = np.isfinite(seen['A'][:, 0]) & np.isfinite(seen['B'][:, 0])
    # a dancer's usual colours: from the frames in which both heads are seen (while one is hidden the
    # model may put its head on the other one, and those frames must not count as usual)
    usual = {label: np.nanmedian(seen[label][both_seen] if both_seen.sum() >= 12 else seen[label], axis=0) for label in 'AB'}
    usual = {label: value / max(value.sum(), 1e-9) for label, value in usual.items()}
    apart = 1.0 - fit(usual['A'], usual['B'])
    wrong, unseen = {}, {}
    for label, other in (('A', 'B'), ('B', 'A')):
        own = np.minimum(seen[label], usual[label]).sum(axis=1)
        unseen[label] = ~np.isfinite(seen[label][:, 0])
        margin = np.where(unseen[label], np.nan, own - np.minimum(seen[label], usual[other]).sum(axis=1))
        wrong[label] = margin < WRONG  # nan compares False
    # wrong: both show the other one's colours (a swap), or one shows the other one's colours while
    # that other one is not to be seen (its head was put on the wrong body)
    bad = (wrong['A'] & (wrong['B'] | unseen['B'])) | (wrong['B'] & (wrong['A'] | unseen['A']))
    spans, start = [], None
    for index, flag in enumerate(list(bad) + [False]):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            if index - start >= 0.5 * fps:
                spans.append((start, index))
            start = None
    return usual, apart, wrong, bad, spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--masks', help='scene folder with the identity mask pass (default: the run\'s scene)')
    parser.add_argument('--self-test', help='first:last frames in which the two dancers are exchanged on purpose')
    parser.add_argument('--output')
    args = parser.parse_args()
    run = Path(args.run)
    record = json.loads((run / 'run.json').read_text())
    scene = Path(args.masks) if args.masks else Path(record['scene'])
    first, total, fps = record['first_frame'], record['frames'], record['fps']
    exchange = set()
    if args.self_test:
        start, stop = (int(v) for v in args.self_test.split(':'))
        exchange = set(range(start - 1, stop))
    seen = head_colours(run, scene, first, total, exchange)
    usual, apart, wrong, bad, spans = judge(seen, fps)
    can_tell = apart > APART
    summary = dict(frames=total, feature='colours around each dancer\'s head', head_seen={label: int(np.isfinite(seen[label][:, 0]).sum()) for label in 'AB'},
                   difference_between_the_two=round(apart, 3), can_tell_apart=bool(can_tell),
                   frames_looking_like_the_other={label: int(wrong[label].sum()) for label in 'AB'},
                   frames_counted_as_wrong=int(bad.sum()),
                   suspected_swaps_seconds=[[round(a / fps, 2), round(b / fps, 2)] for a, b in spans] if can_tell else None)
    bodies = None if exchange else body_colours(run, scene, first, total)
    if bodies is not None:
        body_apart, dressed = judge_bodies(bodies, fps)
        told = body_apart > BODY_APART
        mixed, start = [], None
        for index, flag in enumerate(list(dressed['A'] | dressed['B']) + [False]):
            if flag and start is None:
                start = index
            if not flag and start is not None:
                mixed.append([round(start / fps, 2), round(index / fps, 2)])
                start = None
        summary['outfits'] = dict(feature='colours of each dancer\'s body below the head', difference_between_the_two=round(body_apart, 3),
                                  can_tell_apart=bool(told),
                                  frames_in_the_other_ones_colours={label: int(dressed[label].sum()) for label in 'AB'} if told else None,
                                  mixed_seconds=mixed if told else None)
    print(json.dumps(summary))
    if args.output:
        Path(args.output).write_text(json.dumps(summary))


if __name__ == '__main__':
    main()
