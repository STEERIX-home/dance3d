"""The edit of the challenge video: which camera shows which part of the dance, where the cuts are,
and which effect each cut and each strong beat gets. Nothing in here is chosen by looking at one
video; every choice is a rule on the music and the motion.

    cut_plan.py --scene-inputs DIR/scene_inputs.json --facts motion_facts.json --beats beats.json --output plan.json

Rules:
  cuts       on the music: one shot per bar (4 beats) when a bar lasts 1.2-2.6 s, otherwise per 2 or
             8 beats, whichever is nearest to 1.9 s; no shot shorter than 0.75 s. The bar with the
             strongest beats (not the first or last shot) is cut once more in its middle.
  cameras    the front camera is the base: every second shot is the front, the shots in between go
             round left, right and back (as the front viewer sees it: "left" stands on that viewer's
             left). The last shot is always the front.
  close-ups  where the joints program had to fill in the legs (the source showed the dancer from the
             waist up), the camera comes close and shows the dancer from the thighs up, from the
             beat before that part to the beat after it, so the guessed legs are not shown.
  effects    a section change (into or out of a close-up) gets a white flash, a cut to or from the
             back a whip pan, the extra middle cut a glitch, other cuts by the strength of their beat
             (strong: flash, middle: zoom punch, weak: glitch); strong beats inside a shot a small
             zoom pulse, the strongest one a short shake.
Each shot is made by the video model in a window on its frame grid around the shot (pair_run.py): 17k+5 frames
for H3 (--grid h3), 4k+1 for SCAIL-2 (--grid scail); the window and the frames the 3D render needs (the window
and the frames around it that pair_run.py's in-step check compares with) are written here too.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

OVERLAP = 5
DIRECTIONS = {'front': 0.0, 'left': -90.0, 'right': 90.0, 'back': 180.0}  # yaw about the dancer, from the front camera
NAMES = {'front': '정면', 'left': '왼쪽', 'right': '오른쪽', 'back': '뒤'}
AROUND = ['left', 'right', 'back']
CHOICES = ['zoom', 'flash', 'glitch']  # cut effects chosen by the beat, in the order they take turns
SHORTEST = 0.75  # s


def spans(numbers):
    """[[first, last], ...] of the runs of consecutive numbers."""
    found = []
    for number in numbers:
        if found and number == found[-1][1] + 1:
            found[-1][1] = int(number)
        else:
            found.append([int(number), int(number)])
    return found


def grid_length(length, total, grid='h3'):
    """Frames the video model makes for a shot of `length` frames: H3 17k+5 (at least 22), SCAIL-2 4k+1 (at least 25,
    about a second: the model needs a little motion to settle the character's look)."""
    if grid == 'scail':
        made = max(25, 4 * math.ceil((length - 1) / 4) + 1)
        if made > total:
            made = 4 * ((total - 1) // 4) + 1
    else:
        made = max(22, 17 * math.ceil((length - OVERLAP) / 17) + OVERLAP)
        if made > total:
            made = 17 * ((total - OVERLAP) // 17) + OVERLAP
    if made < length:
        raise ValueError(f'a shot of {length} frames does not fit a video of {total} frames')
    return made


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-inputs', required=True)
    parser.add_argument('--facts', required=True)
    parser.add_argument('--beats', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target-shot', type=float, default=1.9, help='shot length the cuts aim at (s)')
    parser.add_argument('--no-close-ups', action='store_true', help='full body also where the legs were filled in')
    parser.add_argument('--grid', default='h3', choices=['h3', 'scail'], help='frame grid of the video model')
    parser.add_argument('--seconds', type=float, help='plan only the first S seconds of the dance')
    args = parser.parse_args()
    scene = json.loads(Path(args.scene_inputs).read_text())
    facts = json.loads(Path(args.facts).read_text())
    music = json.loads(Path(args.beats).read_text())
    fps, total = float(scene['fps']), int(scene['frames'])
    if args.seconds:
        total = max(25, min(total, int(round(args.seconds * fps))))
    duration = total / fps
    beats = np.asarray([b for b in music['beats'] if 0 <= b < duration])
    strength = np.asarray(music['strength'][:len(music['beats'])])[[i for i, b in enumerate(music['beats']) if 0 <= b < duration]]
    period = 60.0 / music['bpm']

    def frame_of(seconds):
        return int(round(seconds * fps)) + 1

    def beat_strength(seconds):
        if not len(beats):
            return 0.5
        i = int(np.abs(beats - seconds).argmin())
        return float(strength[i]) if abs(beats[i] - seconds) < 0.06 else 0.3

    # beats per shot: 2, 4 or 8, nearest to the target length
    per_shot = min((2, 4, 8), key=lambda k: abs(k * period - args.target_shot) + (0 if k * period >= SHORTEST else 99))
    first_bar = music['bar_phase'] % 2 if per_shot == 2 else music['bar_phase']
    cuts = [float(b) for i, b in enumerate(beats) if i >= first_bar and (i - first_bar) % per_shot == 0]

    # close-ups where the legs were filled in, from the beat before to the beat after
    zones = []
    if not args.no_close_ups:
        for a, b in facts['legs_estimated_spans']:
            b = min(b, duration)  # a part after the end of the video (shortened, or the dancer no longer seen) is not shown
            if b - a < 0.5:
                continue
            start = 0.0 if a < 0.3 or not len(beats) else float(max([x for x in beats if x <= a + 0.15], default=0.0))
            end = duration if b > duration - 0.3 or not len(beats) else float(min([x for x in beats if x >= b - 0.15], default=duration))
            for bar in cuts:  # a bar start just outside the part is the nicer edge
                if a - 0.45 <= bar < start:
                    start = bar
                if end < bar <= b + 0.45:
                    end = bar
            if zones and start <= zones[-1][1] + SHORTEST:
                zones[-1][1] = max(zones[-1][1], end)
            else:
                zones.append([start, end])
    edges = sorted({edge for zone in zones for edge in zone if SHORTEST <= edge <= duration - SHORTEST})
    points = [c for c in cuts if SHORTEST <= c <= duration - SHORTEST and all(abs(c - e) >= SHORTEST for e in edges)]
    points = sorted(set(points) | set(edges))
    bounds = [0.0] + points + [duration]

    def in_zone(a, b):
        inside = sum(max(0.0, min(b, z1) - max(a, z0)) for z0, z1 in zones)
        return inside > 0.5 * (b - a)

    shots = [dict(start=a, end=b) for a, b in zip(bounds[:-1], bounds[1:])]
    # the strongest bar gets one more cut in its middle
    split_at = None
    if per_shot >= 4 and period * per_shot / 2 >= SHORTEST:
        middle = [(i, s) for i, s in enumerate(shots) if 0 < i < len(shots) - 1 and not in_zone(s['start'], s['end'])
                  and abs((s['end'] - s['start']) - per_shot * period) < 0.6 * period]
        if middle:
            def score(item):
                s = item[1]
                inside = [float(x) for j, x in enumerate(strength) if s['start'] - 0.05 <= beats[j] < s['end'] - 0.05]
                return np.mean(inside) if inside else 0.0
            index, chosen = max(middle, key=score)
            half = [b for b in beats if chosen['start'] + 0.05 < b < chosen['end'] - 0.05]
            half = [b for b in half if abs(b - (chosen['start'] + chosen['end']) / 2) < 0.6 * period]
            if half:
                split_at = float(half[0])
                shots[index:index + 1] = [dict(start=chosen['start'], end=split_at), dict(start=split_at, end=chosen['end'])]

    # cameras: front, then one of left/right/back, then front again ...; the last shot is the front
    around, previous = 0, None
    for i, shot in enumerate(shots):
        second_half = split_at is not None and abs(shot['start'] - split_at) < 1e-6
        if i == 0 or (previous != 'front' and not second_half):
            direction = 'front'
        else:
            direction = AROUND[around % len(AROUND)]
            around += 1
        shot['direction'] = previous = direction
    if shots[-1]['direction'] != 'front':
        shots[-1]['direction'] = 'front'
        # two front shots in a row: one shot, unless one of them is a close-up (then the cut is a step closer)
        if len(shots) > 1 and in_zone(shots[-2]['start'], shots[-2]['end']) == in_zone(shots[-1]['start'], shots[-1]['end']):
            shots[-2:] = [dict(start=shots[-2]['start'], end=shots[-1]['end'], direction='front')]

    # where each camera stands, and which side of the body it sees
    pelvis = np.asarray(scene['people'][0]['pelvis'])
    source = np.asarray(scene['source_camera']['location'])
    front = source.mean(0) - pelvis.mean(0)
    front_angle = math.atan2(front[1], front[0])
    facing = np.radians(np.asarray(facts['facing_deg_y_up']))
    forward = np.stack([np.sin(facing), -np.cos(facing)], axis=1)  # y-up world (x, z) -> Blender (x, -z)
    source_index = np.clip(np.round(np.arange(total) / fps * facts['fps']).astype(int), 0, len(forward) - 1)

    plan_shots, render = [], {}
    for i, shot in enumerate(shots):
        framing = 'close' if in_zone(shot['start'], shot['end']) else 'full'
        view = shot['direction'] + ('_close' if framing == 'close' else '')
        first, last = frame_of(shot['start']), (frame_of(shot['end']) - 1 if i < len(shots) - 1 else total)
        length = last - first + 1
        made = grid_length(length, total, args.grid)
        start = int(np.clip(first - (made - length) // 2, 1, total - made + 1))
        yaw = math.radians(DIRECTIONS[shot['direction']]) + front_angle
        towards = np.array([math.cos(yaw), math.sin(yaw)])
        seen = forward[source_index[start - 1:start - 1 + made]] @ towards
        away = [start + i for i in np.flatnonzero(seen < -0.34)]  # frames of the window that show the dancer's back
        plan_shots.append(dict(index=i + 1, name=f'shot_{i + 1:02d}_{view}', view=view, direction=shot['direction'], framing=framing,
                               label=NAMES[shot['direction']], start_s=round(shot['start'], 4), end_s=round(shot['end'], 4),
                               first=first, last=last, frames=length, window=[start, made],
                               back_share=round(float((seen < -0.34).mean()), 3), front_share=round(float((seen > 0.34).mean()), 3),
                               away_frames=spans(away)))
        frames = render.setdefault(view, set())
        # the in-step check looks 24 frames back and 12 ahead (SCAIL-2 chunks made on the time-reversed guide: 24 ahead)
        frames.update(range(max(1, start - 24), min(total, start + made - 1 + (24 if args.grid == 'scail' else 12)) + 1))

    cut_list, effect_turn, previous_kind = [], 0, None
    for before, after in zip(plan_shots[:-1], plan_shots[1:]):
        at = after['start_s']
        power = beat_strength(at)
        if before['framing'] != after['framing']:  # a step closer pushes in, a step back flashes
            kind = 'zoom' if after['framing'] == 'close' else 'flash'
        elif 'back' in (before['direction'], after['direction']):
            kind = 'whip'
        else:
            if split_at is not None and abs(at - split_at) < 1e-6:
                kind = 'glitch'
            else:
                kind = 'flash' if power >= 0.75 else 'zoom' if power >= 0.45 else 'glitch'
            if kind == previous_kind:  # the same effect twice in a row looks repeated: the next one in turn
                kind = CHOICES[(CHOICES.index(kind) + 1) % len(CHOICES)]
        previous_kind = kind
        if kind == 'whip':
            effect_turn += 1
        cut_list.append(dict(frame=after['first'], time=round(at, 4), beat_strength=round(power, 3), transition=kind,
                             whip_to='left' if effect_turn % 2 else 'right', between=[before['name'], after['name']]))
    cut_frames = {cut['frame'] for cut in cut_list}
    pulses = []
    for b, s in zip(beats, strength):
        frame = frame_of(b)
        if s >= 0.7 and all(abs(frame - c) > 3 for c in cut_frames) and 1 < frame < total - 2:
            pulses.append(dict(frame=frame, time=round(float(b), 4), strength=round(float(s), 3)))
    later = [p for p in pulses if p['frame'] > plan_shots[0]['last']]  # not under the title
    if later:
        max(later, key=lambda p: p['strength'])['shake'] = True

    views = {view: dict(direction=view.split('_')[0], yaw_deg=DIRECTIONS[view.split('_')[0]],
                        framing='close' if view.endswith('_close') else 'full') for view in render}
    plan = dict(fps=fps, frames=total, seconds=round(duration, 4), bpm=music['bpm'], beats_per_shot=per_shot, grid=args.grid,
                beat_method=music['method'], front_angle_rad=round(front_angle, 6), close_zones=[[round(a, 3), round(b, 3)] for a, b in zones],
                split_at=split_at, shots=plan_shots, cuts=cut_list, pulses=pulses, views=views,
                render={view: sorted(frames) for view, frames in render.items()},
                front_share=round(sum(s['frames'] for s in plan_shots if s['direction'] == 'front') / total, 3),
                frames_made=sum(s['window'][1] for s in plan_shots))
    Path(args.output).write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    print(f'{len(plan_shots)} shots, {per_shot} beats per shot at {music["bpm"]} per minute, front {plan["front_share"]:.0%} of the time, '
          f'{plan["frames_made"]} frames to make for {total}; close-ups {plan["close_zones"]}')
    for shot in plan_shots:
        print(f'  {shot["start_s"]:6.2f}-{shot["end_s"]:6.2f} s  {shot["view"]:12s} frames {shot["first"]}-{shot["last"]} '
              f'window {shot["window"]} back seen {shot["back_share"]:.0%}')
    for cut in cut_list:
        print(f'  cut {cut["time"]:6.2f} s  {cut["transition"]}')


if __name__ == '__main__':
    main()
