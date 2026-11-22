"""The scenes of the long film (1920x1080)."""
import json

import cv2
import numpy as np

import sources
import widgets as wd
from common import A_COLOUR, B_COLOUR, arrow, cover, covered, phone, stat, sting, video_card
from kit import (AMBER, BG, DIM, FPS, GREEN, GREY, LINE, PANEL, SKY, VIOLET, WHITE, WORK, back_out, box, card, ease, ease_in, ease_out, fit,
                 glow, outline, paste, put, ramp, shadowed, text_layer, window)
from skeleton import figure_panel, orbit_camera, stage_panel
from sources import dance

W, H = 1920, 1080
MAIN = json.loads((WORK / 'main.json').read_text())   # the dance the capture and camera chapters follow
CAPTION_X, CAPTION_Y = 96, 1012
_read = lambda name: json.loads((WORK / name).read_text()) if (WORK / name).exists() else None
DUO, NINE = _read('duo.json'), _read('nine.json')   # the two-dancer run and the challenge run of the robot's dance, once they exist
DUO_BARS = (4, 3, 1)       # the bar of the two-dancer dance at which each of its scenes starts
ROBOT_BARS = (4, 0, 9, 12)  # the same for the robot's dance: dance, graph (where the eyes were found nearly all the time), free, recap


class Group:
    """Scenes that share one piece of music: every scene lasts whole bars (or half bars, kept in step with the dance)."""

    def __init__(self, film, name, bar, chapter=None):
        self.film, self.name, self.bar, self.chapter, self.elapsed = film, name, bar, chapter, 0.0

    def add(self, name, draw, need, at_bar=0.0, lines=(), half=False, **more):
        unit = self.bar / 2 if half else self.bar
        count = int(np.ceil(need / unit - 1e-6))
        offset = self.elapsed / self.bar - at_bar
        assert abs(offset - round(offset)) < 1e-6, f'{name}: the dance would leave the music grid ({self.elapsed / self.bar} vs bar {at_bar})'
        scene = self.film.add(name, draw, count * unit, lines=lines, chapter=more.pop('chapter', self.chapter), at_bar=at_bar,
                              music=dict(group=self.name, bar=self.bar, bars_before=self.elapsed / self.bar), **more)
        scene.bar_length = self.bar
        self.elapsed += count * unit
        return scene


def front_start(source, wanted, low, high):
    """The bar (a whole number between low and high) nearest to `wanted` at which a full front shot of a challenge video starts."""
    if source is None:
        return wanted
    found = []
    for shot in source.shots:
        if shot['direction'] == 'front' and shot['framing'] == 'full':
            bar = round((shot['start_s'] - source.first_bar) / source.bar)
            if low <= bar <= high and abs(source.first_bar + bar * source.bar - shot['start_s']) < 0.25:
                found.append(bar)
    return min(found, key=lambda bar: abs(bar - wanted)) if found else wanted


def front_bars(source, count, first):
    """The bars (whole numbers, from `first` on) at which the next `count` full front shots of a challenge video start."""
    found = []
    for shot in source.shots:
        if shot['direction'] == 'front' and shot['framing'] == 'full':
            bar = round((shot['start_s'] - source.first_bar) / source.bar)
            long_enough = shot['end_s'] - shot['start_s'] > source.bar - 0.2
            if bar >= first and long_enough and abs(source.first_bar + bar * source.bar - shot['start_s']) < 0.25:
                found.append(bar)
    if len(found) < count:
        raise ValueError(f'the challenge video has only {len(found)} front shots of a whole bar from bar {first} on, {count} are needed')
    return found[:count]


def main_time(scene, t):
    """Seconds into the main dance shown at scene time t."""
    challenge = scene.film.main
    return challenge.first_bar + scene.at_bar * challenge.bar + t


# ------------------------------------------------------------------------------------------------ opening
def cold_open(c, t, s):
    film = s.film
    main, d = film.main, main_time(s, t)
    kind = s.kind
    if kind == 'video':
        region = (745, 96, 430, 764)
        phone(c, film.main_dance, d, region, outline_of=film.main_outline)
        figure_panel(c, film.main_dance, d, (1230, 110, 520, 760), yaw_deg=18, glow_strength=0.8, zoom=0.95)
        figure_panel(c, film.main_dance, d, (170, 110, 520, 760), yaw_deg=-28, glow_strength=0.8, zoom=0.95)
        wd.caption(c, ('ONE DANCE VIDEO.', (('VIDEO', GREEN),)), t, 0.02, s.seconds, W / 2, 1040, 104, 'cb', over_picture=False)
    elif kind == 'character':
        x0, wide, tall, gap = 190, 495, 880, 28
        numbers = [main.number(d), main.number(d), main.number(d)]
        if film.duo is not None:
            pictures = [main.made(numbers[0]), film.duo.result('original', film.duo.number(film.duo.clean_from('original', 8.0, s.seconds) + t)),
                        film.duo.result('orbit', film.duo.number(film.duo.clean_from('orbit', 20.0, s.seconds) + t))]
        else:
            pictures = [main.made(numbers[0])] + [p() for p in s.others(film, t)]
        for index, picture in enumerate(pictures):
            video_card(c, picture, (x0 + index * (wide + gap), 44, wide, tall), focus=(0.5, 0.3))
        wd.caption(c, ('ANY CHARACTER.', (('CHARACTER', GREEN),)), t, 0.02, s.seconds, W / 2, 1040, 104, 'cb')
    elif kind == 'camera':
        x0, wide, tall, gap = 190, 495, 880, 28
        source = film.nine or main
        for index, (view, which) in enumerate(s.views):
            shot = source.shot_of(view, which) or source.shot_of('front', 0)
            video_card(c, source.made(source.shot_frame(shot, t)), (x0 + index * (wide + gap), 44, wide, tall), label=wd.VIEW_NAME[shot['direction']],
                       colour=wd.VIEW_COLOUR[shot['direction']], focus=(0.5, 0.3), label_size=22)
        wd.caption(c, ('ANY CAMERA.', (('CAMERA', GREEN),)), t, 0.02, s.seconds, W / 2, 1040, 104, 'cb')
    else:
        robot = film.robot
        picture = robot.picture(robot.footage_time(s.robot_from + t))
        paste(c, fit(picture, W, H, 'cover', (0.62, 0.42)), 0, 0)
        wd.chip(c, 'REAL ROBOT · REAL TIME', 64, 64, GREEN, 1.0, 20)
        wd.caption(c, ('EVEN A REAL ROBOT.', (('ROBOT', GREEN),)), t, 0.02, s.seconds, W / 2, 1040, 104, 'cb')


def logo_sting(c, t, s):
    sting(c, s.film.brand_in, t, first=s.first)


def title(c, t, s):
    film, d = s.film, main_time(s, t)
    figure_panel(c, film.main_dance, d, (1010, 60, 820, 960), yaw_deg=-20 + 14 * t, glow_strength=0.9)
    wd.caption(c, ('FROM ONE\nDANCE VIDEO', (('ONE', GREEN),)), t, 0.1, s.seconds, 120, 300, 132, 'lt', over_picture=False)
    for index, (text, colour) in enumerate((('3D MOTION', GREEN), ('CHARACTER VIDEO', SKY), ('ROBOT MOTION', AMBER))):
        shown = ramp(t, 0.7 + 0.18 * index, 0.95 + 0.18 * index)
        wd.chip(c, text, 124, 640 + index * 62, colour, shown, 24, over_picture=False)


# ------------------------------------------------------------------------------------------------ 01 capture
def capture_in(c, t, s):
    film, d = s.film, main_time(s, t)
    appear = ease_out(ramp(t, 0.0, 0.5))
    phone(c, film.main_dance, d, (300 - (1 - appear) * 40, 150, 430, 764), appear, label='INPUT · ONE VIDEO', outline_of=film.main_outline)
    lift = ramp(t, 1.5, 2.6)
    arrow(c, 790, 532, 960, ramp(t, 1.2, 2.0))
    if lift > 0:
        figure_panel(c, film.main_dance, d, (1010, 70, 800, 940), yaw_deg=-35 + 25 * t, strength=ease(lift), glow_strength=0.9 * lift)
        wd.label(c, 'OUTPUT · THE DANCER IN 3D', 1410, 930, 18, GREY, 'ct', lift)
    wd.caption(c, ('VIDEO → 3D JOINTS', (('3D JOINTS', GREEN),)), t, film.word(s, 'L02', 'video') + 0.4, s.seconds, CAPTION_X, CAPTION_Y, 92, 'lb',
               over_picture=False)


def capture_3d(c, t, s):
    film, d = s.film, main_time(s, t)
    person = film.main_dance
    region = (60, 70, 1120, 950)
    yaw = -30 + 360 * ease(t / s.seconds) * 0.92
    camera = figure_panel(c, person, d, region, yaw_deg=yaw, distance=4.4, glow_strength=1.0)
    # a ruler beside the dancer: real metres from the real floor
    t_metres = film.word(s, 'L03', 'metres') - 0.5
    grown = ease_out(ramp(t, t_metres, t_metres + 0.6))
    anchor = person.anchor_at(d)
    side = np.cross([0.0, 1.0, 0.0], person.forward)
    foot = np.array([anchor[0], 0.0, anchor[2]]) + side * 0.75
    if grown > 0:
        top = foot + np.array([0.0, 1.75 * grown, 0.0])
        line = camera.segment(foot, top)
        if line is not None:
            cv2.line(c, tuple(line[0].astype(int)), tuple(line[1].astype(int)), WHITE, 2, cv2.LINE_AA)
            for k in range(0, 8):
                level = 0.25 * k
                if level > 1.75 * grown:
                    break
                p = camera.project(np.array([foot + [0.0, level, 0.0]]))[0][0]
                major = k % 2 == 0
                cv2.line(c, (int(p[0]), int(p[1])), (int(p[0] + (16 if major else 9)), int(p[1])), WHITE, 2, cv2.LINE_AA)
                if major and k:
                    wd.label(c, f'{level:.1f} m', p[0] + 24, p[1], 18, WHITE, 'lm')
    # feet on the floor: a ring under a foot while it is planted
    t_floor = film.word(s, 'L03', 'floor') - 1.0
    shown = ramp(t, t_floor, t_floor + 0.4)
    if shown > 0:
        contact = person.contact[person.index(d)]
        pose = person.pose(d)
        for planted, ankle, colour in ((contact[0], 7, SKY), (contact[1], 8, AMBER)):
            if planted:
                ground = pose[ankle].copy()
                ground[1] = 0.0
                centre = camera.project(ground[None])[0][0]
                edge = camera.project((ground + side * 0.17)[None])[0][0]
                radius = max(6, int(np.hypot(*(edge - centre))))
                cv2.ellipse(c, (int(centre[0]), int(centre[1])), (radius, max(3, int(radius * 0.38))), 0, 0, 360,
                            tuple(int(v * shown) for v in colour), 3, cv2.LINE_AA)
    x = 1280
    t_joints = film.word(s, 'L03', 'Seventy')
    wd.caption(c, ('REBUILT\nIN 3D', (('3D', GREEN),)), t, film.word(s, 'L03', 'rebuilds') - 0.1, t_joints - 0.1, x, 400, 132, 'lt', over_picture=False)
    count = int(round(person.joints.shape[1] * ease_out(ramp(t, t_joints, t_joints + 0.9))))
    stat(c, x, 190, f'{count}', 'JOINTS · WHOLE BODY, HANDS, FACE', t, t_joints, GREEN, 150)
    stat(c, x, 470, 'METRES', 'WORLD SCALE · NOT PIXELS', t, t_metres + 0.3, WHITE, 96)
    stat(c, x, 700, 'FLOOR', 'CONTACT FOUND FOR EACH FOOT', t, t_floor + 0.3, WHITE, 96)
    if shown > 0:
        for index, (name, colour) in enumerate((('LEFT FOOT', SKY), ('RIGHT FOOT', AMBER))):
            on = person.contact[person.index(d)][index]
            cv2.circle(c, (x + 12 + index * 220, 880), 10, colour if on else DIM, -1, cv2.LINE_AA)
            wd.label(c, name, x + 32 + index * 220, 880, 18, WHITE if on else GREY, 'lm', shown)


def _series(person, joint, axis=1, scale=1.0, smooth=0):
    t = np.arange(person.frames) / person.fps
    return t, person.joints[:, joint, axis] * scale


def capture_curves(c, t, s):
    film, d = s.film, main_time(s, t)
    person, main = film.main_dance, film.main
    figure_panel(c, person, d, (60, 70, 640, 850), yaw_deg=-18 + 6 * t, glow_strength=0.9)
    span = (d - 3.4, d + 1.6)
    tl, left = _series(person, 20)
    _, right = _series(person, 21)
    _, pelvis = _series(person, 0)
    _, lfoot = _series(person, 7)
    _, rfoot = _series(person, 8)
    lateral = person.joints[:, 16] - person.joints[:, 17]
    turn = np.degrees(np.unwrap(np.arctan2(lateral[:, 2], lateral[:, 0])))
    turn = turn - np.median(turn)
    x, w = 740, 1116
    show = lambda k: ease_out(ramp(t, 0.15 + 0.22 * k, 0.6 + 0.22 * k))
    wd.curve_plot(c, (x, 110, w, 262), [dict(t=tl, y=left, colour=SKY, label='LEFT WRIST'), dict(t=tl, y=right, colour=AMBER, label='RIGHT WRIST')],
                  d, span, (0.2, 2.1), 'HEIGHT', show(0), beats=main.beats, values='{:.2f} m')
    wd.curve_plot(c, (x, 392, w, 262), [dict(t=tl, y=lfoot, colour=SKY, label='LEFT ANKLE'), dict(t=tl, y=rfoot, colour=AMBER, label='RIGHT ANKLE'),
                                         dict(t=tl, y=pelvis, colour=GREEN, label='PELVIS')],
                  d, span, (0.0, 1.25), 'HEIGHT', show(1), beats=main.beats, values='{:.2f} m')
    low, high = np.percentile(turn, [1, 99])
    wd.curve_plot(c, (x, 674, w, 262), [dict(t=tl, y=turn, colour=GREEN, label='SHOULDERS TURN')], d, span,
                  (low - 15, high + 15), 'ANGLE', show(2), beats=main.beats, values='{:+.0f}°', zero=0.0)
    wd.caption(c, ('THE DANCE, AS DATA', (('DATA', GREEN),)), t, film.word(s, 'L04', 'This') - 0.1, s.seconds, CAPTION_X, CAPTION_Y + 16, 92, 'lb',
               over_picture=True)


def capture_wall(c, t, s):
    film = s.film
    cols, rows, cw, chh = 5, 3, 348, 250
    for index in range(15):
        number = index + 1
        person = dance(number)
        col, row = index % cols, index // cols
        x, y = 64 + col * 361, 96 + row * 262
        shown = ramp(t, 0.05 + 0.045 * index, 0.3 + 0.045 * index)
        if shown <= 0:
            continue
        box(c, x, y, cw, chh, PANEL, shown, 14)
        outline(c, x, y, cw, chh, 14, LINE, 1, shown)
        start = min(max(0.0, person.seconds - s.seconds - 0.5), 2.0 + (index * 1.7) % 6)
        figure_panel(c, person, start + t, (x + 4, y + 4, cw - 8, chh - 8), yaw_deg=-15 + (index % 5) * 8, strength=shown, glow_strength=0.6 * shown,
                     hands=False, trails=True, zoom=0.86, distance=4.6)
        wd.label(c, f'{number:02d}', x + 16, y + 14, 20, GREEN, 'lt', shown)
        wd.label(c, f'{person.seconds:.0f} s', x + cw - 16, y + 14, 16, GREY, 'rt', shown)
    start = film.word(s, 'L05', 'Forty') - 0.25
    wd.caption(c, (f'{film.joints_run["dances"]} DANCES · {film.joints_run["minutes"]} MIN · HANDS-OFF', (('HANDS-OFF', GREEN),)), t, start, s.seconds,
               CAPTION_X, CAPTION_Y + 16, 92, 'lb', over_picture=False)


def capture_accuracy(c, t, s):
    film, d = s.film, main_time(s, t)
    person = film.main_dance
    phone(c, person, d, (110, 150, 390, 694), 1.0, label='JOINTS LAID BACK OVER THE VIDEO', outline_of=film.main_outline)
    # the body's height as a ruler, and the size of the typical gap on it
    pts = person.pose2d(d)
    grown = ease_out(ramp(t, 0.5, 1.1))
    x, top, bottom = 560, 190, 800
    box(c, x, top + (bottom - top) * (1 - grown), 3, (bottom - top) * grown, WHITE, 1.0)
    if grown > 0.95:
        for y in (top, bottom):
            box(c, x - 10, y, 23, 3, WHITE)
        wd.label(c, 'BODY HEIGHT', x + 24, (top + bottom) / 2 - 14, 18, GREY, 'lm')
        wd.label(c, '= 100 %', x + 24, (top + bottom) / 2 + 14, 18, GREY, 'lm')
    t_number = film.word(s, 'L06', 'typically')
    values = film.accuracy
    typical = float(np.median(values))
    sliver = ease_out(ramp(t, t_number, t_number + 0.5))
    if sliver > 0:
        size = max(3, int(round((bottom - top) * typical / 100)))
        box(c, x - 16, bottom - size, 35, size, GREEN, sliver)
        wd.label(c, f'{typical:.1f} %', x + 24, bottom - size / 2, 20, GREEN, 'lm', sliver)
    wd.bars(c, (860, 150, 996, 520), values, t, 1.0, labels=[f'{k:02d}' for k in range(1, len(values) + 1)],
            title=f'GAP BETWEEN JOINTS AND DANCER · % OF BODY HEIGHT · {len(values)} DANCES', top=2.4, fmt='{:.1f}', colour=GREEN, ticks=(1.0, 2.0),
            tick_fmt='{:.0f} %')
    stat(c, 870, 720, f'{typical:.1f}', f'TYPICAL GAP (MEDIAN OF {len(values)} DANCES · RANGE {min(values):.1f} – {max(values):.1f} %)', t, t_number + 0.2, GREEN,
         150, unit='% of body height')
    wd.caption(c, (f'TYPICAL GAP · {typical:.1f} % OF BODY HEIGHT', ((f'{typical:.1f} %', GREEN),)), t, film.word(s, 'L06', 'one') - 0.1, s.seconds, CAPTION_X,
               CAPTION_Y + 24, 84, 'lb', over_picture=False)


# ------------------------------------------------------------------------------------------------ 02 any camera
def camera_anywhere(c, t, s):
    film, d = s.film, main_time(s, t)
    person = film.main_dance
    yaw = -40 + 400 * ease(ramp(t, 0.2, s.seconds))
    figure_panel(c, person, d, (64, 80, 1130, 930), yaw_deg=yaw, distance=4.6, height=1.5 + 0.5 * np.sin(t * 0.9), glow_strength=1.0)
    wd.camera_map(c, (1260, 190, 596, 596), None, orbit_deg=yaw, title='THE CAMERA, SEEN FROM ABOVE')
    wd.caption(c, ('A CAMERA CAN STAND ANYWHERE', (('ANYWHERE', GREEN),)), t, film.word(s, 'L07', 'camera') - 0.1, s.seconds, CAPTION_X, CAPTION_Y, 92,
               'lb', over_picture=False)


def camera_four(c, t, s):
    film, d = s.film, main_time(s, t)
    person, main = film.main_dance, film.main
    views = (('front', 0), ('left', -90), ('right', 90), ('back', 180))
    wide, tall, gap, x0, y0 = 418, 743, 30, 64, 120
    for index, (view, yaw) in enumerate(views):
        said = dict(s.lines)[f'L07{"cdef"[index]}']
        lit = ramp(t, said - 0.05, said + 0.12)
        x = x0 + index * (wide + gap)
        box(c, x, y0, wide, tall, PANEL, 1.0, 18)
        quad = film.quad(view, main.number(d)) if film.quad else None
        if quad is not None:
            video_card(c, quad, (x, y0, wide, tall), opacity=0.35 + 0.65 * lit)
        else:
            figure_panel(c, person, d, (x + 4, y0 + 4, wide - 8, tall - 8), yaw_deg=yaw, strength=0.35 + 0.65 * lit, glow_strength=0.9 * lit, distance=4.4)
        outline(c, x, y0, wide, tall, 18, wd.VIEW_COLOUR[view] if lit > 0.5 else LINE, 3 if lit > 0.5 else 1)
        pop = back_out(lit) if lit > 0 else 0
        if pop > 0:
            layer = text_layer(wd.VIEW_NAME[view], 'semibold', 84, wd.VIEW_COLOUR[view])
            put(c, layer, x + wide / 2, y0 + tall + 22, min(1.0, lit * 1.5), 'ct', scale=0.8 + 0.2 * pop)


def camera_paint(c, t, s):
    film, d = s.film, main_time(s, t)
    main = film.main
    number = main.number(d)
    shot = main.shots[main.owner[number]]
    # the character picture
    shown = ease_out(ramp(t, 0.0, 0.4))
    ref = film.reference('A', 380)
    card(c, ref, 120, 300 + (1 - shown) * 30, 18, shown, WHITE, 2)
    wd.chip(c, 'CHARACTER PICTURE', 120 + ref.shape[1] / 2, 300 + ref.shape[0] + 20, GREEN, shown, 18, anchor='ct', over_picture=False)
    arrow(c, 120 + ref.shape[1] + 30, 500, 560, ramp(t, 0.5, 1.1))
    # guide and painted result, same frame, same camera
    gx, rx, y, wide, tall = 600, 1110, 96, 478, 850
    paint = film.word(s, 'L08', 'paints')
    done = ease(ramp(t, paint - 0.1, paint + 1.1))
    arrive = ease_out(ramp(t, paint - 0.75, paint - 0.25))     # the second card slides out of the guide
    made, guide = main.made(number), main.guide(number)
    if done >= 1 or guide is None:
        video_card(c, made, (rx, y, wide, tall), label='PAINTED BY A VIDEO MODEL', colour=GREEN)
    elif arrive > 0:
        mixed = fit(guide, wide, tall, 'cover').copy()
        edge = int(tall * done)
        if edge > 0:
            mixed[:edge] = fit(made, wide, tall, 'cover')[:edge]
            cv2.line(mixed, (0, min(tall - 1, edge)), (wide, min(tall - 1, edge)), GREEN, 4)
        card(c, mixed, gx + (rx - gx) * arrive, y, 18, arrive, LINE, 2)
        if done > 0:
            wd.chip(c, 'PAINTED BY A VIDEO MODEL', rx + 14, y + 14, GREEN, 1.0, 18)
    video_card(c, main.guide(number), (gx, y, wide, tall), label='3D GUIDE', colour=SKY)
    # shot by shot
    listed = film.word(s, 'L08', 'shot') - 0.2
    for index, each in enumerate(main.shots):
        shown = ramp(t, listed + 0.05 * index, listed + 0.05 * index + 0.2)
        if shown <= 0:
            continue
        yy = 150 + index * 62
        active = each is shot
        seen = each['index'] <= shot['index']
        colour = wd.VIEW_COLOUR[each['direction']]
        box(c, 1626, yy, 230, 48, colour if active else PANEL, shown, 10)
        if not active:
            outline(c, 1626, yy, 230, 48, 10, LINE, 1, shown)
        wd.label(c, f'{each["index"]:02d}', 1640, yy + 24, 17, BG if active else GREY, 'lm', shown)
        wd.label(c, wd.VIEW_NAME[each['direction']] + (' CLOSE' if each['framing'] == 'close' else ''), 1678, yy + 24, 17,
                 BG if active else (WHITE if seen else GREY), 'lm', shown)
    wd.caption(c, ('PAINTED OVER THE 3D GUIDE', (('3D GUIDE', GREEN),)), t, paint + 0.9, s.seconds, CAPTION_X, CAPTION_Y + 20, 84, 'lb',
               over_picture=False)


def camera_beat(c, t, s):
    film, d = s.film, main_time(s, t)
    main = film.main
    number = main.number(d)
    shot = main.shots[main.owner[number]]
    video_card(c, main.clean.get(number - 1), (735, 84, 450, 800))
    wd.chip(c, wd.VIEW_NAME[shot['direction']], 749, 98, wd.VIEW_COLOUR[shot['direction']], 1.0, 22)
    wd.camera_map(c, (120, 150, 520, 520), shot['direction'], title='CAMERA FILMING NOW')
    # tempo, bar and beat counters
    beat = int(np.searchsorted(main.beats, d + 1e-3))
    since = d - main.beats[max(0, beat - 1)]
    pulse = np.exp(-max(0.0, since) / 0.11)
    stat(c, 1290, 170, f'{main.plan["bpm"]:g}', 'BEATS PER MINUTE · FOUND IN THE MUSIC', t, 0.2, WHITE, 150)
    for k in range(4):
        on = (beat - 1) % 4 == k
        cv2.circle(c, (1306 + k * 62, 470), int(16 + (8 * pulse if on else 0)), GREEN if on else DIM, -1, cv2.LINE_AA)
    wd.label(c, 'BEAT IN THE BAR', 1290, 512, 20, GREY)
    cuts = sum(1 for cut in main.cuts if cut['time'] <= d)
    stat(c, 1290, 590, f'{cuts}', f'CUTS SO FAR · EVERY ONE ON A BEAT', t, 0.4, GREEN, 110)
    span = (main.first_bar + s.at_bar * main.bar - 1.0, min(main.seconds, main.first_bar + s.at_bar * main.bar + s.seconds + 1.0))
    wd.beat_strip(c, (64, 896, W - 128, 168), main.envelope, main.beats, main.bars, main.shots, d, span)
    said = film.word(s, 'L09', 'cuts')
    wd.caption(c, ('CUT ON\nTHE BEAT', (('BEAT', GREEN),)), t, said, s.seconds, 124, 866, 84, 'lb', over_picture=False)


def camera_checks(c, t, s):
    film, d = s.film, main_time(s, t)
    main = film.main
    number = main.number(d)
    shot = main.shots[main.owner[number]]
    video_card(c, main.clean.get(number - 1), (150, 96, 478, 850))
    wd.chip(c, wd.VIEW_NAME[shot['direction']], 164, 110, wd.VIEW_COLOUR[shot['direction']], 1.0, 22)
    checks = main.checks
    shots = checks['shots']
    feet = min((x['framing']['feet_in'] for x in shots if x.get('framing') and 'close' not in x['view']), default=1.0)
    head = min((x['framing']['head_in'] for x in shots if x.get('framing')), default=1.0)
    bad = sum(x['frames_without_the_room'] + x['frames_with_a_mannequin_left'] for x in shots)
    late = sum(x['frames_out_of_step'] for x in shots)
    rows = [('Every cut on a beat', f'{checks["cuts"]["furthest_from_a_beat_s"]:.2f} s off', film.word(s, 'L10', 'every') - 0.15),
            ('Feet in frame, full shots', f'{feet:.0%} of frames', film.word(s, 'L10', 'feet') - 0.15),
            ('Head in frame', f'{head:.0%} of frames', film.word(s, 'L10', 'feet') + 0.25),
            ('Room kept, no guide left over', f'{bad} bad frames', film.word(s, 'L10', 'colours') - 0.5),
            ('In step with the 3D guide', f'{late} frames late', film.word(s, 'L10', 'colours') - 0.15),
            ('Colours steady across shots', 'pass' if not checks['problems'] else 'check', film.word(s, 'L10', 'colours') + 0.2)]
    rows.sort(key=lambda row: row[2])
    x, y = 760, 190
    wd.label(c, 'AUTOMATIC CHECKS · MEASURED ON THIS VIDEO', x, 130, 20, GREY)
    for index, (name, value, at) in enumerate(rows):
        wd.check_row(c, x, y + index * 78, name, value, t, at, width=980, size=34)
    minutes = film.main_minutes
    done = film.word(s, 'L10', 'beat') + 0.3
    stat(c, x, 720, f'{minutes:.0f} min', 'ONE BUTTON · NOBODY TOUCHED IT · ALL CHECKS PASSED', t, done, GREEN, 110)
    wd.caption(c, ('IT CHECKS ITS OWN WORK', (('CHECKS', GREEN),)), t, 0.3, film.word(s, 'L10', 'beat') + 0.2, CAPTION_X + 664, CAPTION_Y + 20, 76, 'lb',
               over_picture=False)


# ------------------------------------------------------------------------------------------------ 03 two dancers
def duo_time(scene, t):
    duo = scene.film.duo
    return duo.first_bar + scene.at_bar * duo.bar + t


def mask_centre(duo, view, who, number, region):
    """Where a dancer's mannequin is inside a card showing the guide picture (cover fit)."""
    mask = duo.mask(view, who, number)
    x, y, w, h = region
    if mask is None:
        return None
    ys, xs = np.nonzero(mask < 128)   # the figure is painted in its colour on white
    if len(xs) < 50:
        return None
    scale = max(w / mask.shape[1], h / mask.shape[0])
    ox, oy = (mask.shape[1] * scale - w) / 2, (mask.shape[0] * scale - h) / 2
    return x + xs.mean() * scale - ox, y + np.percentile(ys, 12) * scale - oy


def two_in(c, t, s):
    film, d = s.film, duo_time(s, t)
    duo = film.duo
    number = duo.number(d)
    each = film.word(s, 'L11', 'Each') - 0.25
    room = ease_out(ramp(t, each, each + 0.5))
    slide = (1 - room) * 470      # the two figures stand in the middle until the room comes in
    stage_panel(c, duo.stage, d, (40 + slide, 110, 900, 800), (A_COLOUR, B_COLOUR), yaw_deg=-25 + 22 * t, distance=5.4, glow_strength=0.9)
    wd.chip(c, 'TWO DANCERS · 3D JOINTS', 64 + slide, 110, GREEN, 1.0, 18, over_picture=False)
    region = (980, 84, 478, 850)
    if room > 0:
        arrow(c, 900, 540, 956, ramp(t, each, each + 0.4))
        figure = film.word(s, 'L11', 'figure') - 0.35
        filled = ease(ramp(t, figure, figure + 0.45))
        plate, guide = duo.plate('orbit', number), duo.guide('orbit', number)
        mixed = guide if filled >= 1 or plate is None else cv2.addWeighted(guide, filled, plate, 1 - filled, 0)
        video_card(c, mixed, (region[0] + (1 - room) * 40, region[1], region[2], region[3]), label='ONE 3D ROOM', colour=GREEN, opacity=room)
    for who, colour, x, y in (('A', A_COLOUR, 1530, 110), ('B', B_COLOUR, 1550, 540)):
        shown = ease_out(ramp(t, figure + 0.3, figure + 0.7)) if room > 0 else 0
        if shown <= 0:
            continue
        ref = film.reference(who, 350)
        card(c, ref, x, y, 16, shown, colour, 4)
        wd.chip(c, f'CHARACTER {who}', x + ref.shape[1] / 2, y + ref.shape[0] + 12, colour, shown, 17, anchor='ct', over_picture=False)
        target = mask_centre(duo, 'orbit', who, number, region)
        link = ease(ramp(t, figure + 0.6, figure + 1.1)) if room > 0 else 0
        if target is not None and link > 0:
            start_point = np.array([x, y + 150.0])
            end_point = start_point + (np.array(target) - start_point) * link
            cv2.line(c, tuple(start_point.astype(int)), tuple(end_point.astype(int)), colour, 3, cv2.LINE_AA)
            if link > 0.95:
                cv2.circle(c, tuple(np.array(target).astype(int)), 20, colour, 3, cv2.LINE_AA)
    wd.caption(c, ('TWO DANCERS · ONE 3D ROOM', (('ONE 3D ROOM', GREEN),)), t, film.word(s, 'L11', 'one', 1) - 0.1, s.seconds, CAPTION_X, CAPTION_Y + 34, 84,
               'lb', over_picture=False)


def two_pass(c, t, s):
    film, d = s.film, duo_time(s, t)
    duo = film.duo
    number = duo.number(d)
    x, y, wide, tall = 721, 84, 478, 850
    guide, result = fit(duo.guide('original', number), wide, tall, 'cover'), fit(duo.result('original', number), wide, tall, 'cover')
    painted = film.word(s, 'L12', 'painted')
    sweep = ease(ramp(t, painted - 0.2, painted + 1.0))
    # a divider that paints the result over the guide, then drifts so that both stay comparable
    drift = 0.5 + 0.5 * np.cos((t - painted - 1.0) * 1.1) if t > painted + 1.0 else 1.0
    edge = int(wide * (1 - sweep) + wide * sweep * (0.0 + 0.36 * (1 - drift)))
    mixed = result.copy()
    if edge > 0:
        mixed[:, :edge] = guide[:, :edge]
        cv2.line(mixed, (min(wide - 1, edge), 0), (min(wide - 1, edge), tall), GREEN, 4)
    card(c, mixed, x, y, 18, 1.0, LINE, 2)
    if edge > 150:
        wd.chip(c, '3D GUIDE', x + 14, y + 14, SKY, 1.0, 18)
    if edge < wide - 330:
        wd.chip(c, 'ONE PASS · BOTH + ROOM', x + wide - 14, y + 14, GREEN, 1.0, 18, anchor='rt')
    for who, colour, yy in (('A', A_COLOUR, 150), ('B', B_COLOUR, 520)):
        ref = film.reference(who, 310)
        card(c, ref, 270 + (260 - ref.shape[1]) / 2, yy, 16, 1.0, colour, 3)
    arrow(c, 560, 500, 680, ramp(t, 0.3, 0.9))
    found = duo.piece_checks['original']     # the program's own checks of every piece it kept
    match = duo.summary.get('original_camera', {}).get('picture_match', {})
    away = [match[w]['away_percent_of_picture_height']['median'] for w in ('A', 'B') if w in match and match[w].get('frames_compared', 0) > 30]
    rows = [('In step with the guide', f'{found["out_of_step"]} frames late', film.word(s, 'L12', 'step') - 0.2)]
    if away:
        rows.append(('In place', f'{max(away):.1f} % of height off', film.word(s, 'L12', 'place') - 0.2))
    rows.append(('Room and light kept', f'{found["off_the_room"]} frames off', film.word(s, 'L12', 'light') - 0.4))
    if duo.reviewed_clean('original') and found['clothes_checked']:   # only said when a look at every frame agrees with the program's checks
        rows.append(('Own head, own clothes', f'{found["wrong_head"] + found["wrong_clothes"] + found["mannequin_left"]} frames',
                     film.word(s, 'L12', 'light') + 0.1))
    wd.label(c, 'AUTOMATIC CHECKS · THIS VIDEO', 1262, 250, 20, GREY)
    for index, (name, value, at) in enumerate(rows):
        wd.check_row(c, 1262, 310 + index * 84, name, value, t, at, width=594, size=27, good=value.startswith('0') or 'height' in value)
    done = film.word(s, 'L12', 'light') + 0.5
    stat(c, 1262, 660, f'{found["pieces"]}', f'PIECES · {found["made_again"]} MADE AGAIN BY THE PROGRAM', t, done, GREEN, 96)
    if 'original' in duo.pieces:
        wd.pieces_strip(c, (1262, 846, 594, 70), duo.pieces['original'], d, duo.seconds, ramp(t, done, done + 0.3))
    wd.caption(c, ('PAINTED IN ONE PASS', (('ONE PASS', GREEN),)), t, painted, s.seconds, CAPTION_X, CAPTION_Y + 34, 84, 'lb', over_picture=False)


def two_camera(c, t, s):
    film, d = s.film, duo_time(s, t)
    duo = film.duo
    number = duo.number(d)
    fly = film.word(s, 'L13', 'or')
    second = ease(ramp(t, fly - 0.1, fly + 0.4))
    colours = (A_COLOUR, B_COLOUR)
    video_card(c, duo.result('original', number), (110, 84, 478, 850), label='THE ORIGINAL CAMERA MOVE', colour=SKY, opacity=1.0 - 0.55 * second)
    path = duo.camera_floor['original']
    wd.stage_map(c, (1118, 84, 420, 410), duo.floor, colours, path, number - 1, 1.0 - 0.55 * second, 'CAMERA PATH · FROM THE VIDEO',
                 reach=2.3, centre=tuple((np.mean(path, axis=0) + np.mean([f.mean(0) for f in duo.floor], axis=0)) / 2), camera_colour=SKY)
    if second > 0:      # the flying camera comes in when it is named
        video_card(c, duo.result('orbit', number), (612 + (1 - second) * 40, 84, 478, 850), label='FLYING ROUND · 360°', colour=GREEN, opacity=second)
        wd.stage_map(c, (1118, 524, 420, 410), duo.floor, colours, duo.camera_floor['orbit'], number - 1, second,
                     'CAMERA PATH · ANY WE LIKE', reach=3.3, centre=duo.set_centre, camera_colour=GREEN)
    for who, colour, yy in (('A', A_COLOUR, 100), ('B', B_COLOUR, 520)):
        ref = film.reference(who, 330)
        card(c, ref, 1580 + (276 - ref.shape[1]) / 2, yy, 16, 1.0, colour, 3)
    wd.caption(c, ('COPY THE SHOT', (('COPY', SKY),)), t, film.word(s, 'L13', 'copy') - 0.1, fly - 0.05, CAPTION_X, CAPTION_Y + 34, 84, 'lb', over_picture=False)
    wd.caption(c, ('OR FLY ALL THE WAY ROUND', (('ROUND', GREEN),)), t, fly, s.seconds, CAPTION_X, CAPTION_Y + 34, 84, 'lb', over_picture=False)


# ------------------------------------------------------------------------------------------------ 04 the robot
def robot_time(scene, t):
    robot = scene.film.robot
    return robot.first_bar + scene.at_bar * robot.bar + t


def robot_in(c, t, s):
    film = s.film
    robot = film.robot
    paste(c, fit(robot.picture(s.footage_from + t), W, H, 'cover', (0.5, 0.45)), 0, 0)
    wd.chip(c, 'REAL ROBOT · REAL TIME', 64, 100, GREEN, 1.0, 20)
    s.scrim = True
    wd.caption(c, ('SAME JOINTS → REAL ROBOT', (('REAL ROBOT', GREEN),)), t, film.word(s, 'L14', 'leave') - 0.1, s.seconds, CAPTION_X, CAPTION_Y + 26, 92, 'lb')


def robot_dance(c, t, s):
    film, d = s.film, robot_time(s, t)
    robot = film.robot
    video_card(c, robot.picture(robot.footage_time(d)), (64, 150, 1130, 636), label='REAL ROBOT · REAL TIME', colour=GREEN, focus=(0.72, 0.42), zoom=1.12)
    box(c, 1230, 150, 300, 533, PANEL, 1.0, 16)
    figure_panel(c, robot.dance, d, (1234, 154, 292, 525), yaw_deg=0, mirror=True, zoom=0.92, glow_strength=0.8, hands=False)
    wd.chip(c, 'THE DANCER · MIRRORED', 1244, 164, SKY, 1.0, 15)
    if film.nine is not None:
        number = film.nine.number(d)
        video_card(c, film.nine.clean.get(number - 1), (1556, 150, 300, 533), mirror=True, zoom=1.2, focus=(0.5, 0.3))
        wd.chip(c, 'CHARACTER · MIRRORED', 1570, 164, GREEN, 1.0, 15)
    else:
        box(c, 1556, 150, 300, 533, PANEL, 1.0, 16)
        figure_panel(c, robot.dance, d, (1560, 154, 292, 525), yaw_deg=70, mirror=True, zoom=0.92, glow_strength=0.8, hands=False)
    at = lambda word: ramp(t, film.word(s, 'L15', word) - 0.05, film.word(s, 'L15', word) + 0.2)
    k = robot.dance.index(d)
    wd.label(c, 'THE DANCER\'S JOINTS, LIVE', 1230, 716, 17, GREY, 'lt', at('Head'))
    wd.meter_row(c, 1230, 778, 'HEAD', GREEN, robot.head_roll[k], -40, 40, f'{robot.head_roll[k]:+.0f}°', at('Head'))
    wd.meter_row(c, 1230, 846, 'BODY', SKY, robot.body_turn[k], -90, 90, f'{robot.body_turn[k]:+.0f}°', at('body'))
    wd.meter_row(c, 1230, 914, 'ARMS', AMBER, (robot.arms[0][k] + robot.arms[1][k]) / 2, -90, 90, f'{(robot.arms[0][k] + robot.arms[1][k]) / 2:+.0f}°', at('antennas'))
    wd.caption(c, ('DRIVEN BY THE DANCER\'S JOINTS', (('JOINTS', GREEN),)), t, film.word(s, 'L15', 'driven') - 0.1, s.seconds, CAPTION_X, CAPTION_Y + 26, 84, 'lb',
               over_picture=False)


def robot_graph(c, t, s):
    film, d = s.film, robot_time(s, t)
    robot = film.robot
    region = (64, 110, 900, 506)
    focus, zoom = (0.80, 0.30), 1.5
    footage = robot.footage_time(d)
    picture = robot.picture(footage)
    video_card(c, picture, region, label='MEASURED ON THE FOOTAGE', colour=SKY, focus=focus, zoom=zoom)
    eyes = robot.eyes(footage)
    if eyes is not None and t > 0.2:
        scale, ox, oy = cover((picture.shape[1], picture.shape[0]), region[2:], focus, zoom)
        a = np.array([region[0] + eyes['left'][0] * scale - ox, region[1] + eyes['left'][1] * scale - oy])
        b = np.array([region[0] + eyes['right'][0] * scale - ox, region[1] + eyes['right'][1] * scale - oy])
        inside = lambda q: region[0] + 20 < q[0] < region[0] + region[2] - 20 and region[1] + 20 < q[1] < region[1] + region[3] - 20
        if inside(a) and inside(b):
            along = (b - a) / max(1.0, np.linalg.norm(b - a))
            cv2.line(c, tuple((a - along * 80).astype(int)), tuple((b + along * 80).astype(int)), SKY, 3, cv2.LINE_AA)
            for point in (a, b):
                cv2.circle(c, tuple(point.astype(int)), int(0.27 * np.linalg.norm(b - a)), SKY, 3, cv2.LINE_AA)
    # the dancer's head, close, with the line through the ears
    box(c, 990, 110, 420, 506, PANEL, 1.0, 16)
    camera = figure_panel(c, robot.dance, d, (994, 114, 412, 498), yaw_deg=0, mirror=True, zoom=2.1, glow_strength=0.8, hands=False, floor=False,
                          trails=False, focus_height=1.42)
    pose = robot.dance.pose(d)
    ears = camera.project(pose[[59, 58]])[0]
    along = (ears[1] - ears[0]) / max(1.0, np.linalg.norm(ears[1] - ears[0]))
    cv2.line(c, tuple((ears[0] - along * 60).astype(int)), tuple((ears[1] + along * 60).astype(int)), GREEN, 3, cv2.LINE_AA)
    wd.chip(c, 'THE DANCER · MIRRORED', 1004, 124, GREEN, 1.0, 15)
    said = film.word(s, 'L16', 'correlation')
    series_t, series_v = robot.tilt_series()
    dancer_at = np.interp(series_t, robot.dance_times, robot.head_roll)
    wd.scatter(c, (1440, 110, 416, 300), dancer_at, series_v, series_t, d, (-40, 40), (-20, 20), SKY, 1.0, 'EVERY MEASURED FRAME',
               'DANCER TILT  ±40°', 'ROBOT  ±20°', fit=robot.fit)
    stat(c, 1446, 424, f'{robot.correlation_shown:.2f}', 'CORRELATION · HEAD TILT', t, said + 0.5, GREEN, 132)
    gain = 40 / 18
    wd.curve_plot(c, (64, 650, W - 128, 300), [dict(t=robot.dance_times, y=robot.head_roll, colour=GREEN, label='DANCER · HEAD TILT FROM THE 3D JOINTS'),
                                                dict(t=series_t, y=series_v, colour=SKY, label='ROBOT · HEAD TILT MEASURED ON THE FOOTAGE', breaks=0.25,
                                                     gain=gain)],
                  d, (d - 5.5, d + 2.0), (-40, 40), None, ease_out(ramp(t, 0.1, 0.6)), beats=robot.beats, values='{:+.0f}°', zero=0.0)
    wd.label(c, 'EACH ON ITS OWN SCALE · DANCER ±40°  ROBOT ±18°', W - 84, 666, 15, GREY, 'rt')
    wd.caption(c, ('THE ROBOT FOLLOWS THE DANCER', (('FOLLOWS', GREEN),)), t, film.word(s, 'L16', 'follows') - 0.1, said + 0.2, CAPTION_X, CAPTION_Y + 34, 76, 'lb',
               over_picture=False)


def robot_free(c, t, s):
    film, d = s.film, robot_time(s, t)
    robot = film.robot
    picture = robot.picture(robot.footage_time(d))
    crop = picture[:, 560:1280]            # the robot with room for its antennas
    if film.nine is not None:
        number = film.nine.number(d)
        video_card(c, film.nine.clean.get(number - 1), (341, 84, 468, 832), label='CHARACTER · MIRRORED', colour=GREEN, mirror=True, zoom=1.25, focus=(0.5, 0.3))
        video_card(c, crop, (839, 84, 740, 832), label='REAL ROBOT · REAL TIME', colour=GREEN, focus=(0.5, 0.4))
    else:
        box(c, 341, 84, 468, 832, PANEL, 1.0, 18)
        figure_panel(c, robot.dance, d, (345, 88, 460, 824), yaw_deg=0, mirror=True, glow_strength=0.9)
        wd.chip(c, 'THE DANCER · MIRRORED', 355, 98, SKY, 1.0, 18)
        video_card(c, crop, (839, 84, 740, 832), label='REAL ROBOT · REAL TIME', colour=GREEN, focus=(0.5, 0.4))
    # (the scene may be one of a row of cuts that share this caption: `since` = seconds since the first of them)
    wd.caption(c, ('SAME DANCE. REAL ROBOT.', (('REAL ROBOT.', GREEN),)), t + getattr(s, 'since', 0.0), 0.3, getattr(s, 'total', s.seconds), W / 2,
               CAPTION_Y + 34, 84, 'cb', over_picture=False)


# ------------------------------------------------------------------------------------------------ the end
def recap(c, t, s):
    film, d = s.film, robot_time(s, t)
    robot, main = film.robot, film.main
    wide, tall, y = 418, 743, 110
    xs = [64 + k * 448 for k in range(4)]
    box(c, xs[0], y, wide, tall, PANEL, 1.0, 18)
    figure_panel(c, robot.dance, d, (xs[0] + 4, y + 4, wide - 8, tall - 8), yaw_deg=-20 + 10 * t, glow_strength=0.9)
    nine = film.nine or main
    video_card(c, nine.clean.get(nine.number(d % nine.seconds) - 1), (xs[1], y, wide, tall))
    if film.duo is not None:
        video_card(c, film.duo.result('original', film.duo.number(4.0 + t)), (xs[2], y, wide, tall))
    else:
        box(c, xs[2], y, wide, tall, PANEL, 1.0, 18)
    video_card(c, robot.picture(robot.footage_time(d))[:, 600:1240], (xs[3], y, wide, tall), focus=(0.5, 0.4))
    names = ('3D JOINTS', 'CHARACTER · ANY CAMERA', 'TWO DANCERS', 'REAL ROBOT')
    for index, name in enumerate(names):
        wd.chip(c, name, xs[index] + wide / 2, y + tall + 18, (GREEN, SKY, VIOLET, AMBER)[index], 1.0, 18, anchor='ct', over_picture=False)
    wd.caption(c, ('ONE VIDEO IN.', (('ONE VIDEO', GREEN),)), t, 0.25, film.word(s, 'L17', 'Characters') - 0.05, W / 2, 1050, 84, 'cb', over_picture=False)
    wd.caption(c, ('CHARACTERS, CAMERAS, ROBOTS OUT.', (('OUT.', GREEN),)), t, film.word(s, 'L17', 'Characters'), s.seconds, W / 2, 1050, 84, 'cb',
               over_picture=False)


def outro(c, t, s):
    sting(c, s.film.brand_in, t, first=s.first, end=s.seconds)


def placeholder(c, t, s):
    wd.label(c, s.name, W / 2, H / 2, 40, GREY, 'cm')


def build(film):
    film.main = sources.Challenge(MAIN['job'], MAIN['name'])
    film.main_dance = dance(MAIN['number'])
    film.main_outline = sources.outlines(MAIN['number'])
    film.main_minutes = MAIN['minutes']
    film.robot = sources.Robot()
    film.robot.correlation_shown = abs(film.robot.correlation)
    film.duo = sources.Duo(DUO['job'], DUO['desktop']) if DUO else None
    film.nine = sources.Challenge(NINE['job'], NINE['name']) if NINE else None
    film.brand_in = sources.brand_clip('brand_m2s', '3-애니메이션/2-전환-MATRIX에서-STEERIX로/matrix-to-steerix-green-x.mp4')
    film.quad = None
    film.accuracy = [round(json.loads((p / '정보.json').read_text())['report']['picture_error_percent_of_body']['median'], 2)
                     for _, p in sorted(sources.DANCES.items())]
    film.joints_run = dict(dances=len(sources.DANCES), minutes=round(json.loads((sources.JOINTS_JOB / 'outputs/runs.json').read_text())[0]))
    # the narrator says these numbers aloud (narration.json): stop if the records say something else
    spoken = dict(joints=(film.main_dance.joints.shape[1], 76), dances=(film.joints_run['dances'], 15), minutes=(film.joints_run['minutes'], 49),
                  typical_gap=(round(float(np.median(film.accuracy)) * 2) / 2, 1.5), correlation=(round(film.robot.correlation_shown, 2), 0.85))
    wrong = {name: pair for name, pair in spoken.items() if pair[0] != pair[1]}
    assert not wrong, f'the narration says other numbers than the records (record, spoken): {wrong}'
    refs = {'A': '/home/steerix-train-server/바탕화면/reze.webp', 'B': '/home/steerix-train-server/바탕화면/makima.webp'}
    cache = {}

    def reference(which, height):
        if (which, height) not in cache:
            cache[which, height] = sources.picture(refs[which], height=height)
        return cache[which, height]

    film.reference = reference
    main = film.main
    P = main.bar
    say = film.spoken

    # ---- opening: four cuts, one bar each
    g = Group(film, 'main', P)
    others = lambda film, t: [lambda: film.main.made(film.main.number(film.main.first_bar + 5 * P + t)),
                              lambda: film.main.made(film.main.number(film.main.first_bar + 8 * P + t))]
    g.add('open_video', cold_open, P, at_bar=3, lines=[('L01a', 0.08)], kind='video', chrome=False)
    g.add('open_character', cold_open, P, at_bar=0, lines=[('L01b', 0.08)], kind='character', others=others, chrome=False, duo_from=8.0)
    g.add('open_camera', cold_open, P, at_bar=0, lines=[('L01c', 0.08)], kind='camera', views=(('left', 0), ('back', 0), ('right', 0)), chrome=False)
    g.add('open_robot', cold_open, P, at_bar=0, lines=[('L01d', 0.08)], kind='robot', robot_from=4.2, chrome=False)
    film.add('logo', logo_sting, 2.5, first=22, flash=False, chrome=False, plain=True, sounds=[('whoosh', 0.0), ('hit', 0.75)], music=None)
    g = Group(film, 'main2', P, chapter=None)
    g.add('title', title, 2 * P, at_bar=2)
    g.chapter = ('01', 'CAPTURE')
    g.add('capture_in', capture_in, say('L02') + 1.1, at_bar=4, lines=[('L02', 0.85)])
    g.add('capture_3d', capture_3d, say('L03') + 0.8, at_bar=2, lines=[('L03', 0.3)])
    g.add('capture_curves', capture_curves, say('L04') + 1.6, at_bar=3, lines=[('L04', 0.3)])
    g.add('capture_wall', capture_wall, say('L05') + 1.2, at_bar=0, lines=[('L05', 0.3)])
    g.add('capture_accuracy', capture_accuracy, say('L06') + 0.9, at_bar=2, lines=[('L06', 0.3)])
    g.chapter = ('02', 'ANY CAMERA')
    g.add('camera_anywhere', camera_anywhere, say('L07') + 1.1, at_bar=4, lines=[('L07', 0.85)])
    beat = P / 4
    g.add('camera_four', camera_four, 3 * P, at_bar=2, lines=[('L07c', 0.02), ('L07d', 0.02 + 2 * beat), ('L07e', 0.02 + 4 * beat), ('L07f', 0.02 + 6 * beat)])
    g.add('camera_paint', camera_paint, say('L08') + 1.3, at_bar=1, lines=[('L08', 0.3)])
    g.add('camera_beat', camera_beat, say('L09') + 3.6, at_bar=2, lines=[('L09', 0.3)])
    g.add('camera_checks', camera_checks, say('L10') + 1.6, at_bar=3, lines=[('L10', 0.3)])
    # ---- 03 two dancers
    duo = film.duo
    if duo is not None:
        g = Group(film, 'duo', duo.bar, chapter=('03', 'TWO DANCERS'))
        g.add('two_in', two_in, say('L11') + 1.1, at_bar=DUO_BARS[0], lines=[('L11', 0.85)])
        g.add('two_pass', two_pass, say('L12') + 1.4, at_bar=DUO_BARS[1], lines=[('L12', 0.3)])
        g.add('two_camera', two_camera, say('L13') + 1.2, at_bar=DUO_BARS[2], lines=[('L13', 0.3)])
        scene = film.scenes[-1]        # the flying camera is shown from the word "or" to the scene's end: that stretch must be a clean one
        shown = (duo.first_bar + DUO_BARS[2] * duo.bar + film.word(scene, 'L13', 'or') - 0.1, duo.first_bar + DUO_BARS[2] * duo.bar + scene.seconds)
        assert duo.is_clean('orbit', *shown) and duo.is_clean('original', shown[1] - scene.seconds, shown[1]), f'two_camera shows a faulty stretch {shown}'
    # ---- 04 the robot
    robot = film.robot
    film.add('robot_in', robot_in, robot.bar * 2, lines=[('L14', 1.0)], chapter=('04', 'REAL ROBOT'), footage_from=5.7,
             music=dict(group='robot', bar=robot.bar, bars_before=0.0))
    g = Group(film, 'robot', robot.bar, chapter=('04', 'REAL ROBOT'))
    g.elapsed = robot.bar * 2
    g.add('robot_dance', robot_dance, say('L15') + 1.2, at_bar=front_start(film.nine, ROBOT_BARS[0], 2, 6), lines=[('L15', 0.3)])
    g.add('robot_graph', robot_graph, say('L16') + 1.0, at_bar=ROBOT_BARS[1], lines=[('L16', 0.3)])
    # robot and character side by side. The character video cuts between its cameras bar by bar, so three bars are shown in
    # which it is filmed from the front (the dance skips the bars between them; robot and character skip together, on the beat)
    fronts = front_bars(film.nine, 3, 2) if film.nine is not None else [ROBOT_BARS[2] + k for k in range(3)]
    for k, bar in enumerate(fronts):
        g.add(f'robot_free_{k + 1}', robot_free, robot.bar, at_bar=bar, since=k * robot.bar, total=len(fronts) * robot.bar)
    g.add('recap', recap, say('L17') + 1.4, at_bar=front_start(film.nine, ROBOT_BARS[3], 10, 13), lines=[('L17', 0.4)], chapter=None)
    film.add('outro', outro, 4.6, first=22, flash=False, chrome=False, plain=True, sounds=[('whoosh', 0.0), ('hit', 0.75)], music=None)
