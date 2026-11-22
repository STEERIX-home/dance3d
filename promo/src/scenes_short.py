"""The scenes of the short film (1080x1920). The parts a phone app covers (top 230, bottom 430, right edge) carry no text."""
import json

import cv2
import numpy as np

import sources
import widgets as wd
from common import A_COLOUR, B_COLOUR, cover, covered, phone, stat, sting, video_card
from kit import (AMBER, BG, DIM, FPS, GREEN, GREY, LINE, PANEL, SKY, VIOLET, WHITE, WORK, back_out, box, card, ease, ease_out, fit, outline,
                 paste, put, ramp, shadowed, text_layer)
from scenes_long import DUO, DUO_BARS, MAIN, NINE, Group, duo_time, front_bars, front_start, robot_time
from skeleton import figure_panel
from sources import dance

W, H = 1080, 1920
TOP, BOTTOM = 250, 1470   # text stays between these lines


def big(c, parts, t, start, end, y, size=96, over_picture=True):
    wd.caption(c, parts, t, start, end, W / 2, y, size, 'cm', over_picture=over_picture, rise=30)


def hook(c, t, s):
    film, d = s.film, robot_time(s, t)
    robot = film.robot
    paste(c, covered(robot.picture(robot.footage_time(d)), (W, 960), (0.84, 0.35), 1.0), 0, 0)
    if film.nine is not None:
        number = film.nine.number(d)
        paste(c, np.ascontiguousarray(covered(film.nine.clean.get(number - 1), (W, 960), (0.5, 0.14), 1.0)[:, ::-1]), 0, 960)
    else:
        box(c, 0, 960, W, 960, PANEL)
        figure_panel(c, robot.dance, d, (0, 960, W, 900), yaw_deg=0, mirror=True, glow_strength=1.0, zoom=1.0)
    box(c, 0, 956, W, 8, GREEN)
    s.scrim = True
    tt = t + getattr(s, 'since', 0.0)     # the hook is two cuts that share the logo's arrival and the caption
    if tt < s.logo_from:   # the intro: the brand's pieces gather into the STEERIX logo where it then stays
        shade = 1 - 0.5 * (1 - ramp(tt, s.logo_from - 0.3, s.logo_from)) * np.clip(1 - np.arange(520, dtype=np.float32) / 520, 0, 1) ** 0.7
        c[:520] = (c[:520] * shade[:, None, None]).astype(np.uint8)
        sting(c, film.brand_in, tt, first=22, scale=0.297, centre=(W / 2, 201), fade=0.08)
    wd.chip(c, 'REAL ROBOT · REAL TIME', 40, 250, GREEN, ramp(tt, 0.9, 1.2), 24)
    big(c, ('COPIED FROM\nONE VIDEO', (('ONE VIDEO', GREEN),)), tt, 0.25, getattr(s, 'total', s.seconds), 960, 112)


def joints(c, t, s):
    film, d = s.film, robot_time(s, t)
    person = film.robot.dance
    phone(c, person, d, (50, 500, 300, 534), ease_out(ramp(t, 0.0, 0.3)), outline_of=film.robot_outline)
    wd.label(c, 'ONE VIDEO', 200, 1060, 20, GREY, 'ct', ramp(t, 0.2, 0.5))
    figure_panel(c, person, d, (330, 420, 750, 900), yaw_deg=-30 + 55 * t, glow_strength=1.0, distance=4.3, strength=ease(ramp(t, 0.5, 1.1)))
    span = (d - 2.6, d + 1.2)
    tl = np.arange(person.frames) / person.fps
    wd.curve_plot(c, (40, 1230, 1000, 300), [dict(t=tl, y=person.joints[:, 20, 1], colour=SKY, label='LEFT WRIST'),
                                              dict(t=tl, y=person.joints[:, 21, 1], colour=AMBER, label='RIGHT WRIST'),
                                              dict(t=tl, y=person.joints[:, 0, 1], colour=GREEN, label='PELVIS')],
                  d, span, (0.2, 2.1), None, ease_out(ramp(t, 0.2, 0.6)), beats=film.robot.beats, values='{:.2f} m')
    big(c, ('VIDEO → 3D JOINTS', (('3D JOINTS', GREEN),)), t, 0.2, s.seconds, 330, 96, over_picture=False)
    wd.label(c, f'{person.joints.shape[1]} JOINTS · IN METRES · FEET ON THE FLOOR', W / 2, 410, 24, GREY, 'cm', ramp(t, 0.6, 0.9))


def cameras(c, t, s):
    film = s.film
    source = film.nine or film.main
    d = source.first_bar + s.at_bar * source.bar + t
    number = source.number(d)
    shot = source.shots[source.owner[number]]
    paste(c, source.clean.get(number - 1), 0, 0)
    s.scrim = True
    age = d - shot['start_s']
    wd.chip(c, wd.VIEW_NAME[shot['direction']] + ' CAMERA', 40 - 30 * (1 - ease_out(ramp(age, 0, 0.18))), 250, wd.VIEW_COLOUR[shot['direction']],
            min(1.0, ramp(age, 0, 0.08) * 2), 46, filled=True)
    span = (source.first_bar + s.at_bar * source.bar - 0.6, source.first_bar + s.at_bar * source.bar + s.seconds + 0.6)
    wd.beat_strip(c, (40, 1250, 1000, 200), source.envelope, source.beats, source.bars, source.shots, d, span, title='BEATS · CUTS')
    big(c, ('CUT ON THE BEAT', (('BEAT', GREEN),)), t, film.word(s, 'S03', 'Cut') - 0.1, s.seconds, 1160, 92)


def two(c, t, s):
    film, d = s.film, duo_time(s, t)
    duo = film.duo
    number = duo.number(d)
    if t < s.fly_at:
        guide, result = duo.guide('original', number), duo.result('original', number)
        sweep = ease(ramp(t, 0.35, 1.3))            # the 3D guide first, then the painted picture comes down over it
        edge = int(H * sweep)
        mixed = cv2.resize(guide, (W, H)).copy()
        if edge > 0:
            mixed[:edge] = result[:edge]
        if 0 < edge < H:
            cv2.line(mixed, (0, edge), (W, edge), GREEN, 6)
        paste(c, mixed, 0, 0)
        wd.chip(c, '3D GUIDE → PAINTED IN ONE PASS', 40, 250, GREEN, 1.0, 24)
    else:
        paste(c, duo.result('orbit', number), 0, 0)
        wd.chip(c, 'THE CAMERA FLIES ROUND', 40, 250, GREEN, 1.0, 24)
        wd.camera_map(c, (720, 330, 320, 320), None, orbit_deg=duo.orbit_angle(number), title='')
    s.scrim = True
    big(c, ('TWO CHARACTERS\nONE 3D ROOM', (('ONE 3D ROOM', GREEN),)), t, film.word(s, 'S04', 'two', 1) - 0.1, s.seconds, 1240, 92)


def robot(c, t, s):
    film, d = s.film, robot_time(s, t)
    rb = film.robot
    region = (40, 280, 1000, 750)
    focus, zoom = (0.84, 0.28), 1.5
    footage = rb.footage_time(d)
    picture = rb.picture(footage)
    video_card(c, picture, region, label='REAL ROBOT · REAL TIME', colour=GREEN, focus=focus, zoom=zoom, label_size=22)
    eyes = rb.eyes(footage)
    if eyes is not None and t > 1.5:
        scale, ox, oy = cover((picture.shape[1], picture.shape[0]), region[2:], focus, zoom)
        a = np.array([region[0] + eyes['left'][0] * scale - ox, region[1] + eyes['left'][1] * scale - oy])
        b = np.array([region[0] + eyes['right'][0] * scale - ox, region[1] + eyes['right'][1] * scale - oy])
        inside = lambda q: region[0] + 20 < q[0] < region[0] + region[2] - 20 and region[1] + 20 < q[1] < region[1] + region[3] - 20
        if inside(a) and inside(b):
            along = (b - a) / max(1.0, np.linalg.norm(b - a))
            cv2.line(c, tuple((a - along * 70).astype(int)), tuple((b + along * 70).astype(int)), SKY, 3, cv2.LINE_AA)
            for point in (a, b):
                cv2.circle(c, tuple(point.astype(int)), int(0.27 * np.linalg.norm(b - a)), SKY, 3, cv2.LINE_AA)
    box(c, 40, 1050, 480, 660, PANEL, 1.0, 18)
    if film.nine is not None:
        video_card(c, film.nine.clean.get(film.nine.number(d) - 1), (40, 1050, 480, 660), mirror=True, focus=(0.5, 0.25), zoom=1.1, label='CHARACTER · MIRRORED',
                   label_size=18)
    else:
        figure_panel(c, rb.dance, d, (44, 1054, 472, 652), yaw_deg=0, mirror=True, zoom=1.0, glow_strength=0.9, hands=False)
        wd.chip(c, 'THE DANCER · MIRRORED', 54, 1064, SKY, 1.0, 18)
    series_t, series_v = rb.tilt_series()
    wd.curve_plot(c, (540, 1050, 500, 300), [dict(t=rb.dance_times, y=rb.head_roll, colour=GREEN, label='DANCER'),
                                             dict(t=series_t, y=series_v, colour=SKY, label='ROBOT', breaks=0.25, gain=40 / 18)],
                  d, (d - 3.2, d + 0.8), (-40, 40), None, ease_out(ramp(t, 1.4, 1.9)), zero=0.0, values='{:+.0f}°')
    shown = ramp(t, 2.6, 2.9)
    put(c, text_layer(f'{rb.correlation_shown:.2f}', 'semibold', 126, GREEN), 548, 1356, shown)
    wd.label(c, 'HEAD TILT CORRELATION', 554, 1512, 20, GREY, 'lt', shown)
    wd.label(c, 'MEASURED ON THIS FOOTAGE', 554, 1544, 20, GREY, 'lt', shown)
    big(c, ('SAME JOINTS → REAL ROBOT', (('REAL ROBOT', GREEN),)), t, 0.15, 2.6, 250, 70, over_picture=False)


def recap(c, t, s):
    film, d = s.film, robot_time(s, t)
    rb = film.robot
    wide, tall = 500, 520
    cells = [(30, 270), (550, 270), (30, 950), (550, 950)]
    box(c, cells[0][0], cells[0][1], wide, tall, PANEL, 1.0, 18)
    figure_panel(c, rb.dance, d, (cells[0][0] + 4, cells[0][1] + 4, wide - 8, tall - 8), yaw_deg=-20 + 12 * t, glow_strength=0.9, hands=False)
    source = film.nine or film.main
    video_card(c, source.clean.get(source.number(d % source.seconds) - 1), (*cells[1], wide, tall), focus=(0.5, 0.12), zoom=1.0)
    if film.duo is not None:
        video_card(c, film.duo.result('original', film.duo.number(4.0 + t)), (*cells[2], wide, tall), focus=(0.5, 0.3))
    else:
        box(c, cells[2][0], cells[2][1], wide, tall, PANEL, 1.0, 18)
    video_card(c, rb.picture(rb.footage_time(d))[:, 600:1240], (*cells[3], wide, tall), focus=(0.5, 0.35))
    for (x, y), name, colour in zip(cells, ('3D JOINTS', 'ANY CAMERA', 'TWO DANCERS', 'REAL ROBOT'), (GREEN, SKY, VIOLET, AMBER)):
        wd.chip(c, name, x + 14, y + 14, colour, 1.0, 20)
    big(c, ('ONE VIDEO IN.', (('ONE VIDEO', GREEN),)), t, 0.15, film.word(s, 'S06', 'Characters') - 0.05, 870, 96, over_picture=False)
    big(c, ('CHARACTERS + ROBOTS OUT.', (('OUT.', GREEN),)), t, film.word(s, 'S06', 'Characters'), s.seconds, 870, 68, over_picture=False)


def outro(c, t, s):
    sting(c, s.film.brand_in, t, first=s.first, scale=1.0, end=s.seconds)


def build(film):
    film.main = sources.Challenge(MAIN['job'], MAIN['name'])
    film.main_dance = dance(MAIN['number'])
    film.robot = sources.Robot()
    film.robot.correlation_shown = abs(film.robot.correlation)
    film.robot_outline = sources.outlines(9)
    film.duo = sources.Duo(DUO['job'], DUO['desktop']) if DUO else None
    film.nine = sources.Challenge(NINE['job'], NINE['name']) if NINE else None
    film.brand_in = sources.brand_clip('brand_m2s', '3-애니메이션/2-전환-MATRIX에서-STEERIX로/matrix-to-steerix-green-x.mp4')
    rb = film.robot
    say = film.spoken
    g = Group(film, 'robot', rb.bar)
    # the hook: robot above, character below, for two bars in which the character video is filmed from the front
    # (the dance skips the bar between them; robot and character skip together, on the beat)
    fronts = front_bars(film.nine, 2, SHORT_BARS['hook']) if film.nine is not None else [SHORT_BARS['hook'], SHORT_BARS['hook'] + 1]
    assert say('S01') + 0.6 <= 2 * rb.bar, 'the first line is longer than the hook'
    g.add('hook_1', hook, rb.bar, at_bar=fronts[0], lines=[('S01', 0.35)], flash=False, logo_from=(62 - 22) / FPS, since=0.0, total=2 * rb.bar,
          sounds=[('swish', 0.35)])
    g.add('hook_2', hook, rb.bar, at_bar=fronts[1], since=rb.bar, total=2 * rb.bar, logo_from=0.0)
    g.add('joints', joints, say('S02') + 0.8, at_bar=SHORT_BARS['joints'], lines=[('S02', 0.3)])
    if film.nine is not None:
        g.add('cameras', cameras, say('S03') + 2.6, at_bar=front_start(film.nine, SHORT_BARS['cameras'], 2, 10), lines=[('S03', 0.3)])
    else:
        g2 = Group(film, 'main', film.main.bar)
        g2.add('cameras', cameras, say('S03') + 2.6, at_bar=1, lines=[('S03', 0.3)])
    if film.duo is not None:
        g3 = Group(film, 'duo', film.duo.bar)
        g3.add('two', two, say('S04') + 1.6, at_bar=1, lines=[('S04', 0.3)], fly_at=2.2)
        scene = film.scenes[-1]        # the two cameras' stretches shown here must be clean ones
        begin = film.duo.first_bar + 1 * film.duo.bar
        assert film.duo.is_clean('original', begin, begin + scene.fly_at) and film.duo.is_clean('orbit', begin + scene.fly_at, begin + scene.seconds), 'the short shows a faulty stretch'
    g = Group(film, 'robot2', rb.bar)
    g.add('robot', robot, say('S05') + 4.8, at_bar=SHORT_BARS['robot'], lines=[('S05', 0.3)])
    g.add('recap', recap, say('S06') + 0.9, at_bar=front_start(film.nine, SHORT_BARS['recap'], 10, 14), lines=[('S06', 0.3)])
    film.add('outro', outro, 3.4, first=22, flash=False, chrome=False, plain=True, sounds=[('whoosh', 0.0), ('hit', 0.75)], music=None)


SHORT_BARS = dict(hook=2, joints=4, cameras=4, robot=1, recap=12)
