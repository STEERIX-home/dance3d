"""Graphs and labels of the dance promo films: curves with a playhead, the beat strip, the camera map, bars, chips, captions."""
import cv2
import numpy as np

from kit import (AMBER, BG, DIM, GREEN, GREY, LINE, PANEL, SKY, VIOLET, WHITE, back_out, box, ease, ease_out, glow, outline, put, ramp,
                 shadowed, text_layer, window)

VIEW_COLOUR = {'front': GREEN, 'left': SKY, 'right': AMBER, 'back': VIOLET}
VIEW_NAME = {'front': 'FRONT', 'left': 'LEFT', 'right': 'RIGHT', 'back': 'BACK'}


def scaled(colour, factor):
    return tuple(int(min(255, c * factor)) for c in colour)


def panel(canvas, region, opacity=1.0, fill=PANEL, edge=LINE, radius=16):
    x, y, w, h = [int(v) for v in region]
    box(canvas, x, y, w, h, fill, opacity, radius)
    outline(canvas, x, y, w, h, radius, edge, 1, opacity)


def chip(canvas, text, x, y, colour=GREEN, opacity=1.0, size=20, anchor='lt', filled=False, over_picture=True):
    """A small label in capitals: a coloured dot and mono text on a dark pill (or dark text on a coloured pill). Returns its width."""
    layer = text_layer(text, 'mono', size, BG if filled else WHITE, spacing=size * 0.08)
    th = layer[1].shape[0]
    w, h = layer[1].shape[1] + int(size * (1.5 if filled else 2.1)), int(size * 1.9)
    x = x - {'l': 0, 'c': w / 2, 'r': w}[anchor[0]]
    y = y - {'t': 0, 'm': h / 2, 'b': h}[anchor[1]]
    if opacity <= 0:
        return w
    box(canvas, x, y, w, h, colour if filled else BG, opacity * (1.0 if filled or not over_picture else 0.78), h // 2)
    if not filled:
        dot = np.empty((h, h, 3), np.uint8)
        dot[:] = colour
        mask = np.zeros((h, h), np.uint8)
        cv2.circle(mask, (h // 2, h // 2), max(3, int(size * 0.26)), 255, -1, cv2.LINE_AA)
        put(canvas, (dot, mask.astype(np.float32) / 255), x + size * 0.85 - h / 2, y, opacity)
    put(canvas, layer, x + size * (0.75 if filled else 1.45), y + (h - th) / 2, opacity)
    return w


def caption(canvas, parts, t, start, end, x, y, size=92, anchor='lb', over_picture=True, kind='semibold', rise=26, colour=WHITE, tracking=0.0):
    """A big caption that slides up into place and fades away. parts: 'text' or (text, ((word, colour), ...))."""
    if not (start - 0.01 <= t <= end + 0.01):
        return
    text, accents = (parts, None) if isinstance(parts, str) else parts
    lines = text.split('\n')
    appear, vanish = ease_out(ramp(t, start, start + 0.28)), 1 - ramp(t, end - 0.2, end)
    opacity = min(appear, vanish)
    step = int(size * 1.12)
    total = step * len(lines)
    top = y - {'t': 0, 'm': total / 2, 'b': total}[anchor[1]]
    for index, line in enumerate(lines):
        delay = ease_out(ramp(t, start + 0.07 * index, start + 0.07 * index + 0.28))
        layer = text_layer(line, kind, size, colour, spacing=tracking, accents=accents)
        lay = shadowed if over_picture else put
        lay(canvas, layer, x, top + index * step + (1 - delay) * rise, min(delay, vanish), anchor[0] + 't')


def label(canvas, text, x, y, size=22, colour=GREY, anchor='lt', opacity=1.0, kind='mono', spacing=None):
    put(canvas, text_layer(text, kind, size, colour, spacing=size * 0.08 if spacing is None else spacing), x, y, opacity, anchor)


def number(canvas, text, x, y, size=64, colour=WHITE, anchor='lt', opacity=1.0):
    put(canvas, text_layer(text, 'semibold', size, colour), x, y, opacity, anchor)


def curve_plot(canvas, region, series, now, span=None, y_range=None, title=None, opacity=1.0, beats=None, legend=True, zero=None,
               values=None, reveal_from=None, background=True, pad=(18, 52, 18, 18)):
    """Curves over time with a playhead at `now`: dim over the whole span, bright up to now.
    series: [dict(t, y, colour, label, width)], span: (t0, t1) shown, values: format of the live number beside each label."""
    x, y, w, h = [int(v) for v in region]
    if opacity <= 0:
        return
    if background:
        panel(canvas, region, opacity * 0.92)
    left, top, right, bottom = pad
    px, py, pw, ph = x + left, y + top, w - left - right, h - top - bottom
    if span is None:
        span = (min(s['t'][0] for s in series), max(s['t'][-1] for s in series))
    if y_range is None:
        low = min(float(np.min(s['y'][(s['t'] >= span[0]) & (s['t'] <= span[1])])) for s in series)
        high = max(float(np.max(s['y'][(s['t'] >= span[0]) & (s['t'] <= span[1])])) for s in series)
        margin = 0.08 * max(1e-6, high - low)
        y_range = (low - margin, high + margin)
    to_x = lambda t: px + (np.asarray(t) - span[0]) / (span[1] - span[0]) * pw
    to_y = lambda v: py + ph - (np.asarray(v) - y_range[0]) / (y_range[1] - y_range[0]) * ph
    for k in range(1, 4):
        gy = int(py + ph * k / 4)
        cv2.line(canvas, (px, gy), (px + pw, gy), scaled(LINE, opacity), 1)
    if zero is not None and y_range[0] < zero < y_range[1]:
        cv2.line(canvas, (px, int(to_y(zero))), (px + pw, int(to_y(zero))), scaled(DIM, opacity), 1)
    if beats is not None:
        for b in beats:
            if span[0] <= b <= span[1]:
                bx = int(to_x(b))
                cv2.line(canvas, (bx, py + ph - 8), (bx, py + ph), scaled(DIM, opacity), 1)

    def draw(layer, offset):
        ox, oy = offset
        for s in series:
            inside = (s['t'] >= span[0] - 0.1) & (s['t'] <= span[1] + 0.1)
            t, v = s['t'][inside], s['y'][inside] * s.get('gain', 1.0)
            if len(t) < 2:
                continue
            pts = np.stack([to_x(t) - ox, np.clip(to_y(v), py - 4, py + ph + 4) - oy], axis=1)
            start = span[0] if reveal_from is None else reveal_from
            done = (t <= now) & (t >= start)
            # a series measured with gaps is drawn in pieces (no line across a gap longer than `breaks` seconds)
            cuts = np.flatnonzero(np.diff(t) > s['breaks']) + 1 if s.get('breaks') else []
            for piece in np.split(np.arange(len(t)), cuts):
                if len(piece) >= 2:
                    cv2.polylines(layer, [np.round(pts[piece] * 16).astype(np.int32)], False, scaled(s['colour'], 0.28 * opacity), 2, cv2.LINE_AA, 4)
                lit = piece[done[piece]]
                if len(lit) >= 2:
                    cv2.polylines(layer, [np.round(pts[lit] * 16).astype(np.int32)], False, scaled(s['colour'], opacity), s.get('width', 3), cv2.LINE_AA, 4)
            if span[0] <= now <= span[1] and (not s.get('breaks') or np.min(np.abs(s['t'] - now)) < s['breaks']):
                value = float(np.interp(now, s['t'], s['y'])) * s.get('gain', 1.0)
                cv2.circle(layer, (int(to_x(now) - ox), int(np.clip(to_y(value), py, py + ph) - oy)), s.get('width', 3) + 4, scaled(WHITE, opacity), -1, cv2.LINE_AA)

    glow(canvas, draw, 0.8 * opacity, sigma=6, region=(px - 12, py - 12, pw + 24, ph + 24))
    if span[0] <= now <= span[1]:
        nx = int(to_x(now))
        box(canvas, nx, py, 1, ph, WHITE, 0.35 * opacity)
    cursor = x + left
    if title:
        layer = text_layer(title, 'mono', 19, GREY, spacing=1.6)
        put(canvas, layer, cursor, y + 16, opacity)
        cursor += layer[1].shape[1] + 26
    if legend:
        for s in series:
            if not s.get('label'):
                continue
            cv2.circle(canvas, (int(cursor + 7), int(y + 28)), 6, scaled(s['colour'], opacity), -1, cv2.LINE_AA)
            text = s['label']
            if values and span[0] <= now:
                text += '  ' + values.format(float(np.interp(min(now, s['t'][-1]), s['t'], s['y'])))
            layer = text_layer(text, 'mono', 19, WHITE, spacing=1.2)
            put(canvas, layer, cursor + 20, y + 16, opacity)
            cursor += layer[1].shape[1] + 44


def beat_strip(canvas, region, envelope, beats, bars, shots, now, span=None, opacity=1.0, title='MUSIC · BEATS · CUTS', lanes=True):
    """The music's onset strength with every beat, the bar starts, and the shots laid on the beats as coloured blocks."""
    x, y, w, h = [int(v) for v in region]
    if opacity <= 0:
        return
    panel(canvas, region, opacity * 0.92)
    px, pw = x + 18, w - 36
    wave_top, wave_h = y + 48, int((h - 66) * (0.56 if lanes else 1.0))
    lane_top = wave_top + wave_h + 12
    lane_h = h - (lane_top - y) - 16
    t, v = envelope
    if span is None:
        span = (0.0, float(t[-1]))
    to_x = lambda s: px + (np.asarray(s) - span[0]) / (span[1] - span[0]) * pw
    inside = (t >= span[0]) & (t <= span[1])
    tt, vv = t[inside], v[inside] / max(1e-9, v[inside].max())
    pts = np.stack([to_x(tt), wave_top + wave_h - vv * (wave_h - 4)], axis=1)
    done = tt <= now

    def draw(layer, offset):
        ox, oy = offset
        if len(pts) >= 2:
            cv2.polylines(layer, [np.round((pts - offset) * 16).astype(np.int32)], False, scaled(GREEN, 0.3 * opacity), 1, cv2.LINE_AA, 4)
        if done.sum() >= 2:
            cv2.polylines(layer, [np.round((pts[done] - offset) * 16).astype(np.int32)], False, scaled(GREEN, opacity), 2, cv2.LINE_AA, 4)

    if done.sum() >= 2:
        area = np.concatenate([pts[done], [[pts[done][-1, 0], wave_top + wave_h], [pts[done][0, 0], wave_top + wave_h]]]).astype(np.int32)
        overlay = canvas[wave_top:wave_top + wave_h + 1, px:px + pw].copy()
        cv2.fillPoly(overlay, [area - np.array([px, wave_top])], scaled(GREEN, 0.5))
        cv2.addWeighted(overlay, 0.3 * opacity, canvas[wave_top:wave_top + wave_h + 1, px:px + pw], 1 - 0.3 * opacity, 0,
                        canvas[wave_top:wave_top + wave_h + 1, px:px + pw])
    glow(canvas, draw, 0.7 * opacity, sigma=5, region=(px - 8, wave_top - 8, pw + 16, wave_h + 16))
    bar_set = {round(b, 3) for b in bars}
    for b in beats:
        if not (span[0] <= b <= span[1]):
            continue
        bx = int(to_x(b))
        strong = any(abs(b - c) < 0.05 for c in bars)
        hit = np.exp(-max(0.0, now - b) / 0.12) if now >= b else 0.0
        tone = scaled(WHITE if strong else GREY, opacity * (0.55 + 0.45 * hit if now >= b else 0.3))
        cv2.line(canvas, (bx, wave_top + wave_h - (16 if strong else 8)), (bx, wave_top + wave_h), tone, 2 if strong else 1, cv2.LINE_AA)
        if hit > 0.05:
            cv2.circle(canvas, (bx, wave_top + wave_h - (20 if strong else 12)), int(3 + 5 * hit), scaled(WHITE, opacity * hit), -1, cv2.LINE_AA)
    if lanes:
        for shot in shots:
            if shot['end_s'] < span[0] or shot['start_s'] > span[1]:
                continue
            x0, x1 = int(to_x(max(span[0], shot['start_s']))) + 2, int(to_x(min(span[1], shot['end_s']))) - 2
            colour = VIEW_COLOUR[shot['direction']]
            active = shot['start_s'] <= now < shot['end_s']
            seen = now >= shot['start_s']
            box(canvas, x0, lane_top, max(2, x1 - x0), lane_h, colour, opacity * (0.95 if active else 0.42 if seen else 0.16), 6)
            if x1 - x0 > 56:
                name = VIEW_NAME[shot['direction']] if x1 - x0 > 92 else VIEW_NAME[shot['direction']][0]
                layer = text_layer(name, 'mono', 17, BG if active else WHITE, spacing=1.4)
                put(canvas, layer, (x0 + x1) / 2, lane_top + lane_h / 2, opacity * (1.0 if seen else 0.5), 'cm')
    if span[0] <= now <= span[1]:
        box(canvas, int(to_x(now)), y + 44, 2, h - 56, WHITE, 0.9 * opacity)
    put(canvas, text_layer(title, 'mono', 19, GREY, spacing=1.6), x + 18, y + 16, opacity)


def camera_map(canvas, region, active, heading_deg=0.0, opacity=1.0, title='CAMERAS', views=('front', 'left', 'right', 'back'), orbit_deg=None):
    """The stage from above: the dancer in the middle and a camera on each side; the one filming now is lit.
    The picture's bottom is the front (where the viewer of the front camera stands). orbit_deg: one camera flying round instead."""
    x, y, w, h = [int(v) for v in region]
    if opacity <= 0:
        return
    panel(canvas, region, opacity * 0.92)
    cx, cy = x + w // 2, y + h // 2 + 14
    radius = int(min(w, h - 40) * 0.34)
    cv2.circle(canvas, (cx, cy), radius, scaled(LINE, opacity), 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(radius * 0.55), scaled(LINE, opacity * 0.7), 1, cv2.LINE_AA)
    for k in range(-2, 3):
        cv2.line(canvas, (cx + k * radius // 2, cy - radius), (cx + k * radius // 2, cy + radius), scaled(LINE, opacity * 0.5), 1)
        cv2.line(canvas, (cx - radius, cy + k * radius // 2), (cx + radius, cy + k * radius // 2), scaled(LINE, opacity * 0.5), 1)
    # the dancer: a disc with a nose towards the front camera (down), turned by heading
    a = np.radians(heading_deg)
    nose = (int(cx + np.sin(a) * 22), int(cy + np.cos(a) * 22))
    cv2.circle(canvas, (cx, cy), 12, scaled(WHITE, opacity), -1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy), nose, scaled(WHITE, opacity), 4, cv2.LINE_AA)
    places = {'front': (0, 1), 'left': (-1, 0), 'right': (1, 0), 'back': (0, -1)}

    def camera_icon(direction, colour, lit, name=None):
        dx, dy = direction
        px, py = cx + dx * (radius + 4), cy + dy * (radius + 4)
        tone = scaled(colour, opacity * (1.0 if lit else 0.45))
        if lit:
            cone = np.array([[px, py], [cx - dy * 30 - dx * 0, cy + dx * 30], [cx + dy * 30, cy - dx * 30]], np.int32)
            overlay = canvas[y:y + h, x:x + w].copy()
            cv2.fillPoly(overlay, [cone - np.array([x, y])], scaled(colour, 0.55))
            cv2.addWeighted(overlay, 0.35 * opacity, canvas[y:y + h, x:x + w], 1 - 0.35 * opacity, 0, canvas[y:y + h, x:x + w])
        size = 13 if lit else 10
        cv2.circle(canvas, (int(px), int(py)), size, tone, -1 if lit else 2, cv2.LINE_AA)
        if lit:
            cv2.circle(canvas, (int(px), int(py)), size + 7, tone, 2, cv2.LINE_AA)
        if name:
            anchor = {(0, 1): 'ct', (0, -1): 'cb', (-1, 0): 'rm', (1, 0): 'lm'}[direction]
            ox, oy = dx * 26, dy * 24
            put(canvas, text_layer(name, 'mono', 17, WHITE if lit else GREY, spacing=1.4), px + ox, py + oy, opacity, anchor)

    if orbit_deg is not None:
        b = np.radians(orbit_deg)
        camera_icon((float(np.sin(b)), float(np.cos(b))), GREEN, True)
    else:
        for view in views:
            camera_icon(places[view], VIEW_COLOUR[view], view == active, VIEW_NAME[view])
    put(canvas, text_layer(title, 'mono', 19, GREY, spacing=1.6), x + 18, y + 16, opacity)


def bars(canvas, region, values, now, start, labels=None, colour=GREEN, opacity=1.0, title=None, top=None, fmt='{:.1f}', highlight=None,
         stagger=0.05, grow=0.5, ticks=(), tick_fmt='{:g}'):
    """Upright bars that grow one after another from `start`."""
    x, y, w, h = [int(v) for v in region]
    if opacity <= 0:
        return
    panel(canvas, region, opacity * 0.92)
    count = len(values)
    px, pw = x + (74 if ticks else 24), w - (98 if ticks else 48)
    base, height = y + h - 44, h - 118
    top = top or max(values) * 1.15
    slot = pw / count
    for tick in ticks:
        ty = int(base - height * tick / top)
        cv2.line(canvas, (px, ty), (px + pw, ty), scaled(LINE, opacity), 1)
        put(canvas, text_layer(tick_fmt.format(tick), 'mono', 15, GREY), px - 12, ty, opacity, 'rm')
    width = int(slot * 0.62)
    for index, value in enumerate(values):
        grown = ease_out(ramp(now, start + index * stagger, start + index * stagger + grow))
        bh = int(height * value / top * grown)
        bx = int(px + slot * index + (slot - width) / 2)
        tone = colour if highlight is None or index == highlight else scaled(colour, 0.55)
        box(canvas, bx, base - bh, width, max(1, bh), tone, opacity, min(6, width // 2))
        if grown > 0.6:
            put(canvas, text_layer(fmt.format(value), 'mono', 15, WHITE), bx + width / 2, base - bh - 8, opacity * ramp(grown, 0.6, 1.0), 'cb')
        if labels:
            put(canvas, text_layer(labels[index], 'mono', 15, GREY), bx + width / 2, base + 10, opacity, 'ct')
    cv2.line(canvas, (px, base), (px + pw, base), scaled(DIM, opacity), 1)
    if title:
        put(canvas, text_layer(title, 'mono', 19, GREY, spacing=1.6), x + 18, y + 16, opacity)


def check_row(canvas, x, y, text, value, now, start, opacity=1.0, width=520, size=24, good=True):
    """One line of an automatic check: a tick that pops in, the name, and the measured value at the right."""
    shown = ramp(now, start, start + 0.3)
    if shown <= 0 or opacity <= 0:
        return
    pop = back_out(shown)
    cv2.circle(canvas, (int(x + 16), int(y + 18)), int(15 * pop), scaled(GREEN if good else AMBER, opacity), -1, cv2.LINE_AA)
    if shown > 0.5:
        tick = np.array([[x + 9, y + 18], [x + 14, y + 24], [x + 24, y + 12]], np.int32)
        cv2.polylines(canvas, [tick], False, BG, 3, cv2.LINE_AA)
    put(canvas, text_layer(text, 'medium', size, WHITE), x + 46, y + 18, opacity * shown, 'lm')
    put(canvas, text_layer(value, 'mono', size - 2, GREEN if good else AMBER), x + width, y + 18, opacity * shown, 'rm')


def progress_line(canvas, x, y, w, share, colour=GREEN, opacity=1.0, height=4):
    box(canvas, x, y, w, height, LINE, opacity, height // 2)
    box(canvas, x, y, max(height, int(w * share)), height, colour, opacity, height // 2)


def scatter(canvas, region, x_values, y_values, times, now, x_range, y_range, colour=SKY, opacity=1.0, title=None, x_name=None, y_name=None, fit=None):
    """Points (x, y) that appear as their time passes: one value against the other. fit: (slope, offset) of a line to draw."""
    x, y, w, h = [int(v) for v in region]
    if opacity <= 0:
        return
    panel(canvas, region, opacity * 0.92)
    px, py, pw, ph = x + 40, y + 46, w - 58, h - 86
    to_x = lambda v: px + (np.asarray(v) - x_range[0]) / (x_range[1] - x_range[0]) * pw
    to_y = lambda v: py + ph - (np.asarray(v) - y_range[0]) / (y_range[1] - y_range[0]) * ph
    cv2.line(canvas, (int(to_x(0)), py), (int(to_x(0)), py + ph), scaled(LINE, opacity), 1)
    cv2.line(canvas, (px, int(to_y(0))), (px + pw, int(to_y(0))), scaled(LINE, opacity), 1)
    seen = times <= now
    if fit is not None and seen.sum() > 10:
        slope, offset = fit
        cv2.line(canvas, (int(to_x(x_range[0])), int(to_y(np.clip(slope * x_range[0] + offset, *y_range)))),
                 (int(to_x(x_range[1])), int(to_y(np.clip(slope * x_range[1] + offset, *y_range)))), scaled(GREEN, 0.7 * opacity), 2, cv2.LINE_AA)
    overlay = np.zeros((ph + 8, pw + 8, 3), np.uint8)
    xs, ys = (to_x(x_values[seen]) - px + 4).astype(int), (to_y(np.clip(y_values[seen], *y_range)) - py + 4).astype(int)
    age = now - times[seen]
    for cx, cy, a in zip(xs, ys, age):
        cv2.circle(overlay, (int(cx), int(cy)), 3 if a > 0.4 else 6, scaled(colour if a > 0.4 else WHITE, opacity * (0.75 if a > 0.4 else 1.0)), -1, cv2.LINE_AA)
    part = canvas[py - 4:py + ph + 4, px - 4:px + pw + 4]
    part[:] = np.maximum(part, overlay[:part.shape[0], :part.shape[1]])
    if title:
        put(canvas, text_layer(title, 'mono', 17, GREY, spacing=1.4), x + 16, y + 14, opacity)
    if x_name:
        put(canvas, text_layer(x_name, 'mono', 15, GREY, spacing=1.2), px + pw / 2, y + h - 12, opacity, 'cb')
    if y_name:
        layer = text_layer(y_name, 'mono', 15, GREY, spacing=1.2)
        turned = (np.ascontiguousarray(np.rot90(layer[0])), np.ascontiguousarray(np.rot90(layer[1])))
        put(canvas, turned, x + 10, py + ph / 2, opacity, 'lm')


def meter_row(canvas, x, y, name, colour, value, low, high, text, shown=1.0, width=626, name_width=250):
    """One live value as a row: its name, a track with a marker at the value (the middle is zero), and the number."""
    if shown <= 0:
        return
    put(canvas, text_layer(name, 'semibold', 44, colour), x, y, shown, 'lm')
    track_x, track_w = x + name_width, width - name_width - 110
    box(canvas, track_x, y - 3, track_w, 6, LINE, shown, 3)
    middle = track_x + track_w * (0 - low) / (high - low)
    box(canvas, middle - 1, y - 12, 2, 24, DIM, shown)
    at = track_x + track_w * (np.clip(value, low, high) - low) / (high - low)
    box(canvas, min(middle, at), y - 3, abs(at - middle), 6, colour, shown, 3)
    cv2.circle(canvas, (int(at), int(y)), 11, scaled(colour, shown), -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(at), int(y)), 5, scaled(WHITE, shown), -1, cv2.LINE_AA)
    put(canvas, text_layer(text, 'mono', 22, WHITE), x + width, y, shown, 'rm')


def pieces_strip(canvas, region, pieces, now, total, opacity=1.0, title='PIECES · EACH ONE CHECKED'):
    """The video as the pieces it was made in: each block lights when the playhead is in it and keeps a tick once passed."""
    x, y, w, h = [int(v) for v in region]
    if opacity <= 0:
        return
    put(canvas, text_layer(title, 'mono', 17, GREY, spacing=1.4), x, y, opacity)
    top, tall = y + 32, h - 32
    for index, (start, end) in enumerate(pieces):
        x0, x1 = x + w * start / total + 2, x + w * end / total - 2
        active, passed = start <= now < end, now >= end
        box(canvas, x0, top, max(3, x1 - x0), tall, GREEN if active else (scaled(GREEN, 0.45) if passed else LINE), opacity, 5)
        if passed and x1 - x0 > 16:
            cx, cy = int((x0 + x1) / 2), int(top + tall / 2)
            tick = np.array([[cx - 6, cy], [cx - 2, cy + 5], [cx + 7, cy - 5]], np.int32)
            cv2.polylines(canvas, [tick], False, scaled(WHITE, opacity), 2, cv2.LINE_AA)


def stage_map(canvas, region, dancers, colours, path, index, opacity=1.0, title='FROM ABOVE', reach=3.4, centre=(0.0, 0.0), camera_colour=GREEN):
    """The room from above: where each dancer stands (with a short trail) and where the camera is on its path.
    dancers: list of (frames, 2) floor positions in metres, path: (frames, 2) camera positions; the front is at the bottom."""
    x, y, w, h = [int(v) for v in region]
    if opacity <= 0:
        return
    panel(canvas, region, opacity * 0.92)
    cx, cy = x + w / 2, y + h / 2 + 14
    scale = (min(w, h - 40) / 2 - 14) / reach
    to = lambda p: (int(cx + (p[0] - centre[0]) * scale), int(cy - (p[1] - centre[1]) * scale))
    for k in range(-int(reach), int(reach) + 1):   # a line every metre
        cv2.line(canvas, to((centre[0] + k, centre[1] - reach)), to((centre[0] + k, centre[1] + reach)), scaled(LINE, opacity * (1.0 if k == 0 else 0.6)), 1)
        cv2.line(canvas, to((centre[0] - reach, centre[1] + k)), to((centre[0] + reach, centre[1] + k)), scaled(LINE, opacity * (1.0 if k == 0 else 0.6)), 1)
    put(canvas, text_layer('1 m', 'mono', 13, GREY), cx - reach * scale + scale / 2, cy + reach * scale + 4, opacity, 'ct')
    cv2.line(canvas, (int(cx - reach * scale), int(cy + reach * scale)), (int(cx - reach * scale + scale), int(cy + reach * scale)), scaled(GREY, opacity), 2)
    index = int(np.clip(index, 0, len(path) - 1))
    points = np.array([to(p) for p in path], np.int32)
    cv2.polylines(canvas, [points], False, scaled(camera_colour, 0.25 * opacity), 2, cv2.LINE_AA)
    if index >= 1:
        cv2.polylines(canvas, [points[:index + 1]], False, scaled(camera_colour, 0.9 * opacity), 3, cv2.LINE_AA)
    middle = np.mean([d[min(index, len(d) - 1)] for d in dancers], axis=0)
    camera = np.array(path[index])
    # the camera's view: a wedge towards the dancers
    aim = middle - camera
    aim = aim / max(1e-6, np.linalg.norm(aim))
    side = np.array([-aim[1], aim[0]])
    far = camera + aim * min(2.6, np.linalg.norm(middle - camera) + 0.7)
    wedge = np.array([to(camera), to(far + side * 0.9), to(far - side * 0.9)], np.int32)
    overlay = canvas[y:y + h, x:x + w].copy()
    cv2.fillPoly(overlay, [wedge - np.array([x, y])], scaled(camera_colour, 0.55))
    cv2.addWeighted(overlay, 0.3 * opacity, canvas[y:y + h, x:x + w], 1 - 0.3 * opacity, 0, canvas[y:y + h, x:x + w])
    for dancer, colour in zip(dancers, colours):
        k = min(index, len(dancer) - 1)
        trail = np.array([to(p) for p in dancer[max(0, k - 30):k + 1]], np.int32)
        if len(trail) >= 2:
            cv2.polylines(canvas, [trail], False, scaled(colour, 0.5 * opacity), 2, cv2.LINE_AA)
        cv2.circle(canvas, to(dancer[k]), 11, scaled(colour, opacity), -1, cv2.LINE_AA)
        cv2.circle(canvas, to(dancer[k]), 11, scaled(WHITE, opacity), 2, cv2.LINE_AA)
    cv2.circle(canvas, to(camera), 12, scaled(camera_colour, opacity), -1, cv2.LINE_AA)
    cv2.circle(canvas, to(camera), 19, scaled(camera_colour, opacity), 2, cv2.LINE_AA)
    if title:
        put(canvas, text_layer(title, 'mono', 17, GREY, spacing=1.4), x + 16, y + 14, opacity)
