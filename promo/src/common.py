"""Scene pieces both films use: picture cards, the phone with the 2D joints, the brand stings."""
import cv2
import numpy as np

import sources
import widgets as wd
from kit import (AMBER, BG, DIM, FPS, GREEN, GREY, LINE, PANEL, SKY, WHITE, box, card, ease, ease_out, fit, outline, paste, put, ramp,
                 text_layer, window)
from skeleton import CENTRE, LEFT, LEFT_FOOT, RIGHT, RIGHT_FOOT

A_COLOUR, B_COLOUR = (235, 140, 60), (70, 70, 235)   # the two-dancer program's mannequins: A blue, B red (BGR)


def cover(source, size, focus=(0.5, 0.5), zoom=1.0):
    """How a picture of size `source` (w, h) fills a card of `size` (w, h): (scale, x offset, y offset) with
    card position = picture position * scale - offset. zoom > 1 goes closer; focus says which part stays (0..1 each way)."""
    sw, sh = source
    w, h = size
    scale = zoom * max(w / sw, h / sh)
    return scale, (sw * scale - w) * focus[0], (sh * scale - h) * focus[1]


def covered(picture, size, focus=(0.5, 0.5), zoom=1.0):
    w, h = size
    scale, ox, oy = cover((picture.shape[1], picture.shape[0]), size, focus, zoom)
    new = (max(w, int(round(picture.shape[1] * scale))), max(h, int(round(picture.shape[0] * scale))))
    resized = cv2.resize(picture, new, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x, y = int(round(min(ox, new[0] - w))), int(round(min(oy, new[1] - h)))
    return resized[y:y + h, x:x + w]


def video_card(canvas, picture, region, radius=18, label=None, colour=GREEN, opacity=1.0, focus=(0.5, 0.5), border=LINE, label_size=18,
               mirror=False, label_anchor='lt', zoom=1.0):
    """A picture filling a rounded card, with an optional chip in its top-left corner."""
    x, y, w, h = [int(v) for v in region]
    if opacity <= 0 or picture is None:
        return
    fitted = covered(picture, (w, h), focus, zoom)
    if mirror:
        fitted = fitted[:, ::-1]
    card(canvas, np.ascontiguousarray(fitted), x, y, radius, opacity, border, 2)
    if label:
        if label_anchor == 'lt':
            wd.chip(canvas, label, x + 14, y + 14, colour, opacity, label_size)
        else:
            wd.chip(canvas, label, x + w / 2, y + h - 14, colour, opacity, label_size, anchor='cb')


def phone(canvas, dance, seconds, region, opacity=1.0, label=None, strength=1.0, outline_of=None):
    """The source video as the program sees it: the dancer's joints where they are in the picture (the footage itself is not shown)."""
    x, y, w, h = [int(v) for v in region]
    if opacity <= 0:
        return
    box(canvas, x, y, w, h, (17, 21, 20), opacity, 34)
    outline(canvas, x, y, w, h, 34, WHITE, 3, opacity)
    box(canvas, x + w / 2 - 46, y + 16, 92, 8, DIM, opacity, 4)   # the speaker slot
    inner = (x + 12, y + 40, w - 24, h - 64)
    scale = min(inner[2] / dance.picture[0], inner[3] / dance.picture[1])
    ox = inner[0] + (inner[2] - dance.picture[0] * scale) / 2
    oy = inner[1] + (inner[3] - dance.picture[1] * scale) / 2
    points = dance.pose2d(seconds) * scale + (ox, oy)
    layer = canvas[y:y + h, x:x + w]
    local = points - (x, y)
    tone = lambda colour: tuple(int(v * opacity * strength) for v in colour)
    if outline_of is not None:   # the dancer as a plain shape: the tracker's outline of the person, no footage
        mask = outline_of(dance.index(seconds))
        size = (int(round(dance.picture[0] * scale)), int(round(dance.picture[1] * scale)))
        shape = cv2.resize(mask * 255, size, interpolation=cv2.INTER_AREA)
        px, py = int(round(ox - x)), int(round(oy - y))
        part = layer[py:py + size[1], px:px + size[0]]
        if part.shape[:2] == shape.shape:
            alpha = (shape.astype(np.float32) / 255.0)[..., None] * opacity
            part[:] = (part * (1 - alpha) + np.array((58, 70, 66), np.float32) * alpha).astype(np.uint8)
            contours, _ = cv2.findContours((shape > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(part, contours, -1, tone((120, 140, 134)), 2, cv2.LINE_AA)
    for bones, colour in ((CENTRE, WHITE), (LEFT, WHITE), (RIGHT, WHITE), (LEFT_FOOT, GREY), (RIGHT_FOOT, GREY)):
        for a, b in bones:
            cv2.line(layer, (int(local[a, 0]), int(local[a, 1])), (int(local[b, 0]), int(local[b, 1])), tone(colour), 3, cv2.LINE_AA)
    for index in range(22):
        cv2.circle(layer, (int(local[index, 0]), int(local[index, 1])), 5, tone(GREEN), -1, cv2.LINE_AA)
    head = (local[58] + local[59]) / 2
    cv2.circle(layer, (int(head[0]), int(head[1])), max(6, int(np.hypot(*(local[58] - local[59])) * 0.75)), tone(WHITE), 3, cv2.LINE_AA)
    # the frame corners a tracker draws round the dancer
    low, high = local[:22].min(0) - 26, local[:22].max(0) + 26
    low, high = np.maximum(low, (14, 44)), np.minimum(high, (w - 14, h - 28))
    for cx, cy, dx, dy in ((low[0], low[1], 1, 1), (high[0], low[1], -1, 1), (low[0], high[1], 1, -1), (high[0], high[1], -1, -1)):
        cv2.line(layer, (int(cx), int(cy)), (int(cx + 26 * dx), int(cy)), tone(GREEN), 2, cv2.LINE_AA)
        cv2.line(layer, (int(cx), int(cy)), (int(cx), int(cy + 26 * dy)), tone(GREEN), 2, cv2.LINE_AA)
    if label:
        wd.chip(canvas, label, x + w / 2, y - 16, GREEN, opacity, 18, anchor='cb', over_picture=False)


def sting(canvas, clip, local, first=0, hold=None, scale=1.0, centre=None, fade=0.12, end=None):
    """A brand animation (black background) played from frame `first`, held on its last frame; added to the canvas."""
    index = first + int(local * FPS)
    picture = clip.get(min(index, clip.count - 1))
    H, W = canvas.shape[:2]
    if scale != 1.0:
        picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    h, w = picture.shape[:2]
    cx, cy = centre or (W / 2, H / 2)
    x, y = int(round(cx - w / 2)), int(round(cy - h / 2))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    gain = ramp(local, 0, fade) * (1 - ramp(local, end - 0.5, end) if end else 1.0)   # fades in, and out at the very end
    part = picture[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) * gain
    canvas[y0:y1, x0:x1] = np.clip(canvas[y0:y1, x0:x1] + part, 0, 255).astype(np.uint8)


def arrow(canvas, x0, y, x1, progress, colour=GREEN, opacity=1.0):
    """A row of dots running from x0 to x1 with an arrow head; `progress` 0..1 draws it in."""
    if progress <= 0 or opacity <= 0:
        return
    end = x0 + (x1 - x0) * ease(progress)
    for px in np.arange(x0, end, 16):
        cv2.circle(canvas, (int(px), int(y)), 3, tuple(int(v * opacity) for v in colour), -1, cv2.LINE_AA)
    if progress > 0.9:
        head = np.array([[x1 + 14, y], [x1 - 6, y - 12], [x1 - 6, y + 12]], np.int32)
        cv2.fillPoly(canvas, [head], tuple(int(v * opacity) for v in colour), cv2.LINE_AA)


def stat(canvas, x, y, value, name, t, start, colour=WHITE, size=120, anchor='l', unit=None, opacity=1.0):
    """A big number with a small name under it, popping in at `start`."""
    shown = ramp(t, start, start + 0.3)
    if shown <= 0:
        return
    rise = (1 - ease_out(shown)) * 22
    layer = text_layer(value, 'semibold', size, colour)
    put(canvas, layer, x, y + rise, shown * opacity, anchor + 't')
    if unit:
        offset = layer[1].shape[1] if anchor == 'l' else 0
        put(canvas, text_layer(unit, 'semibold', int(size * 0.42), GREY), x + offset + 10, y + rise + size * 0.62, shown * opacity, 'lt')
    wd.label(canvas, name, x, y + size * 1.24 + rise, 20, GREY, anchor + 't', shown * opacity)
