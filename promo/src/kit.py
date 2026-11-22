"""Shared pieces of the dance promo films: colours, fonts, text layers, compositing, clip frames, the encoder.

Pictures are OpenCV arrays (BGR, uint8). A layer is (colour BGR uint8, alpha float32 0..1).
"""
import functools
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

JOB = Path('/srv/training/jobs/20261116-041650-dance-promo-video-0df0')
WORK = JOB / 'outputs/work'
BRAND = JOB / 'cache/brand/steerix-matrix-brand'
FONTS = BRAND / '4-폰트/UI 글꼴 (IBM Plex)'
FPS = 24


def bgr(code):
    code = code.lstrip('#')
    return (int(code[4:6], 16), int(code[2:4], 16), int(code[0:2], 16))


GREEN, SKY, AMBER, CORAL = bgr('#00DC7D'), bgr('#53C1DB'), bgr('#F5B536'), bgr('#EF5A64')
WHITE, GREY, DIM, LINE, BG, PANEL = bgr('#EEF2F0'), bgr('#96A09E'), bgr('#4A5553'), bgr('#1C2422'), bgr('#050607'), bgr('#0B0F0E')
VIOLET = bgr('#A78BFA')


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def ease_in(t):
    t = min(1.0, max(0.0, t))
    return t ** 3


def back_out(t, overshoot=1.4):
    t = min(1.0, max(0.0, t)) - 1
    return t * t * ((overshoot + 1) * t + overshoot) + 1


def ramp(t, start, end):
    """0 before start, 1 after end, straight in between."""
    if end <= start:
        return float(t >= end)
    return min(1.0, max(0.0, (t - start) / (end - start)))


def window(t, start, end, fade_in=0.25, fade_out=0.25):
    """1 inside [start, end], with soft edges."""
    return min(ramp(t, start, start + fade_in), 1 - ramp(t, end - fade_out, end))


@functools.lru_cache(maxsize=None)
def font(kind, size):
    name = {'semibold': 'IBMPlexSans-SemiBold-Latin.ttf', 'medium': 'IBMPlexSans-Medium-Latin.ttf', 'regular': 'IBMPlexSans-Regular-Latin.ttf',
            'mono': 'IBMPlexMono-Medium-Latin.ttf', 'mono-regular': 'IBMPlexMono-Regular-Latin.ttf'}[kind]
    return ImageFont.truetype(str(FONTS / name), int(round(size)))


FALLBACK = set('→←↑↓≈✓▶')   # not in the brand font's Latin set: drawn with DejaVu Sans Bold


@functools.lru_cache(maxsize=None)
def fallback_font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', int(round(size * 0.9)))


@functools.lru_cache(maxsize=4096)
def text_layer(text, kind='semibold', size=48, colour=WHITE, spacing=0.0, accents=None):
    """One line of text as a layer. accents: ((word, colour), ...) paints those words (first match each) in another colour.
    spacing: extra pixels between letters (labels in capitals)."""
    f = font(kind, size)
    pieces = [(text, colour)]
    for word, tint in accents or ():
        split = []
        for piece, c in pieces:
            if c == colour and word in piece:
                before, after = piece.split(word, 1)
                split += [(before, c), (word, tint), (after, c)]
            else:
                split.append((piece, c))
        pieces = [(p, c) for p, c in split if p]
    runs = []   # (text, colour, font): the brand font, or the fallback for single signs it lacks
    for piece, c in pieces:
        current = ''
        for ch in piece:
            if ch in FALLBACK:
                if current:
                    runs.append((current, c, f))
                    current = ''
                runs.append((ch, c, fallback_font(size)))
            else:
                current += ch
        if current:
            runs.append((current, c, f))
    probe = ImageDraw.Draw(Image.new('L', (4, 4)))
    widths = [sum(probe.textlength(ch, font=g) + spacing for ch in run) if spacing else probe.textlength(run, font=g) for run, _, g in runs]
    ascent, descent = f.getmetrics()
    width, height = int(np.ceil(sum(widths))) + 4, ascent + descent + 4
    colour_img = np.zeros((height, width, 3), np.uint8)
    alpha_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(alpha_img)
    x = 2.0
    for (run, c, g), w in zip(runs, widths):
        if spacing:
            cx = x
            for ch in run:
                draw.text((cx, 2 + ascent), ch, font=g, fill=255, anchor='ls')
                cx += probe.textlength(ch, font=g) + spacing
        else:
            draw.text((x, 2 + ascent), run, font=g, fill=255, anchor='ls')
        colour_img[:, int(x):int(np.ceil(x + w)) + 1] = c
        x += w
    return colour_img, np.asarray(alpha_img, np.float32) / 255.0


def put(canvas, layer, x, y, opacity=1.0, anchor='lt', scale=1.0):
    """Lay a layer on the canvas. anchor: l/c/r + t/m/b of the layer at (x, y)."""
    colour, alpha = layer
    if scale != 1.0:
        size = (max(1, int(round(colour.shape[1] * scale))), max(1, int(round(colour.shape[0] * scale))))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        colour, alpha = cv2.resize(colour, size, interpolation=interpolation), cv2.resize(alpha, size, interpolation=interpolation)
    h, w = alpha.shape
    x = x - {'l': 0, 'c': w / 2, 'r': w}[anchor[0]]
    y = y - {'t': 0, 'm': h / 2, 'b': h}[anchor[1]]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x0 >= x1 or y0 >= y1 or opacity <= 0:
        return
    a = (alpha[y0 - y:y1 - y, x0 - x:x1 - x] * opacity)[..., None]
    region = canvas[y0:y1, x0:x1]
    region[:] = (region * (1 - a) + colour[y0 - y:y1 - y, x0 - x:x1 - x] * a + 0.5).astype(np.uint8)


def shadowed(canvas, layer, x, y, opacity=1.0, anchor='lt', scale=1.0, shadow=0.5, blur=14, offset=4):
    """A layer with a soft dark shadow under it (text over pictures)."""
    alpha = layer[1]
    h, w = alpha.shape
    left = x - {'l': 0, 'c': w * scale / 2, 'r': w * scale}[anchor[0]]
    top = y - {'t': 0, 'm': h * scale / 2, 'b': h * scale}[anchor[1]]
    pad = int(blur * 2.5)
    soft = _shadow_of(layer, blur, pad)
    put(canvas, soft, left - pad * scale, top - pad * scale + offset, opacity * shadow, 'lt', scale)
    put(canvas, layer, left, top, opacity, 'lt', scale)


_SHADOWS = {}


def _shadow_of(layer, blur, pad):
    """The soft shadow of a layer, `pad` pixels larger on every side (so that the blur is not cut at the layer's edge)."""
    key = (id(layer[1]), blur)
    cached = _SHADOWS.get(key)
    if cached is not None and cached[0] is layer[1]:
        return cached[1]
    padded = cv2.copyMakeBorder(layer[1], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    soft = np.clip(cv2.GaussianBlur(padded, (0, 0), blur) * 1.9, 0, 1)
    result = (np.zeros((*soft.shape, 3), np.uint8), soft)
    if len(_SHADOWS) > 512:
        _SHADOWS.clear()
    _SHADOWS[key] = (layer[1], result)
    return result


def fit(picture, width, height, mode='cover', focus=(0.5, 0.5)):
    """Resize a picture into width x height. cover: fill and crop around focus; contain: fit inside (returns smaller)."""
    h, w = picture.shape[:2]
    scale = max(width / w, height / h) if mode == 'cover' else min(width / w, height / h)
    new = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = cv2.resize(picture, new, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if mode != 'cover':
        return resized
    x = int(round((new[0] - width) * focus[0]))
    y = int(round((new[1] - height) * focus[1]))
    return resized[y:y + height, x:x + width]


def paste(canvas, picture, x, y, opacity=1.0):
    """Paste an opaque picture with its top-left corner at (x, y), clipped."""
    h, w = picture.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x0 >= x1 or y0 >= y1 or opacity <= 0:
        return
    source = picture[y0 - y:y1 - y, x0 - x:x1 - x]
    if opacity >= 1:
        canvas[y0:y1, x0:x1] = source
    else:
        canvas[y0:y1, x0:x1] = (canvas[y0:y1, x0:x1] * (1 - opacity) + source * opacity + 0.5).astype(np.uint8)


@functools.lru_cache(maxsize=64)
def round_mask(width, height, radius):
    mask = np.zeros((height, width), np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
    for cx, cy in ((radius, radius), (width - radius - 1, radius), (radius, height - radius - 1), (width - radius - 1, height - radius - 1)):
        cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32) / 255.0


def card(canvas, picture, x, y, radius=18, opacity=1.0, border=None, border_width=2):
    """A picture with rounded corners (and an optional outline) at (x, y)."""
    h, w = picture.shape[:2]
    put(canvas, (picture, round_mask(w, h, radius)), x, y, opacity)
    if border is not None and opacity > 0:
        outline(canvas, x, y, w, h, radius, border, border_width, opacity)


def outline(canvas, x, y, w, h, radius, colour, width=2, opacity=1.0):
    x, y = int(round(x)), int(round(y))
    pad = width + 2
    layer = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    r = radius
    pts = [(pad + r, pad), (pad + w - r - 1, pad), (pad + w - 1, pad + r), (pad + w - 1, pad + h - r - 1),
           (pad + w - r - 1, pad + h - 1), (pad + r, pad + h - 1), (pad, pad + h - r - 1), (pad, pad + r)]
    for a, b in ((0, 1), (2, 3), (4, 5), (6, 7)):
        cv2.line(layer, pts[a], pts[b], 255, width, cv2.LINE_AA)
    for (cx, cy), start in (((pad + r, pad + r), 180), ((pad + w - r - 1, pad + r), 270), ((pad + w - r - 1, pad + h - r - 1), 0), ((pad + r, pad + h - r - 1), 90)):
        cv2.ellipse(layer, (cx, cy), (r, r), 0, start, start + 90, 255, width, cv2.LINE_AA)
    colour_img = np.empty((*layer.shape, 3), np.uint8)
    colour_img[:] = colour
    put(canvas, (colour_img, layer.astype(np.float32) / 255.0), x - pad, y - pad, opacity)


def box(canvas, x, y, w, h, colour, opacity=1.0, radius=0):
    """A filled rectangle (rounded if radius)."""
    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    if w <= 0 or h <= 0 or opacity <= 0:
        return
    if radius:
        fill = np.empty((h, w, 3), np.uint8)
        fill[:] = colour
        put(canvas, (fill, round_mask(w, h, min(radius, w // 2, h // 2))), x, y, opacity)
        return
    x0, y0, x1, y1 = max(0, x), max(0, y), min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    region = canvas[y0:y1, x0:x1]
    region[:] = (region * (1 - opacity) + np.array(colour, np.float32) * opacity + 0.5).astype(np.uint8)


@functools.lru_cache(maxsize=64)
def feather_mask(width, height, edge):
    """1 in the middle of a width x height area, falling to 0 over `edge` pixels at its sides."""
    fx = np.clip(np.minimum(np.arange(width), width - 1 - np.arange(width)) / max(1, edge), 0, 1)
    fy = np.clip(np.minimum(np.arange(height), height - 1 - np.arange(height)) / max(1, edge), 0, 1)
    smooth = lambda v: v * v * (3 - 2 * v)
    return (smooth(fy)[:, None] * smooth(fx)[None, :]).astype(np.float32)[..., None]


def glow(canvas, draw, strength=1.0, sigma=7, region=None, feather=0):
    """Draw bright lines with a soft halo: `draw(layer)` paints on a black layer, which is blurred, added, then drawn sharp.
    region: (x, y, w, h) part of the canvas the drawing stays in (cheaper); feather: fade the drawing out towards the region's sides."""
    if region is None:
        region = (0, 0, canvas.shape[1], canvas.shape[0])
    x, y, w, h = [int(v) for v in region]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    draw(layer, (x0, y0))
    if feather:
        layer = (layer * feather_mask(x1 - x0, y1 - y0, feather)).astype(np.uint8)
    if strength > 0:
        halo = cv2.GaussianBlur(layer, (0, 0), sigma)
        part = canvas[y0:y1, x0:x1]
        part[:] = np.clip(part.astype(np.int16) + (halo.astype(np.float32) * strength).astype(np.int16), 0, 255).astype(np.uint8)
    part = canvas[y0:y1, x0:x1]
    mask = layer.max(axis=2, keepdims=True).astype(np.float32) / 255.0
    part[:] = np.clip(part * (1 - mask) + layer.astype(np.float32), 0, 255).astype(np.uint8)


class Frames:
    """The frames of a clip, unpacked once into a folder of pictures; frame numbers start at 0."""

    def __init__(self, name, source=None, fps=None, scale=None, start=None, seconds=None, pattern='%05d.jpg', folder=None, first=1):
        self.folder = Path(folder) if folder else WORK / 'frames' / name
        self.pattern, self.first = pattern, first
        if source is not None:
            self.unpack(Path(source), fps, scale, start, seconds)
        record = self.folder / 'done.json'
        self.count = json.loads(record.read_text())['frames'] if record.exists() else len(list(self.folder.glob('*' + Path(pattern).suffix)))
        self._read = functools.lru_cache(maxsize=6)(self._load)

    def unpack(self, source, fps, scale, start, seconds):
        """Unpack the clip unless that was done for this very file (same size and time of writing: a video that was
        made again under the same name is unpacked again). One process at a time (the film is drawn by several)."""
        import fcntl
        import shutil
        pattern = self.pattern
        stamp = [source.stat().st_size, source.stat().st_mtime_ns]
        self.folder.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.folder) + '.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            record = self.folder / 'done.json'
            if record.exists() and json.loads(record.read_text()).get('stamp') == stamp:
                return
            if self.folder.exists():
                shutil.rmtree(self.folder)
            self.folder.mkdir(parents=True)
            command = ['ffmpeg', '-v', 'error', '-nostdin', '-y']
            if start is not None:
                command += ['-ss', f'{start:.4f}']
            command += ['-i', str(source)]
            if seconds is not None:
                command += ['-t', f'{seconds:.4f}']
            filters = []
            if fps:
                filters.append(f'fps={fps}')
            if scale:
                filters.append(f'scale={scale}:flags=lanczos')
            if filters:
                command += ['-vf', ','.join(filters)]
            command += ['-q:v', '2', '-start_number', '1', str(self.folder / pattern)]
            subprocess.run(command, check=True)
            count = len(list(self.folder.glob('*' + Path(pattern).suffix)))
            record.write_text(json.dumps(dict(source=str(source), frames=count, fps=fps, stamp=stamp)))

    def _load(self, index):
        picture = cv2.imread(str(self.folder / (self.pattern % (index + self.first))))
        if picture is None:
            raise FileNotFoundError(self.folder / (self.pattern % (index + self.first)))
        return picture

    def get(self, index):
        return self._read(int(np.clip(index, 0, self.count - 1)))

    def at(self, seconds, fps=FPS):
        return self.get(int(round(seconds * fps)))


def encoder(path, width, height, fps=FPS, crf=15, preset='slow'):
    """ffmpeg fed with raw BGR frames (no sound; the sound is joined afterwards)."""
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', f'{fps:g}', '-i', '-',
               '-c:v', 'libx264', '-crf', str(crf), '-preset', preset, '-pix_fmt', 'yuv420p', '-g', str(int(fps)),
               '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
               '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-movflags', '+faststart', str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE)
