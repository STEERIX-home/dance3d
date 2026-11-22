"""The dance promo films (long 1920x1080 and short 1080x1920, 24 pictures a second).

    film.py long --timeline                     write work/long/timeline.json (scene times, narration, music grid)
    film.py long --stills 3.0,12.5 --out DIR    single pictures at those seconds
    film.py long --range 0 240 --out part.mp4   a stretch of frames as a video without sound
    film.py long --all --out film.mp4 [--workers 6]

A film is a row of scenes. Scenes of one chapter share a "bar": the length of one bar of the dance shown in that
chapter. Every scene lasts a whole number of half bars and the dance shown always stays on the bar grid, so one piece
of music at that tempo fits the whole chapter and every cut falls on a beat.
"""
import argparse
import json
import math
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit import (AMBER, BG, DIM, FPS, GREEN, GREY, JOB, LINE, PANEL, SKY, VIOLET, WHITE, WORK, Frames, back_out, box, card, ease, ease_in,  # noqa: E402
                 ease_out, encoder, fit, glow, outline, paste, put, ramp, shadowed, text_layer, window)
import sources  # noqa: E402
import widgets as wd  # noqa: E402
from skeleton import figure_panel, orbit_camera  # noqa: E402

SIZES = {'long': (1920, 1080), 'short': (1080, 1920)}
VOICE = WORK / 'voice'


class Scene:
    def __init__(self, name, draw, need, lines=(), chapter=None, bar=None, at_bar=0.0, flash=True, chrome=True, **more):
        self.name, self.draw, self.need, self.lines, self.chapter = name, draw, need, list(lines), chapter
        self.bar, self.at_bar, self.flash, self.chrome = bar, at_bar, flash, chrome
        self.__dict__.update(more)
        self.start = self.end = 0.0

    @property
    def seconds(self):
        if self.bar is None:
            return self.need
        half = self.bar / 2
        return math.ceil(self.need / half - 1e-6) * half


class Film:
    def __init__(self, fmt):
        self.fmt = fmt
        self.W, self.H = SIZES[fmt]
        self.words = json.loads((VOICE / 'words.json').read_text())
        self.scenes = []
        self.music = []      # (start, end, bar length, dance time at start, name)
        import importlib
        importlib.import_module(f'scenes_{fmt}').build(self)
        clock = 0.0
        for scene in self.scenes:
            scene.start = clock
            clock = scene.end = clock + scene.seconds
            scene.film = self
        self.seconds = clock
        self.frames = int(round(self.seconds * FPS))
        # chapters: the first scene of each one shows its name large for a moment
        previous = None
        for scene in self.scenes:
            scene.chapter_start = bool(scene.chapter) and scene.chapter != previous
            previous = scene.chapter or previous
        # the backdrop: nearly black, a little lighter in the middle, with a faint green breath at the top
        ys, xs = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        radius = np.sqrt(((xs - self.W / 2) / (self.W * 0.75)) ** 2 + ((ys - self.H * 0.45) / (self.H * 0.75)) ** 2)
        glow_part = np.clip(1 - radius, 0, 1) ** 2
        backdrop = np.empty((self.H, self.W, 3), np.float32)
        backdrop[:] = BG
        backdrop += glow_part[..., None] * np.array([9.0, 12.0, 9.0], np.float32)
        noise = np.random.default_rng(7).normal(0, 0.9, (self.H, self.W, 1)).astype(np.float32)   # breaks up banding
        self.backdrop = np.clip(backdrop + noise, 0, 255).astype(np.uint8)

    def add(self, *args, **kwargs):
        scene = Scene(*args, **kwargs)
        self.scenes.append(scene)
        return scene

    # ---- narration helpers
    def spoken(self, line):
        return self.words[line]['seconds']

    def word(self, scene, line, word, nth=0, default=None):
        """Scene time at which a word of one of the scene's lines starts."""
        offset = dict(scene.lines)[line]
        found = [w for w in self.words[line]['words'] if w['word'].strip('.,:;!?').lower() == word.lower()]
        if len(found) <= nth:
            if default is None:
                raise KeyError(f'{word!r} not in line {line}')
            return offset + default
        return offset + found[nth]['start']

    def scene_at(self, seconds):
        for scene in self.scenes:
            if scene.start <= seconds < scene.end:
                return scene
        return self.scenes[-1]

    def frame(self, number):
        seconds = number / FPS
        scene = self.scene_at(seconds)
        if getattr(scene, 'plain', False):    # the logo stings stand on plain black
            canvas = np.zeros((self.H, self.W, 3), np.uint8)
        else:
            canvas = self.backdrop.copy()
        local = seconds - scene.start
        scene.draw(canvas, local, scene)
        if scene.chrome:
            self.chrome(canvas, scene, local)
        if getattr(scene, 'chapter_start', False) and local < 1.0:
            self.chapter_card(canvas, scene, local)
        if scene.flash and local < 0.21:
            grow = 1.0 + 0.03 * (1 - ease_out(local / 0.21))
            matrix = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), 0, grow)
            canvas = cv2.warpAffine(canvas, matrix, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if scene.flash and local < 3.5 / FPS:
            strength = (0.16, 0.09, 0.04, 0.0)[min(3, int(local * FPS + 0.5))]
            canvas[:] = np.clip(canvas.astype(np.float32) + 255 * strength, 0, 255).astype(np.uint8)
        return canvas

    def chrome(self, canvas, scene, local):
        """The STEERIX logo in a corner and the chapter's name."""
        if getattr(scene, 'scrim', False):
            band = 150 if self.fmt == 'long' else 260
            shade = (1 - np.linspace(0.62, 0.0, band, dtype=np.float32) ** 1.0)[:, None, None]
            canvas[:band] = (canvas[:band] * shade).astype(np.uint8)
        if self.fmt == 'long':
            put(canvas, sources.steerix_logo(24), 64, 46, 0.92)
            if scene.chapter:
                number, name = scene.chapter
                layer = text_layer(name, 'mono', 19, GREY, spacing=2.2)
                put(canvas, layer, self.W - 64, 49, 1.0, 'rt')
                put(canvas, text_layer(number, 'mono', 19, GREEN, spacing=2.2), self.W - 64 - layer[1].shape[1] - 14, 49, 1.0, 'rt')
        elif local >= getattr(scene, 'logo_from', 0.0):   # (a scene may bring the logo in itself, with the brand animation)
            put(canvas, sources.steerix_logo(30), self.W / 2, 186, 0.92, 'ct')

    def chapter_card(self, canvas, scene, local):
        """The chapter's number and name, large over the dimmed first moments of its first scene."""
        number, name = scene.chapter
        shown = min(ease_out(ramp(local, 0.0, 0.16)), 1 - ease(ramp(local, 0.62, 0.95)))
        if shown <= 0:
            return
        canvas[:] = (canvas * (1 - 0.72 * shown)).astype(np.uint8)
        lift = (1 - ease_out(ramp(local, 0.0, 0.2))) * 40 - ease(ramp(local, 0.62, 0.95)) * 50
        size = 150 if self.fmt == 'long' else 112
        put(canvas, text_layer(number, 'mono', int(size * 0.3), GREEN, spacing=6), self.W / 2, self.H / 2 - size * 0.62 + lift, shown, 'cb')
        put(canvas, text_layer(name, 'semibold', size, WHITE, spacing=4), self.W / 2, self.H / 2 + lift, shown, 'cm')
        width = int(260 * ease_out(ramp(local, 0.05, 0.35)))
        box(canvas, self.W / 2 - width / 2, self.H / 2 + size * 0.62 + lift, width, 5, GREEN, shown)

    def timeline(self):
        scenes = []
        for scene in self.scenes:
            scenes.append(dict(name=scene.name, start=round(scene.start, 4), end=round(scene.end, 4), chapter=scene.chapter,
                               bar=scene.bar, lines=[dict(id=line, start=round(scene.start + offset, 4), text=self.words[line]['text'],
                                                          seconds=self.words[line]['seconds']) for line, offset in scene.lines],
                               sounds=[dict(kind=kind, time=round(scene.start + at, 4)) for kind, at in getattr(scene, 'sounds', [])],
                               music=getattr(scene, 'music', None), chapter_start=getattr(scene, 'chapter_start', False)))
        return dict(format=self.fmt, fps=FPS, size=[self.W, self.H], seconds=round(self.seconds, 4), frames=self.frames, scenes=scenes)


def render_range(args):
    fmt, first, last, path = args
    film = Film(fmt)
    out = encoder(path, film.W, film.H)
    for number in range(first, last):
        out.stdin.write(film.frame(number).tobytes())
    out.stdin.close()
    if out.wait():
        raise RuntimeError('ffmpeg failed')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('format', choices=sorted(SIZES))
    parser.add_argument('--timeline', action='store_true')
    parser.add_argument('--stills')
    parser.add_argument('--range', nargs=2, type=int)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--out')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--scale', type=float, default=1.0, help='shrink stills (looking at many at once)')
    args = parser.parse_args()
    film = Film(args.format)
    folder = WORK / args.format
    folder.mkdir(parents=True, exist_ok=True)
    if args.timeline:
        (folder / 'timeline.json').write_text(json.dumps(film.timeline(), indent=1, ensure_ascii=False))
        for scene in film.scenes:
            print(f'{scene.start:7.2f} {scene.end:7.2f} {scene.seconds:6.2f}  {scene.name:14s} {[line for line, _ in scene.lines]}')
        print(f'{film.seconds:.2f} s, {film.frames} frames')
    if args.stills:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for value in args.stills.split(','):
            if value.startswith('@'):   # @scene+seconds
                name, _, offset = value[1:].partition('+')
                seconds = next(s for s in film.scenes if s.name == name).start + float(offset or 0)
            else:
                seconds = float(value)
            started = time.time()
            picture = film.frame(int(round(seconds * FPS)))
            if args.scale != 1.0:
                picture = cv2.resize(picture, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f'{args.format}_{seconds:07.2f}.jpg'), picture, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f'{seconds:.2f} s ({film.scene_at(seconds).name}) in {time.time() - started:.2f} s')
    if args.range:
        render_range((args.format, args.range[0], args.range[1], args.out))
    if args.all:
        parts = folder / 'parts'
        parts.mkdir(exist_ok=True)
        step = math.ceil(film.frames / args.workers / FPS) * FPS   # whole seconds: every part starts on a key picture
        jobs = [(args.format, first, min(film.frames, first + step), str(parts / f'part_{index:02d}.mp4'))
                for index, first in enumerate(range(0, film.frames, step))]
        with multiprocessing.Pool(args.workers) as pool:
            done = pool.map(render_range, jobs)
        listing = parts / 'list.txt'
        listing.write_text(''.join(f"file '{path}'\n" for path in done))
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart',
                        args.out], check=True)
        print(f'{film.frames} frames -> {args.out}')


if __name__ == '__main__':
    main()
