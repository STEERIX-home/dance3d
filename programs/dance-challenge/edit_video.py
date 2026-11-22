"""The finished challenge video: the shots of the cut plan one after another with the plan's effects,
the title, the camera names and the music.

    edit_video.py --plan plan.json --finished DIR --output challenge.mp4 [--music music.m4a] [--title TEXT]
                  [--effects normal|strong|none] [--no-labels] [--compare compare.mp4 --scene DIR]

--finished holds one folder of full-size frames per shot (finish_pair.py --frames-folder); frame k of
a shot's folder is frame (window start + k - 1) of the video. Every effect is worked out from the
plan (cut_plan.py), never chosen here:
  flash    a white flash that fades over five frames, from the first frame of the new shot
  zoom     the new shot starts 10 % (strong: 16 %) larger and settles in under half a second
  whip     the old shot slides out sideways and the new one slides in, blurred by the speed, over six frames
  glitch   colour channels pulled apart and bands of the picture shifted, for four frames around the cut
  pulse    on a strong beat inside a shot the picture grows by 3 % (strong: 5 %) and settles
  shake    on the strongest such beat the picture shakes for a third of a second
  look     a little more contrast and colour, warm light and cool shadows, darker corners
  text     the title over the first shot, the camera's name for a second after each cut, the hashtag
At the end the picture fades to black and the music fades out (encoder below).
--compare writes [3D guide | shot as made | finished] side by side, for checking.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from paths import korean_font as find_korean_font  # Noto Sans CJK KR wherever the system keeps it

LEVELS = {'normal': dict(punch=0.10, pulse=0.03, shake=18.0, glitch=1.0, look=1.0),
          'strong': dict(punch=0.16, pulse=0.05, shake=28.0, glitch=1.5, look=1.3),
          'none': None}
FLASH = [0.85, 0.55, 0.30, 0.15, 0.05]
ACCENT = {'front': (255, 196, 0), 'left': (0, 200, 255), 'right': (255, 80, 160), 'back': (130, 255, 120)}  # RGB


def encoder(path, width, height, fps, audio, duration):
    """ffmpeg fed with raw frames, as composite.encoder of the two-dancer program but without -shortest: the
    sound is already cut to the video's length, and -shortest dropped the last picture when the music ran on
    (11-15, ADDICTED: 349 of 350 frames). Sound shorter than the video just ends early."""
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
               '-r', f'{fps:g}', '-i', '-']
    if audio:  # the music goes on after the dance ends: fade it out instead of cutting it off
        command += ['-ss', '0', '-t', f'{duration:.4f}', '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
                    '-af', f'afade=t=out:st={max(0, duration - 0.5):.3f}:d=0.5', '-c:a', 'aac', '-b:a', '192k']
    command += ['-c:v', 'libx264', '-crf', '15', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def korean_font(weight, size):
    return find_korean_font(size, weight)


def text_picture(text, font, fill=(255, 255, 255), stroke=0, stroke_fill=(0, 0, 0), shadow=10):
    """RGBA picture of the text with a soft drop shadow."""
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    pad = shadow * 2 + 8
    size = (right - left + 2 * pad, bottom - top + 2 * pad)
    picture = Image.new('RGBA', size, (0, 0, 0, 0))
    if shadow:
        dark = Image.new('RGBA', size, (0, 0, 0, 0))
        ImageDraw.Draw(dark).text((pad - left + shadow // 2, pad - top + shadow // 2), text, font=font, fill=(0, 0, 0, 170),
                                  stroke_width=stroke, stroke_fill=(0, 0, 0, 170))
        picture = Image.alpha_composite(picture, dark.filter(ImageFilter.GaussianBlur(shadow / 2)))
    ImageDraw.Draw(picture).text((pad - left, pad - top), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return np.asarray(picture)


def label_picture(text, font, accent):
    """The camera's name on a dark rounded band with a coloured dot."""
    left, top, right, bottom = font.getbbox(text)
    height = bottom - top + 36
    dot = height // 3
    width = right - left + 3 * 22 + dot
    picture = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(picture)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=height // 2, fill=(10, 10, 14, 165))
    draw.ellipse((22, (height - dot) // 2, 22 + dot, (height + dot) // 2), fill=(*accent, 255))
    draw.text((2 * 22 + dot - left, 18 - top), text, font=font, fill=(255, 255, 255, 255))
    return np.asarray(picture)


def lay(frame, picture, centre_x, centre_y, alpha=1.0, scale=1.0):
    """Lay an RGBA picture on the BGR float frame, centred at the given point."""
    if alpha <= 0.003:
        return
    if abs(scale - 1.0) > 1e-3:
        picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = picture.shape[:2]
    x0, y0 = int(round(centre_x - w / 2)), int(round(centre_y - h / 2))
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x0 + w), min(frame.shape[0], y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    part = picture[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32)
    weight = part[..., 3:4] / 255.0 * alpha
    region = frame[fy0:fy1, fx0:fx1]
    region[:] = region * (1 - weight) + part[..., 2::-1] * weight


def zoom_shift(frame, scale, dx=0.0, dy=0.0, centre=(0.5, 0.45)):
    if abs(scale - 1.0) < 1e-4 and abs(dx) < 0.05 and abs(dy) < 0.05:
        return frame
    h, w = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((centre[0] * w, centre[1] * h), 0, scale)
    matrix[:, 2] += (dx, dy)
    return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def blur_sideways(frame, length):
    length = int(length) | 1
    if length < 3:
        return frame
    kernel = np.full((1, length), 1.0 / length, np.float32)
    return cv2.filter2D(frame, -1, kernel, borderType=cv2.BORDER_REFLECT)


def glitch(frame, amount, seed):
    rng = np.random.default_rng(seed)
    h, w = frame.shape[:2]
    out = frame.copy()
    split = int(round(18 * amount))
    if split:
        out[..., 2] = np.roll(frame[..., 2], split, axis=1)
        out[..., 0] = np.roll(frame[..., 0], -split, axis=1)
    for _ in range(int(6 + 6 * amount)):
        top = int(rng.integers(0, h - 20))
        tall = int(rng.integers(12, max(13, int(h * 0.07))))
        out[top:top + tall] = np.roll(out[top:top + tall], int(rng.integers(-60, 61) * amount), axis=1)
    out[::4] *= 1 - 0.12 * min(1.0, amount)  # scan lines
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--finished', required=True, help='folder with one frame folder per shot')
    parser.add_argument('--output', required=True)
    parser.add_argument('--music')
    parser.add_argument('--title', default='')
    parser.add_argument('--effects', default='normal', choices=sorted(LEVELS))
    parser.add_argument('--no-labels', action='store_true')
    parser.add_argument('--compare', help='also write [3D guide | shot as made | finished] here')
    parser.add_argument('--scene', help='3D render folder (needed with --compare)')
    parser.add_argument('--guide-pass', default='pair', help='the render pass the video model followed (H3: pair, SCAIL-2: guide)')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    finished = Path(args.finished)
    fps, total = plan['fps'], plan['frames']
    shots = plan['shots']
    level = LEVELS[args.effects]
    first = cv2.imread(str(finished / shots[0]['name'] / 'frame_0001.png'))
    if first is None:
        raise FileNotFoundError(f'no finished frames in {finished / shots[0]["name"]}')
    height, width = first.shape[:2]

    owner = np.zeros(total + 2, dtype=int)
    for index, shot in enumerate(shots):
        owner[shot['first']:shot['last'] + 1] = index

    def frame_of(index, number):
        """Frame `number` of the video as shot `index` made it (held at the ends of its window)."""
        shot = shots[index]
        start, made = shot['window']
        k = int(np.clip(number - start + 1, 1, made))
        picture = cv2.imread(str(finished / shot['name'] / f'frame_{k:04d}.png'))
        if picture is None:
            raise FileNotFoundError(f'frame {k} of {shot["name"]} is missing')
        return picture.astype(np.float32)

    # what happens where
    cuts = {cut['frame']: cut for cut in plan['cuts']}
    whips = {}
    zoom = np.zeros(total + 2)
    shake = np.zeros((total + 2, 2))
    flash = np.zeros(total + 2)
    glitches = {}
    if level:
        for cut in plan['cuts']:
            c = cut['frame']
            if cut['transition'] == 'whip':
                for k in range(6):
                    whips[c - 3 + k] = (cut, (k + 0.5) / 6)
                flash[c] = max(flash[c], 0.2)
            elif cut['transition'] == 'flash':
                for k, value in enumerate(FLASH):
                    if c + k <= total:
                        flash[c + k] = max(flash[c + k], value)
            elif cut['transition'] == 'zoom':
                for k in range(10):
                    if c + k <= total:
                        zoom[c + k] += level['punch'] * np.exp(-k / 2.5)
            elif cut['transition'] == 'glitch':
                for k, value in zip(range(-1, 3), (0.5, 1.0, 0.7, 0.3)):
                    glitches[c + k] = value * level['glitch']
        for pulse in plan['pulses']:
            p = pulse['frame']
            for k in range(7):
                if p + k <= total:
                    zoom[p + k] += level['pulse'] * pulse['strength'] * np.exp(-k / 2.0)
            if pulse.get('shake'):
                for k in range(8):
                    if p + k <= total:
                        fade = np.exp(-k / 3.0)
                        shake[p + k] = (level['shake'] * np.sin(2.7 * (k + 1)) * fade, level['shake'] * np.cos(3.1 * (k + 1)) * fade)

    # the look: contrast, colour, warm light / cool shadows, darker corners
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.sqrt(((xs - width / 2) / (width / 2)) ** 2 + ((ys - height / 2) / (height / 2)) ** 2) / np.sqrt(2)
    vignette = (1 - 0.28 * np.clip(radius - 0.35, 0, 1) ** 1.5 / 0.65 ** 1.5)[..., None].astype(np.float32)

    def look(frame):
        strength = level['look']
        f = frame / 255.0
        f = 0.5 + (f - 0.5) * (1 + 0.07 * strength)
        lum = f @ np.array([0.114, 0.587, 0.299], np.float32)
        f = lum[..., None] + (f - lum[..., None]) * (1 + 0.12 * strength)
        tone = (lum - 0.5)[..., None] * 0.035 * strength
        f = f + np.concatenate([-tone, np.zeros_like(tone), tone], axis=-1)
        return np.clip(f * (vignette if strength else 1.0), 0, 1) * 255.0

    # text
    title_picture = text_picture(args.title, korean_font('Black', 118), stroke=6, stroke_fill=(20, 20, 30)) if args.title else None
    title_line = text_picture('댄스 챌린지', korean_font('Bold', 58), fill=(255, 214, 64), stroke=3, stroke_fill=(20, 20, 30)) if args.title else None
    # on social sites a hashtag ends at the first space or punctuation mark ("#It'sMe" is only "#It"): letters, digits and _
    hashtag = text_picture('#' + re.sub(r'\W', '', args.title) + '챌린지', korean_font('Bold', 44), stroke=2, stroke_fill=(20, 20, 30),
                           shadow=6) if args.title else None
    labels = {name: label_picture(text, korean_font('Bold', 56), ACCENT[name]) for name, text in
              (('front', '정면'), ('left', '왼쪽'), ('right', '오른쪽'), ('back', '뒤'))}
    title_end = max(shots[0]['last'], int(round(1.2 * fps)))
    title_end = min(title_end, total)

    def texts(frame, number):
        index = owner[number]
        shot = shots[index]
        if title_picture is not None and number <= title_end:
            age, left = number - 1, title_end - number
            alpha = min(1.0, (age + 1) / 5) * min(1.0, (left + 1) / 5)
            scale = 1.0 + 0.25 * max(0.0, 1 - (age + 1) / 5) ** 2
            lay(frame, title_picture, width / 2, height * 0.78, alpha, scale)
            lay(frame, title_line, width / 2, height * 0.78 + 110, alpha)
        elif hashtag is not None:
            alpha = min(1.0, (number - title_end) / 6) * 0.9
            lay(frame, hashtag, width / 2, height - 150, alpha)
        if not args.no_labels and not (title_picture is not None and number <= title_end):
            age = number - shot['first']
            shown = int(round(1.0 * fps))
            if age < shown:
                alpha = min(1.0, (age + 1) / 4) * min(1.0, (shown - age) / 4)
                slide = 60 * (1 - min(1.0, (age + 1) / 4)) ** 2
                picture = labels[shot['direction']]
                lay(frame, picture, 64 + picture.shape[1] / 2 - slide, 170, alpha)

    out = encoder(args.output, width, height, fps, args.music, total / fps)
    compare = None
    if args.compare:
        if not args.scene:
            parser.error('--compare needs --scene')
        compare = encoder(args.compare, (width // 2) * 3, height // 2, fps, args.music, total / fps)
    record = []
    for number in range(1, total + 1):
        index = owner[number]
        made = frame_of(index, number)
        frame = made
        effects = []
        if number in whips:
            cut, progress = whips[number]
            a = frame_of(owner[cut['frame'] - 1], number)
            b = frame_of(owner[cut['frame']], number)
            eased = progress * progress * (3 - 2 * progress)
            moved = int(round(eased * width))
            if cut['whip_to'] == 'left':
                frame = np.concatenate([a[:, moved:], b[:, :moved]], axis=1)
            else:
                frame = np.concatenate([b[:, width - moved:], a[:, :width - moved]], axis=1)
            frame = blur_sideways(frame, 0.22 * width * np.sin(np.pi * progress))
            effects.append('whip')
        if zoom[number] > 1e-4 or np.abs(shake[number]).max() > 0.05:
            frame = zoom_shift(frame, 1.0 + zoom[number], *shake[number])
            effects.append('zoom' if zoom[number] > 1e-4 else 'shake')
        if number in glitches:
            frame = glitch(frame, glitches[number], number)
            effects.append('glitch')
        if level:
            frame = look(frame)
        if flash[number] > 0:
            frame = frame + (255.0 - frame) * flash[number]
            effects.append('flash')
        texts(frame, number)
        if number > total - 8:
            frame = frame * ((total - number) / 8.0)
        final = np.clip(frame + 0.5, 0, 255).astype(np.uint8)
        out.stdin.write(final.tobytes())
        if compare:
            shot = shots[index]
            guide = cv2.imread(str(Path(args.scene) / shot['view'] / args.guide_pass / f'frame_{number:04d}.jpg'))
            if guide is None:
                guide = np.zeros((height, width, 3), np.uint8)
            small = [cv2.resize(picture, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                     for picture in (guide, np.clip(made, 0, 255).astype(np.uint8), final)]
            compare.stdin.write(np.concatenate(small, axis=1).tobytes())
        record.append(dict(frame=number, shot=shots[index]['name'], effects=effects))
    for process in filter(None, (out, compare)):
        process.stdin.close()
        if process.wait():
            raise RuntimeError('ffmpeg failed')
    Path(args.output).with_suffix('.json').write_text(json.dumps(dict(
        frames=total, fps=fps, size=[width, height], effects=args.effects, title=args.title, labels=not args.no_labels,
        music=args.music, cuts=plan['cuts'], frames_with_effects=sum(1 for r in record if r['effects']), per_frame=record), ensure_ascii=False))
    print(f'{total} frames, {total / fps:.2f} s, {len(plan["cuts"])} cuts -> {args.output}')


if __name__ == '__main__':
    main()
