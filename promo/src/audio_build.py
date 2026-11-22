"""The sound of a dance promo film: narration, music cut to the film's bar grid, a few effects, mixed and levelled.

    audio_build.py long|short        (film-audio venv, -I)

Reads work/FORMAT/timeline.json (film.py --timeline), work/voice/*.wav (say.py) and work/music/cues.json
(music_cues.py: every cue already stretched to its dance's tempo, with the time of one bar start).
Writes work/FORMAT/audio/{mix.wav, mix_no_music.wav, narration.wav, music.wav, effects.wav, report.json} and the .srt.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt

WORK = Path('/srv/training/jobs/20261116-041650-dance-promo-video-0df0/outputs/work')
SR = 48000
fmt = sys.argv[1]
timeline = json.loads((WORK / fmt / 'timeline.json').read_text())
total = timeline['seconds']
length = int(round(total * SR))
out = WORK / fmt / 'audio'
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261116)
db = lambda value: 10 ** (value / 20)


def place(track, sound, seconds, gain=1.0):
    start = int(round(seconds * SR))
    if start >= len(track) or start + len(sound) <= 0:
        return
    a, b = max(0, start), min(len(track), start + len(sound))
    track[a:b] += sound[a - start:b - start] * gain


def stereo(mono, pan=0.0):
    return np.stack([mono * np.sqrt((1 - pan) / 2), mono * np.sqrt((1 + pan) / 2)], axis=1)


# ---------------------------------------------------------------- narration
narration = np.zeros((length, 2))
spoken = []   # (start, end, text)
for scene in timeline['scenes']:
    for line in scene['lines']:
        voice, rate = sf.read(WORK / 'voice' / f'{line["id"]}.wav', dtype='float64')
        assert rate == SR
        place(narration, stereo(voice), line['start'])
        spoken.append((line['start'], line['start'] + len(voice) / SR, line['text'], line['id']))
spoken.sort()

# ---------------------------------------------------------------- effects (made here, no samples)
def whoosh(seconds=0.8):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    noise = rng.normal(0, 1, n)
    sweep = np.zeros(n)
    for k, (low, high) in enumerate(((250, 900), (700, 2600), (1800, 6500))):   # three bands opening one after another
        band = sosfilt(butter(2, [low, high], 'bandpass', fs=SR, output='sos'), noise)
        sweep += band * np.clip((t / seconds - 0.18 * k) / 0.55, 0, 1) ** 2
    envelope = (t / seconds) ** 2.2 * np.clip((seconds - t) / 0.03, 0, 1)
    mono = sweep * envelope
    pan = np.linspace(-0.6, 0.5, n)
    return np.stack([mono * np.sqrt((1 - pan) / 2), mono * np.sqrt((1 + pan) / 2)], axis=1) * 0.5


def hit(weight=1.0):
    n = int(1.6 * SR)
    t = np.arange(n) / SR
    sub = np.sin(2 * np.pi * (38 * t + (110 - 38) * 0.07 * (1 - np.exp(-t / 0.07)))) * np.exp(-t / 0.42)
    click = sosfilt(butter(2, 1800, 'highpass', fs=SR, output='sos'), rng.normal(0, 1, n)) * np.exp(-t / 0.012) * 0.5
    tail_l = sosfilt(butter(2, [300, 5000], 'bandpass', fs=SR, output='sos'), rng.normal(0, 1, n)) * np.exp(-t / 0.35) * 0.10
    tail_r = sosfilt(butter(2, [300, 5000], 'bandpass', fs=SR, output='sos'), rng.normal(0, 1, n)) * np.exp(-t / 0.35) * 0.10
    body = (sub * 0.9 + click) * np.clip(t / 0.002, 0, 1)
    return np.stack([body + tail_l, body + tail_r], axis=1) * 0.6 * weight


def tick():
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    mono = np.sin(2 * np.pi * 1900 * t) * np.exp(-t / 0.012) * 0.25
    return stereo(mono)


effects = np.zeros((length, 2))
made = {'whoosh': whoosh(0.75), 'hit': hit(1.0), 'thump': hit(0.45), 'tick': tick(), 'swish': whoosh(0.32) * 1.5}
events = []
for scene in timeline['scenes']:
    for sound in scene.get('sounds', []):
        place(effects, made[sound['kind']], sound['time'])
        events.append((sound['time'], sound['kind']))
    if scene.get('chapter_start'):   # the chapter's name comes up: a short swish that ends on the cut
        place(effects, made['swish'], scene['start'] - 0.30)
        events.append((round(scene['start'] - 0.30, 3), 'swish'))

# ---------------------------------------------------------------- music on the bar grid
music = np.zeros((length, 2))
cues_file = WORK / 'music/cues.json'
cues = json.loads(cues_file.read_text()) if cues_file.exists() else {}
groups = []   # runs of scenes with one piece of music
for scene in timeline['scenes']:
    info = scene.get('music')
    if not info:
        continue
    if groups and groups[-1]['group'] == info['group'] and abs(groups[-1]['end'] - scene['start']) < 1e-3:
        groups[-1]['end'] = scene['end']
    else:
        groups.append(dict(group=info['group'], start=scene['start'] - info['bars_before'] * info['bar'], begin=scene['start'], end=scene['end'],
                           bar=info['bar']))
used = []
for index, group in enumerate(groups):
    cue = cues.get(f'{fmt}:{group["group"]}') or cues.get(group['group'])
    if not cue:
        continue
    sound, rate = sf.read(cue['file'], dtype='float64', always_2d=True)
    assert rate == SR
    # the cue's bar start `cue["bar_start"]` (+ whole bars of the cue) falls on the group's grid origin
    offset = cue['bar_start'] + cue.get('skip_bars', 0) * cue['bar'] + (group['begin'] - group['start'])
    first = int(round(offset * SR))
    count = int(round((group['end'] - group['begin']) * SR))
    piece = sound[first:first + count].copy()
    if len(piece) < count:
        raise SystemExit(f'{group["group"]}: the cue is too short ({len(sound) / SR:.1f} s, needs {offset + count / SR:.1f} s)')
    fade_in, fade_out = int(0.012 * SR), int(0.22 * SR)
    piece[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    piece[-fade_out:] *= (np.cos(np.linspace(0, np.pi / 2, fade_out)) ** 2)[:, None]
    place(music, piece, group['begin'], db(cue.get('gain_db', 0.0)))
    place(effects, made['thump'], group['begin'], 0.8 if index else 0.0)   # a soft thump where a new piece of music starts
    used.append(dict(group=group['group'], file=cue['file'], begin=round(group['begin'], 3), end=round(group['end'], 3), cue_offset=round(offset, 3),
                     bar=group['bar'], grid_origin=round(group['start'], 4)))

# the music steps back while the narrator speaks
duck = np.ones(length)
low = db(-8.5)
for start, end, _, _ in spoken:
    a, b = int((start - 0.10) * SR), int((end + 0.20) * SR)
    ramp_in, ramp_out = int(0.14 * SR), int(0.35 * SR)
    a0, b1 = max(0, a - ramp_in), min(length, b + ramp_out)
    a, b = max(0, a), min(length, b)
    duck[a:b] = np.minimum(duck[a:b], low)
    duck[a0:a] = np.minimum(duck[a0:a], np.linspace(1, low, a - a0))
    duck[b:b1] = np.minimum(duck[b:b1], np.linspace(low, 1, b1 - b))
music *= duck[:, None]

# ---------------------------------------------------------------- levels
meter = pyln.Meter(SR)


def levelled(track, target):
    if np.abs(track).max() < 1e-6:
        return track
    return track * db(target - meter.integrated_loudness(track))


sys.path.insert(0, str(Path.home() / '.local/opt/film-audio/tools'))
import master_cue  # noqa: E402  (its true-peak limiter: short gain dips at the peaks instead of turning everything down)


def limit(track, target, ceiling=-1.5):
    track, _ = master_cue.tp_limit(track, ceiling - 0.1)
    track = levelled(track, target)          # the limiter took a little loudness: once more
    track, _ = master_cue.tp_limit(track, ceiling - 0.1)
    return track


narration = levelled(narration, -17.0)
if np.abs(music).max() > 0:
    music = levelled(music, -21.0)
effects *= db(-9.0)
report = dict(format=fmt, seconds=total, music=used, effects=sorted(events))
for name, parts in (('mix', (narration, music, effects)), ('mix_no_music', (narration, effects))):
    mix = sum(parts)
    target = -16.0 if name == 'mix' else -17.5
    mix = limit(levelled(mix, target), target)
    sf.write(out / f'{name}.wav', mix, SR, subtype='PCM_24')
    report[name] = dict(lufs=round(float(meter.integrated_loudness(mix)), 2), peak_dbfs=round(float(20 * np.log10(np.abs(mix).max() + 1e-12)), 2))
for name, track in (('narration', narration), ('music', music), ('effects', effects)):
    sf.write(out / f'{name}.wav', np.clip(track, -1, 1), SR, subtype='PCM_24')


def stamp(seconds):
    ms = int(round(seconds * 1000))
    return f'{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}'


with open(out / 'subtitles_en.srt', 'w') as handle:
    for index, (start, end, text, _) in enumerate(spoken, 1):
        handle.write(f'{index}\n{stamp(start)} --> {stamp(end + 0.25)}\n{text}\n\n')
report['narration'] = [dict(id=line, start=round(start, 3), end=round(end, 3), text=text) for start, end, text, line in spoken]
(out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(json.dumps({k: report[k] for k in ('mix', 'mix_no_music')}), '| music pieces:', [u['group'] for u in used], '|', len(spoken), 'lines')
