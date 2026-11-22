"""Pick one take per music cue, fit it to its dance's tempo, cut it at a bar start and level it.

    music_cues.py qa        (film-audio venv, -I)   listen-proxy numbers of every raw take -> work/music/qa.json
    music_cues.py make      choose, stretch, cut, level -> work/music/<cue>.wav, work/music/cues.json, work/music/report.json

A take is chosen by numbers only (nobody listens here): no voice, a steady beat, a tempo that two different tempo
estimates agree on and that is close to the wanted one. The wanted bar lengths come from the dances themselves (their
fitted bar grids), so that after the stretch one bar of music lasts exactly one bar of dancing.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

FA = Path.home() / '.local/opt/film-audio'
WORK = Path('/srv/training/jobs/20261116-041650-dance-promo-video-0df0/outputs/work')
MUSIC = WORK / 'music'
sys.path.insert(0, str(FA / 'tools'))
SR = 48000


def bar_of(beats_file):
    record = json.loads(Path(beats_file).read_text())
    bars = np.asarray(record['bars'], float)
    return float(np.polyfit(np.arange(len(bars)), bars, 1)[0])


# the bar length (seconds) each cue must have, and where each group of scenes starts in its cue (whole bars after the cue's start)
TARGET = {'main150': bar_of('/srv/training/jobs/20261115-120933-dance-challenge-48c1/outputs/work/beats.json'),
          'duo129': bar_of(WORK / 'beats/duo_neon.json'),
          'robot120': bar_of(WORK / 'beats/dance09.json')}
GROUPS = {'main': ('main150', 0), 'main2': ('main150', 4), 'duo': ('duo129', 0), 'robot': ('robot120', 0), 'robot2': ('robot120', 8),
          'short:duo': ('duo129', 4)}


def track(path, seconds):
    """(beats, bars, beats per minute) of a sound file, by the challenge program's own beat tracker."""
    record = MUSIC / 'raw' / (Path(path).stem + '.beats.json')
    subprocess.run([sys.executable, '-I', '/srv/training/projects/dance-challenge/beats.py', '--music', str(path), '--seconds', f'{seconds:.3f}',
                    '--output', str(record)], check=True, capture_output=True)
    data = json.loads(record.read_text())
    return np.asarray(data['beats']), np.asarray(data['bars']), data['bpm']


def second_tempo(path, near):
    """A second opinion on the tempo (librosa's estimate, looking near the wanted tempo)."""
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    envelope = librosa.onset.onset_strength(y=y, sr=sr)
    return float(np.atleast_1d(librosa.feature.tempo(onset_envelope=envelope, sr=sr, start_bpm=near))[0])


def bar_phase(path, origin, beat, count):
    """Which beat in four starts the bars (0..3, counted from the grid's first beat): where the harmony changes most
    (chords change on the one) and where the low notes hit hardest. Returns (phase, scores of the four phases)."""
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    hop = 512
    grid = origin + beat * np.arange(count)
    grid = grid[(grid > 0.05) & (grid < len(y) / sr - beat)]
    first = int(round((grid[0] - origin) / beat))
    frames = librosa.time_to_frames(grid, sr=sr, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    per_beat = librosa.util.sync(chroma, frames, aggregate=np.median)       # column k + 1 = from grid beat k to k + 1
    per_beat = per_beat / (np.linalg.norm(per_beat, axis=0, keepdims=True) + 1e-9)
    change = 1 - np.sum(per_beat[:, :-1] * per_beat[:, 1:], axis=0)          # entry k = change of harmony at grid beat k
    low = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))[:14].sum(0)    # below about 150 Hz
    rise = np.maximum(np.diff(low, prepend=low[0]), 0)
    hit = np.array([rise[max(0, f - 2):f + 3].max() for f in frames])
    score = np.zeros(4)
    for values, weight in ((change[:len(grid)], 1.0), (hit, 0.5)):
        by_phase = np.array([values[(p - first) % 4::4].mean() for p in range(4)])
        score += weight * (by_phase - by_phase.mean()) / (by_phase.std() + 1e-9)
    return int(score.argmax()), [round(float(v), 2) for v in score]


def qa():
    takes = sorted((MUSIC / 'raw').glob('*.wav'))
    subprocess.run([str(FA / 'venv/bin/python'), '-I', str(FA / 'tools/check_music.py'), *map(str, takes), '--json', str(MUSIC / 'qa.json')], check=True)


def fit(cue, take, info, bar):
    """Stretch one take to the wanted bar length and cut it at a bar start. Returns (record, None) or (None, why not)."""
    import librosa
    import master_cue
    beat = bar / 4
    fitted = MUSIC / f'{cue}_fitted.wav'
    ratio, rounds = info['stretch'], []
    for _ in range(3):   # stretch; measure the beat of the result; correct what is left
        master_cue.master(dict(src=str(take), out=str(fitted), tempo=ratio, bass_mono_hz=120, width=0.9, lufs=-18, tp=-1.5))
        times, _, _ = track(fitted, sf.info(fitted).duration)
        period, origin = np.polyfit(np.arange(len(times)), times, 1)
        left = min((period / (beat * m) for m in (0.5, 1.0, 2.0)), key=lambda r: abs(np.log(r)))   # > 1: still too slow
        rounds.append(dict(stretch=round(float(ratio), 5), beat_left_over=round(float(left - 1), 5)))
        if abs(left - 1) > 0.02:
            return None, f'after the stretch the beat is {left - 1:+.1%} off: the tempo of this take is not clear'
        if abs(left - 1) < 3e-4:
            break
        ratio *= left
    sound, _ = sf.read(fitted, dtype='float64', always_2d=True)
    origin = float(origin % beat)                       # a grid of the wanted beat through the tracked beats
    count = int((len(sound) / SR - origin) / beat)
    phase, phase_scores = bar_phase(fitted, origin, beat, count)
    first_bar = origin + phase * beat
    grid = first_bar + np.arange(int((len(sound) / SR - first_bar) / bar) - 1) * bar
    energy = np.array([np.sqrt(np.mean(sound[int(g * SR):int((g + bar) * SR)] ** 2)) for g in grid])
    strong = np.flatnonzero(energy > 0.8 * np.median(energy))
    start = float(grid[strong[0]]) if len(strong) else float(grid[0])
    final = MUSIC / f'{cue}.wav'
    master_cue.master(dict(src=str(fitted), out=str(final), start=start, end=len(sound) / SR - 0.5, fade_in=0.004, fade_out=1.0, lufs=-18, tp=-1.5))
    # how far the finished cue's beats are from the grid: by the same tracker, and by another one (librosa)
    check, _, _ = track(final, sf.info(final).duration)
    off = np.abs((check / beat) - np.round(check / beat)) * beat
    y, sr = librosa.load(final, sr=22050, mono=True)
    _, other_beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=240.0 / bar, tightness=100, units='time')
    other_off = (other_beats / beat - np.round(other_beats / beat)) * beat
    middle, spread = float(np.median(other_off)), float(np.percentile(np.abs(other_off - np.median(other_off)), 90))
    record = dict(file=str(final), bar=bar, bar_start=0.0, take=take.name, seconds=round(sf.info(final).duration, 2), cut_at=round(start, 3),
                  rounds=rounds, bar_phase=phase, bar_phase_scores=phase_scores,
                  beats_off_grid_s=dict(median=round(float(np.median(off)), 4), p90=round(float(np.percentile(off, 90)), 4)),
                  second_tracker_off_grid_s=dict(median=round(middle, 4), spread=round(spread, 4)), **info)
    if np.percentile(off, 90) > 0.05:
        return None, f'the beats wander {np.percentile(off, 90):.3f} s from a steady grid'
    if abs(middle) > 0.07 or spread > 0.06:
        return None, f'a second beat tracker puts the beats elsewhere (off by {middle:+.3f} s, spread {spread:.3f} s)'
    return record, None


def make():
    qa_rows = {Path(r['file']).name: r for r in json.loads((MUSIC / 'qa.json').read_text())}
    chosen, report = {}, {}
    for cue, bar in TARGET.items():
        wanted = 240.0 / bar
        ranked = []
        for take in sorted((MUSIC / 'raw').glob(f'{cue}_s*.wav')):
            seconds = sf.info(take).duration
            times, bars, bpm = track(take, seconds)
            period = float(np.polyfit(np.arange(len(times)), times, 1)[0])
            measured = 60.0 / period
            other = second_tempo(take, 120.0)
            agree = min(abs(np.log(other * m / measured)) for m in (0.5, 1.0, 2.0)) < 0.03
            ratio = min((wanted / (measured * m) for m in (0.5, 1.0, 2.0)), key=lambda r: abs(np.log(r)))
            steady = float(np.std(np.diff(times)) / period)
            row = qa_rows.get(take.name, {})
            voice = max((float(v.rsplit(' ', 1)[1]) for v in row.get('vocal_max', [])), default=0.0)
            usable = bool(agree and abs(np.log(ratio)) < 0.08 and voice <= 0.2)
            score = abs(np.log(ratio)) * 12 + steady * 4 + voice * 3 + (0 if usable else 10)
            ranked.append((score, take, dict(measured_bpm=round(measured, 2), second_opinion_bpm=round(other, 2), tempos_agree=bool(agree),
                                             stretch=round(ratio, 4), beat_unsteadiness=round(steady, 4), voice=round(voice, 3), usable=usable,
                                             tags=row.get('tags', [])[:5], key=row.get('key', [''])[0])))
        ranked.sort(key=lambda item: item[0])
        report[cue] = [dict(take=t.name, score=round(s, 3), **info) for s, t, info in ranked]
        for position, (_, take, info) in enumerate(ranked):
            if not info['usable']:
                report[cue][position]['left_out'] = 'tempo not clear or too far from the wanted one, or a voice in it'
                continue
            record, why = fit(cue, take, info, bar)
            if record:
                chosen[cue] = record
                print(cue, json.dumps(record))
                break
            report[cue][position]['left_out'] = why
            print(cue, take.name, 'left out:', why)
        else:
            print(cue, 'NO USABLE TAKE')
    cues = {}
    for group, (cue, skip) in GROUPS.items():
        if cue in chosen:
            cues[group] = dict(file=chosen[cue]['file'], bar=chosen[cue]['bar'], bar_start=0.0, skip_bars=skip)
    (MUSIC / 'cues.json').write_text(json.dumps(cues, indent=1))
    (MUSIC / 'report.json').write_text(json.dumps(dict(chosen=chosen, ranked=report, targets=TARGET), indent=1))
    return chosen


STYLES = {   # for the made-here tracks: chord roots (MIDI) and whether each chord is minor (one chord a bar), and a character
    'main150': dict(chords=((57, True), (53, False), (48, False), (55, False)), seconds=104, build=4,
                    bass=((2, 0), (6, 0), (10, 0), (14, 7)), arp=16, lead=(0, 3, 7, 10, 7, 3, 12, 10), lead_tone=(1.0, 0.45, 0.2, 0.08)),
    'duo129': dict(chords=((52, True), (48, False), (55, False), (50, False)), seconds=56, build=0,
                   bass=((0, 0), (3, 12), (6, 0), (8, 0), (11, 12), (14, 7)), arp=8, lead=(7, 10, 12, 10, 7, 3, 0, 3), lead_tone=(1.0, 0.3, 0.3, 0.1)),
    'robot120': dict(chords=((48, False), (55, False), (57, True), (53, False)), seconds=72, build=0,
                     bass=((0, 0), (2, 12), (8, 0), (10, 12), (14, 7)), arp=8, lead=(12, 7, 4, 7, 12, 14, 16, 12), lead_tone=(1.0, 0.0, 0.35, 0.0, 0.12))}


def synth(cue, bar):
    """A dance track made here from sine waves and noise, exactly at the wanted tempo: kick on every beat, clap on two
    and four, hats, a bass line, soft chords, a plucked pattern and a short tune. Eight-bar blocks: a crash opens each
    block, the tune plays in its second half, and its last bar drops the kick and rises into the next block."""
    from scipy.signal import butter, fftconvolve, sosfilt
    style = STYLES[cue]
    rng = np.random.default_rng(sum(map(ord, cue)))
    beat, six = bar / 4, bar / 16
    bars = int(style['seconds'] / bar)
    length = int(round(bars * bar * SR))
    dry, wet = np.zeros((length + 2 * SR, 2)), np.zeros((length + 2 * SR, 2))     # wet: what gets the room sound
    hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)

    def add(bus, at, mono, gain=1.0, pan=0.0):
        a = int(round(at * SR))
        if a >= length or a < 0:
            return
        mono = mono[:len(bus) - a] * gain
        bus[a:a + len(mono), 0] += mono * np.sqrt((1 - pan) / 2)
        bus[a:a + len(mono), 1] += mono * np.sqrt((1 + pan) / 2)

    def tone(freq, seconds, decay, harmonics=(1.0, 0.3, 0.1), attack=0.004):
        t = np.arange(int(seconds * SR)) / SR
        wave = sum(w * np.sin(2 * np.pi * freq * (k + 1) * t) for k, w in enumerate(harmonics) if w)
        return wave * np.minimum(t / attack, 1) * np.exp(-t * decay)

    def noise(seconds, low=None, high=None, decay=0.0):
        n = rng.normal(0, 1, int(seconds * SR))
        if low and high:
            n = sosfilt(butter(2, [low, high], 'bandpass', fs=SR, output='sos'), n)
        elif low:
            n = sosfilt(butter(2, low, 'highpass', fs=SR, output='sos'), n)
        return n * np.exp(-np.arange(len(n)) / SR * decay)

    t = np.arange(int(0.45 * SR)) / SR
    kick = np.sin(2 * np.pi * (46 * t + 60 * (1 - np.exp(-t * 38)) / 38)) * np.exp(-t * 11) * np.minimum(t / 0.002, 1)
    hat, open_hat = noise(0.07, low=7500, decay=75), noise(0.3, low=6500, decay=14)
    clap, crash = noise(0.2, 900, 6000, decay=24), noise(1.6, low=4000, decay=3.2)
    for index in range(bars):
        at = index * bar
        block = (index - style['build']) % 8 if index >= style['build'] else None      # place inside the eight-bar block
        building = index < style['build']
        last = block == 7
        root, minor = style['chords'][index % 4]
        chord = (root, root + (3 if minor else 4), root + 7, root + 12)
        if block == 0:
            add(wet, at, crash, 0.07)
        # drums
        for b in range(4):
            if not (last and b >= 2):
                add(dry, at + b * beat, kick, 0.44)
            if b in (1, 3) and not (building and index == 0):
                add(wet, at + b * beat, clap, 0.14)
        for k in range(16):
            when = at + k * six
            if k % 4 == 2:
                add(dry, when, hat, 0.075, 0.3)                      # the off-beat hat
            elif k % 2 == 1 and not building:
                add(dry, when, hat, 0.026, -0.3)
            if last and k >= 8:                                       # the kick's place is taken by a rising roll of claps
                add(wet, when, clap, 0.03 + 0.012 * (k - 8))
        if index % 2 == 1 and not last:
            add(dry, at + 3.5 * beat, open_hat, 0.04, 0.2)
        if last or (building and index == style['build'] - 1):        # a rising rush of noise into the next block
            n = int(bar * SR)
            add(wet, at, noise(bar, low=2500) * (np.arange(n) / n) ** 2, 0.05)
        # bass
        for k, step in style['bass']:
            if not (last and k >= 8):
                add(dry, at + k * six, tone(hz(root - 12 + step), six * 1.9, 5.0, (1.0, 0.55, 0.28, 0.12)), 0.17)
        # soft chords, stepping back under each kick
        if not (building and index < 2):
            for k, note in enumerate(chord[:3]):
                pad = tone(hz(note + 12), bar * 1.02, 0.5, (1.0, 0.25, 0.12, 0.05), attack=0.2)
                pad *= 1 - 0.55 * np.exp(-((np.arange(len(pad)) / SR) % beat) / 0.09)
                add(wet, at, pad, 0.07, (-0.7, 0.0, 0.7)[k])
        # the plucked pattern
        if not (building and index < 2):
            count = style['arp']
            for k in range(count):
                note = chord[(0, 2, 1, 2, 3, 2, 1, 2)[k % 8]] + 24
                add(wet, at + k * bar / count, tone(hz(note), 0.4, 8.0, (1.0, 0.4, 0.16)), 0.075 if k % 2 == 0 else 0.05, 0.6 if k % 2 else -0.6)
        # the tune: the second half of every block, two bars long, in the scale of the first chord
        if block is not None and block >= 4 and not last:
            key = style['chords'][0][0]
            half = (block - 4) % 2
            for k in range(4):
                step = style['lead'][half * 4 + k]
                when = at + (k * 4 + (2 if k % 2 else 0)) * six      # a little off the beat every other note
                add(wet, when, tone(hz(key + 24 + step), beat * 1.2, 4.0, style['lead_tone'], attack=0.006), 0.1, 0.15)
    # the room: a short decaying wash of noise that the wet part is played through
    tail = int(0.7 * SR)
    room = rng.normal(0, 1, (tail, 2)) * np.exp(-np.arange(tail) / SR * 7.0)[:, None]
    room = sosfilt(butter(2, [300, 7000], 'bandpass', fs=SR, output='sos'), room, axis=0)
    room /= np.sqrt((room ** 2).sum(0)).mean()
    wash = np.stack([fftconvolve(wet[:, c], room[:, c])[:len(wet)] for c in (0, 1)], axis=1)
    mix = dry + wet + 0.16 * wash
    raw = MUSIC / f'{cue}_made_here_raw.wav'
    sf.write(raw, mix[:length], SR, subtype='PCM_24')
    import master_cue
    final = MUSIC / f'{cue}.wav'
    master_cue.master(dict(src=str(raw), out=str(final), fade_in=0.004, fade_out=1.0, lufs=-18, tp=-1.5))
    return dict(file=str(final), bar=bar, bar_start=0.0, take='made here (sine waves and noise)', seconds=round(length / SR, 2))


def fallback():
    """Fill in a made-here track for every cue that has no chosen take."""
    report_file = MUSIC / 'report.json'
    report = json.loads(report_file.read_text()) if report_file.exists() else dict(chosen={}, ranked={}, targets=TARGET)
    for cue, bar in TARGET.items():
        if cue not in report['chosen'] or (len(sys.argv) > 2 and sys.argv[2] in ('all', cue)):
            report['chosen'][cue] = synth(cue, bar)
            print(cue, 'made here:', report['chosen'][cue]['seconds'], 's')
    cues = {group: dict(file=report['chosen'][cue]['file'], bar=report['chosen'][cue]['bar'], bar_start=0.0, skip_bars=skip)
            for group, (cue, skip) in GROUPS.items() if cue in report['chosen']}
    (MUSIC / 'cues.json').write_text(json.dumps(cues, indent=1))
    report_file.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    {'qa': qa, 'make': make, 'fallback': fallback}[sys.argv[1]]()
