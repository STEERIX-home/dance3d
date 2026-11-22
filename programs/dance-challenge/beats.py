"""Find the beats of the music: tempo, every beat, which beats start a bar, how strong each beat is.

    beats.py --music music.m4a --seconds 14.4 --output beats.json

The sound is read as one channel at 22050 Hz. Where new sound starts, the spectrum grows (onset
strength = how much the log spectrum rose from one moment to the next, summed over frequencies).
The tempo is the beat period at which that strength repeats best (autocorrelation, with a mild
preference for tempos near 120 per minute, so that half and double tempo are not taken). The beats
are then the moments that lie on strong onsets and keep that period as evenly as possible (dynamic
programming, the method of D. Ellis, 2007). The bar starts are the one beat in four on which the
low notes (kick drum, bass) hit hardest.

Without music (or with silence) the beats are a steady 120 per minute from the start, and the
file says so.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

RATE, HOP, WINDOW = 22050, 256, 1024


def read_sound(path, seconds):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-t', f'{seconds:.3f}', '-ac', '1', '-ar', str(RATE),
                          '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype='<f4').astype(np.float64)


def onset_strength(signal):
    """(all frequencies, low frequencies below 180 Hz), one value per HOP samples."""
    count = 1 + (len(signal) - WINDOW) // HOP
    index = np.arange(WINDOW)[None, :] + HOP * np.arange(count)[:, None]
    spectrum = np.abs(np.fft.rfft(signal[index] * np.hanning(WINDOW), axis=1))
    log = np.log1p(1000.0 * spectrum / max(1e-9, spectrum.max()))
    rise = np.maximum(np.diff(log, axis=0), 0.0)
    low = int(180 * WINDOW / RATE) + 1
    strength, bass = rise[:, 1:].mean(1), rise[:, 1:low].mean(1)
    pad = np.zeros(1)
    return np.concatenate([pad, strength]), np.concatenate([pad, bass])


def tempo(envelope, rate):
    """Beats per minute and the score of every tempo tried."""
    centred = envelope - envelope.mean()
    size = 1 << int(np.ceil(np.log2(2 * len(centred))))
    spectrum = np.fft.rfft(centred, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:len(centred)]
    correlation /= max(1e-9, correlation[0])
    candidates = np.arange(60.0, 200.0, 0.5)
    score = []
    for bpm in candidates:
        lag = 60.0 * rate / bpm
        value = np.interp(lag, np.arange(len(correlation)), correlation) + 0.5 * np.interp(2 * lag, np.arange(len(correlation)), correlation)
        score.append(value * np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.9) ** 2))
    score = np.asarray(score)
    return float(candidates[int(score.argmax())]), score


def track(envelope, period, tightness=100.0):
    """Indexes of the beats: on strong onsets, as evenly spaced as `period` allows."""
    local = gaussian_filter1d(envelope, period / 16.0)
    total, back = np.zeros(len(local)), np.full(len(local), -1)
    low, high = int(round(2 * period)), max(1, int(round(period / 2)))
    for t in range(len(local)):
        first, last = max(0, t - low), t - high
        if last < first:
            total[t] = local[t]
            continue
        before = np.arange(first, last + 1)
        value = total[before] - tightness * np.log((t - before) / period) ** 2
        best = int(value.argmax())
        total[t], back[t] = local[t] + value[best], before[best]
    # the last beat: the best total within the last period, among clear peaks
    end = len(local) - 1
    tail = np.arange(max(0, end - int(round(period))), end + 1)
    beat = int(tail[total[tail].argmax()])
    beats = []
    while beat >= 0:
        beats.append(beat)
        beat = int(back[beat])
    return np.asarray(beats[::-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--music', help='sound file; none: a steady 120 per minute')
    parser.add_argument('--seconds', type=float, required=True, help='length of the video the beats are for')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    record = dict(seconds=args.seconds)
    signal = read_sound(args.music, args.seconds) if args.music else np.zeros(0)
    if len(signal) < RATE * 2 or np.sqrt(np.mean(signal ** 2)) < 1e-3:
        period = 0.5
        beats = np.arange(0.0, args.seconds, period)
        record.update(method='steady 120 per minute: no usable music', bpm=120.0, beats=np.round(beats, 4).tolist(),
                      strength=[0.5] * len(beats), energy=[0.5] * len(beats), bar_phase=0)
    else:
        envelope, bass = onset_strength(signal)
        rate = RATE / HOP
        times = (np.arange(len(envelope)) * HOP + WINDOW / 2) / RATE
        cleaned = np.maximum(envelope - gaussian_filter1d(envelope, rate * 0.5), 0.0)
        bpm, _ = tempo(cleaned, rate)
        index = track(cleaned, 60.0 * rate / bpm)
        beats = times[index]
        # beats before the music starts (a silent opening) are kept: the cuts may use them
        near = maximum_filter1d(cleaned, 5)[index]
        strength = np.clip(near / max(1e-9, np.percentile(near, 95)), 0, 1)
        bass_near = maximum_filter1d(bass, 5)[index]
        phase = int(np.argmax([bass_near[p::4].mean() + 0.5 * near[p::4].mean() if len(near[p::4]) else -1 for p in range(4)]))
        edges = np.concatenate([[0], (index[1:] + index[:-1]) // 2, [len(envelope)]])
        loudness = np.array([np.sqrt(np.mean(signal[a * HOP:max(a + 1, b) * HOP] ** 2)) for a, b in zip(edges[:-1], edges[1:])])
        energy = np.clip(loudness / max(1e-9, np.percentile(loudness, 95)), 0, 1)
        intervals = np.diff(beats)
        record.update(method='onset strength + autocorrelation tempo + dynamic programming beats', bpm=round(bpm, 2),
                      beats=np.round(beats, 4).tolist(), strength=np.round(strength, 3).tolist(),
                      energy=np.round(energy, 3).tolist(), bar_phase=phase,
                      evenness=dict(median_s=round(float(np.median(intervals)), 4), p10_s=round(float(np.percentile(intervals, 10)), 4),
                                    p90_s=round(float(np.percentile(intervals, 90)), 4)) if len(intervals) else None)
    record['bars'] = record['beats'][record['bar_phase']::4]
    Path(args.output).write_text(json.dumps(record, indent=1))
    print(f'{record["bpm"]} per minute, {len(record["beats"])} beats, bars start at beat {record["bar_phase"] + 1}; {record["method"]}')


if __name__ == '__main__':
    main()
