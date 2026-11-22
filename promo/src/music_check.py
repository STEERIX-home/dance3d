"""Is the music of a finished film on the film's beat grid? For every piece of music: its beats (tracked in the music
track as mixed) against the grid the scenes were cut on.

    music_check.py long|short      (film-audio venv, -I)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

WORK = Path('/srv/training/jobs/20261116-041650-dance-promo-video-0df0/outputs/work')
fmt = sys.argv[1]
audio = WORK / fmt / 'audio'
report = json.loads((audio / 'report.json').read_text())
music, rate = sf.read(audio / 'music.wav', dtype='float64', always_2d=True)
rows = []
for piece in report['music']:
    a, b = int(piece['begin'] * rate), int(piece['end'] * rate)
    part = audio / f'check_{piece["group"]}.wav'
    sf.write(part, music[a:b], rate)
    record = audio / f'check_{piece["group"]}.json'
    subprocess.run([sys.executable, '-I', '/srv/training/projects/dance-challenge/beats.py', '--music', str(part), '--seconds', f'{(b - a) / rate:.3f}',
                    '--output', str(record)], check=True, capture_output=True)
    data = json.loads(record.read_text())
    beat = piece['bar'] / 4
    beats = np.array(data['beats']) + piece['begin'] - piece['grid_origin']
    off = (beats / beat - np.round(beats / beat)) * beat
    # a tracker may follow the off-beats (hats between the kicks): then every beat is half a beat away, which is the same grid
    half = ((beats - beat / 2) / beat - np.round((beats - beat / 2) / beat)) * beat
    on_off_beats = bool(np.median(np.abs(half)) < np.median(np.abs(off)))
    used = half if on_off_beats else off
    rows.append(dict(group=piece['group'], seconds=round((b - a) / rate, 1), wanted_bpm=round(240 / piece['bar'], 2), tracked_bpm=data['bpm'],
                     beats=len(beats), tracker_followed='off-beats' if on_off_beats else 'beats',
                     beat_off_median_s=round(float(np.median(np.abs(used))), 4), beat_off_p90_s=round(float(np.percentile(np.abs(used), 90)), 4)))
    part.unlink()
(audio / 'music_check.json').write_text(json.dumps(rows, indent=1))
for row in rows:
    print(row)
