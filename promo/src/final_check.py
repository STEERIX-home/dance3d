"""Checks of a finished film file: picture and sound lengths, loudness, black or frozen stretches, and whether the
narration can still be understood in the mix (Whisper listens to the finished sound and the words are compared with the script).

    final_check.py FILM.mp4 TIMELINE.json OUT.json      (film-audio venv, -I)
"""
import json
import os
import re
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')
film, timeline, out = sys.argv[1:4]
timeline = json.load(open(timeline))


def run(*command):
    return subprocess.run(command, capture_output=True, text=True)


probe = json.loads(run('ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', film).stdout)
video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
report = dict(file=film, seconds=float(probe['format']['duration']), size_mb=round(int(probe['format']['size']) / 1e6, 1),
              video=dict(codec=video['codec_name'], width=video['width'], height=video['height'], rate=video['r_frame_rate'],
                         frames=int(video['nb_frames']), seconds=float(video['duration'])),
              audio=dict(codec=audio['codec_name'], rate=audio['sample_rate'], channels=audio['channels'], seconds=float(audio['duration'])),
              planned=dict(frames=timeline['frames'], seconds=timeline['seconds']))
problems = []
if report['video']['frames'] != timeline['frames']:
    problems.append(f'{report["video"]["frames"]} frames, planned {timeline["frames"]}')
if abs(report['audio']['seconds'] - report['video']['seconds']) > 0.06:
    problems.append(f'sound {report["audio"]["seconds"]:.2f} s, picture {report["video"]["seconds"]:.2f} s')
loud = run('ffmpeg', '-hide_banner', '-nostats', '-i', film, '-af', 'ebur128=peak=true', '-f', 'null', '-').stderr
summary = loud[loud.rfind('Summary:'):]
report['loudness'] = dict(integrated_lufs=float(re.search(r'I:\s+(-?[\d.]+) LUFS', summary).group(1)),
                          range_lu=float(re.search(r'LRA:\s+(-?[\d.]+) LU', summary).group(1)),
                          true_peak_dbfs=float(re.search(r'Peak:\s+(-?[\d.]+) dBFS', summary).group(1)))
if report['loudness']['true_peak_dbfs'] > -0.9:
    problems.append(f'true peak {report["loudness"]["true_peak_dbfs"]} dBFS')
detect = run('ffmpeg', '-hide_banner', '-nostats', '-i', film, '-vf', 'blackdetect=d=0.5:pix_th=0.06,freezedetect=n=0.001:d=1.2', '-an', '-f', 'null', '-').stderr
def scene_at(seconds):
    return next((scene['name'] for scene in timeline['scenes'] if scene['start'] - 0.05 <= seconds < scene['end']), timeline['scenes'][-1]['name'])


# the logo scenes are a logo standing still on black: there a dark or still picture is what is wanted
report['black'] = [dict(start=float(a), end=float(b), scene=scene_at(float(a))) for a, b in re.findall(r'black_start:([\d.]+) black_end:([\d.]+)', detect)]
report['frozen'] = [dict(start=float(a), scene=scene_at(float(a))) for a in re.findall(r'freeze_start: ([\d.]+)', detect)]
for kind in ('black', 'frozen'):
    for found in report[kind]:
        if found['scene'] not in ('logo', 'outro'):
            problems.append(f'{kind} picture at {found["start"]:.1f} s in scene {found["scene"]}')
silence = run('ffmpeg', '-hide_banner', '-nostats', '-i', film, '-af', 'silencedetect=n=-50dB:d=1.5', '-vn', '-f', 'null', '-').stderr
report['silent'] = [dict(start=float(a), scene=scene_at(float(a))) for a in re.findall(r'silence_start: ([\d.]+)', silence)]
for found in report['silent']:
    if found['scene'] not in ('logo', 'outro'):
        problems.append(f'no sound for 1.5 s or more from {found["start"]:.1f} s in scene {found["scene"]}')
# the narration, heard back from the finished mix
from faster_whisper import WhisperModel
model = WhisperModel(os.path.expanduser('~/.local/opt/film-audio/downloads/whisper-small.en'), device='cpu', compute_type='int8', cpu_threads=6)
segments, _ = model.transcribe(film, language='en', beam_size=5, condition_on_previous_text=False, vad_filter=True)
heard = ' '.join(s.text.strip() for s in segments)
# the recognizer writes numbers as figures and spells the American way: bring both texts to one form before comparing
SAME = (('76', 'seventy six'), ('15', 'fifteen'), ('49', 'forty nine'), ('1 5', 'one and a half percent'), ('0 85', 'zero point eight five'),
        ('meters', 'metres'), ('colors', 'colours'), ('color', 'colour'))


def norm(text):
    text = ' ' + ' '.join(re.sub(r"[^a-z0-9' ]+", ' ', text.lower().replace('-', ' ')).split()) + ' '
    for heard_as, written in SAME:
        text = text.replace(f' {heard_as} ', f' {written} ')
    return text.split()


script = norm(' '.join(line['text'] for scene in timeline['scenes'] for line in scene['lines']))
got = norm(heard)
d = list(range(len(got) + 1))
for i in range(1, len(script) + 1):
    prev, d[0] = d[0], i
    for j in range(1, len(got) + 1):
        cur = d[j]
        d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (script[i - 1] != got[j - 1]))
        prev = cur
report['narration'] = dict(words=len(script), word_error_rate=round(d[len(got)] / max(1, len(script)), 3), heard=heard)
if report['narration']['word_error_rate'] > 0.08:
    problems.append(f'narration hard to understand in the mix: word error rate {report["narration"]["word_error_rate"]}')
report['problems'] = problems
json.dump(report, open(out, 'w'), indent=1)
print(json.dumps({k: report[k] for k in ('seconds', 'video', 'audio', 'loudness', 'black', 'frozen', 'silent', 'problems')}, indent=1))
print('narration WER', report['narration']['word_error_rate'])
