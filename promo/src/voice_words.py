"""When is each word of the narration spoken? Whisper (small.en, on the processor) listens to every line of a say.py
manifest and writes the word times; it also reports the lines it heard differently from the script.

    voice_words.py VOICE_DIR      (film-audio venv, -I)
"""
import json
import os
import re
import sys
import warnings

warnings.filterwarnings('ignore')
from faster_whisper import WhisperModel

FA = os.path.expanduser('~/.local/opt/film-audio')
folder = sys.argv[1]
manifest = json.load(open(os.path.join(folder, 'manifest.json')))
model = WhisperModel(os.path.join(FA, 'downloads', 'whisper-small.en'), device='cpu', compute_type='int8', cpu_threads=4)
norm = lambda s: re.sub(r"[^a-z0-9' ]+", ' ', s.lower().replace('-', ' ')).split()
out, wrong = {}, 0
for line in manifest['lines']:
    segments, _ = model.transcribe(os.path.join(folder, line['id'] + '.wav'), language='en', word_timestamps=True, beam_size=5,
                                   condition_on_previous_text=False, initial_prompt=line['text'])
    words = [dict(word=w.word.strip(), start=round(w.start, 3), end=round(w.end, 3)) for s in segments for w in s.words]
    heard = ' '.join(w['word'] for w in words)
    same = norm(heard) == norm(line['text'])
    wrong += not same
    out[line['id']] = dict(text=line['text'], seconds=line['duration_s'], heard=heard, same=same, words=words)
    if not same:
        print(f'{line["id"]}: script "{line["text"]}" | heard "{heard}"')
json.dump(out, open(os.path.join(folder, 'words.json'), 'w'), indent=1)
print(f'{len(out)} lines, {wrong} heard differently')
