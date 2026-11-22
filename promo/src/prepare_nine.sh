#!/bin/bash
# After the challenge run of dance 09: a text-free edit of its video for the films, and the note that tells the films where the run is.
#   prepare_nine.sh /srv/training/jobs/<challenge job of dance 09>
set -euo pipefail
NINE=$1
J=/srv/training/jobs/20261116-041650-dance-promo-video-0df0
PY=/srv/training/jobs/20261110-235206-rgbmocap-h3-install-17cf/cache/venv/bin/python
cd /srv/training/projects/dance-challenge
nice -n 19 env PYTHONPATH=/srv/training/projects/duo-scene $PY -B edit_video.py --plan $NINE/outputs/work/plan.json --finished $NINE/outputs/work/finished \
  --output $J/outputs/work/clips/challenge09_clean.mp4 --title '' --effects normal --no-labels
python3 -c "
import json
s = json.load(open('$NINE/outputs/results/summary.json'))
json.dump(dict(job='$NINE', name='challenge09', minutes=s['minutes']), open('$J/outputs/work/nine.json', 'w'))
print('nine.json written,', s['minutes'], 'minutes')"
