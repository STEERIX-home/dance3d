#!/bin/bash
# Render one film, build its sound, and join them:  finish.sh long|short [workers]
# Leaves work/FORMAT/final/{film.mp4, film_no_music.mp4, subtitles_en.srt}.
set -euo pipefail
FMT=$1
WORKERS=${2:-5}
J=/srv/training/jobs/20261116-041650-dance-promo-video-0df0
SRC=$J/outputs/src
W=$J/outputs/work/$FMT
PY=/srv/training/jobs/20261110-235206-rgbmocap-h3-install-17cf/cache/venv/bin/python
FA=$HOME/.local/opt/film-audio
mkdir -p $W/final
cd $SRC
nice -n 19 $PY -B film.py $FMT --timeline > $W/timeline.txt
nice -n 19 $FA/venv/bin/python -I $SRC/audio_build.py $FMT
nice -n 19 $PY -B film.py $FMT --all --out $W/silent.mp4 --workers $WORKERS
for KIND in mix mix_no_music; do
  OUT=$W/final/film.mp4
  [ $KIND = mix_no_music ] && OUT=$W/final/film_no_music.mp4
  ffmpeg -v error -nostdin -y -i $W/silent.mp4 -i $W/audio/$KIND.wav -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 256k -movflags +faststart $OUT
done
cp $W/audio/subtitles_en.srt $W/final/subtitles_en.srt
tail -1 $W/timeline.txt
# checks of the finished file: lengths, loudness, black or frozen stretches, narration heard back, music on the beat grid
nice -n 19 $FA/venv/bin/python -I $SRC/final_check.py $W/final/film.mp4 $W/timeline.json $W/final/check.json | tail -4
nice -n 19 $FA/venv/bin/python -I $SRC/music_check.py $FMT
