#!/bin/bash
# Put the films where the owner looks for them.
#   deliver.sh draft    a first version on the desktop only, marked as such (something to look at while the rest is made)
#   deliver.sh final    the finished set: the promo timeline on the big disk and a copy on the desktop
set -euo pipefail
KIND=${1:-final}
J=/srv/training/jobs/20261116-041650-dance-promo-video-0df0
W=$J/outputs/work
SRC=$J/outputs/src
PY=/srv/training/jobs/20261110-235206-rgbmocap-h3-install-17cf/cache/venv/bin/python
OUT=$J/outputs/final/댄스3D_홍보영상_20261116
TIMELINE=/media/steerix-train-server/HDD_01/SteerixData/홍보영상_타임라인
NAME=23_2026-11-16_댄스3D_캐릭터_로봇_롱폼_숏폼
DESK=$HOME/바탕화면/댄스3D_홍보영상_20261116
MARK=""; [ $KIND = draft ] && MARK="_1차본"
rm -rf "$OUT"
mkdir -p "$OUT/소리" "$OUT/만들기_스크립트"
cp $W/long/final/film.mp4            "$OUT/STEERIX_댄스3D_롱폼_EN_1080p$MARK.mp4"
cp $W/long/final/film_no_music.mp4   "$OUT/STEERIX_댄스3D_롱폼_EN_1080p_음악없음$MARK.mp4"
cp $W/short/final/film.mp4           "$OUT/STEERIX_댄스3D_숏폼_EN_세로$MARK.mp4"
cp $W/short/final/film_no_music.mp4  "$OUT/STEERIX_댄스3D_숏폼_EN_세로_음악없음$MARK.mp4"
cp $W/long/final/subtitles_en.srt    "$OUT/자막_영문_롱폼.srt"
cp $W/short/final/subtitles_en.srt   "$OUT/자막_영문_숏폼.srt"
for F in long short; do
  K=롱폼; [ $F = short ] && K=숏폼
  for S in narration music effects; do cp $W/$F/audio/$S.wav "$OUT/소리/${K}_$S.wav"; done
done
cp $SRC/*.py $SRC/*.sh $SRC/*.json "$OUT/만들기_스크립트/"
cd $SRC
nice -n 19 $PY -B make_thumbnail.py "$OUT"
nice -n 19 $PY -B make_explain.py "$OUT" $KIND
( cd "$OUT" && sha256sum *.mp4 *.srt *.jpg 설명.html > sha256.txt )
if [ $KIND = final ]; then   # the timeline on the big disk
  mkdir -p "$TIMELINE/$NAME"
  cp -r "$OUT"/. "$TIMELINE/$NAME/"
  if ! grep -q "^## $NAME" "$TIMELINE/목록.md"; then
    { echo; echo "## $NAME"; ( cd "$OUT" && ls *.mp4 | sed 's/^/- /' ); } >> "$TIMELINE/목록.md"
  fi
  ( cd "$TIMELINE/$NAME" && sha256sum -c sha256.txt --quiet ) && echo "the copy on the big disk matches"
fi
rm -rf "$DESK"
mkdir -p "$DESK"
cp -r "$OUT"/. "$DESK/"
( cd "$DESK" && sha256sum -c sha256.txt --quiet ) && echo "the desktop copy matches"
ls -la "$DESK"
