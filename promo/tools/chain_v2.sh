#!/bin/bash
# After the two-dancer page run has ended and the GPU is free:
#  1. make the music takes (a few minutes, under this job's own GPU lease),
#  2. press the challenge page's button for dance 09 with reze.webp, the way a person would.
J=/srv/training/jobs/20261116-041650-dance-promo-video-0df0
UIPY=/srv/training/jobs/20261110-074853-h3-character-swap-install-77ab/cache/venv/bin/python
FA=$HOME/.local/opt/film-audio
LOG=$J/logs/chain.log
say() { echo "$(date +%H:%M:%S) $*" >> $LOG; }
say "waiting for the two-dancer run to end"
while pgrep -f 'duo_pipeline\.py' > /dev/null; do sleep 15; done
for i in $(seq 30); do grep -qE 'FINAL|ERROR' $J/logs/duo-page-run.log && break; sleep 3; done
say "two-dancer run ended: $(tail -c 300 $J/logs/duo-page-run.log | tr '\n' ' ')"
$HOME/bin/duo-dance stop >> $LOG 2>&1
for i in $(seq 60); do
  $HOME/gpu-dashboard/gpu-agent.py available 2>/dev/null | grep -q '"count": 1' && break
  sleep 5
done
say "GPU: $($HOME/gpu-dashboard/gpu-agent.py available 2>/dev/null | tr -d '\n ')"
ssd-guard check >> $LOG 2>&1 || { say "ssd-guard refused; not starting"; exit 1; }
say "music takes"
timeout 1500 /usr/bin/python3 -B $J/outputs/src/lease_run.py dance-promo-music -- \
  systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=1G \
  $FA/venv-acestep/bin/python -I $FA/tools/acestep_gen.py $J/outputs/src/music_jobs.json $J/outputs/work/music/raw > $J/logs/acestep.log 2>&1
say "music takes ended with $? : $(ls $J/outputs/work/music/raw/*.wav 2>/dev/null | wc -l) files"
for i in $(seq 60); do
  $HOME/gpu-dashboard/gpu-agent.py available 2>/dev/null | grep -q '"count": 1' && break
  sleep 5
done
$HOME/bin/dance-challenge start >> $LOG 2>&1
for i in $(seq 60); do ss -ltn 2>/dev/null | grep -q ':7865 ' && break; sleep 2; done
sleep 4
say "starting the challenge run for dance 09"
$UIPY -B $J/outputs/tools/page_run.py run '09_퇴사할게여   (20261114-124936-dance-joints-f011)' /home/steerix-train-server/바탕화면/reze.webp '' > $J/logs/challenge09-page-run.log 2>&1
say "challenge page run returned: $(tail -c 300 $J/logs/challenge09-page-run.log | tr '\n' ' ')"
