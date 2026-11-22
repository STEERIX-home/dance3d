# 댄스 3D 홍보 영상 (긴 것 · 짧은 것) — 2026-11-16

춤 영상 하나에서 3D 관절을 뽑고, 그 관절로 캐릭터 영상(1인 챌린지, 2인)과 로봇 춤을 만드는 과정을 보여 주는 영상 두 편.
긴 것은 가로 1920×1080, 짧은 것은 세로 1080×1920, 둘 다 1초에 24장. 목소리와 큰 자막은 영어.

## 다시 만들기

    outputs/src/finish.sh long      # 장면 그리기 + 소리 + 합치기 + 검사  → outputs/work/long/final/
    outputs/src/finish.sh short     #                                      → outputs/work/short/final/
    outputs/src/deliver.sh final    # 바탕화면과 홍보영상_타임라인(HDD)에 넣기  (draft: 바탕화면에만, 1차본 표시)

그림은 `…rgbmocap-h3-install-17cf/cache/venv/bin/python`, 소리는 `~/.local/opt/film-audio/venv/bin/python -I`로 돈다(스크립트에 적혀 있다).
GPU는 쓰지 않는다(음악 후보를 생성 모델로 만들 때만 `lease_run.py`로 임대를 받아 썼다).

## 파일 (`outputs/src/`)

| 파일 | 하는 일 |
| --- | --- |
| `film.py` | 영상 한 편 = 장면의 줄. 장면 길이·대본 시각을 정하고 장면 함수를 불러 한 장씩 그려 ffmpeg로 보낸다. `--timeline`, `--stills`, `--all` |
| `scenes_long.py`, `scenes_short.py` | 장면들. 장마다 그 장에 나오는 춤의 한 마디 길이를 단위로 써서, 장면이 마디 수만큼 이어지고 컷이 박자 위에 온다 |
| `kit.py`, `widgets.py`, `common.py`, `skeleton.py` | 글자·카드·그래프(관절 곡선, 박자 띠, 카메라 지도, 막대, 산점도), 3D 관절 그림 |
| `sources.py` | 자료 읽기: 챌린지 실행, 2인 실행, 로봇 영상(눈 추적 결과), 브랜드 애니메이션 |
| `narration.json`, `voice_words.py` | 대본 → `say.py`(Kokoro)로 목소리, Whisper로 낱말 시각(자막과 그림을 말에 맞추는 데 쓴다) |
| `music_cues.py`, `music_jobs.json` | 음악. `qa`/`make`: 생성 모델(ACE-Step)로 만든 후보를 수치로 골라 춤 빠르기에 맞춤. `fallback`: 여기서 직접 만든 곡 |
| `audio_build.py`, `music_check.py`, `final_check.py` | 목소리+음악+효과음 섞기, 음악 박자가 영상 격자 위인지, 완성 파일 검사(길이·소리 크기·검은 화면·목소리 되듣기) |
| `robot_track.py`, `robot_antennas.py`, `robot_compare.py` | 로봇 영상에서 두 눈(렌즈)과 더듬이를 찾고, 춤 관절과 견주기 |
| `duo_checks_probe*.py`, `duo_chunk_probe.py` | 2인 프로그램의 새 검사(옷 색)를 정하려고 끝난 실행에 대 본 것 |
| `duo_review.py` | 끝난 2인 영상을 처음부터 끝까지 보기 위한 그림(0.5초마다 3D 화면과 나란히, 두 사람을 가까이). 판정은 하지 않는다 |
| `prepare_nine.sh` | 09번 춤 챌린지 결과에서 글자 없는 편집본(`work/clips/challenge09_clean.mp4`)을 만들고 `work/nine.json`을 쓴다 |
| `make_explain.py`, `make_thumbnail.py`, `deliver.sh` | 설명 문서, 썸네일, 전달 |

## 결과

- `outputs/final/댄스3D_홍보영상_20261116/` (바탕화면 `~/바탕화면/댄스3D_홍보영상_20261116/`과 HDD `홍보영상_타임라인/23_2026-11-16_댄스3D_캐릭터_로봇_롱폼_숏폼/`에 같은 것)
- 긴 것 2:21(3392장), 짧은 것 0:37(888장). 완성 검사 문제 없음, 목소리 되듣기 단어 오류 0 %, 소리 크기 -16 LUFS.

## 자료

- 춤 관절: `~/바탕화면/춤 관절 정보 만들기/20261114-124936-dance-joints-f011` (15개 춤). 1·2장은 02번, 로봇은 09번.
- 1인 챌린지: `work/main.json`(02번 춤 실행), `work/nine.json`(09번 춤 실행, 있으면 로봇 옆에 캐릭터가 나온다).
- 2인: `work/duo.json`(실행과 결과 폴더), `work/duo_clean.json`(끝까지 보고 맞다고 확인한 구간. 장면이 이 밖을 쓰면 멈춘다).
- 로봇: `~/바탕화면/danceC.MP4`. 음악이 시작하는 곳(13.05초)은 소리를 09번 춤 원본과 맞춰서 찾았다(`scratchpad/tools/audio_match.py`와 같은 방법).
- 브랜드: `cache/brand/steerix-matrix-brand` (github.com/kraizer-ai/steerix-matrix-brand).

## 지킨 것

- 원본 유튜브 영상과 그 노래는 넣지 않았다. 입력 영상은 사람의 윤곽과 관절 선으로만 보여 준다.
- 화면의 숫자는 실행 기록에서 읽는다(손으로 적은 숫자 없음).
- 2인 영상은 끝까지 보고 맞게 나온 구간만 쓴다.
- 대본이 말하는 숫자(관절 76개, 춤 15개, 49분, 1.5 %, 0.85)가 기록과 다르면 장면 만들기가 멈춘다(`scenes_long.py`).
- `kit.Frames`는 영상 파일의 크기와 수정 시각을 기억한다. 같은 이름으로 다시 만든 영상(2인 프로그램의 이어서 하기)은 다시 풀어 쓴다.
