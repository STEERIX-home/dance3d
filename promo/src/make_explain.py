"""The Korean description page that goes with the films (one self-contained HTML file with pictures inside).

    make_explain.py DELIVER_FOLDER

Everything it states is read from the records of the runs and of this job; nothing is typed in as a number.
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from film import FPS, Film  # noqa: E402
from kit import WORK  # noqa: E402

deliver = Path(sys.argv[1])
draft = len(sys.argv) > 2 and sys.argv[2] == 'draft'
films = {fmt: Film(fmt) for fmt in ('long', 'short')}
checks = {fmt: json.loads((WORK / fmt / 'final/check.json').read_text()) for fmt in films}
music = json.loads((WORK / 'music/report.json').read_text()) if (WORK / 'music/report.json').exists() else None
long, short = films['long'], films['short']
robot, main, duo, nine = long.robot, long.main, long.duo, long.nine


def picture(film, name, offset, width):
    scene = next(s for s in film.scenes if s.name == name)
    frame = film.frame(int(round((scene.start + offset) * FPS)))
    height = int(round(frame.shape[0] * width / frame.shape[1]))
    ok, data = cv2.imencode('.jpg', cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 84])
    return 'data:image/jpeg;base64,' + base64.b64encode(data.tobytes()).decode()


def clock(seconds):
    return f'{int(seconds // 60)}:{int(seconds % 60):02d}'


def chapters(film):
    rows, current = [], None
    names = {'open': '첫머리 (네 컷)', 'logo': 'STEERIX 로고', 'title': '제목 화면', 'outro': 'STEERIX 로고 (끝)', 'recap': '정리', 'hook': '첫머리', 'joints': '영상 → 3D 관절',
             'cameras': '카메라 네 방향, 박자에 맞춘 컷', 'two': '두 사람 → 두 캐릭터', 'robot': '진짜 로봇'}
    chapter = {'CAPTURE': '1장 · 영상에서 3D 관절 뽑기', 'ANY CAMERA': '2장 · 3D라서 어느 쪽에서나 찍기', 'TWO DANCERS': '3장 · 두 사람, 한 3D 방',
               'REAL ROBOT': '4장 · 같은 관절로 진짜 로봇'}
    for scene in film.scenes:
        if scene.chapter:
            label = chapter[scene.chapter[1]]
        else:
            key = scene.name.split('_')[0]
            label = names.get(scene.name, names.get(key, scene.name))
        if rows and rows[-1][0] == label:
            rows[-1][2] = scene.end
        else:
            rows.append([label, scene.start, scene.end])
    return rows


def files():
    rows = []
    for path in sorted(deliver.iterdir()):
        if path.suffix in ('.mp4', '.srt', '.jpg') and path.is_file():
            note = ''
            if path.suffix == '.mp4':
                seconds = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                                               capture_output=True, text=True).stdout)
                note = f'{clock(seconds)} · {path.stat().st_size / 1e6:.0f} MB'
            rows.append((path.name, note))
    return rows


CHAPTER_OF = {'main150': '첫머리와 1·2장', 'duo129': '3장(두 사람)', 'robot120': '4장(로봇)과 정리'}


def music_text():
    """What the music really is, from the records of music_cues.py (report.json: the takes tried and what was used)."""
    if not music or not music['chosen']:
        return '<li>음악: 없음.</li>'
    tried = sum(len(rows) for rows in music.get('ranked', {}).values())
    rows = []
    for name, info in music['chosen'].items():
        tempo = 240 / info['bar']
        passed = [row['take'] for row in music.get('ranked', {}).get(name, []) if row.get('usable') and not row.get('left_out')]
        if 'measured_bpm' in info:
            off = info['beats_off_grid_s']['median'] * 1000
            rows.append(f'{CHAPTER_OF.get(name, name)}: 생성 모델(ACE-Step, 가사 없음)로 만든 곡 <code>{info["take"]}</code>. '
                        f'만들어진 빠르기 1분에 {info["measured_bpm"]:.1f}박을 춤과 같은 {tempo:.1f}박으로 맞췄고, 박자가 기준 격자에서 벗어난 정도는 중간값 {off:.0f}밀리초입니다.')
        else:
            why = (f'생성 모델이 만든 후보 가운데 수치 검사를 통과한 곡이 하나 있었지만(<code>{passed[0]}</code>), 영상 전체의 소리를 한 가지로 맞추려고 쓰지 않았습니다'
                   if passed else '생성 모델이 만든 후보 가운데 빠르기가 맞는 곡이 없었습니다')
            rows.append(f'{CHAPTER_OF.get(name, name)}: 이 프로그램이 사인파와 잡음으로 직접 만든 곡(1분에 {tempo:.1f}박)을 썼습니다. {why}.')
    made_here = all('measured_bpm' not in info for info in music['chosen'].values())
    # how far the generated takes were from the tempo asked for, cue by cue
    asked = ', '.join(f'1분에 {240 / music["targets"][name]:.0f}박이 필요한 장을 위해 만든 {len(rows_)}곡은 {min(r["measured_bpm"] for r in rows_):.0f}~{max(r["measured_bpm"] for r in rows_):.0f}박'
                      for name, rows_ in music.get('ranked', {}).items() if rows_)

    def worst(fmt):   # the largest median distance of the music's beats from the film's grid, in milliseconds (music_check.py)
        return max(row['beat_off_median_s'] for row in json.loads((WORK / fmt / 'audio/music_check.json').read_text())) * 1000
    return ('<li>음악: 원본 춤 영상의 노래는 쓰지 않고, 이 컴퓨터에서 새로 만든 연주곡을 장마다 그 장의 춤과 같은 빠르기로 맞춰 넣었습니다(컷이 박자 위에 오도록).'
            + (f' 먼저 음악 생성 모델(ACE-Step)로 후보 {tried}곡을 만들었는데, 부탁한 빠르기대로 나오지 않은 곡이 많았습니다({asked}).' if tried else '')
            + '<ul>' + ''.join(f'<li>{row}</li>' for row in rows) + '</ul>'
            + ('세 곡 모두 드럼·베이스·화음·짧은 가락으로 된 단순한 전자음 곡입니다. ' if made_here else '')
            + f'완성된 소리에서 음악의 박자를 다시 찾아 영상의 박자 격자와 견주면, 벗어난 정도의 중간값이 긴 영상 {worst("long"):.0f}밀리초, 짧은 영상 {worst("short"):.0f}밀리초 이하입니다. '
            + '저는 소리를 들을 수 없어서 곡이 듣기에 어떤지는 확인하지 못했습니다. 수치(목소리 없음, 박자가 고름, 빠르기가 맞음, 소리 크기)만 확인했습니다. '
            '들어 보시고 마음에 안 들면 <code>_음악없음</code> 판에 다른 음악을 얹으면 됩니다. '
            '생성 모델이 만든 후보는 작업 폴더의 <code>outputs/work/music/raw/</code>에 있습니다.</li>')


def duo_text():
    if not duo:
        return ''
    review = json.loads((WORK / 'duo_clean.json').read_text())[Path(duo.job).name]
    spans = lambda view: ', '.join(f'{a:.1f}~{b:.1f}초' for a, b in review[view])
    return ('<h2>2인 캐릭터 영상을 끝까지 본 결과</h2><p>두 영상(원본과 같은 카메라, 360도 회전)을 처음부터 끝까지 0.5초 간격으로 3D 안내 화면과 나란히 놓고 보았고, '
            '문제가 시작되고 끝나는 곳은 0.25초 간격으로 다시 보았습니다.</p><ul>' + ''.join(f'<li>{line}</li>' for line in review.get('findings_ko', []))
            + f'<li>홍보 영상에는 맞게 나온 구간만 썼습니다 (원본과 같은 카메라 {spans("original")}, 360도 회전 {spans("orbit")}). '
            '장면을 만들 때 프로그램이 이 구간 밖을 쓰면 멈추게 해 두었습니다.</li></ul>')


_free = [scene for scene in long.scenes if scene.name.startswith('robot_free')]
side_by_side = f'{clock(_free[0].start)}~{clock(_free[-1].end)}' if _free else ''
duo_lines = duo_text()
music_lines = music_text()
draft_note = ('<p style="border:1px solid #f5b536;padding:10px 14px;border-radius:8px">지금은 <b>1차본</b>입니다. 로봇 장에서 로봇 옆에 같은 춤을 추는 캐릭터 영상'
              '(09번 춤의 1인 챌린지, 만드는 중)이 아직 없어 그 자리에 관절 그림이 들어가 있습니다. 끝나면 이 폴더를 완성본으로 바꿉니다.</p>') if draft else ''
accuracy = long.accuracy
summary = duo.summary if duo else {}
original = summary.get('videos', {}).get('original', {})
table = lambda rows: ''.join(f'<tr><td>{clock(a)} ~ {clock(b)}</td><td>{label}</td></tr>' for label, a, b in rows)
html = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>댄스 3D 홍보 영상 설명</title>
<style>
body {{ font-family: "Noto Sans CJK KR", "Malgun Gothic", sans-serif; background: #0b0f0e; color: #e8eeec; max-width: 980px; margin: 32px auto; padding: 0 20px; line-height: 1.65; }}
h1 {{ font-size: 28px; }} h2 {{ font-size: 21px; margin-top: 36px; border-left: 5px solid #00dc7d; padding-left: 10px; }}
table {{ border-collapse: collapse; width: 100%; }} td, th {{ border-bottom: 1px solid #24302d; padding: 7px 8px; text-align: left; vertical-align: top; }} td:first-child {{ width: 30%; }}
img {{ max-width: 100%; border-radius: 10px; border: 1px solid #24302d; }} .row {{ display: flex; gap: 12px; flex-wrap: wrap; }} .row > div {{ flex: 1; min-width: 260px; }}
.tall img {{ max-height: 520px; width: auto; }} code {{ color: #9fe8c9; }} .note {{ color: #9fb0ab; font-size: 14px; }}
</style></head><body>
<h1>댄스 영상 → 3D → 캐릭터 영상 · 2인 캐릭터 · 로봇 춤 — 홍보 영상 (긴 것, 짧은 것)</h1>
<p class="note">만든 날 2026-11-16 · 작업 폴더 <code>/srv/training/jobs/20261116-041650-dance-promo-video-0df0</code></p>
{draft_note}

<h2>들어 있는 파일</h2>
<table>{''.join(f'<tr><td><code>{name}</code></td><td>{note}</td></tr>' for name, note in files())}</table>
<p>긴 영상은 가로 1920×1080, 짧은 영상은 세로 1080×1920입니다. 둘 다 1초에 24장입니다(만든 춤 영상이 24장이라 30장으로 바꾸면 춤이 끊겨 보입니다).
목소리와 큰 자막은 영어입니다. <code>_음악없음</code> 파일은 목소리와 효과음만 있습니다. <code>.srt</code>는 영어 전체 자막입니다.</p>

<h2>긴 영상 구성 ({clock(long.seconds)})</h2>
<table>{table(chapters(long))}</table>
<div class="row"><div><img src="{picture(long, 'capture_curves', 4.0, 640)}"></div><div><img src="{picture(long, 'camera_beat', 5.0, 640)}"></div></div>
<div class="row"><div><img src="{picture(long, 'two_pass', 5.5, 640) if duo else picture(long, 'camera_paint', 6.5, 640)}"></div><div><img src="{picture(long, 'robot_graph', 7.0, 640)}"></div></div>

<h2>짧은 영상 구성 ({clock(short.seconds)})</h2>
<table>{table(chapters(short))}</table>
<div class="row tall"><div><img src="{picture(short, 'hook_2', 0.9, 360)}"></div><div><img src="{picture(short, 'cameras', 4.5, 360)}"></div><div><img src="{picture(short, 'robot', 5.0, 360)}"></div></div>

<h2>영상에 쓴 자료</h2>
<ul>
<li>춤 관절: 「춤 관절 정보」 실행 <code>20261114-124936-dance-joints-f011</code>의 15개 춤. 1·2장과 숫자는 02번 춤, 로봇 장은 09번 춤입니다.</li>
<li>1인 캐릭터 영상(2장): 「댄스 챌린지」 화면에서 손대지 않고 돌린 실행 <code>{Path(main.job).name}</code> (reze.webp, {long.main_minutes:.0f}분).</li>
{f'<li>2인 캐릭터 영상(3장): 「2인 캐릭터 춤」 화면에서 오늘 새로 돌린 실행 <code>{Path(duo.job).name}</code>(103분, 손대지 않음). 그 결과에서 문제를 찾아 프로그램에 검사를 넣은 뒤, 화면의 「멈춘 작업 이어서 하기」로 한 번 더 돌렸습니다(49분, 아래 참고). 왼쪽 사람 = reze.webp, 오른쪽 사람 = makima.webp. 카메라는 원본과 같은 카메라와 360도 회전 두 가지입니다.</li>' if duo else ''}
{f'<li>로봇 옆의 캐릭터(4장, 짧은 영상의 첫머리와 카메라 장면): 09번 춤으로 「댄스 챌린지」 화면에서 오늘 새로 돌린 실행 <code>{Path(nine.job).name}</code>(reze.webp, 76분, 손대지 않음, 조각 33개 모두 첫 시도에 통과). 완성 영상을 0.5초 간격으로, 컷 16곳은 앞뒤로 보았습니다.</li>' if nine else ''}
<li>로봇: 바탕화면의 <code>danceC.MP4</code> 그대로입니다(속도를 바꾸지 않았습니다). 소리를 맞춰 보니 이 영상의 13.05초부터 09번 춤 음악이 나옵니다.</li>
<li>인트로·아웃트로: <code>kraizer-ai/steerix-matrix-brand</code>의 전환 애니메이션에서 MATRIX 글자가 사라진 뒤(조각이 모여 STEERIX 로고가 되는 부분)만 썼습니다.
글꼴(IBM Plex)과 초록색(#00DC7D)도 그 저장소 것입니다.</li>
</ul>

{duo_lines}

<h2>영상 속 숫자의 근거</h2>
<table>
<tr><td>관절 76개, 미터 단위, 바닥 기준</td><td>관절 결과 파일(results.npz)의 형식 그대로입니다.</td></tr>
<tr><td>춤 15개, 49분, 손대지 않음</td><td>관절 실행 f011의 기록(15개 모두 성공, 49분).</td></tr>
<tr><td>몸 높이의 약 1.5 %</td><td>관절을 원본 화면에 다시 겹쳤을 때의 차이. 춤마다 중간값 {min(accuracy):.1f}~{max(accuracy):.1f} %, 15개의 중간값 {float(np.median(accuracy)):.1f} %.</td></tr>
<tr><td>1분에 {main.plan["bpm"]:g}박, 컷 {len(main.cuts)}개 모두 박자 위</td><td>챌린지 실행의 자동 검사 기록(박자에서 가장 먼 컷 {main.checks["cuts"]["furthest_from_a_beat_s"]:.2f}초).</td></tr>
<tr><td>자동 검사 여섯 줄, {long.main_minutes:.0f}분</td><td>같은 실행의 검사 기록과 걸린 시간.</td></tr>
{f'<tr><td>2인 영상의 검사 줄, 조각 {original.get("chunks", "")}개</td><td>오늘 실행의 기록(조각마다 프로그램이 한 검사)과 결과 요약. 「자리」는 원본 영상 속 사람의 자리와 비교한 값입니다. 화면에는 제가 끝까지 보고 맞다고 확인한 줄만 넣었습니다.</td></tr>' if original else ''}
<tr><td>로봇 고개 기울기의 상관 {abs(robot.correlation):.2f}</td><td>로봇 영상에서 두 눈(렌즈)을 장마다 찾아 그 둘을 잇는 선의 기울기를 재고, 춤 관절의 귀를 잇는 선의 기울기와 비교했습니다.
눈을 찾은 {robot.measured}장으로 계산했고, 로봇이 음악보다 {robot.LAG:.2f}초 늦게 움직이는 것을 맞췄습니다. 로봇은 춤추는 사람을 거울처럼 따라 해서, 로봇 옆의 관절과 캐릭터는 좌우를 뒤집어 보여 줍니다(화면에 MIRRORED라고 적었습니다).
로봇 기울기는 사람의 약 {robot.fit[0]:.1f}배 크기라서 그래프는 각자 눈금으로 그렸고 그렇게 적었습니다.</td></tr>
</table>

<h2>소리</h2>
<ul>
<li>목소리: 이 컴퓨터에서 돌린 음성 합성(Kokoro, af_heart). 완성된 소리를 음성 인식기에 다시 들려 보아 대본과 맞는지 확인했습니다
(긴 영상 단어 오류 {checks["long"]["narration"]["word_error_rate"]:.1%}, 짧은 영상 {checks["short"]["narration"]["word_error_rate"]:.1%}).</li>
{music_lines}
<li>크기: 긴 영상 {checks["long"]["loudness"]["integrated_lufs"]:.1f} LUFS, 짧은 영상 {checks["short"]["loudness"]["integrated_lufs"]:.1f} LUFS.</li>
</ul>

<h2>알아 둘 점</h2>
<ul>
<li>원본 유튜브 춤 영상과 그 노래는 영상에 넣지 않았습니다. 입력 영상은 전화기 모양 안에 사람의 윤곽(추적 프로그램이 딴 외곽선)과 관절 선으로만 보여 줍니다.</li>
<li>캐릭터 영상은 영상 모델이 3D 안내 화면 위에 그린 것입니다. 그림에 없는 옆모습·뒷모습과 2인 영상의 360도 회전에서 원본에 안 찍힌 부분은 모델이 지어낸 것입니다.</li>
<li>캐릭터 영상을 만드는 H3 계열 모델의 사용 허락 지역에서 한국·미국·유럽연합·영국이 빠져 있습니다. 공개하기 전에 확인이 필요합니다.</li>
<li>로봇과 캐릭터를 나란히 놓은 장면(긴 영상 {side_by_side}, 짧은 영상 첫머리)은 캐릭터 영상이 정면에서 찍힌 마디만 골라 이었습니다. 로봇 영상도 같은 마디로 건너뛰며, 속도는 바꾸지 않았습니다.</li>
<li>로봇은 고개 기울기만 재서 확인했습니다. 몸통과 더듬이는 영상에서 움직이는 것이 보이지만 수치로 확인한 것은 아닙니다(더듬이는 일부 장에서만 잴 수 있었습니다).</li>
<li>영상에 로봇 제품 이름, 모델 이름, 노래 제목은 넣지 않았습니다. 로봇 영상 속 태블릿 화면에는 한글 앱 화면이 그대로 보입니다.</li>
<li>인트로를 STEERIX → MATRIX 전환으로 바꾸려면 <code>scenes_long.py</code>, <code>scenes_short.py</code>의 로고 장면 한 줄만 바꾸면 됩니다.</li>
</ul>

<h2>다시 만들기</h2>
<p><code>만들기_스크립트/finish.sh long</code>, <code>finish.sh short</code> (작업 폴더의 <code>outputs/src</code>와 같은 파일). 장면은 <code>scenes_long.py</code>, <code>scenes_short.py</code>, 대본은 <code>narration.json</code>입니다.</p>
</body></html>
'''
(deliver / '설명.html').write_text(html)
print('written', deliver / '설명.html', f'{len(html) / 1e6:.1f} MB')
