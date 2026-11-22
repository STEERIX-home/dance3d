"""The Korean summary of a run of dance_joints.py: what came out of every video, what to look out
for, and what the files hold. Written from the step reports alone (world_one.py's report.json)."""
import json
import time
from pathlib import Path

# when a note is written (thresholds on world_one.py's report)
SEEN_NOTE = 0.97          # share of frames in which the dancer was reliably seen
FLOOR_NOTE = 0.5          # share of frames in which "up" came from the floor seen in the picture
PICTURE_NOTE = (4.0, 8.0)  # joints against the picture, percent of the body's height: median, worst tenth
FEET_NOTE = 0.05          # planted feet above or below the floor (metres)
TURN_NOTE = 30.0          # the camera turned about the vertical by more than this (degrees); smaller turns can be measuring error


def seconds_text(span):
    return f'{span[0]:.1f}~{span[1]:.1f}초'


def video_notes(about, report):
    """Things to know about one video's result, in plain Korean."""
    notes = []
    if about.get('flipped'):
        notes.append(f'좌우를 되돌렸습니다({about.get("flipped_because") or "설정"}). 화면 속 거울모드 모습이 아니라 '
                     '실제로 춘 방향의 안무입니다.')
    kinds = {'first seen pose held': '사람이 또렷이 보이기 전이라 처음 보인 자세로 채움',
             'last seen pose held': '사람이 또렷이 보이지 않아 마지막으로 보인 자세로 채움',
             'bridged between the seen frames': '사람이 또렷이 보이지 않아 앞뒤 장면을 이어 추정'}
    whole = [estimate['seconds'] for estimate in report.get('estimated', [])]
    for estimate in report.get('estimated', []):
        if estimate['seconds'][1] - estimate['seconds'][0] >= 0.2:
            notes.append(f'{seconds_text(estimate["seconds"])}: {kinds.get(estimate["kind"], estimate["kind"])}.')
    legs = report.get('legs', {})
    if legs.get('never_seen'):
        notes.append('다리가 한 번도 화면에 또렷이 나오지 않아, 다리 관절과 골반 높이는 동작 추정의 짐작입니다.')
    else:
        # the parts not already filled as a whole body
        filled = []
        for first, stop in (span['seconds'] for span in legs.get('filled', [])):
            for a, b in sorted(whole):
                if a <= first < b:
                    first = b
                if a < stop <= b:
                    stop = a
            if stop - first >= 0.2:
                filled.append([first, stop])
        if filled:
            notes.append('다리가 화면 밖이거나 가려진 구간은 다리 관절과 골반 높이를 앞뒤 장면에서 이어 붙였습니다(몸의 나머지는 그대로): '
                         + ', '.join(seconds_text(span) for span in filled) + '.')
    held = report.get('camera_held', [])
    still = [step['seconds'][0] for step in held if step['why'].endswith('without the background moving')]
    edits = report.get('cuts', {}).get('seconds', []) + [step['seconds'][0] for step in held
                                                        if step['why'] != 'cut' and step['seconds'][0] not in still]

    def times(values):
        return ', '.join(f'{seconds:.1f}초' for seconds in sorted(values)[:8]) + (' ...' if len(values) > 8 else '')

    if edits:
        notes.append(f'편집으로 장면이나 배경이 바뀌는 곳이 {len(edits)}군데 있습니다({times(edits)}). 그곳의 화면 변화는 '
                     '카메라 움직임으로 보지 않고, 춤추는 사람의 방향과 자리가 이어지게 했습니다.')
    if still:
        notes.append(f'카메라가 돈 것처럼 잰 곳 가운데 배경은 그만큼 움직이지 않은 {len(still)}곳({times(still)})은 '
                     '카메라 움직임으로 보지 않았습니다(조명이 바뀌거나 배경에 무늬가 적을 때 생깁니다).')
    gravity = report.get('gravity', {})
    if gravity.get('feet_check', {}).get('verdict') == 'the floor seen in the picture is not the one the dancer stands on':
        notes.append('화면 속 바닥이 사람의 발과 맞지 않아(합성한 배경, 반사가 심한 바닥 등), 위쪽 방향은 바닥 대신 '
                     '동작 추정이 몸에서 잡은 방향을 썼습니다. 카메라가 기울어 찍혔다면 몸도 그만큼 기울어 있을 수 있습니다.')
    elif 'turned_deg' not in gravity:
        notes.append('화면에서 바닥을 찾지 못해, 위쪽 방향은 동작 추정이 몸에서 잡은 그대로입니다. '
                     '카메라가 기울어 찍혔다면 몸도 그만큼 기울어 있을 수 있습니다.')
    elif report.get('floor_trusted_share', 1.0) < FLOOR_NOTE:
        notes.append(f'바닥이 또렷이 보인 장면이 {report["floor_trusted_share"] * 100:.0f} %라, 나머지 장면의 위쪽 방향은 '
                     '동작 추정이 몸에서 잡은 그대로입니다.')
    error = report.get('picture_error_percent_of_body', {})
    if error.get('median') is not None and (error['median'] > PICTURE_NOTE[0] or error['p90'] > PICTURE_NOTE[1]):
        notes.append(f'관절이 화면 속 사람과 어긋나는 장면이 있습니다(몸 높이의 {error["median"]:.1f} %, 나쁜 장면 10 %는 '
                     f'{error["p90"]:.1f} % 이상). 검증용 영상에서 확인해 주세요.')
    feet = (report.get('planted_feet') or {}).get('height_m', {})
    if feet.get('median') is not None and abs(feet['median']) > FEET_NOTE:
        notes.append(f'바닥을 디딘 발이 바닥에서 {abs(feet["median"]) * 100:.0f} cm {"떠" if feet["median"] > 0 else "묻혀"} 있습니다.')
    turn = report.get('camera', {}).get('turn_range_deg', 0.0)
    if turn > TURN_NOTE:
        notes.append(f'카메라가 좌우로 약 {turn:.0f}° 돈 것으로 쟀습니다. 카메라가 돈 만큼은 빼고 사람의 움직임만 남겼습니다.')
    return notes


def video_lines(about, report):
    seen = report['reliable_frames'] / report['frames']
    gravity = report.get('gravity', {})
    roll = gravity.get('camera_roll_deg')
    error = report.get('picture_error_percent_of_body', {})
    pelvis = report['pelvis_height_m']
    lines = [f'  길이 {about["seconds"]:.1f}초, 초당 {about["fps"]:.4g}장, {about["width"]}×{about["height"]}',
             f'  사람이 또렷이 보인 장면 {seen * 100:.0f} %, 바닥을 기준으로 바로 세운 장면 {report.get("floor_trusted_share", 0) * 100:.0f} %'
             + (f', 카메라 좌우 기울기 {roll["min"]:+.0f}°~{roll["max"]:+.0f}°' if roll else '')]
    if error.get('median') is not None:
        lines.append(f'  관절과 화면 속 사람의 차이: 몸 높이의 {error["median"]:.1f} % (중간값), 나쁜 장면 10 %는 {error["p90"]:.1f} % 이상')
    lines.append(f'  골반 높이(위아래 움직임) {pelvis["p5"]:.2f}~{pelvis["p95"]:.2f} m')
    return lines


def write_summary(target, job, options, state, abouts, reports, problems, runs):
    """결과_안내.txt, 결과_요약.json and 파일_설명.txt in the run's results folder. runs: minutes of every
    run of this job (a stopped run that was continued has more than one)."""
    target = Path(target)
    done = [video for video in options['videos'] if state[video['number']]['state'] == 'done']
    failed = [video for video in options['videos'] if state[video['number']]['state'] == 'failed']
    lines = [f'춤 관절 정보 - {time.strftime("%Y-%m-%d %H:%M")} (작업 {Path(job).name})', '',
             f'영상 {len(options["videos"])}개 가운데 {len(done)}개를 만들었습니다'
             + (f', {len(failed)}개는 실패했습니다' if failed else '') + f'. 걸린 시간 {max(1, round(sum(runs)))}분'
             + (f' (실행 {len(runs)}번: ' + ' + '.join(f'{round(m)}분' if m >= 1 else '1분 미만' for m in runs) + ')' if len(runs) > 1 else '')
             + '.',
             '영상 폴더마다 results.npz, motion.bvh, motion_amass.npz, 관절_정면과_옆.mp4, 정보.json이 있습니다. 형식은 파일_설명.txt에 있습니다.',
             '원본 영상이 보이는 확인 영상은 검증용_원본포함_공유금지 폴더에만 있습니다. 밖으로 내보내지 마세요.', '']
    for problem in problems:
        lines.append(f'목록에서 읽지 못한 줄: {problem}')
    if problems:
        lines.append('')
    summary = dict(job=str(job), videos=[])
    for video in options['videos']:
        number, about = video['number'], abouts.get(video['number'])
        title = (about or {}).get('title') or video['title'] or video['link']
        lines.append(f'{number:02d} {title}')
        if about and about.get('youtube') and about['youtube'].get('title'):
            lines.append(f'  유튜브 제목: {about["youtube"]["title"]}')
        entry = dict(number=number, title=title, link=video['link'], state=state[number]['state'])
        if number in reports:
            report = reports[number]
            notes = video_notes(about, report)
            lines += video_lines(about, report) + [f'  알림: {note}' for note in notes]
            entry.update(about=about, notes=notes, report=report)
        else:
            error = state[number].get('error') or '끝나지 않았습니다.'
            lines.append('  실패: ' + error.splitlines()[0])
            lines += ['    ' + line for line in error.splitlines()[1:3]]
            entry.update(error=error)
        lines.append('')
        summary['videos'].append(entry)
    (target / '결과_안내.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (target / '결과_요약.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    names = next((report['joint_names'] for report in reports.values() if report.get('joint_names')), None)
    (target / '파일_설명.txt').write_text(files_text(names), encoding='utf-8')


def files_text(names):
    text = """파일 설명 (영상 폴더마다 같은 형식)

results.npz
  「3D 모캡 H3」(1인 프로그램)의 outputs/mocap/results.npz와 같은 이름, 같은 모양의 값이 들어 있습니다.
  파이썬에서 읽기:  d = numpy.load('results.npz');  d['joints_global'] ...
  L = 장면 수, d['fps'] = 초당 장면 수 (원본 영상 그대로. 1인 프로그램 파일은 24로 맞춘 값입니다)

  joints_global   (L, 76, 3)  관절 76개의 위치. 미터. y가 위쪽, 바닥이 y = 0, x와 z는 바닥 위의 방향
  joint_names     (76,)       관절 이름 (아래 목록)
  smplx_global_*  같은 동작의 SMPL-X 값: global_orient (L,3) 몸 전체의 방향, body_pose (L,63) 관절 21개의 회전,
                  left_hand_pose / right_hand_pose (L,45) 손가락, transl (L,3) 위치, betas (L,10) 체형.
                  회전은 축-각도(라디안) 형식
  foot_contact    (L, 2)  왼발, 오른발이 바닥을 디뎠는지
  smplx_incam_*, joints_incam, joints_2d, K_fullimg, bbx_xyxy
                  카메라를 기준으로 본 같은 값과 화면 위치(픽셀)
  더 넣은 값      reliable (L,) 사람이 또렷이 보인 장면 / estimated (L,) 1이면 앞뒤 장면을 이어 추정한 장면 /
                  floor_trust (L,) 화면 속 바닥을 기준으로 삼은 정도 0~1 / turned_to_floor_deg (L,) 바닥 기준으로 세우며 돌린 각도 /
                  camera_R_c2w, camera_center (L,3,3), (L,3) 카메라의 방향과 위치 / flipped_left_right 좌우를 되돌린 영상인지

  1인 프로그램의 파일과 다른 점
  - global 값은 화면에 보이는 바닥을 기준으로 바로 세운 값입니다. 카메라가 기울어 찍혀도 몸이 기울지 않습니다.
  - 동작 추정이 뭉갠 빠른 움직임(튕기기, 흔들기)을 화면 속 관절 위치로 되살렸습니다.
  - 사람이 또렷이 보이지 않은 장면은 추정하지 않고 앞뒤 장면을 이어 붙였습니다(estimated = 1).
  - 거울모드 영상은 좌우를 되돌렸습니다(flipped_left_right). 화면 위치 값도 되돌린 화면 기준입니다.

motion.bvh        같은 동작의 뼈대 애니메이션. SMPL-X 관절 55개, 센티미터, y가 위쪽. 블렌더로 가져올 때 크기 0.01.
motion_amass.npz  같은 동작의 AMASS SMPL-X 형식(z가 위쪽). 사람 모양 로봇으로 동작을 옮기는 도구(GMR 등)가 읽는 형식.
관절_정면과_옆.mp4  관절만 정면과 옆에서 본 영상. 원본 화면은 없습니다. 파랑 = 왼쪽, 빨강 = 오른쪽, 회색 = 추정한 장면.
정보.json          영상 정보, 알림, 자동 확인 수치.
"""
    if names:
        text += '\n관절 목록 (joints_global의 두 번째 번호)\n'
        text += '\n'.join(f'  {index:2d} {name}' for index, name in enumerate(names)) + '\n'
    return text
