"""Checks of a finished challenge run, written to outputs/results/checks.json:

  video     length, frame rate, picture size and sound of the finished file, against the plan
  cuts      every cut on a beat of the music (at most one frame away), the front shown at least half the time
  shots     what pair_run.py's own checks left in the frames each shot shows (the room of the guide
            lost, a mannequin left, the character behind or ahead of the guide), and how much of each
            shot is a repeated frame (a shot that ran past its window holds its last frame)
  framing   where each camera cuts the body (camera-path.json of blender_views.py): full shots show
            the feet, close-ups leave the knees out, the head stays in, nothing comes near the lens
  look      whether the character keeps its look from shot to shot: the colours of the finished frame
            inside the dancer's outline (the mask of the 3D scene, which the video model follows) above the
            close-up's lower edge (20 cm under the pelvis); the share of colours that the other shots of the
            same framing hardly have
  stills    one picture every 2 s of [3D guide | finished] for looking through (check_stills/)

    check_challenge.py --job JOB
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def colours(frame, outline, lowest=1.0):
    """Hue-saturation-value histogram (normalised) of the frame inside the dancer's outline (the scene's mask
    picture: the dancer flat on white), a little inside its edge, above the height `lowest` of the picture
    (0 = top, 1 = bottom)."""
    small = cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA)
    character = (cv2.resize(outline, (270, 480), interpolation=cv2.INTER_AREA).min(-1) < 200).astype(np.uint8)
    character = cv2.erode(character, np.ones((5, 5), np.uint8))
    character[int(round(np.clip(lowest, 0.0, 1.0) * 480)):] = 0
    if character.mean() < 0.01:
        return None
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], character, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return histogram / max(1.0, histogram.sum())


def seen_at(row, point, manifest):
    """Height in the picture (0 = top, 1 = bottom) at which the camera of one camera-path.json row sees a point."""
    w, x, y, z = row['rotation_quaternion']
    turn = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    local = (np.asarray(point, dtype=float) - np.asarray(row['location'])) @ turn
    width, height = manifest['resolution']
    tan_y = manifest['sensor_width_mm'] / 2 / manifest['lens_mm'] * height / width
    return float((1 - local[1] / -local[2] / tan_y) / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = Path(args.job)
    work, results = job / 'outputs/work', job / 'outputs/results'
    plan = json.loads((work / 'plan.json').read_text())
    beats = json.loads((work / 'beats.json').read_text())
    fps, total = plan['fps'], plan['frames']
    pelvis = np.asarray(json.loads((work / 'scene_inputs/scene_inputs.json').read_text())['people'][0]['pelvis'])
    report, problems = dict(), []

    info = probe(results / 'challenge.mp4')
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    sound = [s for s in info['streams'] if s['codec_type'] == 'audio']
    frames = int(video.get('nb_frames') or 0)
    report['video'] = dict(width=int(video['width']), height=int(video['height']), frames=frames,
                           seconds=round(float(info['format']['duration']), 3), sound=bool(sound),
                           plan_frames=total, plan_seconds=plan['seconds'])
    if frames != total:
        problems.append(f'완성 영상의 장 수({frames})가 계획({total})과 다릅니다.')
    inputs = json.loads((job / 'outputs/input/inputs.json').read_text())
    if inputs['music']['file'] and not sound:
        problems.append('완성 영상에 소리가 없습니다.')

    times = np.asarray(beats['beats'])
    away = [round(float(np.abs(times - cut['time']).min()), 4) for cut in plan['cuts']] if len(times) else []
    report['cuts'] = dict(count=len(plan['cuts']), furthest_from_a_beat_s=max(away) if away else None, front_share=plan['front_share'],
                          bpm=beats['bpm'], beat_method=beats['method'])
    if away and max(away) > 1.0 / fps + 1e-6:
        problems.append(f'박자에서 {max(away):.2f}초 벗어난 컷이 있습니다.')
    if plan['front_share'] < 0.5:
        problems.append(f'정면이 나오는 시간이 {plan["front_share"]:.0%}로 절반이 안 됩니다.')

    shots, signatures, paths, guides = [], {}, {}, {}
    for shot in plan['shots']:
        run = json.loads((work / 'gen' / shot['name'] / 'run.json').read_text())
        guides[shot['name']] = run.get('guide', 'pair')  # the pass the video model followed (H3: pair, SCAIL-2: guide)
        shown = set(range(shot['first'], shot['last'] + 1))
        kept = {entry['chunk']: entry for entry in run['runs']}
        left = dict(room=set(), mannequin=set(), step=set(), face=set(), hair=set())
        faces_checked, hair_checked, hair = False, False, []
        for entry in kept.values():
            tries = {t['attempt']: t for t in entry['tries']}
            final = tries.get(entry['kept_try'], {})
            left['room'] |= set(final.get('frames_off_the_room', [])) & shown
            left['mannequin'] |= set(final.get('frames_with_a_mannequin_left', [])) & shown
            left['step'] |= set(final.get('frames_out_of_step', [])) & shown
            left['face'] |= set(final.get('frames_with_a_face_from_behind', [])) & shown
            faces_checked = 'frames_with_a_face_from_behind' in final  # runs made before the face check have no such list
            left['hair'] |= set(final.get('frames_with_added_hair', [])) & shown
            hair_checked = 'frames_with_added_hair' in final  # likewise for the hair check (qa_hair.py)
            hair += [value for number, value in entry.get('added_hair', []) if number in shown]
        start, made = shot['window']
        held = sum(1 for number in shown if not start <= number <= start + made - 1)
        entry = dict(name=shot['name'], view=shot['view'], seconds=[shot['start_s'], shot['end_s']],
                     tries=sum(len(e['tries']) for e in kept.values()), chunks=len(kept),
                     chunk_frames=run.get('chunk_frames'), generation_seconds=[e['seconds'] for e in kept.values()],
                     frames_without_the_room=len(left['room']), frames_with_a_mannequin_left=len(left['mannequin']),
                     frames_out_of_step=len(left['step']), held_frames=held)
        if faces_checked:
            entry['frames_with_a_face_from_behind'] = len(left['face'])
        if hair_checked:
            entry['frames_with_added_hair'] = len(left['hair'])
            entry['added_hair_median'] = round(float(np.median(hair)), 3) if hair else None
            entry['hair_limit'] = run.get('hair_limit')
        shots.append(entry)
        if shot['view'] not in paths:
            paths[shot['view']] = json.loads((work / 'scene' / shot['view'] / 'camera-path.json').read_text())
        mine = [paths[shot['view']][number - 1] for number in sorted(shown)]
        if 'knees_out' in mine[0]:  # cameras made with the knee points of the body
            share = {key: round(float(np.mean([row[key] for row in mine])), 3) for key in ('feet_in', 'head_in', 'knees_out')}
            entry['framing'] = dict(share, nearest_m=min(row['nearest_m'] for row in mine))
            where = f'{shot["label"]}{" 가까이" if shot["framing"] == "close" else ""} 장면({shot["start_s"]:.1f}~{shot["end_s"]:.1f}초)'
            if shot['framing'] == 'full' and share['feet_in'] < 0.9:
                problems.append(f'{where}에서 발이 화면 밖으로 나간 장이 {1 - share["feet_in"]:.0%}입니다.')
            if shot['framing'] == 'close' and share['knees_out'] < 0.75:
                problems.append(f'{where}에서 무릎이 보이는 장이 {1 - share["knees_out"]:.0%}입니다 (가까이 장면은 허벅지 위만 보여야 합니다).')
            if share['head_in'] < 0.95:
                problems.append(f'{where}에서 머리가 화면 밖으로 나간 장이 {1 - share["head_in"]:.0%}입니다.')
            if entry['framing']['nearest_m'] < 0.12:
                problems.append(f'{where}에서 몸이 카메라 {entry["framing"]["nearest_m"] * 100:.0f}cm 앞까지 다가갑니다.')
        for kind, text in (('room', '방이 사라진'), ('mannequin', '마네킹이 남은'), ('step', '동작이 안내 화면과 어긋난'),
                           ('face', '뒷머리에 얼굴이 그려진'), ('hair', '그림에 없는 머리카락(묶은 꼬리·땋은 머리 등)이 그려진')):
            if len(left[kind]) >= 3:
                problems.append(f'{shot["label"]} 장면({shot["start_s"]:.1f}~{shot["end_s"]:.1f}초)에 {text} 장이 {len(left[kind])}장 남았습니다.')
        # colours of the character in this shot, above the close-up's lower edge
        manifest = json.loads((work / 'scene' / shot['view'] / 'manifest.json').read_text())
        values = []
        for number in sorted(shown)[::4]:
            frame = cv2.imread(str(work / 'finished' / shot['name'] / f'frame_{number - start + 1:04d}.png'))
            outline = cv2.imread(str(work / 'scene' / shot['view'] / 'mask' / f'frame_{number:04d}.png'))
            if frame is not None and outline is not None:
                thigh = pelvis[number - 1] - np.array([0.0, 0.0, 0.20])
                value = colours(frame, outline, seen_at(paths[shot['view']][number - 1], thigh, manifest))
                if value is not None:
                    values.append(value)
        if values:
            signatures[shot['name']] = np.mean(values, axis=0)
    report['shots'] = shots
    if len(signatures) >= 2:
        # a close-up shows the face and arms larger than a full shot does, so the share of each colour differs
        # by framing: each shot is held against the other shots of its framing (all others if it is alone),
        # and what counts is how much of it is colours that the others hardly have (under 0.4 %)
        framing = {s['name']: s['framing'] for s in plan['shots']}
        novel = {}
        for name, signature in signatures.items():
            others = [n for n in signatures if n != name and framing[n] == framing[name]] or [n for n in signatures if n != name]
            usual = np.median(np.stack([signatures[n] for n in others]), axis=0)
            novel[name] = float(signature[usual < 0.004].sum())
        report['look'] = dict(new_colour_share=dict((name, round(value, 3)) for name, value in novel.items()),
                              note='share of the character (inside the outline, above the thighs) in colours the other shots '
                                   'of the same framing hardly have; above 0.30 the character likely changed its look')
        for name, value in novel.items():
            if value > 0.30:
                shot = next(s for s in plan['shots'] if s['name'] == name)
                problems.append(f'{shot["label"]}{" 가까이" if shot["framing"] == "close" else ""} 장면'
                                f'({shot["start_s"]:.1f}~{shot["end_s"]:.1f}초)의 캐릭터에 다른 장면에 없는 색이 {value:.0%} 있습니다 '
                                f'(옷이나 모습이 바뀌었을 수 있음).')

    stills = results / 'check_stills'
    stills.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(results / 'challenge.mp4'))
    wanted = {int(round(t * fps)) for t in np.arange(0, total / fps, 2.0)} | {total - 1}
    number, written = 0, []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if number in wanted:
            shot = next(s for s in plan['shots'] if s['first'] <= number + 1 <= s['last'])
            guide = cv2.imread(str(work / 'scene' / shot['view'] / guides[shot['name']] / f'frame_{number + 1:04d}.jpg'))
            pair = np.concatenate([cv2.resize(guide, (540, 960)), cv2.resize(frame, (540, 960))], axis=1)
            file = stills / f'{number / fps:05.1f}s_{shot["view"]}.jpg'
            cv2.imwrite(str(file), pair, [cv2.IMWRITE_JPEG_QUALITY, 88])
            written.append(file.name)
        number += 1
    report['stills'] = written
    report['problems'] = problems
    report['passed'] = not problems
    (results / 'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(dict(passed=report['passed'], problems=problems, video=report['video'], cuts=report['cuts']), ensure_ascii=False))


if __name__ == '__main__':
    main()
