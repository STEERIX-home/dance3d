"""Result page of one run of duo_pipeline.py, written from the run's own records.

Nobody writes anything for it: every number comes from a file the steps left behind, every
sentence is fixed text that is shown or not depending on those numbers.

    make_summary.py --job JOB_DIR

Outputs in JOB/outputs/results: summary.json, 결과_요약.html (no frame of the source video in it),
guide_and_result_small.mp4 (3D guide next to the result, first camera), source_and_result.mp4
(source next to the original-view result; shows the source, for checking only)
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from make_report import strip

CAMERAS = {'original': ('1_원본과_같은_카메라', '원본과 같은 카메라'), 'orbit': ('2_360도_회전', '360도 회전'),
           'dynamic': ('3_다이내믹_카메라', '다이내믹 카메라'), 'front': ('4_정면', '정면')}
SIDES = {'A': '왼쪽 사람', 'B': '오른쪽 사람'}
STYLE = '''
:root { --bg:#f6f7f9; --card:#ffffff; --ink:#1b1f24; --dim:#5d6672; --line:#dfe3e8; --accent:#0b7a5a; --warn:#9a5b00; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171b; --card:#1d2126; --ink:#e7eaee; --dim:#9aa4b0; --line:#30363d; --accent:#3ecf9f; --warn:#e3a84a; } }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.7 "Noto Sans CJK KR","Noto Sans KR","Malgun Gothic",sans-serif; }
main { max-width:1080px; margin:0 auto; padding:24px 16px 80px; }
h1 { font-size:26px; line-height:1.3; margin:8px 0 4px; }
h2 { font-size:20px; margin:32px 0 10px; padding-top:14px; border-top:2px solid var(--line); }
h3 { font-size:17px; margin:20px 0 6px; }
p, li { max-width:860px; }
.dim { color:var(--dim); font-size:14px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 18px; margin:12px 0; }
.key { border-left:5px solid var(--accent); }
.warn { border-left:5px solid var(--warn); }
table { border-collapse:collapse; width:100%; margin:10px 0; font-size:15px; display:block; overflow-x:auto; }
th, td { border:1px solid var(--line); padding:7px 10px; text-align:left; vertical-align:top; }
th { background:var(--bg); white-space:nowrap; }
img { max-width:100%; border-radius:8px; border:1px solid var(--line); display:block; margin:8px 0; }
code { background:var(--bg); border:1px solid var(--line); border-radius:4px; padding:1px 5px; font-size:13px; word-break:break-all; }
'''


def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def spans_text(spans):
    return ', '.join(f'{first:g}~{stop:g}초' for first, stop in spans)


def original_camera(scene, world, people):
    """What the page says about the "original" view: how the source camera was found and how closely
    the 3D scene, seen through it, repeats the source picture. Returns (record, sentences, warnings)."""
    camera, gravity = scene.get('source_camera') or {}, (world or {}).get('gravity') or {}
    if not camera.get('roll_deg'):
        return {}, [], []
    fps, frames = scene['fps'], scene['frames']
    roll = camera['roll_deg']
    from_floor = gravity.get('source', '').startswith('floor')
    record = dict(roll_deg=roll, up_from='floor seen in the picture' if from_floor else 'the bodies', picture_match=camera.get('picture_match'),
                  moved_for_size_m=camera.get('moved_for_size_m', 0.0), goes_with_size_of={})
    lines = ['원본 영상의 카메라가 장면마다 어디에서 어느 쪽을 어떤 기울기로 찍었는지를 다시 계산해 그대로 썼습니다. '
             '그래서 이 영상에는 원본에 찍힌 부분만 나옵니다.']
    warnings = []
    if from_floor:
        lines.append(f'카메라가 좌우로 기운 각도는 원본 화면에 보이는 바닥에서 쟀습니다. 이 영상에서는 {roll["min"]:+.0f}°에서 {roll["max"]:+.0f}° 사이를 오갑니다.')
        total = gravity.get('frames_fully_trusted', 0) + gravity.get('frames_not_trusted', 0)
        if total and gravity.get('frames_not_trusted', 0) / max(total, 1) > 0.2:
            warnings.append('원본과 같은 카메라: 원본 화면에 바닥이 보이지 않는 구간이 길어서 그 구간의 카메라 기울기는 몸의 자세에서 추정했습니다. '
                            '원본 카메라가 좌우로 기우는 장면이면 방 대신 사람이 기울어 보일 수 있습니다.')
    else:
        warnings.append('원본과 같은 카메라: 원본 화면에서 바닥을 찾지 못해 카메라 기울기를 몸의 자세에서 추정했습니다. '
                        '원본 카메라가 좌우로 기우는 영상이면 방 대신 사람이 기울어 보일 수 있습니다.')
    for label, match in (camera.get('picture_match') or {}).items():
        if match.get('frames_compared', 0) >= 0.2 * frames:
            away, ratio = match['away_percent_of_picture_height'], match['size_ratio']
            lines.append(f'{SIDES[label]}이 화면에서 차지하는 자리(골반과 머리)는 원본 화면과 화면 높이의 {away["median"]:.1f} %(중앙값) 차이이고, '
                         f'크기는 원본의 {ratio["median"]:.2f}배입니다.')
    resized = {label: person['size_factor'] for label, person in people.items() if abs(person['size_factor'] - 1) > 1e-3}
    if resized and camera.get('follows'):
        for label in resized:
            has = np.asarray(camera['follows'][label]) >= 0.5
            edges = np.flatnonzero(np.diff(np.concatenate([[0], has.astype(int), [0]])))
            record['goes_with_size_of'][label] = [[round(first / fps, 1), round(stop / fps, 1)]
                                                  for first, stop in zip(edges[::2].tolist(), edges[1::2].tolist())]
        close = {label: spans for label, spans in record['goes_with_size_of'].items() if spans}
        if close:
            parts = ', '.join(f'{SIDES[label]}의 캐릭터 {spans_text(spans)}' for label, spans in close.items())
            lines.append(f'키를 바꾼 캐릭터가 화면을 채우는 구간({parts})에는 카메라가 그 캐릭터의 키에 맞춰 높이와 거리를 바꿔 원본과 같은 구도로 찍습니다'
                         f'(가장 많이 옮긴 곳 {camera["moved_for_size_m"]:.2f} m). 그 밖에서는 원본 카메라 그대로이고, 키를 바꾼 캐릭터는 그 비율만큼 '
                         '작거나 크게 보입니다.')
        else:
            lines.append('키를 바꾼 캐릭터는 원본 카메라 그대로 찍혀서 그 비율만큼 작거나 크게 보입니다.')
    return record, lines, warnings


def quick_motion(rows, least=1.5):
    """From the rows of a motion comparison: over all windows in which the reference's quick
    sideways head motion is large enough to measure, the middle size ratio and agreement."""
    out = {}
    for label in 'AB':
        kept = [row['x'] for row in rows or [] if row['band'] == f'{label}/head' and row['x']['reference_rms_px'] >= least
                and row['x']['correlation'] is not None]
        if kept:
            out[label] = dict(windows=len(kept), ratio=round(float(np.median([x['ratio'] for x in kept])), 2),
                              agreement=round(float(np.median([x['correlation'] for x in kept])), 2))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    job = args.job
    work, results = job / 'outputs/work', job / 'outputs/results'
    options = load(job / 'outputs/options.json')
    inputs = load(job / 'outputs/input/inputs.json')
    tracks = load(work / 'tracks/tracks.json')
    scene = load(work / 'scene_inputs/scene_inputs.json')
    cameras = [camera for camera in options['cameras'] if (results / f'duo_{camera}.mp4').exists()]
    warnings, videos = [], {}

    people = {}
    for person in scene['people']:
        estimated = np.asarray(person['estimated'])
        people[person['label']] = dict(shown_height_cm=round(person['standing_height_m'] * 100), size_factor=round(person['shown_size'], 3),
                                       frames_estimated=int((estimated > 0).sum()), frames=len(estimated))
        if (estimated > 0).mean() > 0.02:
            warnings.append(f'{SIDES[person["label"]]}은 {scene["frames"]}장 가운데 {int((estimated > 0).sum())}장에서 원본에 보이지 않아 '
                            f'그 동안의 동작은 앞뒤 동작으로 이어 붙인 추정입니다.')
    for label, legs in (scene.get('legs_planted') or {}).items():
        if legs:
            warnings.append(f'{SIDES[label]}의 다리는 처음 {max(leg["seconds"][1] for leg in legs):.1f}초 동안 화면 밖이어서 바닥에 선 것으로 두었습니다.')
    free = [CAMERAS[camera][1] for camera in options['cameras'] if camera != 'original']
    for label, spans in (scene.get('legs_out_of_the_picture') or {}).items():
        total = sum(stop - first for first, stop in spans)
        if total >= 1.0:
            same = ' 원본과 같은 카메라에는 그 부분이 나오지 않습니다.' if 'original' in options['cameras'] else ''
            where = (f'그 동안의 다리 동작은 추정이어서 {" · ".join(free)} 카메라에서는 어색할 수 있습니다.{same}' if free
                     else '그 동안의 다리 동작은 추정이지만, 원본과 같은 카메라에는 그 부분이 나오지 않습니다.')
            warnings.append(f'{SIDES[label]}의 발은 원본에서 {scene["duration_seconds"]:.1f}초 가운데 {total:.1f}초 동안 화면 밖이거나 가려져 있었습니다'
                            f'({spans_text(spans)}). {where}')
    same_camera, camera_lines, camera_warnings = original_camera(scene, load(work / 'world/report.json'), people)
    if 'original' in options['cameras']:
        warnings += camera_warnings
    references = {label: load(work / f'ref_{label}/reference.json', {}) for label in 'AB'}

    for camera in cameras:
        run = work / 'gen' / camera
        record, faces = load(run / 'run.json'), load(run / 'faces.json', {})
        identity, seams, motion = load(run / 'identity_check.json', {}), load(run / 'seams.json', []), load(run / 'motion_check.json', {})
        title = CAMERAS[camera][1]
        retried, left = 0, 0
        def failed(entry):
            return (len(entry['frames_off_the_room']) + len(entry.get('frames_with_the_wrong_head', []))
                    + len(entry.get('frames_in_the_other_ones_colours', []))
                    + len(entry.get('frames_with_a_mannequin_left', [])) + len(entry.get('frames_out_of_step', [])))

        late = []
        for chunk in record['runs']:
            tries = sorted(chunk.get('tries', []), key=lambda entry: entry['attempt'])
            kept = next((entry for entry in tries if entry['attempt'] == chunk.get('kept_try')), None)
            if tries and failed(tries[0]) >= 3:
                retried += 1
                left += failed(kept) if kept else 0
            if kept and kept.get('frames_out_of_step'):
                first = record['first_frame']
                late.append((round((kept['frames_out_of_step'][0] - first) / record['fps'], 1),
                             round((kept['frames_out_of_step'][-1] - first + 1) / record['fps'], 1),
                             abs(kept['out_of_step_by_frames']) / record['fps'], kept['out_of_step_by_frames'] < 0))
        if left:
            warnings.append(f'{title}: 방이 사라지거나, 머리가 뒤바뀌거나, 옷이 서로 바뀌거나, 인형이 그대로 남거나, 동작이 3D 화면과 어긋난 장면이 '
                            f'다시 만든 뒤에도 {left}장 남았습니다.')
        for start, stop, seconds, behind in late:
            warnings.append(f'{title}: {start:g}~{stop:g}초에는 캐릭터의 동작이 3D 화면보다 약 {seconds:.1f}초 {"늦습니다" if behind else "빠릅니다"}. '
                            '빠르게 도는 동작 뒤에 생성 모델이 가끔 이렇게 밀리며, 다시 만들어도 남은 경우입니다.')
        if identity.get('can_tell_apart') is False:
            warnings.append(f'{title}: 두 캐릭터의 머리 색이 비슷해서 서로 바뀌었는지를 자동으로 확인하지 못했습니다.')
        elif identity.get('suspected_swaps_seconds'):
            spans = ', '.join(f'{a}~{b}초' for a, b in identity['suspected_swaps_seconds'])
            warnings.append(f'{title}: 두 캐릭터가 서로 바뀐 것으로 의심되는 구간이 있습니다 ({spans}).')
        outfits = identity.get('outfits') or {}
        if outfits.get('mixed_seconds'):
            spans = ', '.join(f'{a}~{b}초' for a, b in outfits['mixed_seconds'])
            warnings.append(f'{title}: 한 캐릭터가 다른 캐릭터의 옷 색으로 그려진 구간이 있습니다 ({spans}). 한 사람이 다른 사람 뒤에 가려진 동안 '
                            '생성 모델이 두 캐릭터의 옷을 섞는 일이 있고, 다시 만들어도 남은 경우입니다.')
        rough = [seam for seam in seams if seam['clearly_different_share'] > 0.02]
        if rough:
            warnings.append(f'{title}: 구간 이음매 {len(rough)}곳에서 앞뒤 구간이 눈에 띄게 다릅니다 '
                            f'({", ".join(str(seam["at_seconds"]) + "초" for seam in rough)}).')
        face_rows = {}
        for label, face in (faces.get('faces') or {}).items():
            if 'not_corrected' in face:
                face_rows[label] = dict(corrected=0, note='그림에서 얼굴을 찾지 못해 보정하지 않음')
            else:
                face_rows[label] = dict(corrected=face['corrected_frames'], seen=face['frames_with_a_clear_face'],
                                        match_as_generated=None if not face['match_as_generated'] else round(face['match_as_generated']['median'], 2),
                                        match_after=None if face['mean_similarity_after'] is None else round(face['mean_similarity_after'], 2))
        videos[camera] = dict(file=CAMERAS[camera][0] + '.mp4', title=title, frames=record['frames'], seconds=round(record['frames'] / record['fps'], 2),
                              made_at=[record['width'], record['height']], base_model=record['models']['unet'],
                              generation_minutes=round(sum(chunk['seconds'] for chunk in record['runs']) / 60, 1),
                              chunks=len(record['runs']), chunks_made_again=retried, frames_off_the_room_left=left,
                              identity=identity, seams=seams, faces=face_rows, follows_guide=quick_motion(motion.get('rows')))

    # the finished original-view video against the source video itself (quick sideways head motion in 4 s pieces).
    # On 11-13 this was 0.19 for both dancers and I took it for an effect of hair and clothes and left it off the
    # page; in fact those videos were 0.65 s behind their guide in long stretches. In step it is 0.44-0.58.
    against_source = quick_motion(load(work / 'diag/original_vs_source.json'))
    if 'original' in cameras and against_source:
        camera_lines.append('원본 영상과 직접 견준 빠른 머리 움직임(1초에 3번보다 빠른 좌우 움직임): '
                            + ', '.join(f'{SIDES[label]} 크기 {value["ratio"]}배 · 일치도 {value["agreement"]}' for label, value in against_source.items())
                            + '. 일치도는 1이 완전히 같은 것이고, 머리카락과 옷이 원본과 달라서 1이 되지는 않습니다.')
        weak = [SIDES[label] for label, value in against_source.items() if value['agreement'] < 0.3]
        if weak:
            warnings.append(f'원본과 같은 카메라: 원본 영상과 견준 빠른 움직임의 일치도가 낮습니다({", ".join(weak)}). 동작이 원본과 어긋난 구간이 '
                            '있을 수 있으니 검증용 폴더의 「원본과_결과_나란히」 영상으로 확인해 주세요.')
    for label, face in next(iter(videos.values()))['faces'].items() if videos else []:
        if face.get('note'):
            warnings.append(f'{SIDES[label]}의 캐릭터: {face["note"]} (그린 캐릭터는 대개 그렇습니다).')

    # the source next to the original-view result (shows the source: for checking only)
    if 'original' in cameras:
        def label(text):
            from paths import font  # Noto Sans CJK wherever the system keeps it
            return f"drawtext=fontfile={font()}:text='{text}':x=40:y=40:fontsize=56:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=18"
        portrait = inputs['video']['portrait']
        half = '810:1440' if portrait else '1280:720'
        stack = 'hstack' if portrait else 'vstack'
        command = ['ffmpeg', '-v', 'error', '-y', '-i', str(job / 'outputs/input/source.mp4'), '-i', str(results / 'duo_original.mp4'),
                   '-filter_complex', f'[0:v]fps=24,scale={half},{label("원본")}[a];[1:v]scale={half},{label("결과")}[b];[a][b]{stack}=2:shortest=1[v]',
                   '-map', '[v]']
        if inputs['video']['has_audio']:
            command += ['-map', '1:a:0', '-c:a', 'copy']
        command += ['-c:v', 'libx264', '-crf', '21', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(results / 'source_and_result.mp4')]
        subprocess.run(command, check=True)

    # the 3D guide next to the result, small, for the first camera only (the full-size ones stay in the job)
    if cameras:
        first = results / f'duo_{cameras[0]}_guide_and_result.mp4'
        if first.exists():
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(first), '-vf', 'scale=iw*3/4:ih*3/4', '-c:v', 'libx264', '-crf', '21',
                            '-preset', 'medium', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart',
                            str(results / 'guide_and_result_small.mp4')], check=True)

    summary = dict(job=str(job), options=options, source=inputs['video'], characters=inputs['characters'],
                   assignment=dict(rule='두 사람이 가장 또렷이 떨어져 보이는 장면에서 왼쪽 사람이 A, 오른쪽 사람이 B',
                                   anchor_seconds=round(tracks['anchor_frame'] / tracks['fps'], 2),
                                   A=inputs['characters'][0]['source'], B=inputs['characters'][1]['source']),
                   people=people, clip=dict(frames=scene['frames'], seconds=round(scene['duration_seconds'], 2), kept_apart=scene.get('kept_apart')),
                   references={label: dict(surroundings_kept=ref.get('keep_background'), plain_share=ref.get('plain_surroundings_share'))
                               for label, ref in references.items()},
                   videos=videos, warnings=warnings, original_camera=same_camera,
                   original_view_against_source=against_source)
    (results / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    # ---------- the page ----------
    def face_cell(video):
        parts = []
        for label, face in video['faces'].items():
            parts.append(f'{SIDES[label]}: ' + (face['note'] if face.get('note') else
                                                f'{face["corrected"]}장 보정 (생성 직후 닮음 {face["match_as_generated"]} → 보정 뒤 {face["match_after"]})'))
        return '<br>'.join(parts) or '-'

    def identity_cell(video):
        identity = video['identity']
        if not identity:
            return '-'
        if identity.get('can_tell_apart') is False:
            return '확인 못 함 (색이 비슷함)'
        spans = identity.get('suspected_swaps_seconds') or []
        text = '바뀐 구간 없음' if not spans else '의심 구간 ' + ', '.join(f'{a}~{b}초' for a, b in spans)
        outfits = identity.get('outfits') or {}
        if outfits.get('can_tell_apart') is False:
            text += '<br>옷: 확인 못 함 (색이 비슷함)'
        elif outfits.get('mixed_seconds') is not None:
            text += '<br>옷: ' + ('섞인 구간 없음' if not outfits['mixed_seconds'] else
                                 '섞인 구간 ' + ', '.join(f'{a}~{b}초' for a, b in outfits['mixed_seconds']))
        return text

    def motion_cell(found):
        return '<br>'.join(f'{SIDES[label]}: 크기 {value["ratio"]}배 · 일치도 {value["agreement"]} ({value["windows"]}구간)'
                           for label, value in found.items()) or '잴 만큼 빠른 움직임 없음'

    rows = ''.join(
        f'<tr><td><b>{video["file"]}</b><br><span class="dim">{video["title"]}</span></td><td>{video["seconds"]}초 · {video["frames"]}장</td>'
        f'<td>{video["generation_minutes"]:.0f}분' + (f'<br>다시 만든 구간 {video["chunks_made_again"]}' if video['chunks_made_again'] else '') + '</td>'
        f'<td>{identity_cell(video)}</td><td>{face_cell(video)}</td><td>{motion_cell(video["follows_guide"])}</td></tr>'
        for video in videos.values())
    figures = ''.join(f'<h3>{video["title"]}</h3><img src="data:image/jpeg;base64,'
                      f'{strip(results / f"duo_{camera}.mp4", [video["seconds"] * share for share in (0.04, 0.2, 0.36, 0.52, 0.68, 0.84, 0.97)])}" alt="{video["title"]}">'
                      for camera, video in videos.items())
    source = inputs['video']
    heights = ' · '.join(f'{SIDES[label]} {value["shown_height_cm"]} cm' + (' (지정)' if label in options['heights'] else ' (원본 그대로)')
                         for label, value in people.items())
    base = {'official': '공식 H3', 'singularity': 'Singularity V2'}[options['base']]
    used = summary['clip']['seconds']
    if source.get('start'):
        taken = f'원본 영상의 {source["start"]:g}초 지점부터 {source["duration"]:.1f}초 길이'
    elif source.get('seconds'):
        taken = f'원본 영상의 앞 {source["duration"]:.1f}초'
    else:
        taken = f'영상 전체 {source["duration"]:.1f}초'
    used_text = (f'{taken}를 썼습니다' if used >= source['duration'] - 0.3
                 else f'{taken} 가운데 두 사람이 함께 보이는 {used:.1f}초를 썼습니다')
    page = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>2인 캐릭터 춤 결과</title><style>{STYLE}</style></head><body><main>
<h1>두 사람 춤 → 두 캐릭터 (고정 3D 방)</h1>
<p class="dim">작업 폴더 <code>{job}</code> · 이 문서는 실행 기록에서 자동으로 만들어졌습니다.</p>
<div class="card key"><b>입력과 배정</b>
<p>영상 {source["width"]}×{source["height"]}, 초당 {source["fps"]:.0f}장. {used_text}.<br>
두 사람이 가장 또렷이 떨어져 보이는 장면(원본 영상의 {(source.get("start") or 0) + summary["assignment"]["anchor_seconds"]:.1f}초)에서 <b>왼쪽 사람</b>에게 <code>{Path(summary["assignment"]["A"]).name}</code>,
<b>오른쪽 사람</b>에게 <code>{Path(summary["assignment"]["B"]).name}</code>를 입혔습니다. 반대로 하려면 두 그림의 자리를 바꿔 넣고 다시 실행합니다.<br>
캐릭터 키: {heights} · 본체: {base}</p></div>
''' + (f'<div class="card warn"><b>자동 확인에서 나온 것</b><ul>{"".join(f"<li>{text}</li>" for text in warnings)}</ul></div>' if warnings else
       '<div class="card key"><b>자동 확인</b><p>알려 드릴 문제가 나오지 않았습니다.</p></div>') + f'''
<h2>결과 영상</h2>
''' + (f'<div class="card"><b>원본과 같은 카메라</b><p>{"<br>".join(camera_lines)}</p></div>' if 'original' in videos and camera_lines else '') + f'''
<table><tr><th>파일</th><th>길이</th><th>생성 시간</th><th>누가 누구인지</th><th>얼굴</th><th>3D 화면의 빠른 머리 움직임을 따라간 정도</th></tr>{rows}</table>
<p class="dim">화면 전체(두 캐릭터, 그림자, 방)가 한 번의 생성에서 나옵니다. {next(iter(videos.values()))["made_at"][0]}×{next(iter(videos.values()))["made_at"][1]}로 만들어 최종 크기로 키웠습니다.
"크기"는 3D 화면 대비 배율(1이 같음), "일치도"는 두 움직임이 같이 가는 정도(1이 완전히 같음)이며, 1초에 3번보다 빠른 좌우 움직임이 잴 수 있을 만큼 큰 구간만 셉니다.
''' + (f'3D 화면의 빠른 움직임은 생성 모델이 줄이는 만큼을 메우려고 미리 {scene["quick_motion"]["gain"]:g}배로 키워 둔 것이라, 이 칸의 크기가 {1 / scene["quick_motion"]["gain"]:.1f}배쯤이면 실제 동작과 같은 크기입니다.' if scene.get('quick_motion') else '') + '''</p>
''' + f'''
{figures}
<h2>알아 둘 점</h2>
<ul>
<li>원본에서 가려지거나 화면 밖에 있던 몸의 동작은 추정입니다.</li>
<li>두 사람이 든 화면 전체를 한 번에 만들기 때문에, 사람이 화면에서 작게 보이는 카메라일수록 얼굴과 옷의 세밀함이 떨어집니다.</li>
<li>얼굴 보정은 캐릭터 그림에서 얼굴 하나를 또렷이 찾았을 때만 하며, 뒷모습·옆모습·가려진 얼굴에는 하지 않습니다.</li>
<li>H3 계열 모델의 사용 허락 지역에서 한국·미국·유럽연합·영국이 빠져 있습니다.</li>
<li>원본 영상이 보이는 파일은 <code>검증용_원본포함_공유금지</code> 폴더에만 있습니다.</li>
</ul>
</main></body></html>'''
    (results / '결과_요약.html').write_text(page)
    print(f'summary: {len(videos)} videos, {len(warnings)} notes -> {results / "결과_요약.html"}')


if __name__ == '__main__':
    main()
