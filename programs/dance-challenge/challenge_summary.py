"""The result note of a challenge run, in Korean (결과_안내.txt), and the same as data (summary.json).

    challenge_summary.py --job JOB
"""
import argparse
import json
import time
from pathlib import Path

TRANSITIONS = {'flash': '하얀 번쩍임', 'zoom': '확대했다 돌아오기', 'whip': '옆으로 휙 넘기기', 'glitch': '화면 깨짐 효과'}
EFFECTS = {'normal': '보통', 'strong': '강하게', 'none': '없음(컷만)'}


def seconds(value):
    return f'{value:.1f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = Path(args.job)
    work, results = job / 'outputs/work', job / 'outputs/results'
    options = json.loads((job / 'outputs/options.json').read_text())
    inputs = json.loads((job / 'outputs/input/inputs.json').read_text())
    info = json.loads((job / 'outputs/input/info.json').read_text())
    plan = json.loads((work / 'plan.json').read_text())
    checks = json.loads((results / 'checks.json').read_text())
    description = (work / 'describe/description_A.txt').read_text().strip() if (work / 'describe/description_A.txt').exists() else ''
    backend = options.get('backend', 'h3')  # runs made before the SCAIL-2 backend were all H3
    tier = json.loads((work / 'tier.json').read_text()) if (work / 'tier.json').exists() else None
    title = inputs['title']
    name = '_'.join(title.split()) if title else '댄스'
    stages = {p.stem: json.loads(p.read_text()) for p in (work / 'stages').glob('*.json')}
    spent = sum(s.get('seconds', 0) for s in stages.values())
    character = inputs['character']
    if '/gradio/' in character['source']:  # the page hands over its own temporary copy of the upload: only the name means anything
        picture = f'{Path(character["source"]).name} (화면에서 올린 그림)'
    else:
        picture = character['source']

    lines = [f'댄스 챌린지 영상 - {title or "(제목 없음)"}', f'만든 때: {time.strftime("%Y-%m-%d %H:%M")}', '',
             f'캐릭터 그림: {picture}', f'  작업에 쓴 사본: {job / "outputs/input" / character["file"]}',
             f'관절 결과: {inputs["joints"]["source"]} ({inputs["joints"]["seconds"]:.1f}초)',
             '음악: ' + ('관절을 뽑은 춤 영상의 소리' if inputs['music']['file'] and inputs['music']['from_dance_video'] else
                       inputs['music']['source'] if inputs['music']['file'] else '없음 (소리 없는 영상)')
             + (f', 1분에 {plan["bpm"]:g}박' if inputs['music']['file'] else ''), '',
             '결과 파일',
             f'  {name}_챌린지.mp4   완성 영상 ({checks["video"]["width"]}x{checks["video"]["height"]}, 1초에 {plan["fps"]:g}장, '
             f'{checks["video"]["seconds"]:.1f}초, 효과: {EFFECTS[options["effects"]]})',
             '  3D화면_만든장면_완성_나란히.mp4   [3D 안내 화면 | 영상 모델이 만든 장면 | 효과를 넣은 완성] 확인용',
             ('영상 모델: SCAIL-2 (MIT 사용 허락), ' + (f'{tier["behaves_as_gb"]:g} GB 그래픽 카드 등급, 모델 파일 {tier["unet"]}, '
                                                   f'만든 크기 {tier["size"][0]}x{tier["size"][1]}' if tier else '')
              if backend == 'scail' else '영상 모델: MiniMax H3'), '',
             f'컷 구성 ({len(plan["shots"])}장면, 정면이 나오는 시간 {plan["front_share"]:.0%})']
    cuts = {cut['between'][1]: cut for cut in plan['cuts']}
    for shot in plan['shots']:
        cut = cuts.get(shot['name'])
        lines.append(f'  {seconds(shot["start_s"])}~{seconds(shot["end_s"])}초  {shot["label"]}'
                     + (' (가까이, 허벅지 위)' if shot['framing'] == 'close' else '')
                     + (f'  - 들어올 때: {TRANSITIONS[cut["transition"]]}' if cut and options['effects'] != 'none' else ''))
    steady = plan['beat_method'].startswith('steady')  # beats.py found no usable music and laid out an even beat
    lines += ['', '프로그램이 스스로 한 것',
              (f'- 쓸 수 있는 음악이 없어 1분에 {plan["bpm"]:g}박으로 정하고 {plan["beats_per_shot"]}박마다 컷을 나눴습니다.' if steady else
               f'- 음악에서 박자를 찾아(1분에 {plan["bpm"]:g}박) {plan["beats_per_shot"]}박마다 컷을 나눴습니다.')
              + ('' if plan.get('split_at') is None else
                 ' 마디 하나는 가운데서 한 번 더 잘랐습니다.' if steady else ' 가장 센 마디 하나는 가운데서 한 번 더 잘랐습니다.'),
              '- 정면을 기준으로, 정면 사이사이에 왼쪽·오른쪽·뒤 장면을 차례로 넣었습니다. 마지막 장면은 정면입니다.',
              '  (왼쪽·오른쪽은 정면에서 보는 사람의 왼쪽·오른쪽입니다.)']
    close = [shot for shot in plan['shots'] if shot['framing'] == 'close']
    if close:  # the shots really filmed close (a filled-in part shorter than half a shot stays in a full shot)
        lines.append('- 관절 결과에서 다리를 이어 붙인 부분이 들어 있는 장면('
                     + ', '.join(f'{seconds(shot["start_s"])}~{seconds(shot["end_s"])}초' for shot in close)
                     + ')은 추정한 다리가 보이지 않게 허벅지 위만 가까이 찍었습니다.')
    if options.get('seconds'):
        lines.append(f'- 춤의 앞 {float(options["seconds"]):g}초만 만들었습니다 (--seconds).')
    elif checks['video']['seconds'] < inputs['joints']['seconds'] - 0.2:  # the scene stops at the last frame the dancer was seen
        lines.append(f'- 관절 결과는 {inputs["joints"]["seconds"]:.1f}초인데, {checks["video"]["seconds"]:.1f}초 뒤로는 원본 영상에서 '
                     '춤추는 사람이 제대로 보이지 않아(관절을 추정만 한 부분) 거기서 영상을 끝냈습니다.')
    if options['effects'] != 'none':
        lines.append(('- 컷마다 장면 전환 효과를 넣' if steady else '- 컷마다 박자의 세기에 따라 장면 전환 효과를 넣')
                     + ('고, 센 박자에는 화면이 살짝 커졌다 돌아오게 했습니다.' if plan['pulses'] else '었습니다.'))
    if description:  # the text is English (the video model reads English), so the Korean note only says where it is
        lines.append('- 그림을 읽는 작은 모델이 캐릭터의 생김새를 글로 적어 영상 모델에 함께 줬습니다 '
                     '(영어 글이라 결과_요약.json에만 적었습니다).')
    lines += ['', '확인한 것']
    lines.append(f'- 완성 영상: {checks["video"]["frames"]}장, {checks["video"]["seconds"]:.2f}초, 소리 '
                 + ('있음' if checks['video']['sound'] else '없음'))
    if checks['cuts']['furthest_from_a_beat_s'] is not None:
        lines.append(f'- 모든 컷이 박자 위에 있습니다 (박자에서 가장 먼 컷: {checks["cuts"]["furthest_from_a_beat_s"]:.2f}초).')
    named = {s['name']: s for s in plan['shots']}
    for shot in checks['shots']:
        planned = named[shot['name']]
        line = (f'- {planned["label"]}{" 가까이" if planned["framing"] == "close" else ""} '
                f'{seconds(shot["seconds"][0])}~{seconds(shot["seconds"][1])}초: 조각 {shot["chunks"]}개, '
                + ('다시 만든 것 없음' if shot['tries'] == shot['chunks'] else f'다시 만든 것 {shot["tries"] - shot["chunks"]}번')
                + f', 남은 문제 장 - 방 사라짐 {shot["frames_without_the_room"]}, '
                f'인형 남음 {shot["frames_with_a_mannequin_left"]}, 동작 어긋남 {shot["frames_out_of_step"]}'
                + (f', 뒷머리 얼굴 {shot["frames_with_a_face_from_behind"]}' if 'frames_with_a_face_from_behind' in shot else '')
                + (f', 그림에 없는 머리카락 {shot["frames_with_added_hair"]}' if 'frames_with_added_hair' in shot else ''))
        framing = shot.get('framing')
        if framing:
            line += (f'; 무릎이 화면 밖인 장 {framing["knees_out"]:.0%}' if planned['framing'] == 'close' else
                     f'; 발까지 화면 안인 장 {framing["feet_in"]:.0%}') + f', 머리가 화면 안인 장 {framing["head_in"]:.0%}'
        lines.append(line)
    if checks['problems']:
        lines += ['', '남은 문제'] + [f'- {problem}' for problem in checks['problems']]
    else:
        lines.append('- 프로그램의 검사를 모두 통과했습니다.')
    lines += ['', '알아 둘 점',
              '- 그림에 없는 옆모습과 뒷모습은 영상 모델이 그림을 보고 지어낸 것입니다.',
              '- 장면마다 따로 만들어서, 장면이 바뀔 때 옷 주름이나 머리 모양이 조금 다를 수 있습니다.',
              *(['- H3 계열 모델의 사용 허락 지역에서 한국·미국·유럽연합·영국이 빠져 있습니다.'] if backend == 'h3' else
                ['- SCAIL-2 는 MIT 사용 허락이라 지역 제한이 없습니다. 만든 크기에서 1080x1920 으로 키웠습니다.']),
              f'- 걸린 시간(단계 합): {spent / 60:.0f}분', f'- 작업 폴더: {job}']
    (results / '결과_안내.txt').write_text('\n'.join(lines) + '\n')
    (results / 'summary.json').write_text(json.dumps(dict(
        title=title, job=str(job), options=options, inputs=inputs, source=dict(title=info.get('title'), link=info.get('link')),
        plan=dict(shots=[{k: s[k] for k in ('name', 'view', 'start_s', 'end_s', 'window', 'back_share')} for s in plan['shots']],
                  cuts=plan['cuts'], bpm=plan['bpm'], front_share=plan['front_share'], close_zones=plan['close_zones']),
        checks=checks, description=description, backend=backend, tier=tier, minutes=round(spent / 60, 1),
        stage_seconds={name: stage.get('seconds') for name, stage in sorted(stages.items())}), ensure_ascii=False, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
