"""Local Korean page for challenge.py: a joints result and a character picture in, one button, the
challenge video out.

The page only starts challenge.py and shows how far it is; everything that decides anything is in
the program, so the same run can be started from a terminal with the same result.
Started by challenge_control.py (which gives it a run folder for its own small files). Results are read from
and written to the results folder of the config (results_dir); runs are in runs_dir.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'duo-scene'))
import paths  # noqa: E402

PORT = 7865
TITLE = '댄스 챌린지'
STATE = Path(os.environ['CHALLENGE_UI_STATE'])
STATE.mkdir(parents=True, exist_ok=True)
RESULTS = paths.RESULTS  # the challenge videos (one folder per run) and the joints results of dance-joints
EFFECTS = {'보통': 'normal', '강하게': 'strong', '없음 (컷만)': 'none'}
STATES = {'running': '진행 중', 'done': '끝', 'failed': '멈춤'}


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, TypeError):
        return False


def active():
    record = STATE / 'active.json'
    data = json.loads(record.read_text()) if record.exists() else {}
    return data if alive(data.get('pid')) else {}


def results():
    """The joints results in the results folder (runs of dance-joints), newest run first: {shown name: folder}."""
    found = {}
    runs = sorted((p for p in RESULTS.glob('*') if p.is_dir() and (p / '결과_안내.txt').exists() and any(p.glob('[0-9][0-9]_*/results.npz'))),
                  key=lambda p: p.name, reverse=True)
    for run in runs:
        for video in sorted(run.glob('[0-9][0-9]_*')):
            if (video / 'results.npz').is_file():
                found[f'{video.name}   ({run.name})'] = str(video)
    return found


def job_of(log):
    for line in log.read_text(errors='replace').splitlines()[:5]:
        if '] run /' in line:
            return Path(line.split('] run ', 1)[1].split(' (backend', 1)[0].strip())
    return None


def follow(process, log):
    """Yield (status text, video, note, job) while the program runs and once more at its end."""
    job = None
    while True:
        running = process.poll() is None
        if job is None:
            job = job_of(log)
            if job:
                (STATE / 'last_job.txt').write_text(str(job))
        progress = {}
        if job and (job / 'outputs/progress.json').exists():
            try:
                progress = json.loads((job / 'outputs/progress.json').read_text())
            except ValueError:  # being written this instant
                progress = {}
        minutes = (time.time() - progress.get('started', time.time())) / 60
        if running:
            count = f' · {progress.get("steps_done", 0)}/{progress["steps_total"]}단계' if progress.get('steps_total') else ''
            yield f'**진행 중**{count} · {minutes:.0f}분 지남\n\n지금: {progress.get("step", "준비 중")}', None, None, str(job or '')
            time.sleep(3)
            continue
        target = RESULTS / job.name if job else None
        if progress.get('state') == 'done' and target and target.exists():
            video = next(iter(sorted(target.glob('*_챌린지.mp4'))), None)
            note = target / '결과_안내.txt'
            yield (f'**끝났습니다** · {minutes:.0f}분 걸림\n\n결과 폴더: `{target}`', str(video) if video else None,
                   str(note) if note.exists() else None, str(job))
        else:
            error = progress.get('error') or '\n'.join(log.read_text(errors='replace').splitlines()[-8:])
            yield (f'**멈췄습니다**\n\n```\n{error}\n```\n\n고칠 것이 없으면 「멈춘 작업 이어서 하기」를 누르면 끝난 단계는 건너뛰고 이어서 합니다.',
                   None, None, str(job or ''))
        return


def launch(command):
    if active():
        raise gr.Error('이미 실행 중인 작업이 있습니다. 끝나거나 중지한 뒤 다시 시작해 주세요.')
    log = STATE / f'run-{time.strftime("%Y%m%d-%H%M%S")}.log'
    with log.open('w') as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
    (STATE / 'active.json').write_text(json.dumps(dict(pid=process.pid, log=str(log), started=time.time())))
    return process, log


def run(joints, picture, music, use_title, title, effects, labels):
    folder = results().get(joints, joints)
    if not folder or not Path(folder).exists():
        raise gr.Error('관절 결과를 골라 주세요. 「춤 관절 정보」로 만든 영상 폴더입니다.')
    if not picture:
        raise gr.Error('캐릭터 그림을 넣어 주세요. 캐릭터 한 명의 전신 그림이면 됩니다.')
    command = [sys.executable, '-B', str(ROOT / 'challenge.py'), '--joints', folder, '--character', picture,
               '--effects', EFFECTS[effects]]
    if music:
        command += ['--music', music]
    if not use_title:
        command += ['--title', '']
    elif title and title.strip():
        command += ['--title', title.strip()]
    if not labels:
        command.append('--no-labels')
    process, log = launch(command)
    yield from follow(process, log)


def newest_job():
    """The newest run of this program in the runs folder (its folders carry the start time in their names); '' if there is none."""
    jobs = sorted(path.name for path in paths.RUNS.glob('*-dance-challenge-*')
                  if '-ui-' not in path.name and (path / 'outputs/options.json').exists())
    return str(paths.RUNS / jobs[-1]) if jobs else ''


def resume(job):
    if not job and (STATE / 'last_job.txt').exists():  # the page was reloaded: the last run started from here
        job = (STATE / 'last_job.txt').read_text().strip()
    if not job:  # the page itself was started anew since (it keeps its notes per start): the newest run there is
        job = newest_job()
    if not job or not (Path(job) / 'outputs/options.json').exists():
        raise gr.Error('이어서 할 작업이 없습니다.')
    process, log = launch([sys.executable, '-B', str(ROOT / 'challenge.py'), '--job', job])
    yield from follow(process, log)


def stop():
    data = active()
    if not data:
        return '실행 중인 작업이 없습니다.'
    os.killpg(data['pid'], signal.SIGTERM)
    return '중지를 요청했습니다. 지금 단계가 멈추고 GPU를 돌려줍니다. 끝난 단계의 결과는 남아 있어 이어서 할 수 있습니다.'


def refreshed():
    choices = list(results())
    return gr.update(choices=choices, value=choices[0] if choices else None)


with gr.Blocks(title=TITLE, analytics_enabled=False,
               theme=gr.themes.Soft(primary_hue='rose', font=['Arial', 'sans-serif'], font_mono=['monospace']),
               css='.gradio-container {max-width: 1320px !important;}') as demo:
    gr.Markdown('# 댄스 챌린지 만들기\n'
                '「춤 관절 정보」로 뽑은 춤 동작에 캐릭터를 입혀, 3D 무대에서 **정면을 기준으로 왼쪽·오른쪽·뒤 장면을 음악 박자에 맞춰 '
                '잘라 넣고 효과를 준** 세로 챌린지 영상을 만듭니다. **고르고 전체 실행을 누르면 끝까지 알아서 합니다.**\n\n'
                f'결과는 결과 폴더 `{RESULTS}` 에 작업마다 새 폴더로 저장됩니다.')
    with gr.Row():
        with gr.Column():
            choices = list(results())
            joints = gr.Dropdown(choices, value=choices[0] if choices else None, allow_custom_value=True,
                                 label='1 · 춤 동작 (「춤 관절 정보 만들기」의 영상 폴더)')
            refresh = gr.Button('목록 새로 읽기', size='sm')
            picture = gr.Image(label='2 · 캐릭터 그림 (한 명, 전신)', type='filepath', sources=['upload'], height=360)
            with gr.Accordion('더 고르기 (안 바꿔도 됩니다)', open=False):
                music = gr.File(label='다른 음악 (비우면 동작을 뽑은 춤 영상의 소리)', type='filepath', file_types=['audio', 'video'])
                use_title = gr.Checkbox(value=True, label='첫머리에 제목 넣기')
                title = gr.Textbox(label='제목 (비우면 관절 결과의 제목, 예: L을가져가)')
                effects = gr.Radio(list(EFFECTS), value='보통', label='효과의 세기')
                labels = gr.Checkbox(value=True, label='장면마다 정면·왼쪽·오른쪽·뒤 이름 넣기')
            full = gr.Button('전체 실행 · 3D 무대 → 캐릭터 입히기 → 컷 편집과 효과 → 결과 폴더에 저장', variant='primary')
            with gr.Row():
                again = gr.Button('멈춘 작업 이어서 하기')
                halt = gr.Button('현재 작업 중지', variant='stop')
            gr.Markdown('15초 춤 하나에 30~50분쯤 걸립니다(대부분 캐릭터 입히기). 다리가 화면 밖이라 관절을 이어 붙인 구간은 '
                        '허벅지 위만 가까이 찍어 추정한 다리를 보이지 않게 합니다. 그림에 없는 옆모습과 뒷모습은 영상 모델이 지어냅니다.')
        with gr.Column():
            status = gr.Markdown('춤 동작과 캐릭터 그림을 고르고 전체 실행을 누르세요.')
            video = gr.Video(label='완성 영상', interactive=False)
            note = gr.File(label='결과 안내 (컷 구성과 자동 확인 결과)')
    current = gr.State('')
    outputs = [status, video, note, current]
    refresh.click(refreshed, outputs=joints, queue=False)
    full.click(run, [joints, picture, music, use_title, title, effects, labels], outputs, concurrency_id='program', concurrency_limit=1,
               api_name='run')
    again.click(resume, [current], outputs, concurrency_id='program', concurrency_limit=1, api_name='resume')
    halt.click(stop, outputs=status, queue=False, api_name='stop')

if __name__ == '__main__':
    demo.queue(max_size=2).launch(server_name='127.0.0.1', server_port=PORT, share=False, allowed_paths=[str(RESULTS)],
                                  show_error=True, inbrowser=False)
