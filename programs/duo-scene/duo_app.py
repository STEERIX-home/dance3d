"""Local Korean page for the two-dancer program: a video and two pictures in, one button, videos out.

The page only starts duo_pipeline.py and shows how far it is; everything that decides anything is
in the pipeline, so the same run can be started from a terminal with the same result.
Started by duo_control.py (which gives it a run folder for its own small files).
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
sys.path.insert(0, str(ROOT))
import paths  # noqa: E402

PORT = 7863
TITLE = '2인 캐릭터 춤'
STATE = Path(os.environ['DUO_UI_STATE'])
STATE.mkdir(parents=True, exist_ok=True)
RESULTS = paths.RESULTS  # one folder per run in the results folder of the config (results_dir)
CAMERAS = {'원본과 같은 카메라': 'original', '360도 회전': 'orbit', '다이내믹 카메라': 'dynamic', '정면': 'front'}
FILES = {'original': '1_원본과_같은_카메라.mp4', 'orbit': '2_360도_회전.mp4', 'dynamic': '3_다이내믹_카메라.mp4', 'front': '4_정면.mp4'}
BASES = {'공식 H3 · 3D로 만든 방을 그대로 둡니다': 'official',
         'Singularity V2 · 방을 실제 연습실처럼 다시 그립니다 (빠른 동작이 조금 무뎌집니다)': 'singularity'}


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


def job_of(log):
    """The run folder folder the pipeline printed when it started."""
    for line in log.read_text(errors='replace').splitlines()[:5]:
        if '] job /' in line:
            return Path(line.split('] job ', 1)[1].strip())
    return None


def follow(process, log):
    """Yield (status text, videos, summary file, folder) while the pipeline runs and once more at its end."""
    empty = [None] * len(FILES)
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
            share = f' · {progress["steps_done"]}/{progress["steps_total"]} 단계' if progress.get('steps_total') else ''
            yield f'**진행 중**{share} · {minutes:.0f}분 지남\n\n지금: {progress.get("step", "시작하는 중")}', *empty, None, str(job or '')
            time.sleep(3)
            continue
        if progress.get('state') == 'done':
            target = RESULTS / job.name
            videos = [str(target / name) if (target / name).exists() else None for name in FILES.values()]
            summary = target / '결과_요약.html'
            yield (f'**끝났습니다** · {minutes:.0f}분 걸림\n\n결과 폴더: `{target}`\n\n`결과_요약.html`에 자동 확인 결과가 있습니다.',
                   *videos, str(summary) if summary.exists() else None, str(job))
        else:
            error = progress.get('error') or '\n'.join(log.read_text(errors='replace').splitlines()[-8:])
            yield (f'**멈췄습니다**\n\n```\n{error}\n```\n\n고칠 것이 없으면 「멈춘 작업 이어서 하기」를 누르면 끝난 단계는 건너뛰고 이어서 합니다.',
                   *empty, None, str(job or ''))
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


def run(video, left, right, left_cm, right_cm, left_outfit, right_outfit, cameras, base, start, seconds):
    if not video or not left or not right:
        raise gr.Error('춤 영상 하나와 캐릭터 그림 두 장을 모두 넣어 주세요.')
    if not cameras:
        raise gr.Error('만들 카메라를 하나 이상 골라 주세요.')
    command = [sys.executable, '-B', str(ROOT / 'duo_pipeline.py'), '--video', video, '--character', left, '--character', right,
               '--cameras', ','.join(CAMERAS[name] for name in cameras), '--base', BASES[base], '--start', str(float(start or 0))]
    if seconds:
        command += ['--seconds', str(float(seconds))]
    for number, centimetres in ((1, left_cm), (2, right_cm)):
        if centimetres:
            command += ['--height', f'{number}={float(centimetres):g}']
    for number, outfit in ((1, left_outfit), (2, right_outfit)):
        if outfit and outfit.strip():
            command += ['--outfit', f'{number}={" ".join(outfit.split())}']
    process, log = launch(command)
    yield from follow(process, log)


def newest_job():
    """The newest run of this program (its job folders carry the start time in their names); '' if there is none."""
    jobs = sorted(path.name for path in paths.RUNS.glob('*-duo-dance-*')
                  if '-ui-' not in path.name and (path / 'outputs/options.json').exists())
    return str(paths.RUNS / jobs[-1]) if jobs else ''


def resume(job):
    if not job and (STATE / 'last_job.txt').exists():  # the page was reloaded: the last run started from here
        job = (STATE / 'last_job.txt').read_text().strip()
    if not job:  # the page itself was started anew since (it keeps its notes per start): the newest run there is
        job = newest_job()
    if not job or not (Path(job) / 'outputs/options.json').exists():
        raise gr.Error('이어서 할 작업이 없습니다.')
    process, log = launch([sys.executable, '-B', str(ROOT / 'duo_pipeline.py'), '--job', job])
    yield from follow(process, log)


def stop():
    data = active()
    if not data:
        return '실행 중인 작업이 없습니다.'
    os.killpg(data['pid'], signal.SIGTERM)
    return '중지를 요청했습니다. 지금 단계가 멈추고 GPU를 돌려줍니다. 끝난 단계의 결과는 남아 있어 이어서 할 수 있습니다.'


with gr.Blocks(title=TITLE, analytics_enabled=False,
               theme=gr.themes.Soft(primary_hue='indigo', font=['Arial', 'sans-serif'], font_mono=['monospace']),
               css='.gradio-container {max-width: 1320px !important;}') as demo:
    gr.Markdown('# 두 사람 춤 영상 + 캐릭터 그림 두 장 → 두 캐릭터의 춤\n'
                '두 사람의 춤을 3D로 뽑아 고정된 3D 연습실에 세우고, 각 사람에게 캐릭터를 입혀 원본 영상과 같은 카메라로 찍습니다. '
                '**영상과 그림 두 장을 넣고 전체 실행을 누르면 끝까지 알아서 합니다.**\n\n'
                f'결과는 결과 폴더 `{RESULTS}` 에 작업마다 새 폴더로 저장됩니다.')
    with gr.Row():
        with gr.Column():
            video = gr.Video(label='1 · 두 사람이 춤추는 영상', sources=['upload'])
            with gr.Row():
                left = gr.Image(label='2 · 왼쪽 사람에게 입힐 캐릭터', type='filepath', sources=['upload'])
                right = gr.Image(label='3 · 오른쪽 사람에게 입힐 캐릭터', type='filepath', sources=['upload'])
            gr.Markdown('왼쪽·오른쪽은 두 사람이 가장 또렷이 떨어져 보이는 장면 기준입니다(결과 폴더의 `사람_배정_확인` 영상에서 볼 수 있습니다). '
                        '두 사람의 전신이 보이고 장면이 끊기지 않는 영상이 좋습니다. 캐릭터는 한 명만 나온 그림을 씁니다.')
            with gr.Row():
                left_cm = gr.Number(label='왼쪽 캐릭터 키 (cm, 비우면 원본 사람 크기)', value=None, minimum=50, maximum=250)
                right_cm = gr.Number(label='오른쪽 캐릭터 키 (cm, 비우면 원본 사람 크기)', value=None, minimum=50, maximum=250)
            with gr.Accordion('옷 바꾸기 (선택)', open=False):
                gr.Markdown('비워 두면 그림 속 옷 그대로입니다. 적으면 얼굴·머리·몸은 그림대로 두고 옷만 바꿉니다. '
                            '영어로 적는 것이 가장 잘 듣습니다. 예: `a red hoodie, blue jeans and white sneakers`')
                with gr.Row():
                    left_outfit = gr.Textbox(label='왼쪽 캐릭터의 옷', value='', lines=1)
                    right_outfit = gr.Textbox(label='오른쪽 캐릭터의 옷', value='', lines=1)
            cameras = gr.CheckboxGroup(list(CAMERAS), value=list(CAMERAS)[:1], label='만들 카메라 (하나에 약 50분)')
            gr.Markdown('**원본과 같은 카메라**는 원본 영상의 카메라 움직임(위치·방향·기울기)을 그대로 따라가서, 원본에 찍힌 부분만 보여 줍니다. '
                        '나머지 세 카메라는 원본에 찍히지 않은 부분(화면 밖의 다리, 가려진 몸)도 보여 주는데 그 부분은 추정이라 어색할 수 있습니다.')
            base = gr.Radio(list(BASES), value=list(BASES)[0], label='H3 본체')
            with gr.Accordion('영상의 일부만 쓰기 (선택)', open=False):
                with gr.Row():
                    start = gr.Number(label='시작 위치 (초)', value=0, minimum=0)
                    seconds = gr.Number(label='길이 (초, 비우면 끝까지)', value=None, minimum=2.5)
            full = gr.Button('전체 실행 · 춤 추출 → 캐릭터 입히기 → 결과 폴더에 저장', variant='primary')
            with gr.Row():
                again = gr.Button('멈춘 작업 이어서 하기')
                halt = gr.Button('현재 작업 중지', variant='stop')
        with gr.Column():
            status = gr.Markdown('영상과 그림을 넣고 전체 실행을 누르세요.')
            results = [gr.Video(label=name) for name in CAMERAS]
            summary = gr.File(label='결과 요약 (자동 확인 결과)')
            gr.Markdown('만든 영상은 하나씩 끝나는 대로 결과 폴더에 먼저 들어갑니다. 원본에서 가려지거나 화면 밖이던 동작은 추정이고, '
                        '표정은 원본 사람의 표정을 옮긴 것이 아니라 모델이 춤에 맞춰 만든 것입니다. '
                        '얼굴 보정은 캐릭터 그림에서 얼굴 하나를 또렷이 찾았을 때만 합니다.')
    current = gr.State('')
    outputs = [status, *results, summary, current]
    full.click(run, [video, left, right, left_cm, right_cm, left_outfit, right_outfit, cameras, base, start, seconds], outputs,
               concurrency_id='pipeline',
               concurrency_limit=1, api_name='run')
    again.click(resume, [current], outputs, concurrency_id='pipeline', concurrency_limit=1, api_name='resume')
    halt.click(stop, outputs=status, queue=False, api_name='stop')

if __name__ == '__main__':
    demo.queue(max_size=2).launch(server_name='127.0.0.1', server_port=PORT, share=False, allowed_paths=[str(RESULTS)],
                                  max_file_size='2gb', show_error=True, inbrowser=False)
