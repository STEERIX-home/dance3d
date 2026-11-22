"""Local Korean page for dance_joints.py: a list of dance videos in, one button, joint files out.

The page only starts dance_joints.py and shows how far it is; everything that decides anything is
in the program, so the same run can be started from a terminal with the same result.
Started by joints_control.py (which gives it a run folder for its own small files).
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

PORT = 7864
TITLE = '춤 관절 정보'
STATE = Path(os.environ['JOINTS_UI_STATE'])
STATE.mkdir(parents=True, exist_ok=True)
RESULTS = paths.RESULTS  # one folder per run in the results folder of the config (results_dir)
MIRROR = {'제목에 거울모드가 있으면 되돌리기 (실제로 춘 방향)': 'auto', '화면 그대로': 'keep', '모두 되돌리기': 'flip'}
STATES = {'waiting': '기다림', 'running': '진행 중', 'done': '끝', 'failed': '실패'}


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
    """The run folder folder the program printed when it started."""
    for line in log.read_text(errors='replace').splitlines()[:5]:
        if '] job /' in line:
            return Path(line.split('] job ', 1)[1].split(':')[0].strip())
    return None


def table(progress):
    rows = ['| 번호 | 제목 | 상태 | 지금 |', '| --- | --- | --- | --- |']
    for video in progress.get('videos', []):
        step = (video.get('error') or '').splitlines()[0] if video.get('state') == 'failed' else video.get('step', '')
        rows.append(f'| {video["number"]:02d} | {video["title"] or "(유튜브 제목)"} | {STATES.get(video["state"], video["state"])} | {step} |')
    return '\n'.join(rows)


def follow(process, log):
    """Yield (status text, summary file, folder) while the program runs and once more at its end."""
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
            count = f' · {progress.get("done", 0)}/{progress["total"]}개 끝' if progress.get('total') else ''
            yield f'**진행 중**{count} · {minutes:.0f}분 지남\n\n{table(progress)}', None, str(job or '')
            time.sleep(3)
            continue
        target = RESULTS / job.name if job else None
        summary = target / '결과_안내.txt' if target else None
        if progress.get('state') == 'done':
            yield (f'**끝났습니다** · {minutes:.0f}분 걸림\n\n결과 폴더: `{target}`\n\n{table(progress)}',
                   str(summary) if summary and summary.exists() else None, str(job))
        else:
            error = progress.get('error') or '\n'.join(log.read_text(errors='replace').splitlines()[-8:])
            yield (f'**멈췄습니다**\n\n```\n{error}\n```\n\n{table(progress)}\n\n'
                   '고칠 것이 없으면 「멈춘 작업 이어서 하기」를 누르면 끝난 단계는 건너뛰고 이어서 합니다.',
                   str(summary) if summary and summary.exists() else None, str(job or ''))
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


def run(text, mirror):
    lines = [line for line in (text or '').splitlines() if line.strip() and not line.strip().startswith('#')]
    if not lines:
        raise gr.Error('영상 목록을 넣어 주세요. 한 줄에 제목과 유튜브 링크 하나씩입니다.')
    listing = STATE / f'list-{time.strftime("%Y%m%d-%H%M%S")}.txt'
    listing.write_text(text.strip() + '\n', encoding='utf-8')
    process, log = launch([sys.executable, '-B', str(ROOT / 'dance_joints.py'), '--list', str(listing), '--mirror', MIRROR[mirror]])
    yield from follow(process, log)


def resume(job):
    if not job and (STATE / 'last_job.txt').exists():  # the page was reloaded: the last run started from here
        job = (STATE / 'last_job.txt').read_text().strip()
    if not job or not (Path(job) / 'outputs/options.json').exists():
        raise gr.Error('이어서 할 작업이 없습니다.')
    process, log = launch([sys.executable, '-B', str(ROOT / 'dance_joints.py'), '--job', job])
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
    gr.Markdown('# 춤 영상 목록 → 영상마다 한 사람의 3D 관절 움직임\n'
                '영상마다 화면에 크게 나오는 한 사람의 춤을 3D 관절로 뽑아, 동작 변환기에 넣을 파일로 만듭니다. '
                '**목록을 넣고 전체 실행을 누르면 끝까지 알아서 합니다.**\n\n'
                f'결과는 결과 폴더 `{RESULTS}` 에 작업마다 새 폴더로 저장됩니다.')
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='영상 목록 (한 줄에 영상 하나: 제목 + 유튜브 링크)', lines=16,
                              placeholder='NEON https://youtube.com/shorts/VIDEO_ID_1\nL을가져가 https://youtube.com/shorts/VIDEO_ID_2')
            gr.Markdown('제목을 비우면 유튜브 제목을 씁니다. 링크 대신 이 컴퓨터의 영상 파일 경로를 써도 됩니다. '
                        '30초 쇼츠 하나에 몇 분 걸립니다.')
            mirror = gr.Radio(list(MIRROR), value=list(MIRROR)[0], label='좌우 되돌리기')
            gr.Markdown('거울모드 영상은 좌우가 뒤집힌 화면이라, 그대로 뽑으면 왼손·오른손이 바뀐 안무가 됩니다. '
                        '기본은 제목에 "거울모드", "mirror mode", "mirrored" 같은 말이 있는 영상만 되돌려 실제로 춘 방향으로 뽑습니다.')
            full = gr.Button('전체 실행 · 내려받기 → 3D 관절 뽑기 → 결과 폴더에 저장', variant='primary')
            with gr.Row():
                again = gr.Button('멈춘 작업 이어서 하기')
                halt = gr.Button('현재 작업 중지', variant='stop')
        with gr.Column():
            status = gr.Markdown('목록을 넣고 전체 실행을 누르세요.')
            summary = gr.File(label='결과 안내 (영상마다 나온 것과 알아 둘 것)')
            gr.Markdown('끝난 영상은 하나씩 결과 폴더에 먼저 들어갑니다. 화면에 나오지 않은 몸(화면 밖의 다리, 가려진 팔)은 추정이고, '
                        '어느 구간인지 결과 안내에 적힙니다. 원본 위에 관절을 그린 확인 영상은 `검증용_원본포함_공유금지` 폴더에만 있습니다.')
    current = gr.State('')
    outputs = [status, summary, current]
    full.click(run, [text, mirror], outputs, concurrency_id='program', concurrency_limit=1, api_name='run')
    again.click(resume, [current], outputs, concurrency_id='program', concurrency_limit=1, api_name='resume')
    halt.click(stop, outputs=status, queue=False, api_name='stop')

if __name__ == '__main__':
    demo.queue(max_size=2).launch(server_name='127.0.0.1', server_port=PORT, share=False, allowed_paths=[str(RESULTS)],
                                  show_error=True, inbrowser=False)
