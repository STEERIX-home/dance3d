#!/usr/bin/python3
"""Start and stop the two-dancer page (duo_app.py) as a user service.

    duo_control.py start [--open] | stop | status       what the launcher buttons call
    duo_control.py serve | cleanup                       what the service itself runs

Where the tools are comes from the config file (programs/duo-scene/paths.py). The page gets a run folder
of its own for its small files (uploads, run logs); every run it starts makes its own run folder
and holds its own GPU lease when the config names a gpu_helper (duo_pipeline.py).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import paths  # noqa: E402
from run_support import new_run, register  # noqa: E402

SERVICE = 'duo-dance.service'
URL = 'http://127.0.0.1:7863'
OWNER = 'duo-dance'


def runtime():
    """Where the page notes which run folder it uses: the user's runtime folder, else the runs folder."""
    base = os.environ.get('XDG_RUNTIME_DIR')
    return (Path(base) if base else paths.RUNS / '.pages') / 'duo-dance'


def systemctl(*args, check=True):
    return subprocess.run(['systemctl', '--user', *args], capture_output=True, text=True, check=check)


def healthy():
    try:
        with urllib.request.urlopen(URL + '/config', timeout=2) as response:
            return json.load(response).get('title') == '2인 캐릭터 춤'
    except (OSError, ValueError):
        return False


def serve():
    job = new_run(OWNER, 'duo-dance-ui')
    state = runtime() / 'server.json'
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(dict(job=str(job), url=URL)))
    env = os.environ.copy()
    env.update(DUO_UI_STATE=str(job / 'outputs/ui'), GRADIO_TEMP_DIR=str(job / 'cache/gradio'), GRADIO_ANALYTICS_ENABLED='False',
               TMPDIR=str(job / 'cache/tmp'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1')
    (job / 'cache/tmp').mkdir(parents=True, exist_ok=True)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (job / 'logs/ui.log').open('a') as log:
        child = subprocess.Popen([str(paths.UI_PYTHON), '-B', str(ROOT / 'duo_app.py')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None and not stopped:
                time.sleep(.5)
            time.sleep(.3)  # a stop of the service reaches the page and this process at the same moment
            if not stopped and child.poll() is not None and child.returncode not in (0, -signal.SIGTERM):
                raise RuntimeError('화면 실행 실패: ' + str(job / 'logs/ui.log'))
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def cleanup():
    """After the service stopped: a run it started is asked to stop (it releases its own lease and
    registers its own job), and the page's job is registered."""
    file = runtime() / 'server.json'
    if not file.is_file():
        return
    state = json.loads(file.read_text())
    active = Path(state['job']) / 'outputs/ui/active.json'
    if active.is_file():
        pid = json.loads(active.read_text()).get('pid')
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, TypeError, PermissionError):
            pass
    register(Path(state['job']))
    file.write_text(json.dumps({**state, 'stopped': True}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop', 'serve', 'cleanup', 'status'])
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    if args.action == 'serve':
        serve()
    elif args.action == 'cleanup':
        cleanup()
    elif args.action == 'start':
        systemctl('start', SERVICE)
        for _ in range(180):
            if healthy():
                if args.open:
                    subprocess.Popen(['xdg-open', URL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                print(URL)
                return
            if systemctl('is-active', '--quiet', SERVICE, check=False).returncode:
                raise RuntimeError('화면 서버를 시작하지 못했습니다. ' + str(runtime() / 'server.json') + '에 적힌 작업 폴더의 logs/ui.log를 '
                                   '확인해 주세요. / The page server did not start: see logs/ui.log of the run folder named there.')
            time.sleep(.5)
        raise RuntimeError('화면 준비 시간이 초과됐습니다.')
    elif args.action == 'stop':
        systemctl('stop', SERVICE)
        print('2인 캐릭터 춤 화면을 중지했습니다. 실행 중이던 작업에는 중지를 요청했습니다.')
    else:
        print(json.dumps(dict(running=healthy(), url=URL), ensure_ascii=False))


if __name__ == '__main__':
    main()
