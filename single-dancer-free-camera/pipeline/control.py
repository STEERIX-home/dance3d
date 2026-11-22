#!/usr/bin/python3
import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import runner
from settings import ROOT, SERVICE, UI_PYTHON, URL

STATE = runner.RUNTIME / 'server.json'


def process_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def systemctl(*args, check=True):
    return subprocess.run(['systemctl','--user',*args],capture_output=True,text=True,check=check)


def healthy():
    try:
        with urllib.request.urlopen(URL+'/config',timeout=2) as response:
            return json.load(response).get('title') == '3D 모캡 → H3 캐릭터 스왑'
    except (OSError, ValueError):
        return False


def serve():
    runner.preflight(10)
    job = Path(runner.helper('job-init','--owner',runner.OWNER,'--job','rgbmocap-ui')['job_dir'])
    runner.write_json(STATE,{'job':str(job),'url':URL})
    env = runner.environment(job)
    env.update(RGBMOCAP_UI_JOB=str(job),GRADIO_TEMP_DIR=str(job/'cache/gradio'),
               GRADIO_ANALYTICS_ENABLED='False')
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (job/'logs/ui.log').open('a') as log:
        child = subprocess.Popen([str(UI_PYTHON),str(ROOT/'app.py')],cwd=ROOT,env=env,
                                 stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None and not stopped:
                time.sleep(.5)
            if child.poll() is not None and child.returncode:
                raise RuntimeError('UI 실행 실패: '+str(job/'logs/ui.log'))
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def cleanup():
    # systemd has stopped this service's entire cgroup before ExecStopPost.
    active = runner.RUNTIME/'active.json'
    if active.is_file():
        data = json.loads(active.read_text())
        producer_alive = process_alive(data.get('pid'))
        if not data.get('finished') and not producer_alive:
            job = Path(data['job'])
            record = job/'outputs/h3-run.json'
            if record.is_file() and runner.h3_running():
                try:
                    runner.cancel_h3_prompt(json.loads(record.read_text())['prompt_id'])
                except Exception as error:
                    runner.write_json(job/'outputs/stop-error.json',{'error':str(error)})
            leases = runner.helper('leases')
            for lease in leases.get('leases',[]) if isinstance(leases,dict) else leases:
                if lease.get('owner')==runner.OWNER and lease.get('job')==job.name:
                    runner.helper('release',lease['id'])
            try:
                runner.helper('artifact-register','--path',job)
            except RuntimeError as error:
                if 'already registered' not in str(error):
                    raise
            runner.write_json(active,{'job':str(job),'finished':True,'stopped':True})
    if STATE.is_file():
        state=json.loads(STATE.read_text())
        try:
            runner.helper('artifact-register','--path',state['job'])
        except RuntimeError as error:
            if 'already registered' not in str(error):
                raise
        runner.write_json(STATE,{**state,'stopped':True})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','stop','serve','cleanup','status'])
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args()
    if args.action=='serve':
        serve()
    elif args.action=='cleanup':
        cleanup()
    elif args.action=='start':
        systemctl('start',SERVICE)
        for _ in range(180):
            if healthy():
                if args.open:
                    subprocess.Popen(['xdg-open',URL],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
                print(URL)
                return
            if systemctl('is-active','--quiet',SERVICE,check=False).returncode:
                raise RuntimeError('3D UI 서버 시작 실패. '+str(STATE)+'의 작업 로그를 확인해 주세요.')
            time.sleep(.5)
        raise RuntimeError('UI 준비 시간이 초과됐습니다.')
    elif args.action=='stop':
        systemctl('stop',SERVICE)
        h3_stopped = not runner.h3_running()
        if runner.h3_running():
            queue=runner.get_json(runner.H3_URL+'/queue')
            owned_prompt = None
            active = runner.RUNTIME/'active.json'
            if active.is_file():
                data = json.loads(active.read_text())
                record = Path(data['job'])/'outputs/h3-run.json'
                if record.is_file() and not process_alive(data.get('pid')):
                    owned_prompt = json.loads(record.read_text())['prompt_id']
            ids = {item[1] for item in queue['queue_running']+queue['queue_pending']}
            if not ids or (owned_prompt is not None and ids <= {owned_prompt}):
                runner.h3_control('stop')
                h3_stopped = True
        print('3D 모캡 UI와 H3 서버를 중지했습니다.' if h3_stopped else
              '3D 모캡 UI를 중지했습니다. 다른 작업이 있는 H3 서버는 계속 실행합니다.')
    else:
        print(json.dumps({'running':healthy(),'url':URL,'missing_models':runner.missing_models()},ensure_ascii=False))


if __name__=='__main__':
    main()
