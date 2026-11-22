#!/usr/bin/python3
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from config import ROOT, INSTALL, COMFY, MODELS, EXAMPLES, PYTHON, PORT, URL, OWNER, SERVICE
from runtime import helper, preflight, new_job, environment, register, write_json, missing_models

RUNTIME=Path(os.environ.get('XDG_RUNTIME_DIR',f'/run/user/{os.getuid()}'))/'h3-character-swap'
STATE=RUNTIME/'state.json'


def query(path):
    with urllib.request.urlopen(URL+path,timeout=2) as r:
        return json.load(r)


def healthy():
    try:
        data=query('/system_stats')
        return 'comfyui_version' in data.get('system',{})
    except (OSError,ValueError):
        return False


def group_alive(pid):
    try:
        os.killpg(pid,0)
        return True
    except ProcessLookupError:
        return False


def cleanup(session):
    data=helper('leases')
    current=data.get('leases',[]) if isinstance(data,dict) else data
    for lease in current:
        if lease.get('owner')==OWNER and lease.get('job')==session.name:
            helper('release',lease['id'])
    register(session)


def serve():
    RUNTIME.mkdir(mode=0o700,parents=True,exist_ok=True)
    previous=Path(json.loads(STATE.read_text())['session']) if STATE.exists() else None
    if previous:
        carry_bytes=sum(p.stat().st_size for relative in ('outputs/input','outputs/user')
                        for p in (previous/relative).rglob('*') if p.is_file())
        preflight(20+carry_bytes/1024**3)
    session=new_job('h3-character-swap')
    write_json(STATE,{'session':str(session),'starting':True})
    env=environment(session)
    missing=missing_models()
    mode='setup' if missing else 'gpu'
    lease=None
    child=None
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        if not missing:
            if not helper('available').get('available'):
                raise RuntimeError('사용 가능한 GPU가 없습니다. 다른 작업이 끝난 뒤 다시 시작해 주세요.')
            acquired=helper('acquire','--owner',OWNER,'--job',session.name,'--count',1,'--ttl',7200)
            lease={'id':acquired['id'],'gpu_ids':acquired['gpu_ids']}
            del acquired
            write_json(session/'outputs/lease.json',lease)
            env['CUDA_VISIBLE_DEVICES']=','.join(map(str,lease['gpu_ids']))
            env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        else:
            env['CUDA_VISIBLE_DEVICES']=''
        if previous:
            for relative in ('outputs/input','outputs/user'):
                if (previous/relative).is_dir():
                    shutil.copytree(previous/relative,session/relative,dirs_exist_ok=True)
        for relative in ['outputs/input','outputs/results','outputs/user/default/workflows','cache/comfy-temp','cache/previews']:
            (session/relative).mkdir(parents=True,exist_ok=True)
        MODELS.mkdir(parents=True,exist_ok=True)
        for preset in (ROOT/'workflows').glob('*.json'):
            target=session/'outputs/user/default/workflows'/preset.name
            if not target.exists():
                shutil.copy2(preset,target)
        for sample in EXAMPLES.glob('h3-demo-*'):
            target=session/'outputs/input'/sample.name
            if not target.exists():
                shutil.copy2(sample,target)
        settings_file=session/'outputs/user/default/comfy.settings.json'
        if not settings_file.exists():
            write_json(settings_file,{
                'Comfy.ColorPalette':'dark','Comfy.Locale':'ko','Comfy.Workflow.WorkflowTabsPosition':'Topbar',
                'Comfy.UseNewMenu':'Top','Comfy.User.ShowSettingsButton':True,
                'Comfy.TutorialCompleted':True,'LiteGraph.Canvas.MinFontSizeForLOD':0})
        cmd=[str(PYTHON),str(COMFY/'main.py'),'--listen','127.0.0.1','--port',str(PORT),
             '--models-directory',str(MODELS),'--input-directory',str(session/'outputs/input'),
             '--output-directory',str(session/'outputs/results'),'--user-directory',str(session/'outputs/user'),
             '--temp-directory',str(session/'cache/comfy-temp'),'--previews-directory',str(session/'cache/previews'),
             '--database-url','sqlite:///'+str(session/'cache/comfy.sqlite'),'--max-upload-size','1024',
             '--offline','--disable-auto-launch','--disable-all-custom-nodes','--whitelist-custom-nodes',
             'ComfyUI-KJNodes','h3_local','--log-stdout']
        if missing:
            cmd+=['--cpu']
        else:
            cmd+=['--use-pytorch-cross-attention','--reserve-vram','2']
        write_json(session/'outputs/session.json',{'mode':mode,'missing_models':missing,'command':cmd})
        with open(session/'logs/server.log','a') as log:
            child=subprocess.Popen(cmd,cwd=COMFY,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_json(STATE,{'session':str(session),'pid':child.pid,'process_group':child.pid,'mode':mode,'url':URL})
            renew_at=time.monotonic()+600
            while child.poll() is None and not stopped:
                if lease and time.monotonic()>renew_at:
                    helper('renew',lease['id'],'--ttl',7200)
                    renew_at=time.monotonic()+600
                time.sleep(.5)
            if not stopped and child.poll() is not None:
                raise RuntimeError(f'ComfyUI 종료 코드 {child.returncode}: {session}/logs/server.log')
    finally:
        if child:
            if group_alive(child.pid):
                os.killpg(child.pid,signal.SIGTERM)
            deadline=time.monotonic()+20
            while group_alive(child.pid) and time.monotonic()<deadline:
                child.poll()
                time.sleep(.25)
            if group_alive(child.pid):
                os.killpg(child.pid,signal.SIGKILL)
            child.wait()
        if lease:
            try:
                helper('release',lease['id'])
            except RuntimeError:
                pass  # ExecStopPost retries after all owned processes exit.
        register(session)
        write_json(STATE,{'session':str(session),'stopped':True,'url':URL})


def systemctl(*args,check=True):
    return subprocess.run(['systemctl','--user',*args],check=check,capture_output=True,text=True)


def start(open_browser):
    if systemctl('is-active','--quiet',SERVICE,check=False).returncode:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:
                sock.bind(('127.0.0.1',PORT))
            except OSError as e:
                raise RuntimeError(f'{PORT} 포트를 다른 프로그램이 사용하고 있습니다.') from e
    systemctl('start',SERVICE)
    for _ in range(180):
        if healthy():
            if open_browser:
                subprocess.Popen(['xdg-open',URL],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            print(URL)
            return
        if systemctl('is-active','--quiet',SERVICE,check=False).returncode:
            state=json.loads(STATE.read_text()) if STATE.exists() else {}
            raise RuntimeError('서버 시작 실패. 로그: '+str(state.get('session',''))+'/logs/server.log')
        time.sleep(.5)
    raise RuntimeError('서버 준비 시간이 초과됐습니다. 서버 로그를 확인해 주세요.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','stop','serve','cleanup','status'])
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args()
    if args.action=='serve':
        serve()
    elif args.action=='start':
        start(args.open)
    elif args.action=='stop':
        systemctl('stop',SERVICE)
        print('H3 서버를 중지했습니다.')
    elif args.action=='cleanup':
        if STATE.is_file():
            cleanup(Path(json.loads(STATE.read_text())['session']))
    else:
        print(json.dumps({'running':healthy(),'url':URL,'missing_models':missing_models()},ensure_ascii=False))

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print(str(e),file=sys.stderr)
        raise SystemExit(1)
