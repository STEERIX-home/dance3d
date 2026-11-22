import json
import os
import subprocess
import urllib.request
from pathlib import Path
from config import HELPER, OWNER, MODEL_INSTALL, MODELS, MODEL_FILES, PYTHON_HEADERS


def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    temp.replace(path)


def helper(*args):
    result=subprocess.run([str(HELPER),*map(str,args)],capture_output=True,text=True,timeout=30)
    if result.returncode:
        raise RuntimeError(f'GPU 관리 명령 {args[0]} 실패: {result.stderr.strip()[:400]}')
    return json.loads(result.stdout)


def preflight(estimate_gib=20):
    with urllib.request.urlopen(os.environ.get('GPU_DASHBOARD_URL','http://127.0.0.1:8080').rstrip('/')+'/api/v1/status',timeout=10) as response:
        status=json.load(response)
    disk=status['host']['disk']
    if disk['free_gib']-estimate_gib<disk['reserve_target_gib']:
        raise RuntimeError('예상 출력과 최소 여유 공간 5%를 확보할 수 없습니다.')
    return status


def new_job(name):
    preflight()
    return Path(helper('job-init','--owner',OWNER,'--job',name)['job_dir'])


def register(job):
    try:
        helper('artifact-register','--path',job)
        write_json(job/'outputs/retention.json',{'registered':True})
    except RuntimeError as e:
        if 'already registered' not in str(e):
            write_json(job/'outputs/retention.json',{'registered':False,'error':str(e)})


def missing_models():
    manifest=MODEL_INSTALL/'outputs/models.json'
    info=json.loads(manifest.read_text()) if manifest.is_file() else {}
    missing=[]
    for repo,filename,folder in MODEL_FILES:
        p=MODELS/folder/Path(filename).name
        record=info.get(filename,{})
        if not p.is_file() or not record.get('complete') or p.stat().st_size!=record.get('size_bytes'):
            missing.append(p.name)
    return missing


def environment(job):
    env=os.environ.copy()
    env['CPATH']=os.pathsep.join(filter(None,[str(PYTHON_HEADERS/'python3.12'),str(PYTHON_HEADERS),env.get('CPATH')]))
    mapping={'TMPDIR':'cache/tmp','XDG_CACHE_HOME':'cache/xdg','HF_HOME':'cache/hf',
             'HF_HUB_CACHE':'cache/hf/hub','TORCH_HOME':'cache/torch','TORCHINDUCTOR_CACHE_DIR':'cache/inductor',
             'TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda','MPLCONFIGDIR':'cache/matplotlib'}
    for key,relative in mapping.items():
        p=job/relative
        p.mkdir(parents=True,exist_ok=True)
        env[key]=str(p)
    env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1',
                'TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1',
                'OMP_NUM_THREADS':'6','MKL_NUM_THREADS':'6','H3_JOB_DIR':str(job)})
    return env
