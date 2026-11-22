"""Managed local pipeline. Never run a GPU subprocess without its own dashboard lease."""
import contextlib
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from settings import (BLENDER, CHECKPOINTS, EXMR, EXMR_PYTHON, H3_ROOT, H3_URL, HELPER,
                      INSTALL, JOBS, JPEG_LIB, OWNER, REQUIRED_MODELS, ROOT, missing_models, model_dir,
                      FACE_MODELS, FACE_PYTHON, UI_PYTHON, H3_FRAMES)

RUNTIME = Path(os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')) / OWNER
RUNTIME.mkdir(mode=0o700, parents=True, exist_ok=True)
CANCEL = threading.Event()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def helper(*args):
    result = subprocess.run([str(HELPER), *map(str, args)], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f'GPU 관리 명령 {args[0]} 실패: {result.stderr[-500:]}')
    return json.loads(result.stdout)


def get_json(url, data=None):
    request = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'서버 요청 실패 ({error.code}): {error.read().decode()[-1800:]}') from None


def preflight(estimate_gib=30):
    status = get_json(os.environ.get('GPU_DASHBOARD_URL', 'http://127.0.0.1:8080').rstrip('/') + '/api/v1/status')
    disk = status['host']['disk']
    if disk['free_gib'] - estimate_gib < disk['reserve_target_gib']:
        raise RuntimeError('예상 출력 공간과 서버 최소 여유 공간 5%를 확보할 수 없습니다.')
    return status


@contextlib.contextmanager
def managed_job(name):
    preflight()
    job = Path(helper('job-init', '--owner', OWNER, '--job', name)['job_dir'])
    write_json(RUNTIME / 'active.json', {'job': str(job), 'stage': name, 'pid': os.getpid()})
    try:
        yield job
    except BaseException as error:
        write_json(job / 'outputs/error.json', {'error': str(error), 'type': type(error).__name__})
        raise
    finally:
        try:
            helper('artifact-register', '--path', job)
        finally:
            write_json(RUNTIME / 'active.json', {'job': str(job), 'finished': True})


@contextlib.contextmanager
def exclusive():
    with (RUNTIME / 'pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('다른 3D → H3 작업이 실행 중입니다.') from None
        CANCEL.clear()
        yield


def environment(job, gpu_ids=()):
    env = os.environ.copy()
    paths = {'TMPDIR': 'cache/tmp', 'XDG_CACHE_HOME': 'cache/xdg', 'HF_HOME': 'cache/hf',
             'HF_HUB_CACHE': 'cache/hf/hub', 'TORCH_HOME': 'cache/torch',
             'TRITON_CACHE_DIR': 'cache/triton', 'TORCHINDUCTOR_CACHE_DIR': 'cache/inductor',
             'CUDA_CACHE_PATH': 'cache/cuda', 'MPLCONFIGDIR': 'cache/matplotlib',
             'YOLO_CONFIG_DIR': 'cache/yolo', 'BLENDER_USER_CONFIG': 'cache/blender/config',
             'BLENDER_USER_SCRIPTS': 'cache/blender/scripts',
             'BLENDER_USER_EXTENSIONS': 'cache/blender/extensions'}
    for key, relative in paths.items():
        directory = job / relative
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    # DINOv3 source is pinned by ExMR's downloader. Copy code, never credentials.
    source = CHECKPOINTS.parent / 'cache/torch/hub/facebookresearch_dinov3_main'
    target = job / 'cache/torch/hub/facebookresearch_dinov3_main'
    if source.is_dir() and not target.exists():
        shutil.copytree(source, target)
    env.update(PYTHONPATH=str(EXMR), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1',
               EXMR_CHECKPOINT_DIR=str(CHECKPOINTS), CUDA_VISIBLE_DEVICES=','.join(map(str, gpu_ids)),
               CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6',
               OPENBLAS_NUM_THREADS='6')
    env['LD_LIBRARY_PATH'] = str(JPEG_LIB) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    return env


def run(command, job, env, label, report=lambda text: None):
    log_path = job / 'logs' / (label + '.log')
    with log_path.open('a') as log:
        log.write('COMMAND ' + json.dumps(list(map(str, command))) + '\n')
        log.flush()
        process = subprocess.Popen(list(map(str, command)), cwd=job, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            last_report = 0
            while process.poll() is None:
                if CANCEL.is_set():
                    raise RuntimeError('사용자가 작업을 중지했습니다. 생성된 중간 파일은 보존됩니다.')
                if time.monotonic() - last_report >= 5:
                    report(label + ' 진행 중')
                    last_report = time.monotonic()
                time.sleep(0.5)
            if process.returncode:
                tail = log_path.read_text(errors='replace')[-1800:]
                raise RuntimeError(f'{label} 실패. 로그: {log_path}\n{tail}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def h3_running():
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'h3-character-swap.service']).returncode == 0


def h3_control(action):
    completed = subprocess.run(['/usr/bin/python3', str(H3_ROOT / 'control.py'), action],
                               capture_output=True, text=True, timeout=150)
    if completed.returncode:
        raise RuntimeError('H3 서버 ' + action + ' 실패: ' + completed.stderr[-800:])


@contextlib.contextmanager
def gpu_stage(job, report):
    was_running = h3_running()
    lease = None
    done = threading.Event()
    renewer = None
    renew_error = []
    try:
        if was_running:
            queue = get_json(H3_URL + '/queue')
            if queue['queue_running'] or queue['queue_pending']:
                raise RuntimeError('H3에서 다른 영상이 실행 또는 대기 중입니다. 완료 후 다시 실행해 주세요.')
            report('비어 있는 H3 서버의 GPU를 3D 동작 추출 단계로 전환합니다.')
            h3_control('stop')
        if not helper('available').get('available'):
            raise RuntimeError('현재 사용 가능한 GPU가 없습니다. 다른 작업은 중지하지 않습니다.')
        acquired = helper('acquire', '--owner', OWNER, '--job', job.name, '--count', 1, '--ttl', 7200)
        lease = {key: acquired[key] for key in ('id', 'gpu_ids')}
        del acquired
        write_json(job / 'outputs/lease.json', lease)
        def renew():
            while not done.wait(600):
                try:
                    helper('renew', lease['id'], '--ttl', 7200)
                except Exception as error:
                    renew_error.append(str(error))
                    CANCEL.set()
                    return
        renewer = threading.Thread(target=renew, daemon=True)
        renewer.start()
        yield environment(job, lease['gpu_ids'])
        if renew_error:
            raise RuntimeError('GPU 임대 갱신 실패: ' + renew_error[0])
    finally:
        done.set()
        if renewer:
            renewer.join(timeout=95)
        if lease:
            helper('release', lease['id'])
        if was_running and not h3_running():
            try:
                h3_control('start')
            except Exception as error:
                write_json(job / 'outputs/h3-restart.json', {'error': str(error)})


def probe_video(path, count=False):
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= 1024**3:
        raise ValueError('영상은 비어 있지 않은 1 GiB 이하 파일이어야 합니다.')
    command = ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json']
    if count:
        command += ['-count_frames']
    result = subprocess.run([*command, str(path)], capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise ValueError('영상을 읽을 수 없습니다. 손상되지 않은 MP4/MOV/WebM을 사용해 주세요.')
    data = json.loads(result.stdout)
    stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
    if not stream:
        raise ValueError('영상 트랙이 없습니다.')
    duration = float(stream.get('duration') or data.get('format', {}).get('duration') or 0)
    numerator, denominator = map(float, stream.get('avg_frame_rate', '0/1').split('/'))
    fps = numerator / denominator if denominator else 0
    if not math.isfinite(duration) or duration <= 0 or not 1 <= fps <= 240:
        raise ValueError('영상 길이 또는 프레임 속도를 확인할 수 없습니다.')
    if min(stream['width'], stream['height']) < 128 or max(stream['width'], stream['height']) > 8192:
        raise ValueError('영상의 가로·세로는 128~8192 픽셀 범위여야 합니다.')
    return dict(width=stream['width'], height=stream['height'], duration=duration, fps=fps,
                frames=int(stream.get('nb_read_frames') or stream.get('nb_frames') or 0),
                audio=any(s.get('codec_type') == 'audio' for s in data['streams']))


def validate_options(video, start, frames, width, height, yaw, elevation, zoom):
    info = probe_video(video)
    if frames == 0:
        from full_video import full_duration
        full_duration(video)
        if start != 0:
            raise ValueError('영상 끝까지 모드에서는 시작 위치를 0초로 설정해 주세요.')
    elif frames not in H3_FRAMES:
        raise ValueError('지원하지 않는 H3 프레임 수입니다.')
    if not math.isfinite(start) or start < 0 or start + frames/24 > info['duration'] + 1/info['fps']:
        raise ValueError(f'선택 구간이 영상 길이 {info["duration"]:.2f}초를 벗어납니다.')
    if (width, height) not in ((1344, 768), (768, 1344), (992, 992), (768, 768)):
        raise ValueError('지원하지 않는 출력 해상도입니다.')
    if frames > 124 and (width, height) not in ((768, 768), (768, 1344)):
        raise ValueError('긴 구간은 정사각형 768 또는 세로 768×1344 해상도를 사용해 주세요.')
    if not (-180 <= yaw <= 180 and -10 <= elevation <= 60 and 0.5 <= zoom <= 1.5):
        raise ValueError('카메라 설정이 유효하지 않습니다.')
    return info


def validate_image(path, target=None):
    from PIL import Image, ImageOps
    path = Path(path or '')
    if not path.is_file() or not 0 < path.stat().st_size <= 50*1024**2:
        raise ValueError('캐릭터 이미지는 50 MiB 이하 PNG/JPEG/WebP 파일이어야 합니다.')
    with Image.open(path) as im:
        # Phone HDR JPEGs may be MPO containers: the first frame is the normal
        # color photo and later frames are gain/depth maps, not animation.
        if im.format not in ('PNG', 'JPEG', 'WEBP', 'MPO') or (
                im.format != 'MPO' and getattr(im, 'n_frames', 1) != 1):
            raise ValueError('정지 PNG/JPEG/WebP 이미지가 필요합니다.')
        im.seek(0)
        if min(im.size) < 128 or max(im.size) > 8192 or im.width*im.height > 32_000_000:
            raise ValueError('이미지는 최소 128 픽셀, 최대 8192 픽셀·32 메가픽셀 이하여야 합니다.')
        im.load()
        im = ImageOps.exif_transpose(im).convert('RGBA')
        if target:
            background = Image.new('RGBA', im.size, 'white')
            background.alpha_composite(im)
            background.convert('RGB').save(target)
        return {'width': im.width, 'height': im.height}


def require_models():
    global CHECKPOINTS
    CHECKPOINTS = model_dir()
    missing = missing_models()
    if missing:
        raise RuntimeError('3D 추출에 필요한 모델이 없습니다: ' + ', '.join(missing)
                           + '\n화면의 모델 준비 안내에서 공식 파일과 접근 승인을 먼저 준비해 주세요.')
    if not BLENDER.is_file() or not EXMR_PYTHON.is_file():
        raise RuntimeError('ExMR/Blender 설치가 완료되지 않았습니다.')
    if not (CHECKPOINTS.parent / 'cache/torch/hub/facebookresearch_dinov3_main/hubconf.py').is_file():
        raise RuntimeError('DINOv3 코드 캐시가 없습니다. 모델 등록 도구를 다시 실행해 주세요.')


def render(job, mesh, frames, width, height, yaw, elevation, zoom, env, report, audio=None, orbit=False):
    out = job / 'outputs/blender'
    run([BLENDER, '-b', '-t', '6', '--python', ROOT/'blender_guide.py', '--', '--mesh', mesh,
         '--output', out, '--width', width, '--height', height, '--yaw', yaw,
         '--elevation', elevation, '--zoom', zoom, '--orbit-degrees', 360 if orbit else 0],
        job, env, 'blender', report)
    guide = job / 'outputs/guide.mp4'
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '24',
               '-i', out/'frames/frame_%04d.png']
    if audio:
        command += ['-i', audio, '-map', '0:v:0', '-map', '1:a:0?']
    command += ['-frames:v', frames, '-t', frames/24, '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ar', '48000',
                '-threads', '4', '-movflags', '+faststart', guide]
    run(command, job, env, 'guide-encode', report)
    info = probe_video(guide, count=True)
    if info['frames'] != frames or abs(info['fps']-24) > 0.001:
        raise RuntimeError('Blender 가이드 프레임 수 또는 FPS 검증 실패')
    return guide


def reconstruct(video, start, frames, static_cam, ground, width, height, yaw, elevation, zoom,
                report=lambda text: None, *, orbit=False):
    info = validate_options(video, start, frames, width, height, yaw, elevation, zoom)
    if frames == 0:
        from full_video import reconstruct_entire
        return reconstruct_entire(video,report,width=width,height=height,orbit=orbit,
            static_cam=static_cam,ground=ground,yaw=yaw,elevation=elevation,zoom=zoom)
    require_models()
    with managed_job('rgbmocap-3d-guide') as job:
        inputs = job / 'outputs/inputs'
        inputs.mkdir()
        source = inputs / ('source' + Path(video).suffix.lower())
        shutil.copy2(video, source)
        env = environment(job)
        clip = inputs / 'motion.mp4'
        run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-ss', start,
             '-i', source, '-map', '0:v:0', '-map', '0:a:0?', '-vf',
             'fps=24,scale=1920:1920:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1',
             '-frames:v', frames, '-t', frames/24, '-c:v', 'libx264', '-crf', '18',
             '-c:a', 'aac', '-threads', '4', '-filter_threads', '4', clip], job, env, 'input-check', report)
        if probe_video(clip, count=True)['frames'] != frames:
            raise ValueError('입력 영상의 선택 구간에서 필요한 프레임을 모두 읽지 못했습니다.')
        exmr_out = job / 'outputs/mocap'
        settings = dict(input=info, start=start, frames=frames, fps=24, static_cam=static_cam,
                        ground=ground, width=width, height=height, yaw=yaw, elevation=elevation,
                        zoom=zoom, orbit=bool(orbit), motion_retimed=False)
        write_json(job / 'outputs/settings.json', settings)
        with gpu_stage(job, report) as gpu_env:
            command = [EXMR_PYTHON, EXMR/'tools/demo.py', '--video', clip, '--output_dir', exmr_out,
                       '--checkpoint_dir', CHECKPOINTS, '--export_fps', '24', '--no_render', '--keep_frames']
            if static_cam:
                command += ['--static_cam']
            if ground:
                command += ['--snap_to_ground']
            run(command, job, gpu_env, 'exmr', report)
            mesh = job / 'outputs/animated_mesh.npz'
            run([EXMR_PYTHON, ROOT/'mesh_export.py', '--source', exmr_out/'results.npz',
                 '--model', CHECKPOINTS/REQUIRED_MODELS['SMPL-X 인체 모델'], '--output', mesh,
                 '--frames', frames], job, env, 'mesh-export', report)
            render(job, mesh, frames, width, height, yaw, elevation, zoom, gpu_env, report, clip, orbit)
        write_json(job / 'outputs/completed.json', {'stage': '3d-guide', 'settings': settings})
        return job


def trusted_job(value):
    path = Path(value).resolve()
    path.relative_to(JOBS.resolve())
    manifest = json.loads((path / '.gpu-job.json').read_text())
    if manifest.get('owner') != OWNER:
        raise ValueError('이 UI에서 만든 3D 작업만 다시 렌더할 수 있습니다.')
    return path


def rerender(previous, yaw, elevation, zoom, report=lambda text: None, *, orbit=False):
    previous = trusted_job(previous)
    settings = json.loads((previous / 'outputs/settings.json').read_text())
    mesh = previous / 'outputs/animated_mesh.npz'
    if not mesh.is_file():
        raise ValueError('3D 모션 파일이 없습니다. 먼저 1단계를 실행해 주세요.')
    if not (-180 <= yaw <= 180 and -10 <= elevation <= 60 and 0.5 <= zoom <= 1.5):
        raise ValueError('카메라 설정이 유효하지 않습니다.')
    with managed_job('rgbmocap-camera-render') as job:
        settings.update(yaw=yaw, elevation=elevation, zoom=zoom, orbit=bool(orbit))
        write_json(job / 'outputs/settings.json', settings)
        shutil.copy2(mesh, job / 'outputs/animated_mesh.npz')
        for name in ('mocap', 'inputs'):
            shutil.copytree(previous / 'outputs' / name, job / 'outputs' / name,
                            ignore=shutil.ignore_patterns('cache', 'frames'))
        with gpu_stage(job, report) as env:
            render(job, mesh, settings['frames'], settings['width'], settings['height'], yaw,
                   elevation, zoom, env, report, previous / 'outputs/inputs/motion.mp4', orbit)
        write_json(job / 'outputs/completed.json', {'stage': '3d-guide', 'source_job': str(previous)})
        return job


def make_h3_prompt(image_name, video_name, width, height, frames, prompt, seed, prefix, face_name=None,
                   *, reference_detail=False, anchor_name=None):
    graph = json.loads((ROOT / 'h3_prompt.json').read_text())
    graph['137']['inputs']['image'] = image_name
    graph['138']['inputs']['value'] = prompt
    graph['129']['inputs']['noise_seed'] = int(seed)
    graph['132']['inputs']['value'] = frames/24
    # Fixed dimensions and length avoid rounding differences in dynamic graph nodes.
    graph['136']['inputs'].update(width=width, height=height, length=frames)
    graph['136']['inputs']['ref_image_size'] = 'max' if reference_detail else 'match'
    graph['152']['inputs'].update(file=video_name, start_seconds=0, width=width, height=height, length=frames)
    graph['152']['inputs'].pop('video-preview', None)
    graph['92']['inputs']['filename_prefix'] = prefix
    if face_name:
        graph['153'] = {'class_type': 'LoadImage', 'inputs': {'image': face_name},
                        '_meta': {'title': 'Picture 2 · source face detail'}}
        graph['136']['inputs']['ref_images.ref_image_1'] = ['153', 0]
        graph['138']['inputs']['value'] += (' <Picture 2> is a close-up of the same character. '
            'Use this exact facial identity: preserve the eye shape, nose, mouth, jaw and facial proportions. '
            'Use <Picture 1> for the complete outfit and hairstyle. The close-up is an identity reference; '
            'preserve the exact camera framing and all dance movements from <Video 1>, including intentional close shots.')
    if anchor_name:
        graph['154'] = {'class_type':'LoadVideo','inputs':dict(file=anchor_name),
            '_meta':{'title':'Previous five generated frames'}}
        graph['156'] = {'class_type':'GetVideoComponents','inputs':dict(video=['154',0])}
        graph['155'] = {'class_type':'MiniMaxH3AddGuide','inputs':dict(
            positive=['136',0],latent=['136',1],vae=['119',0],image=['156',0],frame_idx=0),
            '_meta':{'title':'Continue the same character and camera'}}
        graph['126']['inputs']['conditioning'] = ['155',0]
    return graph


def require_face_models():
    names = ['models/buffalo_l/det_10g.onnx', 'models/buffalo_l/w600k_r50.onnx', 'inswapper_128.onnx']
    if not FACE_PYTHON.is_file() or any(not (FACE_MODELS/name).is_file() for name in names):
        raise RuntimeError('얼굴 보정 환경이 준비되지 않았습니다.')


def prepare_face(job, image, report, *, reference_detail=False):
    require_face_models()
    run([FACE_PYTHON, ROOT/'face_identity.py', 'prepare', '--models', FACE_MODELS,
         '--image', image, '--output', job/'outputs', *(['--detail'] if reference_detail else [])],
        job, environment(job), '얼굴-입력-검사', report)
    return job/'outputs/face-reference.png'


def cancel_h3_prompt(prompt_id):
    queue = get_json(H3_URL + '/queue')
    if any(item[1] == prompt_id for item in queue['queue_pending']):
        get_json(H3_URL + '/queue', {'delete': [prompt_id]})
    if any(item[1] == prompt_id for item in queue['queue_running']):
        get_json(H3_URL + '/interrupt', {})
    # Interrupt is asynchronous. Give this prompt time to finish so the server's
    # lease can be released by h3_session; do not wait for unrelated work.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        queue = get_json(H3_URL + '/queue')
        if not any(item[1] == prompt_id for item in queue['queue_running'] + queue['queue_pending']):
            return
        time.sleep(.5)


@contextlib.contextmanager
def h3_session(report):
    was_running = h3_running()
    try:
        if not was_running:
            report('H3 서버 시작')
            h3_control('start')
        queue = get_json(H3_URL + '/queue')
        if queue['queue_running'] or queue['queue_pending']:
            raise RuntimeError('H3에서 다른 작업이 실행 또는 대기 중입니다. 완료 후 다시 실행해 주세요.')
        state = json.loads((Path('/run/user')/str(os.getuid())/'h3-character-swap/state.json').read_text())
        session = Path(state['session'])
        if session.resolve().parent != JOBS.resolve() or state.get('mode') != 'gpu':
            raise RuntimeError('H3 GPU 서버의 관리 상태를 확인할 수 없습니다.')
        yield session
    finally:
        if not was_running and h3_running():
            # Stop only an idle server this call started. Never stop someone else's prompt.
            queue = get_json(H3_URL + '/queue')
            if not queue['queue_running'] and not queue['queue_pending']:
                h3_control('stop')


def swap(guide, character, prompt, seed, report=lambda text: None, *,
         preserve_face=True, face_image=None, reference_detail=False,
         anchor_video=None, correct_face=True):
    info = probe_video(guide, count=True)
    frames, width, height = info['frames'], info['width'], info['height']
    if (not 56 <= frames <= 226 or frames % 17 != 5
            or abs(info['fps']-24) > .001 or width % 32 or height % 32):
        raise ValueError('H3 가이드는 24 fps, 56~226 중 17n+5 프레임, 가로·세로 32의 배수여야 합니다.')
    if frames > 124 and (width, height) not in ((768, 768), (768, 1344)):
        raise ValueError('긴 구간은 정사각형 768 또는 세로 768×1344 해상도를 사용해 주세요.')
    if width*height > 1344*768:
        raise ValueError('이 RTX 5090 프리셋은 약 1 메가픽셀 이하 가이드를 지원합니다.')
    if anchor_video:
        anchor_info = probe_video(anchor_video,count=True)
        if (anchor_info['frames'] != 5 or abs(anchor_info['fps']-24) > .001
                or (anchor_info['width'],anchor_info['height']) != (width,height)):
            raise ValueError('연결 참고는 같은 해상도의 24 fps·5프레임 영상이어야 합니다.')
    validate_image(character)
    if face_image:
        validate_image(face_image)
    if not prompt.strip() or len(prompt) > 6000 or '<Video 1>' not in prompt or '<Picture 1>' not in prompt:
        raise ValueError('프롬프트에 <Video 1>과 <Picture 1> 참조를 포함해 주세요 (최대 6000자).')
    if not 0 <= seed <= 2**53-1:
        raise ValueError('시드 범위가 유효하지 않습니다.')
    with managed_job('rgbmocap-h3-swap') as job:
        env = environment(job)
        inputs = job/'outputs/inputs'
        inputs.mkdir()
        image = inputs/'character.png'
        validate_image(character, image)
        source_face = image
        if face_image:
            source_face = inputs/'face-source.png'
            validate_image(face_image, source_face)
        face = prepare_face(job, source_face, report, reference_detail=reference_detail) if preserve_face else None
        video = inputs/'guide.mp4'
        shutil.copy2(guide, video)
        raw = job/'outputs/character-swap-raw.mp4'
        with h3_session(report) as session:
            image_name, video_name = job.name+'-character.png', job.name+'-guide.mp4'
            shutil.copy2(image, session/'outputs/input'/image_name)
            shutil.copy2(video, session/'outputs/input'/video_name)
            face_name = None
            if face:
                face_name = job.name+'-face.png'
                shutil.copy2(face, session/'outputs/input'/face_name)
            anchor_name = None
            if anchor_video:
                anchor_name = job.name+'-previous.mp4'
                shutil.copy2(anchor_video,inputs/'previous.mp4')
                shutil.copy2(anchor_video,session/'outputs/input'/anchor_name)
            graph = make_h3_prompt(image_name, video_name, width, height, frames, prompt, seed,
                                   'video/'+job.name+'/result', face_name, reference_detail=reference_detail,
                                   anchor_name=anchor_name)
            write_json(job/'outputs/h3-prompt.json', graph)
            response = get_json(H3_URL+'/prompt', {'prompt': graph, 'client_id': job.name})
            prompt_id = response['prompt_id']
            write_json(job/'outputs/h3-run.json', {'prompt_id': prompt_id, 'session': str(session)})
            success = False
            try:
                started = time.monotonic()
                while time.monotonic()-started < 7200:
                    if CANCEL.is_set():
                        raise RuntimeError('사용자가 캐릭터 교체를 중지했습니다.')
                    entry = get_json(H3_URL+'/history/'+prompt_id).get(prompt_id)
                    if entry:
                        write_json(job/'outputs/h3-history.json', entry)
                        if entry['status']['status_str'] != 'success':
                            raise RuntimeError('H3 실행 실패. 기록: '+str(job/'outputs/h3-history.json'))
                        files = []
                        for output in entry.get('outputs', {}).values():
                            for records in output.values():
                                if isinstance(records, list):
                                    for record in records:
                                        if (isinstance(record, dict) and record.get('type') == 'output'
                                                and str(record.get('filename', '')).endswith('.mp4')):
                                            candidate = session/'outputs/results'/record.get('subfolder', '')/record['filename']
                                            candidate.resolve().relative_to((session/'outputs/results').resolve())
                                            files.append(candidate)
                        if not files:
                            raise RuntimeError('H3 완료 기록에 결과 MP4가 없습니다.')
                        shutil.copy2(files[0], raw)
                        success = True
                        break
                    report(f'H3 캐릭터 교체 진행 중 · {int(time.monotonic()-started)}초')
                    time.sleep(3)
                else:
                    raise RuntimeError('H3 실행 시간 2시간을 초과했습니다.')
            finally:
                if not success:
                    cancel_h3_prompt(prompt_id)
        output_info = probe_video(raw, count=True)
        if (output_info['frames'] != frames or abs(output_info['fps']-24) > .001
                or (output_info['width'], output_info['height']) != (width, height)):
            raise RuntimeError('H3 출력 크기 또는 프레임 수가 요청과 다릅니다.')
        final_source = raw
        face_report = None
        if preserve_face and correct_face:
            report('원본 사진의 얼굴 반영 · 얼굴이 보이는 프레임만 보정')
            run([FACE_PYTHON, ROOT/'face_identity.py', 'correct', '--models', FACE_MODELS,
                 '--image', source_face, '--video', raw, '--output', job/'outputs'],
                job, env, '얼굴-보정', report)
            final_source = job/'outputs/face-corrected-silent.mp4'
            face_report = json.loads((job/'outputs/face-report.json').read_text())
        result = job/'outputs/character-swap.mp4'
        # Keep the uploaded dance's soundtrack. Do not replace it with H3-generated sound.
        run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', final_source, '-i', video,
             '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'copy', '-c:a', 'aac', '-t', frames/24,
             '-movflags', '+faststart', result], job, env, '원본-오디오-결합', report)
        checked = probe_video(result, count=True)
        if checked['frames'] != frames or checked['audio'] != info['audio']:
            raise RuntimeError('최종 영상 프레임 또는 원본 오디오 검증 실패')
        run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', result, '-f', 'null', '-'],
            job, env, 'result-decode', report)
        write_json(job/'outputs/completed.json', {'stage': 'h3-swap', 'output': checked,
                   'preserve_face': preserve_face, 'reference_detail': reference_detail,
                   'face_correction': bool(preserve_face and correct_face),
                   'continuation_anchor': bool(anchor_video), 'corrected_frames':
                   face_report['corrected_frames'] if face_report else 0})
        return job


def make_bundle(swap_job, guide_job=None, report=lambda text: None):
    """Separate managed output: editable 3D assets, reference, final and comparison video."""
    swap_job = trusted_job(swap_job)
    if guide_job:
        guide_job = trusted_job(guide_job)
    with managed_job('dance-result-bundle') as job:
        from result_bundle import copy_inputs, export_desktop
        copy_inputs(job, swap_job, guide_job)
        run([UI_PYTHON, ROOT/'result_bundle.py', '--job', job], job, environment(job),
            '입력-과정-결과-통합영상', report)
        for name in ('최종_캐릭터_춤.mp4', '입력부터_결과까지.mp4'):
            run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', job/'outputs'/name,
                 '-f', 'null', '-'], job, environment(job), 'bundle-decode', report)
        destination = export_desktop(job)
        write_json(job/'outputs/completed.json', {'stage': 'bundle', 'desktop': str(destination),
                   'guide_job': str(guide_job) if guide_job else None, 'swap_job': str(swap_job)})
        return job, destination
