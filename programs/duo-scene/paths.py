"""Every location the dance programs use, read from ONE config file. Nothing in the programs names a
computer, a user or a folder of its own: the config says where each installed tool and model is.

The config file is found as
    1. the file named by the environment variable DANCE3D_CONFIG, or
    2. config.local.json at the top of the repository (the installer writes it; it is not in git).
config.example.json at the top of the repository explains every key.

A run's own writable files (frames, logs, caches) are all inside its run folder, which the pipeline
that started a step hands over in the environment variable DUO_JOB.

The locations are read only when a program asks for one (so `--help` works without a config), and a
missing key or a path that does not exist stops the program with one sentence that names the key
and the install step that provides it.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent        # programs/duo-scene
PROGRAMS = ROOT.parent                        # programs/
REPO = PROGRAMS.parent                        # the repository
CONFIG_VARIABLE = 'DANCE3D_CONFIG'

# key -> (what it is, the install step that provides it). Messages are Korean first, English after.
KEYS = {
    'python.motion': ('동작 프로그램 환경의 파이썬 / Python of the motion environment (ExMR, SAM 3, MoGe)',
                      '설치: 동작 환경 / install step: motion environment'),
    'python.comfy': ('ComfyUI 환경의 파이썬 / Python of the ComfyUI environment', '설치: ComfyUI / install step: ComfyUI'),
    'python.face': ('얼굴 보정 환경의 파이썬 / Python of the face environment (InsightFace)',
                    '설치: 얼굴 환경 / install step: face environment'),
    'python.ui': ('화면(Gradio)용 파이썬 / Python for the local pages (Gradio)', '설치: ComfyUI (같은 환경) / install step: ComfyUI'),
    'blender': ('Blender 실행 파일 / the Blender program', '설치: Blender 4.5 / install step: Blender 4.5'),
    'comfyui': ('ComfyUI 프로그램 폴더 / the ComfyUI program folder', '설치: ComfyUI / install step: ComfyUI'),
    'comfy_custom_nodes': ('ComfyUI 추가 노드 폴더 목록 / list of ComfyUI custom node folders to enable',
                           '설치: ComfyUI-GGUF / install step: ComfyUI-GGUF'),
    'comfy_pydeps': ('ComfyUI 추가 패키지 폴더 / extra Python packages for ComfyUI (e.g. gguf)',
                     '설치: ComfyUI-GGUF / install step: ComfyUI-GGUF'),
    'exmr': ('ExMR 프로그램 폴더 / the ExMR program folder', '설치: 동작 환경 / install step: motion environment'),
    'smplx': ('SMPL-X 몸 모델 파일(SMPLX_NEUTRAL.npz) / the SMPL-X body model file',
              'smpl-x.is.tue.mpg.de 등록 후 받기 / install step: SMPL-X (register at smpl-x.is.tue.mpg.de first)'),
    'models.comfy': ('ComfyUI 모델 폴더 (SCAIL-2) / the ComfyUI models folder (SCAIL-2 files)',
                     '설치: SCAIL-2 모델 / install step: SCAIL-2 models'),
    'models.h3': ('MiniMax H3 모델 폴더 / the MiniMax H3 models folder',
                  '설치: H3 모델 (한국·미국·EU·영국 사용 불가 약관) / install step: H3 models (licence excludes Korea, US, EU, UK)'),
    'models.h3_extra': ('다른 H3 바탕 모델 폴더 / folder with other H3 base models', '선택 / optional'),
    'models.sam3': ('SAM 3 가중치 (sam3.pt) / SAM 3 weights', '설치: SAM 3 / install step: SAM 3'),
    'models.moge': ('MoGe 가중치 (model.pt) / MoGe weights', '설치: MoGe / install step: MoGe'),
    'models.detector': ('사람 찾기 모델 (yolo11x.pt) / person detector weights', '설치: 동작 모델 / install step: motion models'),
    'models.motion': ('ExMR 모델 폴더 (exmr/, body_models/, yolo/ ...) / the ExMR checkpoint folder',
                      '설치: 동작 모델 / install step: motion models'),
    'models.face': ('InsightFace 모델 폴더 (buffalo_l, inswapper_128.onnx) / the InsightFace models folder',
                    '설치: 얼굴 모델 / install step: face models'),
    'models.describer': ('캐릭터 설명 모델 폴더 (Qwen2.5-VL-7B-Instruct) / the picture-describing model folder',
                         '선택 / optional'),
    'pydeps': ('동작 프로그램 추가 패키지 폴더 / extra Python packages for the motion tools', '선택 / optional'),
    'python_headers': ('Python.h 가 있는 폴더 (시스템에 python3-dev 가 없을 때) / folder holding Python.h', '선택 / optional'),
    'yt_dlp': ('yt-dlp 실행 파일 / the yt-dlp program', '설치: yt-dlp / install step: yt-dlp'),
    'runs_dir': ('실행 폴더를 만들 곳 / where each run gets its own folder', 'config 에 적기 / write it into the config'),
    'results_dir': ('결과를 저장할 곳 / where finished results are saved', 'config 에 적기 / write it into the config'),
    'gpu_helper': ('GPU 임대 도우미 (gpu-agent.py) / the GPU lease helper', '선택 / optional'),
}
OPTIONAL = {'models.h3', 'models.h3_extra', 'models.describer', 'pydeps', 'comfy_pydeps', 'python_headers', 'gpu_helper',
            'comfy_custom_nodes', 'python.ui', 'yt_dlp'}
FOLDERS = {'comfyui', 'exmr', 'models.comfy', 'models.h3', 'models.h3_extra', 'models.face', 'models.describer',
           'models.motion', 'pydeps', 'comfy_pydeps', 'python_headers'}
# Other H3 base models the user may choose (file names inside a diffusion_models folder ComfyUI reads).
BASE_MODELS = {'official': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors',
               'singularity': 'Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors'}


class ConfigError(RuntimeError):
    """The config is missing, unreadable, lacks a key, or names a path that does not exist."""


_config = None
_config_file = None


def config_file():
    """The config file that is used, or None if there is none."""
    given = os.environ.get(CONFIG_VARIABLE)
    if given:
        return Path(given).expanduser()
    local = REPO / 'config.local.json'
    return local if local.is_file() else None


def config():
    """The config as a dict (read once)."""
    global _config, _config_file
    if _config is None:
        file = config_file()
        if file is None:
            raise ConfigError(f'설정 파일이 없습니다. 설치 프로그램이 {REPO / "config.local.json"} 를 만들거나 환경 변수 {CONFIG_VARIABLE} '
                              f'로 설정 파일을 알려 주세요 (보기: {REPO / "config.example.json"}). / No config file: run the installer, '
                              f'which writes {REPO / "config.local.json"}, or set {CONFIG_VARIABLE} to a config file '
                              f'(see config.example.json).')
        if not file.is_file():
            raise ConfigError(f'설정 파일이 없습니다: {file} ({CONFIG_VARIABLE}) / The config file named by {CONFIG_VARIABLE} '
                              f'does not exist: {file}')
        try:
            _config = json.loads(file.read_text(encoding='utf-8'))
        except ValueError as error:
            raise ConfigError(f'설정 파일을 읽을 수 없습니다: {file} ({error}) / The config file is not valid JSON: {file} ({error})')
        _config_file = file
    return _config


def raw(key, default=None):
    """The config value of a dotted key ('models.comfy'), or `default` if it is absent or null."""
    value = config()
    for part in key.split('.'):
        if not isinstance(value, dict) or value.get(part) is None:
            return default
        value = value[part]
    return value


def missing_text(key, path=None):
    what, step = KEYS.get(key, (key, ''))
    if path is None:
        return f'설정에 "{key}" 가 없습니다 - {what}. {step} / config key "{key}" is missing ({_config_file}).'
    return f'설정 "{key}" 의 경로가 없습니다: {path} - {what}. {step} / config key "{key}" points to a missing path.'


def location(key, required=None):
    """The path of a config key (~ expanded, relative to the config file's folder), checked to exist.
    required=None: as the key is (OPTIONAL keys give None when absent); True: the run needs it now.
    A required key that is absent, or any path that does not exist, stops with a clear sentence."""
    if required is None:
        required = key not in OPTIONAL
    value = raw(key)
    if value in (None, ''):
        if required:
            raise ConfigError(missing_text(key))
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (_config_file.parent / path)
    if not path.exists():
        raise ConfigError(missing_text(key, path))
    if key in FOLDERS and not path.is_dir():
        raise ConfigError(f'설정 "{key}" 는 폴더여야 합니다: {path} / config key "{key}" must be a folder.')
    return path


def require(*keys):
    """Check several keys at the start of a run: every problem in one message."""
    config()  # no config file at all: that one sentence, not one per key
    problems = []
    for key in keys:
        try:
            if key == 'smplx':
                _smplx()
            elif key == 'models.detector':
                _detector()
            elif key in ('runs_dir', 'results_dir'):
                _folder_setting(key)
            elif key == 'comfy_custom_nodes':
                custom_nodes()
            else:
                location(key, required=True)
        except ConfigError as error:
            problems.append(str(error))
    if problems:
        raise ConfigError('\n'.join(problems))


def custom_nodes():
    """The custom node folders ComfyUI should load (config list comfy_custom_nodes), checked to exist."""
    folders = raw('comfy_custom_nodes', [])
    if isinstance(folders, str):
        folders = [folders]
    out = []
    for folder in folders:
        path = Path(str(folder)).expanduser()
        if not path.is_absolute():
            path = _config_file.parent / path
        if not path.is_dir():
            raise ConfigError(missing_text('comfy_custom_nodes', path))
        out.append(path)
    return out


def vram_setting():
    """vram_gb of the config: 'auto' or a number of GB."""
    value = raw('vram_gb', 'auto')
    if value == 'auto':
        return 'auto'
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(f'설정 "vram_gb" 는 "auto" 나 숫자여야 합니다: {value!r} / config key "vram_gb" must be "auto" or a number.')


def _smplx():
    path = location('smplx', required=False)
    if path is None:  # the body model inside the ExMR checkpoint folder, where ExMR keeps it
        motion = location('models.motion', required=False)
        if motion is not None and (motion / 'body_models/smplx/SMPLX_NEUTRAL.npz').is_file():
            return motion / 'body_models/smplx/SMPLX_NEUTRAL.npz'
        raise ConfigError(missing_text('smplx'))
    if path.is_dir():
        for candidate in (path / 'SMPLX_NEUTRAL.npz', path / 'smplx/SMPLX_NEUTRAL.npz', path / 'body_models/smplx/SMPLX_NEUTRAL.npz'):
            if candidate.is_file():
                return candidate
        raise ConfigError(missing_text('smplx', path / 'SMPLX_NEUTRAL.npz'))
    return path


def _detector():
    path = location('models.detector', required=False)
    if path is None:
        motion = location('models.motion', required=False)
        if motion is not None and (motion / 'yolo/yolo11x.pt').is_file():
            return motion / 'yolo/yolo11x.pt'
        raise ConfigError(missing_text('models.detector'))
    return path


def _run_folder():
    given = os.environ.get('DUO_JOB')
    if given:
        return Path(given)
    try:  # a tool started on its own (not by a pipeline) writes into one fixed folder of the runs folder
        return Path(str(raw('runs_dir') or Path.cwd())).expanduser() / 'standalone'
    except ConfigError:
        return Path.cwd() / 'dance3d-standalone'


def _optional(key):
    return lambda: location(key, required=False)


def _folder_setting(key):
    """runs_dir / results_dir: a folder the programs write into (made when missing)."""
    value = raw(key)
    if not value:
        raise ConfigError(missing_text(key))
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _config_file.parent / path
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ConfigError(f'설정 "{key}" 의 폴더를 만들 수 없습니다: {path} ({error}) / cannot create the folder of config key "{key}".')
    return path


# name exported to the programs -> how it is found (only worked out when a program asks for it)
_NAMES = {
    'JOB': _run_folder,
    'EXMR': lambda: location('exmr'),
    'EXMR_PYTHON': lambda: location('python.motion'),
    'BLENDER': lambda: location('blender'),
    'SMPLX': _smplx,
    'BODY_MODELS': lambda: location('models.motion'),
    'COMFY': lambda: location('comfyui'),
    'COMFY_PYTHON': lambda: location('python.comfy'),
    'COMFY_MODELS': lambda: location('models.comfy'),
    'H3_MODELS': _optional('models.h3'),
    'EXTRA_MODELS': _optional('models.h3_extra'),
    'HELPER': _optional('gpu_helper'),
    'SAM3': lambda: location('models.sam3'),
    'MOGE': lambda: location('models.moge'),
    'DETECTOR': _detector,
    'FACE_PYTHON': lambda: location('python.face'),
    'FACE_MODELS': lambda: location('models.face'),
    'DESCRIBER': _optional('models.describer'),
    'UI_PYTHON': lambda: location('python.ui', required=False) or location('python.comfy'),
    'PYDEPS': _optional('pydeps'),
    'COMFY_PYDEPS': _optional('comfy_pydeps'),
    'PYTHON_HEADERS': _optional('python_headers'),
    'RESULTS': lambda: _folder_setting('results_dir'),
    'RUNS': lambda: _folder_setting('runs_dir'),
}


def __getattr__(name):
    if name in _NAMES:
        value = _NAMES[name]()
        if name != 'JOB':  # the run folder may be set later by the pipeline (DUO_JOB)
            globals()[name] = value
        return value
    raise AttributeError(f'module paths has no attribute {name}')


def cache_environment():
    """Environment that keeps every cache and temporary file inside the run folder."""
    job = _run_folder()
    env = os.environ.copy()
    mapping = {'TMPDIR': 'cache/tmp', 'XDG_CACHE_HOME': 'cache/xdg', 'HF_HUB_CACHE': 'cache/hf/hub',
               'TORCH_HOME': 'cache/torch', 'TORCHINDUCTOR_CACHE_DIR': 'cache/inductor',
               'TRITON_CACHE_DIR': 'cache/triton', 'CUDA_CACHE_PATH': 'cache/cuda',
               'MPLCONFIGDIR': 'cache/matplotlib', 'YOLO_CONFIG_DIR': 'cache/ultralytics',
               'NUMBA_CACHE_DIR': 'cache/numba'}
    for key, relative in mapping.items():
        path = job / relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    pydeps = location('pydeps', required=False)
    env.update({'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                'DO_NOT_TRACK': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'DUO_JOB': str(job),
                # extra packages live outside the shared interpreters, which are never modified
                'YOLO_AUTOINSTALL': 'false', 'UV_CACHE_DIR': str(job / 'cache/uv'), 'UV_LINK_MODE': 'copy',
                'PYTHONPATH': os.pathsep.join(filter(None, [str(pydeps) if pydeps else None, str(ROOT), env.get('PYTHONPATH')]))})
    if _config_file is not None:
        env[CONFIG_VARIABLE] = str(_config_file)  # every step reads the same config
    return env


def font(weight='Bold'):
    """A Noto Sans CJK font file (Korean text in pictures): the usual Linux place, else what fontconfig finds."""
    for folder in (Path('/usr/share/fonts/opentype/noto'), Path('/usr/share/fonts/noto-cjk'), Path('/usr/share/fonts/google-noto-cjk')):
        for name in (f'NotoSansCJK-{weight}.ttc', f'NotoSansCJKkr-{weight}.otf'):
            if (folder / name).is_file():
                return folder / name
    import subprocess
    try:
        found = subprocess.run(['fc-match', '-f', '%{file}', f'Noto Sans CJK KR:style={weight}'], capture_output=True, text=True,
                               timeout=20).stdout.strip()
        if found and Path(found).is_file() and 'CJK' in found:
            return Path(found)
    except (OSError, subprocess.SubprocessError):
        pass
    raise ConfigError('한글 글꼴(Noto Sans CJK)이 없습니다. fonts-noto-cjk 를 설치해 주세요. / The Korean font Noto Sans CJK is missing: '
                      'install the package fonts-noto-cjk.')


def korean_font(size, weight='Bold'):
    """A PIL font of this size with Korean letters (the KR face of Noto Sans CJK)."""
    from PIL import ImageFont
    file = font(weight)
    for index in range(10):
        try:
            candidate = ImageFont.truetype(str(file), size, index=index)
        except OSError:
            break
        if 'KR' in candidate.getname()[0]:
            return candidate
    return ImageFont.truetype(str(file), size)
