import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def configured_path(name, default):
    return Path(os.environ.get(name, str(default))).expanduser().resolve()

JOBS = configured_path('RGBMOCAP_JOBS', '/srv/training/jobs')
INSTALL = configured_path('RGBMOCAP_INSTALL', JOBS / 'SET_RGBMOCAP_INSTALL')
EXMR = configured_path('EXMR_ROOT', ROOT.parent / 'external/ExMR')
EXMR_PYTHON = configured_path('EXMR_PYTHON', INSTALL / 'cache/venv/bin/python')
CHECKPOINTS = configured_path('RGBMOCAP_CHECKPOINTS', INSTALL / 'checkpoints')
BLENDER = configured_path('BLENDER_BIN', INSTALL / 'cache/runtime/blender-4.5.16-linux-x64/blender')
JPEG_LIB = configured_path('JPEG_LIB', INSTALL / 'cache/runtime/libjpeg/opt/libjpeg-turbo/lib64')
UI_PYTHON = configured_path('RGBMOCAP_UI_PYTHON', sys.executable)
H3_ROOT = configured_path('H3_ROOT', ROOT.parent / 'integrations/h3')
H3_URL = os.environ.get('H3_URL', 'http://127.0.0.1:8188')
HELPER = configured_path('GPU_AGENT_HELPER', Path.home() / 'gpu-dashboard/gpu-agent.py')
OWNER = 'rgbmocap-h3'
PORT = int(os.environ.get('RGBMOCAP_PORT', '7862'))
H3_FRAMES = (56, 73, 90, 124, 226)
URL = f'http://127.0.0.1:{PORT}'
SERVICE = 'rgbmocap-h3.service'
FACE_INSTALL = configured_path('FACE_INSTALL', INSTALL)
FACE_PYTHON = configured_path('FACE_PYTHON', FACE_INSTALL / 'cache/face-venv/bin/python')
FACE_MODELS = configured_path('FACE_MODELS', FACE_INSTALL / 'checkpoints/face')
DESKTOP_RESULTS = configured_path('RGBMOCAP_DESKTOP_RESULTS', Path.home() / 'Desktop/Free Camera Results')

PROMPT = ('Replace the entire gray human mannequin in <Video 1> with the character in '
          '<Picture 1>. Match the reference character\'s face, hairstyle, skin, outfit, hands '
          'and footwear. Follow the mannequin\'s full body motion, hand gestures and foot '
          'placement and dance timing exactly, including every moving pose. Preserve the '
          'camera movement and full orbit, framing, floor and background of <Video 1>. '
          'Keep the reference outfit opaque and securely covering the body while dancing. '
          'Maintain the same face, hairstyle and character throughout the video.')

REQUIRED_MODELS = {
    'ExMR': 'exmr/exmr.safetensors',
    'SMPL-X 인체 모델': 'body_models/smplx/SMPLX_NEUTRAL.npz',
    'YOLO': 'yolo/yolo11m.pt',
    'Sapiens2': 'sapiens2/sapiens2_1b_pose.safetensors',
    'SAM 3D Body 가중치': 'sam-3d-body-dinov3/model.ckpt',
    'SAM 3D Body 설정': 'sam-3d-body-dinov3/model_config.yaml',
    'SAM 3D Body MHR': 'sam-3d-body-dinov3/assets/mhr_model.pt',
    'HaMeR': 'hamer/hamer_ckpts/checkpoints/hamer.ckpt',
    'HaMeR 설정': 'hamer/hamer_ckpts/model_config.yaml',
    'HaMeR 손 평균': 'hamer/data/mano_mean_params.npz',
    'MANO 손 모델': 'hamer/data/mano/MANO_RIGHT.pkl',
}


def model_dir():
    location = ROOT / 'model-location.json'
    if location.is_file():
        path = Path(json.loads(location.read_text())['checkpoints']).resolve()
        path.relative_to(JOBS.resolve())
        return path
    return CHECKPOINTS


def missing_models():
    root = model_dir()
    return [name for name, relative in REQUIRED_MODELS.items()
            if not (root / relative).is_file() or (root / relative).stat().st_size == 0]
