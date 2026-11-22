import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def configured_path(name, default):
    return Path(os.environ.get(name, str(default))).expanduser().resolve()

INSTALL = configured_path('H3_INSTALL', Path('/srv/training/jobs/SET_H3_INSTALL'))
MODEL_INSTALL = configured_path('H3_MODEL_INSTALL', INSTALL)
COMFY = configured_path('COMFYUI_ROOT', ROOT / 'ComfyUI')
PYTHON = configured_path('H3_PYTHON', INSTALL / 'cache/venv/bin/python')
MODELS = configured_path('H3_MODELS', MODEL_INSTALL / 'checkpoints/models')
EXAMPLES = configured_path('H3_EXAMPLES', MODEL_INSTALL / 'outputs/examples')
PYTHON_HEADERS = configured_path('PYTHON_HEADERS', Path(sys.base_prefix) / 'include')
HELPER = configured_path('GPU_AGENT_HELPER', Path.home() / 'gpu-dashboard/gpu-agent.py')
PORT = 8188
URL = f'http://127.0.0.1:{PORT}'
OWNER = 'h3-character-swap'
SERVICE = 'h3-character-swap.service'
MODEL_FILES = [
 ('Comfy-Org/MiniMax-H3', 'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'diffusion_models'),
 ('Comfy-Org/MiniMax-H3', 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'text_encoders'),
 ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_video_vae_int8_convrot.safetensors', 'vae'),
 ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_audio_vae_fp32.safetensors', 'vae'),
 ('akatz-ai/MiniMax-H3-Character-Swap-LoRA', 'h3_character_swap_v1.5.safetensors', 'loras'),
 ('lightx2v/Minimax-h3-Turbo', 'minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors', 'loras'),
]
