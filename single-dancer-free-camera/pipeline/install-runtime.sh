#!/usr/bin/env bash
set -euo pipefail
ROOT="${RGBMOCAP_INSTALL:?Set RGBMOCAP_INSTALL to a managed job-init directory}"
EXMR="${EXMR_ROOT:?Set EXMR_ROOT to the ExMR checkout}"
export UV_CACHE_DIR="$ROOT/cache/uv"
export TMPDIR="$ROOT/cache/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HOME="$ROOT/cache/hf" TORCH_HOME="$ROOT/cache/torch"
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=''
PY="$ROOT/cache/venv/bin/python"
uv pip install --python "$PY" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" -r "$EXMR/requirements.txt" -c "$EXMR/constraints.txt"
uv pip install --python "$PY" --no-deps \
 'git+https://github.com/geopavlakos/hamer.git@3a01849f4148352e9260b69bf28b65d1671a4905' \
 'git+https://github.com/facebookresearch/sapiens2.git@7e5bae88456ac418ff0e58e74106c9fe192055d4'
uv pip install --python "$PY" --no-deps 'https://huggingface.co/twehrbein/ExMR/resolve/03e36730ba7466ef09fcd98523fec6d9c2da8a0a/wheels/nvdiffrast-0.4.0%2Bpt2.8.0cu128-cp312-cp312-linux_x86_64.whl#sha256=ca26d44f4f198e8fb460f86f355c5fdaeaac6319e15a5551ea31397719c0d563'
# ExMR is used through PYTHONPATH; no build products are written into the source tree.
uv pip freeze --python "$PY" > "$ROOT/outputs/installed-packages.txt"
printf 'RUNTIME_INSTALL_COMPLETE\n'
