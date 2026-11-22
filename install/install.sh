#!/usr/bin/env bash
# Installer of the dance programs for Linux (Ubuntu 22.04 / 24.04) and for Windows through WSL2 (Ubuntu).
# It needs no sudo: it checks the system packages and prints the one apt line to run when some are missing.
#
#   install/install.sh                      the challenge video (= --challenge), SCAIL-2 tier from the card (--vram auto)
#   install/install.sh --joints             also video -> 3D joints (ExMR, SAM 3D Body, HaMeR, Sapiens2, YOLO, MoGe)
#   install/install.sh --all                --joints + the picture describer + the relight LoRA (not H3)
#   install/install.sh --h3                 also MiniMax H3 (optional; its licence excludes KR, US, EU, UK)
#   install/install.sh --check              only check the prerequisites
#
# Options: --vram 8|12|16|24|auto  --accept <licence> (repeatable)  --models-from <dir> (repeatable)  --copy
#          --no-verify  --no-models  --runs-dir <dir>  --results-dir <dir>
# Install root: $DANCE3D_HOME, default <repo>/.dance3d. Everything it installs stays in there; the config the
# programs read is written to <repo>/config.local.json. Running it again goes on where it stopped.
set -Eeuo pipefail

# ---------- pinned versions (the versions the programs ran with on the training server) ----------
UV_VERSION=0.13.1
UV_SHA256=eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220  # uv-x86_64-unknown-linux-gnu.tar.gz
PYTHON_VERSION=3.12.15
EXMR_URL=https://github.com/twehrbein/ExMR.git
EXMR_COMMIT=eac72e47339b404c0443f5fb323412e2ce432410
SAM3D_BODY_URL=https://github.com/facebookresearch/sam-3d-body.git
SAM3D_BODY_COMMIT=b5c765a0d89d789985e186d396315e7590887b94  # ExMR's submodule third_party/sam-3d-body
COMFYUI_URL=https://github.com/Comfy-Org/ComfyUI.git
COMFYUI_COMMIT=f4f3dedddea4e7ac8e79589a1362649fc2bf3377  # 0.44.0: SCAIL-2 nodes built in
GGUF_URL=https://github.com/city96/ComfyUI-GGUF.git
GGUF_COMMIT=6ea2651e7df66d7585f6ffee804b20e92fb38b8a
BLENDER_VERSION=4.5.16
BLENDER_URL=https://download.blender.org/release/Blender4.5/blender-4.5.16-linux-x64.tar.xz
BLENDER_SHA256=23432b064977908beb1ab4b7f32f024e7331105c875c187dce002aff84b10c11  # download.blender.org/.../blender-4.5.16.sha256
MOTION_TORCH=(torch==2.8.0 torchvision==0.23.0)
MOTION_TORCH_INDEX=https://download.pytorch.org/whl/cu128
DRIVER_MIN=580  # ComfyUI's PyTorch 2.13 is a CUDA 13.0 build (the motion environment's CUDA 12.8 needs 570)
APT_PACKAGES=(git curl ca-certificates xz-utils ffmpeg build-essential fonts-noto-cjk libvulkan1 libx11-6 libxrender1
              libxfixes3 libxi6 libxkbcommon0 libegl1 libgl1)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${DANCE3D_HOME:-$REPO/.dance3d}"
PINS="$REPO/install/pins"

say() { printf '\n== %s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die() { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }
trap 'die "line $LINENO failed: $BASH_COMMAND (run the installer again to go on; it skips what is done)"' ERR

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

LEVEL_JOINTS=0 LEVEL_ALL=0 LEVEL_H3=0 CHECK_ONLY=0 NO_MODELS=0 VRAM=auto RUNS_DIR="" RESULTS_DIR=""
MODEL_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --challenge) ;;
    --joints) LEVEL_JOINTS=1 ;;
    --all) LEVEL_ALL=1 ;;
    --h3) LEVEL_H3=1 ;;
    --check) CHECK_ONLY=1 ;;
    --no-models) NO_MODELS=1 ;;
    --vram) [ $# -ge 2 ] || usage 1; VRAM="$2"; shift ;;
    --accept|--models-from) [ $# -ge 2 ] || usage 1; MODEL_ARGS+=("$1" "$2"); shift ;;
    --copy|--no-verify) MODEL_ARGS+=("$1") ;;
    --runs-dir) [ $# -ge 2 ] || usage 1; RUNS_DIR="$2"; shift ;;
    --results-dir) [ $# -ge 2 ] || usage 1; RESULTS_DIR="$2"; shift ;;
    -h|--help) usage 0 ;;
    *) printf 'unknown option: %s\n\n' "$1"; usage 1 ;;
  esac
  shift
done
case "$VRAM" in auto|8|12|16|24) ;; *) die "--vram takes 8, 12, 16, 24 or auto (got $VRAM)" ;; esac
LEVELS=(--challenge)
[ "$LEVEL_JOINTS" = 1 ] && LEVELS+=(--joints)
[ "$LEVEL_ALL" = 1 ] && LEVELS+=(--all)
[ "$LEVEL_H3" = 1 ] && LEVELS+=(--h3)
WANT_JOINTS=$(( LEVEL_JOINTS || LEVEL_ALL ))

# everything the installer writes stays in the install root (also uv's cache, Python and temporary files)
export UV_CACHE_DIR="$ROOT/cache/uv" UV_PYTHON_INSTALL_DIR="$ROOT/python" UV_PYTHON_PREFERENCE=only-managed
export UV_NO_CONFIG=1 UV_HTTP_TIMEOUT=300 TMPDIR="$ROOT/cache/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
UV="$ROOT/bin/uv"

# ---------- 1. prerequisites ----------
check_prerequisites() {
  say "Checking the prerequisites"
  local problems=() missing=() pkg
  [ "$(uname -s)" = Linux ] || die "this installer runs on Linux; on Windows run it inside WSL2 (see README.md)"
  [ "$(uname -m)" = x86_64 ] || die "only x86_64 is supported (found $(uname -m))"
  local os_name="unknown" os_version=""
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    os_name="$(. /etc/os-release; echo "${PRETTY_NAME:-$NAME}")"; os_version="$(. /etc/os-release; echo "${VERSION_ID:-}")"
  fi
  note "system: $os_name"
  case "$os_version" in 22.04|24.04) ;; *) note "WARNING: written for Ubuntu 22.04 / 24.04; $os_name is untested" ;; esac
  IS_WSL=0
  local on_windows=""
  if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=1 on_windows=" (on Windows)"
    grep -qi 'microsoft-standard' /proc/sys/kernel/osrelease 2>/dev/null ||
      problems+=("This is WSL1, which has no GPU: convert it with 'wsl --set-version <distro> 2' in Windows PowerShell")
    note "Windows Subsystem for Linux 2"
    case "$REPO" in /mnt/[a-z]/*) problems+=("The repository is on a Windows drive ($REPO): clone it inside WSL, e.g. into ~/, where files are fast and keep Linux permissions") ;; esac
  fi
  if command -v dpkg-query >/dev/null 2>&1; then
    for pkg in "${APT_PACKAGES[@]}"; do
      dpkg-query -W -f='${db:Status-Status}' "$pkg" 2>/dev/null | grep -qx installed || missing+=("$pkg")
    done
  else
    for pkg in git curl xz ffmpeg ffprobe gcc g++; do command -v "$pkg" >/dev/null 2>&1 || missing+=("$pkg"); done
  fi
  for pkg in git curl tar xz ffmpeg ffprobe gcc g++ sha256sum; do
    command -v "$pkg" >/dev/null 2>&1 || problems+=("The command '$pkg' is not there")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    if command -v dpkg-query >/dev/null 2>&1; then
      problems+=("System packages are missing. Run this (it needs your password), then this installer again:
      sudo apt-get update && sudo apt-get install -y ${missing[*]}")
    else
      problems+=("Not a Debian/Ubuntu system: install ${missing[*]} with your package manager")
    fi
  fi
  local smi=""
  smi="$(command -v nvidia-smi || true)"
  [ -z "$smi" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ] && smi=/usr/lib/wsl/lib/nvidia-smi
  if [ -z "$smi" ]; then
    if [ "$IS_WSL" = 1 ]; then
      problems+=("No NVIDIA driver is visible in WSL (nvidia-smi). Install the NVIDIA driver $DRIVER_MIN or newer on Windows (not inside WSL), then run 'wsl --shutdown' and open Ubuntu again")
    else
      problems+=("No NVIDIA driver (nvidia-smi not found). Install the NVIDIA driver $DRIVER_MIN or newer, e.g. 'sudo ubuntu-drivers install', then reboot")
    fi
  else
    local driver memory name
    driver="$("$smi" --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    memory="$("$smi" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
    name="$("$smi" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    if [ -z "$driver" ]; then
      problems+=("nvidia-smi found no graphics card")
    else
      note "graphics card: $name, ${memory:-?} MiB, driver $driver"
      [ "${driver%%.*}" -ge "$DRIVER_MIN" ] 2>/dev/null ||
        problems+=("The NVIDIA driver $driver is older than $DRIVER_MIN (needed by PyTorch 2.13 / CUDA 13.0 of ComfyUI): update the driver$on_windows")
      [ "${memory:-0}" -ge 7600 ] 2>/dev/null || problems+=("The card has ${memory:-?} MiB; SCAIL-2 needs at least 8 GB")
    fi
  fi
  mkdir -p "$ROOT"
  local free_gb need_gb
  free_gb=$(( $(df -Pk "$ROOT" | awk 'NR==2 {print $4}') / 1048576 ))
  need_gb=$(( 20 + 33 + (WANT_JOINTS ? 14 : 0) + (LEVEL_ALL ? 18 : 0) + (LEVEL_H3 ? 43 : 0) ))
  note "disk: $free_gb GB free at $ROOT; this choice needs about $need_gb GB, and each run needs 20-60 GB more"
  [ "$free_gb" -ge "$need_gb" ] || problems+=("Not enough disk space: about $need_gb GB needed at $ROOT, $free_gb GB free (set DANCE3D_HOME to another disk)")
  local ram_gb
  ram_gb=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
  note "memory: $ram_gb GB"
  local wsl_memory=""
  [ "$IS_WSL" = 1 ] && wsl_memory=" (WSL gets half of the PC's memory unless %UserProfile%\\.wslconfig gives it more)"
  [ "$ram_gb" -ge 30 ] || note "WARNING: less than 32 GB of memory; the video model may not fit$wsl_memory"
  if [ ${#problems[@]} -gt 0 ]; then
    printf '\nPlease fix first:\n' >&2
    local p
    for p in "${problems[@]}"; do printf ' - %s\n' "$p" >&2; done
    exit 1
  fi
  note "all prerequisites are there"
}

# ---------- helpers ----------
fetch() {  # url file sha256: download (resumes) and check
  local url="$1" out="$2" sha="$3"
  if [ -f "$out" ] && [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sha" ]; then return 0; fi
  mkdir -p "$(dirname "$out")"
  note "downloading $url"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$out.part" "$url" || curl -fL --retry 5 --retry-delay 5 -o "$out.part" "$url"
  [ "$(sha256sum "$out.part" | cut -d' ' -f1)" = "$sha" ] || { rm -f "$out.part"; die "checksum mismatch for $url (removed; run again)"; }
  mv "$out.part" "$out"
}

checkout() {  # url folder commit: the folder holds exactly that commit (shallow fetch by commit)
  local url="$1" dir="$2" commit="$3"
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
    note "$(basename "$dir") $commit (already there)"
    return 0
  fi
  if [ -d "$dir/.git" ] && [ -n "$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    die "$dir has local changes; move it away, then run again"
  fi
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote remove origin 2>/dev/null || true
  git -C "$dir" remote add origin "$url"
  git -C "$dir" fetch -q --depth 1 origin "$commit"
  git -C "$dir" -c advice.detachedHead=false checkout -q --force --detach FETCH_HEAD
  [ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ] || die "$dir: could not check out $commit"
  note "$(basename "$dir") $commit"
}

stamp_of() { cat "$@" | sha256sum | cut -c1-16; }
done_already() { [ -f "$ROOT/state/$1" ] && [ "$(cat "$ROOT/state/$1")" = "$2" ]; }
mark_done() { mkdir -p "$ROOT/state"; printf '%s' "$2" > "$ROOT/state/$1"; }

make_env() {  # name: a fresh virtual environment with the pinned Python (kept when it is already there)
  local env="$ROOT/envs/$1"
  if [ ! -x "$env/bin/python" ]; then "$UV" venv -q --python "$PYTHON_VERSION" "$env" >&2; fi
  printf '%s' "$env/bin/python"
}

# ---------- 2. uv and Python ----------
install_uv() {
  say "uv $UV_VERSION and Python $PYTHON_VERSION"
  mkdir -p "$ROOT"/{bin,downloads,cache/tmp,state,src,envs}
  if [ ! -x "$UV" ] || [ "$("$UV" --version 2>/dev/null | cut -d' ' -f2)" != "$UV_VERSION" ]; then
    local archive="$ROOT/downloads/uv-$UV_VERSION-x86_64-unknown-linux-gnu.tar.gz"
    fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" "$archive" "$UV_SHA256"
    tar -xzf "$archive" -C "$ROOT/cache/tmp"
    mv -f "$ROOT/cache/tmp/uv-x86_64-unknown-linux-gnu/uv" "$ROOT/cache/tmp/uv-x86_64-unknown-linux-gnu/uvx" "$ROOT/bin/"
  fi
  note "$("$UV" --version)"
  "$UV" python install -q "$PYTHON_VERSION"
  PYTHON="$("$UV" python find "$PYTHON_VERSION")"
  note "Python $PYTHON_VERSION: $PYTHON"
}

# ---------- 3. program sources ----------
install_sources() {
  say "Program sources (pinned commits)"
  checkout "$EXMR_URL" "$ROOT/src/ExMR" "$EXMR_COMMIT"
  local recorded
  recorded="$(git -C "$ROOT/src/ExMR" ls-tree HEAD third_party/sam-3d-body | awk '{print $3}')"
  [ "$recorded" = "$SAM3D_BODY_COMMIT" ] || die "ExMR records sam-3d-body $recorded, the installer expects $SAM3D_BODY_COMMIT"
  checkout "$SAM3D_BODY_URL" "$ROOT/src/ExMR/third_party/sam-3d-body" "$SAM3D_BODY_COMMIT"
  checkout "$COMFYUI_URL" "$ROOT/src/ComfyUI" "$COMFYUI_COMMIT"
  checkout "$GGUF_URL" "$ROOT/src/ComfyUI/custom_nodes/ComfyUI-GGUF" "$GGUF_COMMIT"
}

# ---------- 4. Python environments ----------
install_envs() {
  local key py
  say "Motion environment (ExMR, SAM 3, MoGe; PyTorch 2.8.0 for CUDA 12.8)"
  key="$(stamp_of "$PINS/motion-extra.txt" "$ROOT/src/ExMR/requirements.txt" "$ROOT/src/ExMR/constraints.txt" <(echo "${MOTION_TORCH[*]} $PYTHON_VERSION"))"
  if done_already env-motion "$key"; then note "already installed"; else
    py="$(make_env motion)"
    "$UV" pip install --python "$py" "${MOTION_TORCH[@]}" --index-url "$MOTION_TORCH_INDEX"
    "$UV" pip install --python "$py" -r "$ROOT/src/ExMR/requirements.txt" -c "$ROOT/src/ExMR/constraints.txt"
    "$UV" pip install --python "$py" --no-deps -r "$PINS/motion-extra.txt"
    mark_done env-motion "$key"
  fi

  say "ComfyUI environment (PyTorch 2.13.0 for CUDA 13.0, gguf)"
  key="$(stamp_of "$PINS/comfy-constraints.txt" "$PINS/comfy-extra.txt" "$ROOT/src/ComfyUI/requirements.txt" <(echo "$PYTHON_VERSION"))"
  if done_already env-comfy "$key"; then note "already installed"; else
    py="$(make_env comfy)"
    "$UV" pip install --python "$py" -r "$ROOT/src/ComfyUI/requirements.txt" -r "$PINS/comfy-extra.txt" -c "$PINS/comfy-constraints.txt"
    mark_done env-comfy "$key"
  fi

  say "Face environment (InsightFace 0.7.3 and ONNX Runtime on the CPU; InsightFace is compiled)"
  key="$(stamp_of "$PINS/face-requirements.txt" "$PINS/face-constraints.txt" <(echo "$PYTHON_VERSION"))"
  if done_already env-face "$key"; then note "already installed"; else
    py="$(make_env face)"
    # InsightFace's source package needs numpy and Cython while it is built
    "$UV" pip install --python "$py" -c "$PINS/face-constraints.txt" numpy Cython setuptools wheel
    "$UV" pip install --python "$py" --no-build-isolation -r "$PINS/face-requirements.txt" -c "$PINS/face-constraints.txt"
    mark_done env-face "$key"
  fi

  if [ "$WANT_JOINTS" = 1 ]; then
    say "yt-dlp (YouTube links in the joints program)"
    key="$(stamp_of "$PINS/tools-requirements.txt" "$PINS/tools-constraints.txt" <(echo "$PYTHON_VERSION"))"
    if done_already env-tools "$key"; then note "already installed"; else
      py="$(make_env tools)"
      "$UV" pip install --python "$py" -r "$PINS/tools-requirements.txt" -c "$PINS/tools-constraints.txt"
      mark_done env-tools "$key"
    fi
  fi
}

# ---------- 5. Blender ----------
install_blender() {
  say "Blender $BLENDER_VERSION"
  local dir="$ROOT/blender-$BLENDER_VERSION-linux-x64"
  if done_already blender "$BLENDER_SHA256" && [ -x "$dir/blender" ]; then note "already installed"; else
    fetch "$BLENDER_URL" "$ROOT/downloads/blender-$BLENDER_VERSION-linux-x64.tar.xz" "$BLENDER_SHA256"
    rm -rf "$dir"
    tar -xJf "$ROOT/downloads/blender-$BLENDER_VERSION-linux-x64.tar.xz" -C "$ROOT"
    rm -f "$ROOT/downloads/blender-$BLENDER_VERSION-linux-x64.tar.xz"
    mark_done blender "$BLENDER_SHA256"
  fi
  local libs
  libs="$(ldd "$dir/blender" 2>/dev/null | awk '/not found/ {print $1}' | tr '\n' ' ')"
  [ -z "$libs" ] || die "Blender misses system libraries: $libs(see the apt line of the prerequisite check)"
  note "$dir/blender"
}

# ---------- 6. models ----------
install_models() {
  say "Model files (licences are asked once each)"
  local code=0
  "$PYTHON" -I -B "$REPO/install/get_models.py" --root "$ROOT" "${LEVELS[@]}" --vram "$VRAM" "${MODEL_ARGS[@]}" || code=$?
  [ "$code" = 0 ] || [ "$code" = 2 ] || die "the model download failed (exit code $code); run again to go on"
  MODELS_COMPLETE=$(( code == 0 ))
}

# ---------- 7. config + smoke tests ----------
write_config() {
  say "Config for the programs"
  local args=(--root "$ROOT" --repo "$REPO" --vram "$VRAM")
  [ "$WANT_JOINTS" = 1 ] && args+=(--joints)
  [ "$LEVEL_ALL" = 1 ] && args+=(--all)
  [ "$LEVEL_H3" = 1 ] && args+=(--h3)
  [ -n "$RUNS_DIR" ] && args+=(--runs-dir "$RUNS_DIR")
  [ -n "$RESULTS_DIR" ] && args+=(--results-dir "$RESULTS_DIR")
  "$PYTHON" -I -B "$REPO/install/write_config.py" "${args[@]}"
}

smoke_tests() {
  say "Quick tests (CPU only; the graphics card is not used here)"
  export CUDA_VISIBLE_DEVICES=""
  "$ROOT/envs/motion/bin/python" -B -c "
import sys; sys.path[:0] = ['$ROOT/src/ExMR', '$ROOT/src/ExMR/third_party/sam-3d-body']
import torch, torchvision, ultralytics, smplx, clip, moge.model.v2, exmr.pipeline
from ultralytics.models.sam import SAM3SemanticPredictor
print('   motion: torch', torch.__version__, 'CUDA build', torch.version.cuda, '| ultralytics', ultralytics.__version__)"
  "$ROOT/envs/comfy/bin/python" -B -c "
import torch, gguf, transformers, av
print('   comfy: torch', torch.__version__, 'CUDA build', torch.version.cuda, '| gguf', gguf.__version__ if hasattr(gguf, '__version__') else 'ok')"
  local test="$ROOT/cache/comfy-test"
  mkdir -p "$test"/{input,output,user,temp,models}
  ( cd "$ROOT/src/ComfyUI" && "$ROOT/envs/comfy/bin/python" -B main.py --cpu --quick-test-for-ci --disable-all-custom-nodes \
      --whitelist-custom-nodes ComfyUI-GGUF --models-directory "$test/models" --input-directory "$test/input" \
      --output-directory "$test/output" --user-directory "$test/user" --temp-directory "$test/temp" \
      --database-url "sqlite:///$test/comfy.sqlite" --disable-auto-launch > "$ROOT/cache/comfy-test.log" 2>&1 ) ||
    die "ComfyUI did not start (log: $ROOT/cache/comfy-test.log)"
  grep -q 'ComfyUI-GGUF' "$ROOT/cache/comfy-test.log" || note "WARNING: ComfyUI-GGUF not named in $ROOT/cache/comfy-test.log"
  note "comfy: ComfyUI starts and loads its nodes (log: $ROOT/cache/comfy-test.log)"
  "$ROOT/envs/face/bin/python" -B -c "
import insightface, onnxruntime, cv2
print('   face: insightface', insightface.__version__, '| onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers())"
  "$ROOT/blender-$BLENDER_VERSION-linux-x64/blender" -b --factory-startup --python-exit-code 1 \
      --python-expr "import bpy; print('   blender', bpy.app.version_string)" 2>/dev/null | grep '^   blender' ||
    die "Blender did not start"
  unset CUDA_VISIBLE_DEVICES
}

check_prerequisites
[ "$CHECK_ONLY" = 1 ] && exit 0
install_uv
install_sources
install_envs
install_blender
MODELS_COMPLETE=0
if [ "$NO_MODELS" = 1 ]; then note "models skipped (--no-models)"; else install_models; fi
write_config
smoke_tests
say "Done"
note "config: $REPO/config.local.json   install root: $ROOT ($(du -sh "$ROOT" 2>/dev/null | cut -f1))"
if [ "$MODELS_COMPLETE" = 1 ]; then
  note "next: ./run.sh check, then ./run.sh joints <dance video> and ./run.sh challenge --joints <its result folder> --character <picture>"
else
  note "Some model files are still missing (listed above): fix what it says, then run this installer again."
  note "./run.sh check shows what is missing at any time."
fi
