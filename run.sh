#!/usr/bin/env bash
# One entry for the dance programs (after install/install.sh). The config is <repo>/config.local.json, or the
# file named by DANCE3D_CONFIG.
#
#   ./run.sh check                                  config, files, graphics card: what is missing and how to get it
#   ./run.sh joints <video> [<video> ...]           dance videos (files or YouTube links) -> 3D joints of one dancer each
#   ./run.sh joints --list <list.txt>               the same from a list: one video per line, "title  file-or-link"
#   ./run.sh challenge --joints <folder> --character <picture> [--vram 8|12|16|24] [--backend scail|h3] [...]
#                                                   3D joints + one character picture -> challenge video cut on the beat
#   ./run.sh duo --video <video> --character <left> --character <right> [...]
#                                                   two dancers of one video -> two characters in one 3D room
#
# Every program also takes --job <run folder> to go on with a stopped run, and --help for all its options.
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${DANCE3D_CONFIG:-$REPO/config.local.json}"
export DANCE3D_CONFIG="$CONFIG" PYTHONDONTWRITEBYTECODE=1

die() { printf 'STOPPED: %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# any Python 3 reads the config: the system's, else the one the installer put into the install root
HOST_PY="$(command -v python3 || true)"
if [ -z "$HOST_PY" ]; then
  HOST_PY="$(find "${DANCE3D_HOME:-$REPO/.dance3d}/python" -maxdepth 3 -path '*/bin/python3' 2>/dev/null | head -n 1 || true)"
fi
[ -n "$HOST_PY" ] || die "no python3 found (Ubuntu: sudo apt-get install -y python3)"

config_value() {  # dotted key -> value ('' when absent or null)
  "$HOST_PY" -I -B -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value.get(part) if isinstance(value, dict) else None
print("" if value is None else (json.dumps(value) if isinstance(value, (list, dict)) else value))' "$CONFIG" "$1"
}

need_config() {
  [ -f "$CONFIG" ] || die "no config at $CONFIG: run install/install.sh first (or set DANCE3D_CONFIG)"
  PY_MOTION="$(config_value python.motion)"
  [ -n "$PY_MOTION" ] && [ -x "$PY_MOTION" ] || die "python.motion of $CONFIG is missing: run install/install.sh"
}

check() {
  [ -f "$CONFIG" ] || die "no config at $CONFIG: run install/install.sh first (or set DANCE3D_CONFIG)"
  local problems=0
  printf 'config: %s\n\n' "$CONFIG"
  printf 'tools and folders:\n'
  local key value
  for key in python.motion python.comfy python.face blender comfyui exmr smplx models.comfy models.sam3 models.face \
             models.motion models.moge models.detector models.h3 models.describer yt_dlp; do
    value="$(config_value "$key")"
    if [ -z "$value" ]; then
      case "$key" in
        models.moge|models.detector|yt_dlp) printf '  --    %-17s not installed (needed for: joints; install/install.sh --joints)\n' "$key" ;;
        models.h3) printf '  --    %-17s not installed (optional H3 backend; install/install.sh --h3)\n' "$key" ;;
        models.describer) printf '  --    %-17s not installed (optional; install/install.sh --all)\n' "$key" ;;
        *) printf '  MISS  %-17s not in the config: run install/install.sh\n' "$key"; problems=$((problems + 1)) ;;
      esac
    elif [ -e "$value" ]; then
      printf '  ok    %-17s %s\n' "$key" "$value"
    else
      printf '  MISS  %-17s %s does not exist\n' "$key" "$value"; problems=$((problems + 1))
    fi
  done
  local nodes
  nodes="$(config_value comfy_custom_nodes)"
  case "$nodes" in *ComfyUI-GGUF*) printf '  ok    %-17s %s\n' comfy_custom_nodes "$nodes" ;;
    *) printf '  MISS  %-17s ComfyUI-GGUF is not listed (needed for the 8/12/16 GB tiers)\n' comfy_custom_nodes; problems=$((problems + 1)) ;; esac
  for key in ffmpeg ffprobe; do
    if command -v "$key" >/dev/null 2>&1; then
      printf '  ok    %-17s %s\n' "$key" "$(command -v "$key")"
    else
      printf '  MISS  %-17s sudo apt-get install -y ffmpeg\n' "$key"; problems=$((problems + 1))
    fi
  done

  printf '\nmodel files:\n'
  local levels=(--challenge) vram code=0
  [ -n "$(config_value models.moge)" ] && levels+=(--joints)
  [ -n "$(config_value models.h3)" ] && levels+=(--h3)
  [ -n "$(config_value models.describer)" ] && levels+=(--extra describer)
  vram="$(config_value vram_gb)"; [ -n "$vram" ] || vram=auto
  "$HOST_PY" -I -B "$REPO/install/get_models.py" --check --config "$CONFIG" --vram "$vram" "${levels[@]}" | sed -n '/^Check:/,$p' || code=$?
  [ "$code" = 0 ] || problems=$((problems + 1))

  printf '\ngraphics card:\n'
  local smi
  smi="$(command -v nvidia-smi || true)"
  [ -z "$smi" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ] && smi=/usr/lib/wsl/lib/nvidia-smi
  if [ -n "$smi" ] && "$smi" --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  ok    /'; then
    [ -n "${CUDA_VISIBLE_DEVICES+x}" ] && printf '  CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  else
    printf '  MISS  no NVIDIA graphics card is visible (nvidia-smi)\n'; problems=$((problems + 1))
  fi
  if [ "$problems" -gt 0 ]; then printf '\n%s problem(s); fix them, then run ./run.sh check again.\n' "$problems"; exit 1; fi
  printf '\nEverything the programs need is there.\n'
}

is_link() { case "$1" in http://*|https://*|www.*|youtu.be/*|youtube.com/*|m.youtube.com/*) return 0 ;; *) return 1 ;; esac; }

joints() {
  need_config
  local program="$REPO/programs/dance-joints/dance_joints.py" arg
  for arg in "$@"; do
    case "$arg" in --list|--list=*|--job|--job=*|-h|--help) exec "$PY_MOTION" -B "$program" "$@" ;; esac
  done
  local videos=() rest=()
  while [ $# -gt 0 ]; do
    case "$1" in --*) rest=("$@"); break ;; *) videos+=("$1"); shift ;; esac
  done
  [ ${#videos[@]} -gt 0 ] || usage 1
  # the program reads "title  file-or-link" lines split at spaces: each file gets a link without spaces
  local runs folder list number=0 video name link
  runs="$(config_value runs_dir)"
  [ -n "$runs" ] || die "runs_dir is not in $CONFIG"
  folder="$runs/inputs/$(date +%Y%m%d-%H%M%S)"
  case "$folder" in *[[:space:]]*) die "runs_dir must not contain spaces ($runs): change runs_dir in $CONFIG" ;; esac
  mkdir -p "$folder"
  list="$folder/videos.txt"
  : > "$list"
  for video in "${videos[@]}"; do
    number=$((number + 1))
    if is_link "$video"; then
      printf 'video%02d %s\n' "$number" "$video" >> "$list"
      continue
    fi
    [ -f "$video" ] || die "not a video file: $video"
    name="$(basename "$video")"
    link="$folder/$(printf '%02d' "$number").${name##*.}"
    ln -sfn "$(cd "$(dirname "$video")" && pwd)/$name" "$link"
    printf '%s %s\n' "${name%.*}" "$link" >> "$list"
  done
  printf 'video list: %s\n' "$list"
  exec "$PY_MOTION" -B "$program" --list "$list" "${rest[@]}"
}

[ $# -gt 0 ] || usage 1
command="$1"; shift
case "$command" in
  check) check "$@" ;;
  joints) joints "$@" ;;
  challenge) need_config; exec "$PY_MOTION" -B "$REPO/programs/dance-challenge/challenge.py" "$@" ;;
  duo) need_config; exec "$PY_MOTION" -B "$REPO/programs/duo-scene/duo_pipeline.py" "$@" ;;
  -h|--help|help) usage 0 ;;
  *) printf 'unknown command: %s\n\n' "$command"; usage 1 ;;
esac
