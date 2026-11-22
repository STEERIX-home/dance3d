#!/usr/bin/env python3
"""Write <repo>/config.local.json: where the installer put every tool and model (keys: config.example.json).

    python3 install/write_config.py --root <install root> --repo <repo> [--joints] [--all] [--h3] [--vram 12]

Keys that are already in the file and that the installer does not manage (e.g. gpu_helper, managed_jobs,
python.ui, models.h3_extra) are kept, and a path you set by hand is not replaced by null. The old file is kept
as config.local.json.bak. Only the Python standard library is used.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_models  # noqa: E402  (the SCAIL-2 tier choice is the same as for the downloads)


def merge(old, new):
    """new over old; a None in new never replaces a value in old; dicts are merged key by key."""
    out = dict(old)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(old.get(key), dict):
            out[key] = merge(old[key], value)
        elif value is not None or old.get(key) is None:
            out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--joints', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--h3', action='store_true')
    parser.add_argument('--vram', default='auto')
    parser.add_argument('--runs-dir')
    parser.add_argument('--results-dir')
    args = parser.parse_args()
    root, repo = Path(args.root).absolute(), Path(args.repo).absolute()
    models = root / 'models'
    joints = args.joints or args.all

    def there(path):
        """the path if it exists, else None (so a program names the missing key and its install step)"""
        return str(path) if path.exists() else None

    new = {
        '_about': ['Written by install/install.sh; the keys are explained in config.example.json.',
                   'Re-running the installer keeps keys it does not manage (e.g. gpu_helper) and paths you changed.'],
        'python': {'motion': there(root / 'envs/motion/bin/python'), 'comfy': there(root / 'envs/comfy/bin/python'),
                   'face': there(root / 'envs/face/bin/python')},
        'blender': there(root / 'blender-4.5.16-linux-x64/blender'),
        'comfyui': there(root / 'src/ComfyUI'),
        'comfy_custom_nodes': [p for p in [there(root / 'src/ComfyUI/custom_nodes/ComfyUI-GGUF')] if p],
        'exmr': there(root / 'src/ExMR'),
        'smplx': there(models / 'motion/body_models/smplx/SMPLX_NEUTRAL.npz'),
        'models': {
            'comfy': there(models / 'comfy'),
            'h3': there(models / 'h3') if args.h3 else None,
            'sam3': there(models / 'sam3/sam3.pt'),
            'moge': there(models / 'moge/model.pt') if joints else None,
            'detector': there(models / 'motion/yolo/yolo11x.pt') if joints else None,
            'motion': there(models / 'motion'),
            'face': there(models / 'face'),
            'describer': there(models / 'describer/qwen2.5-vl-7b-instruct') if args.all else None,
        },
        'yt_dlp': there(root / 'envs/tools/bin/yt-dlp') if joints else None,
        'runs_dir': str(Path(args.runs_dir).expanduser().absolute()) if args.runs_dir else str(root / 'runs'),
        'results_dir': str(Path(args.results_dir).expanduser().absolute()) if args.results_dir else str(repo / 'results'),
        'vram_gb': get_models.choose_tier(args.vram),
    }
    target = repo / 'config.local.json'
    old = {}
    if target.is_file():
        try:
            old = json.loads(target.read_text(encoding='utf-8'))
        except ValueError:
            print(f'   {target} was not valid JSON; it is kept as config.local.json.bak and written again')
        shutil.copyfile(target, target.with_name('config.local.json.bak'))
    config = merge(old, new)
    config['_about'] = new['_about']
    target.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    missing = [key for key, value in [('python.motion', config['python'].get('motion')), ('python.comfy', config['python'].get('comfy')),
                                      ('python.face', config['python'].get('face')), ('blender', config.get('blender')),
                                      ('comfyui', config.get('comfyui')), ('exmr', config.get('exmr')), ('smplx', config.get('smplx')),
                                      ('models.sam3', config['models'].get('sam3')), ('models.face', config['models'].get('face'))]
               if not value]
    print(f'   wrote {target} (SCAIL-2 tier {config["vram_gb"]} GB)')
    if missing:
        print(f'   still empty (not installed yet): {", ".join(missing)}')


if __name__ == '__main__':
    main()
