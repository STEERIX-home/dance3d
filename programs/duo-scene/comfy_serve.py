#!/usr/bin/env python3
"""Start a private ComfyUI for this run (a pipeline starts it; it holds the run's GPU while it is up).

    comfy_serve.py [--vram-cap-gb GB [--vram-os-share-gb GB]] [more ComfyUI options]

It uses the installed ComfyUI program and interpreter read-only (config: comfyui, python.comfy). The
models come from the config's model folders (models.comfy, models.h3, models.h3_extra and the run's own
checkpoints/models); inputs, outputs, temporary files and the database all live in the run folder,
and it listens on localhost only.

Custom nodes: only the folders the config lists (comfy_custom_nodes, e.g. ComfyUI-GGUF for the SCAIL-2
GGUF files) are loaded; whatever else sits in ComfyUI's own custom_nodes folder is not.
Memory: --vram-cap-gb makes ComfyUI run as on a card of that size (comfy_capped.py; the SCAIL-2 pipeline
passes it when its --vram tier is smaller than the card, scail_tiers.py), with ComfyUI's own default
reserve. Without it, and without --reserve-vram among the options, 1.5 GB of the card are kept free (what
the H3 runs were made with).
"""
import os
import sys
from pathlib import Path

import paths
from paths import COMFY, COMFY_MODELS, COMFY_PYTHON, JOB

PORT = int(os.environ.get('DANCE3D_COMFY_PORT', '8189'))


def main():
    work = JOB / 'outputs/work/comfy'
    for relative in ('input', 'output'):
        (work / relative).mkdir(parents=True, exist_ok=True)
    for relative in ('comfy-user', 'comfy-temp', 'comfy-previews'):
        (JOB / 'cache' / relative).mkdir(parents=True, exist_ok=True)
    kinds = ('diffusion_models', 'unet', 'loras', 'text_encoders', 'vae', 'clip_vision')
    text = ''
    # the H3 models of the single-dancer program, other H3 base models, and models downloaded into this run
    for name, more in (('h3', paths.location('models.h3', required=False)), ('extra', paths.location('models.h3_extra', required=False)),
                       ('own', JOB / 'checkpoints/models')):
        if more is not None and more.is_dir() and more.resolve() != COMFY_MODELS.resolve():
            found = [kind for kind in kinds if (more / kind).is_dir()]
            if found:
                text += f'{name}:\n  base_path: {more}\n' + ''.join(f'  {kind}: {kind}\n' for kind in found)
    nodes = paths.custom_nodes()
    if nodes:  # each listed folder is one node pack; ComfyUI searches their parent folders and loads only these by name
        parents = sorted({str(folder.parent) for folder in nodes})
        text += 'custom:\n  custom_nodes: |\n' + ''.join(f'    {parent}\n' for parent in parents)
    extra = JOB / 'cache/comfy-extra-models.yaml'
    extra.write_text(text or '{}\n')
    env = os.environ.copy()
    headers = paths.location('python_headers', required=False)
    if headers is not None:  # Triton compiles a small helper and needs Python.h
        include = [str(path) for path in sorted(headers.glob('python3*')) if path.is_dir()] + [str(headers)]
        env['CPATH'] = os.pathsep.join(filter(None, include + [env.get('CPATH')]))
    env['HF_HUB_OFFLINE'] = env['TRANSFORMERS_OFFLINE'] = '1'
    # the motion tools' extra packages are not for ComfyUI; its own extra packages (comfy_pydeps) are
    extra_packages = paths.location('comfy_pydeps', required=False)
    env.pop('PYTHONPATH', None)
    if extra_packages is not None:
        env['PYTHONPATH'] = str(extra_packages)
    options = sys.argv[1:]
    cap = {}
    for name in ('--vram-cap-gb', '--vram-os-share-gb'):  # ours, not ComfyUI's: taken out of the options
        if name in options:
            at = options.index(name)
            cap[name] = options[at + 1]
            del options[at:at + 2]
    if '--vram-cap-gb' in cap:
        env['DANCE3D_VRAM_CAP_GB'] = cap['--vram-cap-gb']
        env['DANCE3D_VRAM_OS_SHARE_GB'] = cap.get('--vram-os-share-gb', '0.6')
        start = [str(COMFY_PYTHON), '-B', str(Path(__file__).resolve().parent / 'comfy_capped.py'), str(COMFY / 'main.py')]
    else:
        start = [str(COMFY_PYTHON), '-B', str(COMFY / 'main.py')]
        if '--reserve-vram' not in options:
            options = ['--reserve-vram', '1.5'] + options
    command = [*start, '--listen', '127.0.0.1', '--port', str(PORT),
               '--models-directory', str(COMFY_MODELS), '--extra-model-paths-config', str(extra),
               '--input-directory', str(work / 'input'), '--output-directory', str(work / 'output'),
               '--user-directory', str(JOB / 'cache/comfy-user'), '--temp-directory', str(JOB / 'cache/comfy-temp'),
               '--previews-directory', str(JOB / 'cache/comfy-previews'),
               '--database-url', 'sqlite:///' + str(JOB / 'cache/comfy.sqlite'),
               '--offline', '--disable-auto-launch', '--disable-all-custom-nodes', '--preview-method', 'none',
               '--use-pytorch-cross-attention', '--log-stdout']
    if nodes:
        command += ['--whitelist-custom-nodes', *[folder.name for folder in nodes]]
    command += options
    os.chdir(COMFY)
    os.execve(command[0], command, env)


if __name__ == '__main__':
    main()
