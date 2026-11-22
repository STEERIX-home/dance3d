#!/usr/bin/python3
"""Import user-obtained body assets; use the author's downloader and checksums throughout."""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import runner
from settings import EXMR, EXMR_PYTHON, REQUIRED_MODELS, ROOT, model_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smplx-zip', type=Path)
    parser.add_argument('--mano-zip', type=Path)
    parser.add_argument('--sam-dir', type=Path, help='An already authorized SAM 3D Body download')
    args = parser.parse_args()
    for path in (args.smplx_zip, args.mano_zip):
        if path and not path.is_file():
            parser.error(f'파일 없음: {path}')
    if args.sam_dir:
        for relative in ('model.ckpt','model_config.yaml','assets/mhr_model.pt'):
            if not (args.sam_dir / relative).is_file():
                parser.error(f'SAM 파일 없음: {args.sam_dir / relative}')
    with runner.exclusive(), runner.managed_job('rgbmocap-models') as job:
        previous = model_dir()
        target = job / 'checkpoints'
        # Reuse immutable verified weights without duplicating many gigabytes.
        for source in previous.rglob('*'):
            if source.is_file() and not any(part.startswith('.') for part in source.relative_to(previous).parts):
                dest = target / source.relative_to(previous)
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, dest)
                except OSError:
                    shutil.copy2(source, dest)
        if args.sam_dir:
            for relative in ('model.ckpt','model_config.yaml','assets/mhr_model.pt'):
                dest = target/'sam-3d-body-dinov3'/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Do not overwrite a hardlink to an earlier completed artifact.
                if not dest.exists():
                    shutil.copy2(args.sam_dir/relative, dest)
        env = runner.environment(job)
        env.pop('HF_HUB_OFFLINE', None)
        env.pop('TRANSFORMERS_OFFLINE', None)
        env.update(HF_TOKEN_PATH=os.environ.get('HF_TOKEN_PATH', str(Path.home()/'.cache/huggingface/token')),
                   HF_HUB_DISABLE_XET='1')
        command = [str(EXMR_PYTHON), str(EXMR/'tools/download_models.py'),
                   '--checkpoint_dir', str(target), '--verify']
        for flag, path in (('--smplx_zip', args.smplx_zip), ('--mano_zip', args.mano_zip)):
            if path:
                command += [flag, str(path.resolve())]
        print('모델 다운로드·검사 로그:', job/'logs/models.log', flush=True)
        with (job/'logs/models.log').open('w') as log:
            result = subprocess.run(command, env=env, cwd=job, stdout=log, stderr=subprocess.STDOUT)
        missing = [name for name, relative in REQUIRED_MODELS.items() if not (target/relative).is_file()]
        if result.returncode or missing:
            raise RuntimeError('모델 등록 미완료: '+', '.join(missing)+'. 로그: '+str(job/'logs/models.log'))
        runner.write_json(job/'outputs/model-validation.json',{'verified_by':'ExMR official SHA-256 downloader',
                                                              'files':REQUIRED_MODELS})
        # This is installation configuration, not generated run data.
        runner.write_json(ROOT/'model-location.json',{'checkpoints':str(target)})
        print('모델 등록 완료. UI의 모델 상태 새로고침을 누르세요:', target)


if __name__=='__main__':
    main()
