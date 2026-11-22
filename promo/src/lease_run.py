#!/usr/bin/env python3
"""Run one command under a managed GPU lease: acquire, renew while it runs, always release.

    lease_run.py LABEL -- command args...

The command sees only the leased physical GPU (CUDA_VISIBLE_DEVICES) and keeps its caches in this job.
The lease token is never printed.
"""
import json
import os
import signal
import subprocess
import sys
import time

HELPER = '/home/steerix-train-server/gpu-dashboard/gpu-agent.py'
JOB = '/srv/training/jobs/20261116-041650-dance-promo-video-0df0'
OWNER = 'claude-code'


def helper(*args):
    result = subprocess.run([HELPER, *map(str, args)], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f'gpu-agent {args[0]} failed: {result.stderr.strip()[:300]}')
    return json.loads(result.stdout)


def main():
    split = sys.argv.index('--')
    label, command = sys.argv[1], sys.argv[split + 1:]
    if not helper('available').get('available'):
        raise RuntimeError('No GPU is available: another lease or CUDA process holds it')
    acquired = helper('acquire', '--owner', OWNER, '--job', label, '--count', 1, '--ttl', 3600)
    lease_id, gpu_ids = acquired['id'], acquired['gpu_ids']
    del acquired
    env = dict(os.environ)
    for name, folder in (('XDG_CACHE_HOME', 'cache/xdg'), ('TORCH_HOME', 'cache/torch'), ('HF_HOME', 'cache/hf'), ('NUMBA_CACHE_DIR', 'cache/numba'),
                         ('MPLCONFIGDIR', 'cache/mpl'), ('TMPDIR', 'cache/tmp')):
        os.makedirs(os.path.join(JOB, folder), exist_ok=True)
        env[name] = os.path.join(JOB, folder)
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    print(f'[lease] acquired {lease_id} for {label}: GPU {gpu_ids}', flush=True)
    child = None
    try:
        child = subprocess.Popen(command, env=env, start_new_session=True)

        def forward(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        renew_at = time.monotonic() + 600
        while child.poll() is None:
            if time.monotonic() > renew_at:
                helper('renew', lease_id, '--ttl', 3600)
                renew_at = time.monotonic() + 600
            time.sleep(1)
        return child.returncode
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        try:
            helper('release', lease_id)
            print(f'[lease] released {lease_id}', flush=True)
        except RuntimeError as error:
            print(f'[lease] RELEASE FAILED {lease_id}: {error}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
