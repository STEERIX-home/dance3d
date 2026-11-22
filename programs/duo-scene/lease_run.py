#!/usr/bin/env python3
"""Run one command on the run's GPU: with a gpu_helper in the config, under a lease (acquired, renewed while it
runs, always released); without one, on the GPU the environment shows.

    lease_run.py LABEL -- command args...

The command sees only the leased physical GPU (CUDA_VISIBLE_DEVICES) and keeps its caches in the run folder
(DUO_JOB). The lease token is never printed.
"""
import os
import signal
import subprocess
import sys

from paths import cache_environment
from run_support import Lease

OWNER = 'claude-duo'


def main():
    split = sys.argv.index('--')
    label, command = sys.argv[1], sys.argv[split + 1:]
    lease = Lease(OWNER, label)
    env = cache_environment()
    env.update(lease.environment())
    print(f'[lease] {label}: GPU {lease.gpus or "as the environment shows"}', flush=True)
    child = None
    try:
        child = subprocess.Popen(command, env=env, start_new_session=True)

        def forward(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        return child.wait()
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        lease.release()
        print(f'[lease] released ({label})', flush=True)


if __name__ == '__main__':
    sys.exit(main())
