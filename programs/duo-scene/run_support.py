"""What every pipeline of the dance programs needs around its steps, the same everywhere:

  run folder   one folder per run with outputs/, logs/, cache/, checkpoints/. By default a dated folder
               under runs_dir of the config; with managed_jobs (and a gpu_helper) the helper's job-init
               makes it and artifact-register records it when the run ends.
  GPU          with a gpu_helper in the config, one lease for the whole run (renewed in the background,
               always released) and only the leased GPU is used; without one, the GPU the environment
               shows (CUDA_VISIBLE_DEVICES, or the first one).
  file cache   on Linux, the file cache of the program's own memory group is given back before the video
               model starts (see free_cache); skipped where that does not exist.
Pure Python (no third-party packages): the pipelines run with any Python 3.
"""
import json
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path

import paths

LAYOUT = ('outputs', 'logs', 'cache', 'checkpoints')


class Failed(RuntimeError):
    """A step failed; the message is what the user is shown."""


def stamp():
    return time.strftime('%H:%M:%S')


def helper(*args):
    """Call the GPU lease helper of the config (gpu_helper)."""
    program = paths.location('gpu_helper', required=True)
    result = subprocess.run([str(program), *map(str, args)], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise Failed(f'gpu-agent {args[0]}: {result.stderr.strip()[:300]}')
    return json.loads(result.stdout)


def managed():
    """True when runs are managed jobs of the GPU helper (config: managed_jobs true and a gpu_helper)."""
    return bool(paths.raw('managed_jobs', False)) and paths.location('gpu_helper', required=False) is not None


def new_run(owner, name):
    """A new, empty run folder (its path is spelled exactly as it is used from then on)."""
    if managed():
        return Path(helper('job-init', '--owner', owner, '--job', name)['job_dir'])
    runs = paths.RUNS
    folder = runs / f'{time.strftime("%Y%m%d-%H%M%S")}-{name}-{secrets.token_hex(2)}'
    for part in LAYOUT:
        (folder / part).mkdir(parents=True, exist_ok=True)
    return folder


def given_run(text):
    """The run folder the user named to go on with. Never resolved: a run is recognised by the spelling of
    its paths, and resolving links can change it (a jobs folder that is a link to another disk)."""
    folder = Path(os.path.abspath(os.path.expanduser(text)))
    if not (folder / 'outputs/options.json').is_file():
        raise Failed(f'이어서 할 작업 폴더가 아닙니다 (outputs/options.json 이 없음): {folder} / not a run folder of this program: {folder}')
    return folder


def register(folder):
    """Record a finished run with the GPU helper (managed jobs only)."""
    if not managed():
        return
    try:
        helper('artifact-register', '--path', folder)
    except Exception as error:  # a run that was continued is already registered
        if 'already' not in str(error):
            print(f'[{stamp()}] artifact-register: {error}', flush=True)


class Lease:
    """The GPU for the whole run. With a gpu_helper: one lease, renewed in the background, always released.
    Without one: the GPUs the environment already shows (nothing to take or give back)."""

    def __init__(self, owner, label, ttl=3600):
        self.id, self.stop, self.thread = None, threading.Event(), None
        if paths.location('gpu_helper', required=False) is None:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
            self.gpus = [part for part in visible.split(',') if part.strip()] if visible else []
            return
        if not helper('available').get('available'):
            raise Failed('GPU를 쓸 수 없습니다. 다른 작업이 GPU를 쓰고 있습니다. / No GPU is free: another job holds it.')
        acquired = helper('acquire', '--owner', owner, '--job', label, '--count', 1, '--ttl', ttl)
        self.id, self.gpus, self.ttl = acquired['id'], [str(g) for g in acquired['gpu_ids']], ttl
        self.thread = threading.Thread(target=self._renew, daemon=True)
        self.thread.start()

    def environment(self):
        """What the steps' environment gets: only the leased GPU (nothing when no helper decides)."""
        if self.id is None:
            return {}
        return dict(CUDA_VISIBLE_DEVICES=','.join(self.gpus), CUDA_DEVICE_ORDER='PCI_BUS_ID')

    def _renew(self):
        while not self.stop.wait(600):
            try:
                helper('renew', self.id, '--ttl', self.ttl)
            except Exception as error:  # the next turn tries again; a lost lease shows up in the dashboard
                print(f'[{stamp()}] lease renew failed: {error}', flush=True)

    def release(self):
        self.stop.set()
        if self.id is None:
            return
        try:
            helper('release', self.id)
        except Exception as error:
            print(f'[{stamp()}] lease release failed: {error}', flush=True)


def free_cache():
    """Give back the file cache that earlier work left in this program's memory group (Linux cgroup v2).
    A group with a memory limit (a page's service) otherwise pushes the video model's files out of the cache
    between steps, and every chunk reads them again from the disk (11-15: 117 s -> 200 s a chunk on a page's
    second run). Only cached copies of files are dropped (no swapping out). Skipped quietly where the
    kernel has no such files (other systems, cgroup v1, a group without memory control)."""
    try:
        line = Path('/proc/self/cgroup').read_text().strip().splitlines()
    except OSError:
        return
    entry = next((item for item in line if item.startswith('0::')), None)
    if entry is None:
        return
    group = Path('/sys/fs/cgroup') / entry.split('::', 1)[1].lstrip('/')
    if not (group / 'memory.reclaim').exists() or not (group / 'memory.stat').exists():
        return
    try:
        stat = dict(row.split() for row in (group / 'memory.stat').read_text().splitlines())
        before = int(stat['file'])
        try:
            (group / 'memory.reclaim').write_text(f'{before} swappiness=0')
        except OSError:  # the kernel freed less than asked (or knows no swappiness): what it freed is still freed
            pass
        after = int(dict(row.split() for row in (group / 'memory.stat').read_text().splitlines())['file'])
        print(f'[{stamp()}] file cache of the memory group: {before / 2**30:.1f} -> {after / 2**30:.1f} GiB', flush=True)
    except (OSError, ValueError, KeyError):
        return


def _gpu_processes():
    """{pid: MiB} of every process that holds memory on the GPUs (compute and graphics), from nvidia-smi -q;
    {} if nvidia-smi is missing. --query-compute-apps would leave out graphics programs such as Blender's EEVEE."""
    try:
        text = subprocess.run(['nvidia-smi', '-q', '-d', 'PIDS'], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    found, pid = {}, None
    for line in text.splitlines():
        key, _, value = line.partition(':')
        key, value = key.strip(), value.strip()
        if key == 'Process ID':
            pid = int(value) if value.isdigit() else None
        elif key == 'Used GPU Memory' and pid is not None:
            number = value.split()[0]
            if number.isdigit():
                found[pid] = found.get(pid, 0) + int(number)
            pid = None
    return found


def _children_of(root):
    """The process ids of `root` and of all processes started under it (from /proc)."""
    parents = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                parents[int(entry.name)] = int((entry / 'stat').read_text().rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
    ours, grew = {root}, True
    while grew:
        grew = False
        for pid, parent in parents.items():
            if parent in ours and pid not in ours:
                ours.add(pid)
                grew = True
    return ours


def _role(pid):
    """A short name of what a process is: the Python script it runs, or its program's name."""
    try:
        words = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except OSError:
        return f'pid {pid}'
    words = [word.decode(errors='replace') for word in words if word]
    script = next((Path(word).name for word in words if word.endswith('.py')), None)
    return script or (Path(words[0]).name if words else f'pid {pid}')


class MemoryWatch:
    """Graphics memory of this run, sampled every `every` seconds while the run goes on: what this run's own
    processes hold (each one and all together, also per step) and what the whole card holds. Written to
    `path` (JSON) now and then and at the end. This is the real use, measured from outside the programs."""

    def __init__(self, path, every=1.0):
        self.path, self.every, self.step = Path(path), every, 'start'
        self.root = os.getpid()
        self.peak, self.card_peak, self.card_start = 0, 0, None
        self.by_step, self.by_role, self.series = {}, {}, []
        self.stop, self.thread = threading.Event(), None
        self.started = time.time()

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *error):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=30)
        self.write()

    def _card_used(self):
        try:
            rows = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=20).stdout.split()
            return max(int(float(value)) for value in rows) if rows else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def _run(self):
        last_write = time.time()
        while not self.stop.is_set():
            used = _gpu_processes()
            ours = _children_of(self.root) if used else set()
            mine = {pid: mib for pid, mib in used.items() if pid in ours}
            total = sum(mine.values())
            card = self._card_used()
            if self.card_start is None and card is not None:
                self.card_start = card
            self.peak = max(self.peak, total)
            if card is not None:
                self.card_peak = max(self.card_peak, card)
            step = self.step
            self.by_step[step] = max(self.by_step.get(step, 0), total)
            for pid, mib in mine.items():
                role = _role(pid)
                self.by_role[role] = max(self.by_role.get(role, 0), mib)
            if not self.series or self.series[-1][1:] != [total, card, step] or time.time() - self.series[-1][0] - self.started >= 10:
                self.series.append([round(time.time() - self.started, 1), total, card, step])
            if time.time() - last_write > 30:
                self.write()
                last_write = time.time()
            self.stop.wait(self.every)

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(
            note='MiB, sampled every %.1f s with nvidia-smi -q -d PIDS (compute and graphics processes); "run" = the processes '
                 'of this run only, "card" = everything on the card (memory.used, includes the desktop and other programs)' % self.every,
            run_peak_mib=self.peak, card_peak_mib=self.card_peak, card_at_start_mib=self.card_start,
            peak_by_step_mib=self.by_step, peak_by_program_mib=self.by_role,
            series_columns=['seconds', 'run_mib', 'card_mib', 'step'], series=self.series), indent=1))


def gpu_memory_gib(gpus=None):
    """Total memory of the (first) GPU the run uses, in GiB, from nvidia-smi; None if it cannot be read."""
    command = ['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits']
    try:
        rows = subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    totals = {}
    for row in rows:
        try:
            index, total = [part.strip() for part in row.split(',')]
            totals[index] = float(total) / 1024
        except ValueError:
            continue
    if not totals:
        return None
    visible = [g for g in (gpus or []) if g in totals]
    if not visible and os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        visible = [g.strip() for g in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if g.strip() in totals]
    return totals[visible[0]] if visible else next(iter(totals.values()))
