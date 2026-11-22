#!/usr/bin/env python3
"""Run ComfyUI as if its graphics card had only DANCE3D_VRAM_CAP_GB of memory: the --vram tier of the dance
programs on a larger card, or a card that other programs share.

    comfy_capped.py COMFYUI/main.py [ComfyUI options]        (comfy_serve.py starts it so)

ComfyUI decides how much of each model it keeps on the card from the card's total and free memory
(torch.cuda.mem_get_info). Here the card reads as one of the cap's size on which a desktop holds
DANCE3D_VRAM_OS_SHARE_GB (what a real card of that size loses to the screen and the driver), and whose
free memory is the rest of the cap that this process does not hold yet; never more than is really free.
ComfyUI's dynamic-VRAM library (comfy_aimdo) gets the same budget for the model weights it pages in.
What the process holds is read from the driver (NVML), as nvidia-smi shows it.

Nothing else is changed: ComfyUI runs with its own defaults. Without DANCE3D_VRAM_CAP_GB this only runs
main.py. This is a budget ComfyUI keeps, not a wall: memory it cannot plan for (a model's working memory)
can still go past it, so the pipelines measure what the run really holds (run_support.MemoryWatch).
"""
import ctypes
import os
import runpy
import sys
from pathlib import Path

GIB = 1024 ** 3
NOT_AVAILABLE = 2 ** 64 - 1  # NVML's "value not available" (e.g. per-process memory on some Windows drivers)


class _Process(ctypes.Structure):  # nvmlProcessInfo_t, as nvmlDeviceGetComputeRunningProcesses_v3 fills it
    _fields_ = [('pid', ctypes.c_uint), ('usedGpuMemory', ctypes.c_ulonglong), ('gpuInstanceId', ctypes.c_uint),
                ('computeInstanceId', ctypes.c_uint)]


class Card:
    """This process's graphics card as the driver sees it (NVML, the library behind nvidia-smi)."""

    def __init__(self):
        self.nvml = ctypes.CDLL('nvml.dll' if os.name == 'nt' else 'libnvidia-ml.so.1')
        if self.nvml.nvmlInit_v2():
            raise OSError('nvmlInit_v2 failed')
        self.handle = ctypes.c_void_p()
        first = (os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0].strip()
        if first.startswith(('GPU-', 'MIG-')):
            failed = self.nvml.nvmlDeviceGetHandleByUUID(first.encode(), ctypes.byref(self.handle))
        else:  # an index: NVML counts in PCI order, as CUDA does with CUDA_DEVICE_ORDER=PCI_BUS_ID (set below)
            failed = self.nvml.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(int(first)), ctypes.byref(self.handle))
        if failed:
            raise OSError(f'NVML has no card for CUDA_VISIBLE_DEVICES={first}')

    def held_here(self):
        """Bytes this process holds on the card (0 before its first use), or None if the driver does not say."""
        size = 128
        table = (_Process * size)()
        count = ctypes.c_uint(size)
        if self.nvml.nvmlDeviceGetComputeRunningProcesses_v3(self.handle, ctypes.byref(count), table):
            return None
        used = [table[i].usedGpuMemory for i in range(count.value) if table[i].pid == os.getpid()]
        if any(value == NOT_AVAILABLE for value in used):
            return None
        return sum(used)


def install(cap, os_share):
    """Make ComfyUI see a card of `cap` bytes with `os_share` bytes of it taken by the desktop."""
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    try:
        card = Card()
    except (OSError, ValueError) as error:
        card = None
        print(f'[comfy_capped] NVML not usable ({error}): counting only what PyTorch holds', flush=True)
    state = dict(total=None)

    def headroom():
        """comfy_aimdo keeps this much of the real card free of this process's own pages."""
        import comfy_aimdo.control as control
        if control.lib is None or state['total'] is None:
            return
        default = control.get_simple_vram_headroom() if hasattr(control, 'get_simple_vram_headroom') else 0
        state.setdefault('aimdo_default', default)
        wanted = int(max(0, state['total'] - cap) + os_share + state['aimdo_default'])
        control.set_simple_vram_headroom(wanted)
        print(f'[comfy_capped] dynamic VRAM budget: card {state["total"] / GIB:.2f} GiB, this process up to '
              f'{(state["total"] - wanted) / GIB:.2f} GiB', flush=True)

    import comfy_aimdo.control as control  # loads no torch, so ComfyUI's own order of imports is kept
    original_init = control.init

    def init(*args, **kwargs):
        done = original_init(*args, **kwargs)
        if done:
            headroom()
        return done

    control.init = init

    def patch_torch():
        import torch
        if not torch.cuda.is_available():  # ComfyUI on the processor (--cpu, no card): nothing to cap
            print('[comfy_capped] no CUDA card: ComfyUI runs uncapped', flush=True)
            return
        real = torch.cuda.mem_get_info

        def mem_get_info(device=None):
            free, total = real(device)
            held = card.held_here() if card is not None else None
            reserved = torch.cuda.memory_reserved(device)
            if held is None or (held == 0 and reserved > 0):  # no word from the driver: PyTorch's part and a usual context
                held = reserved + GIB // 2
            return max(0, min(free, cap - os_share - held)), min(total, cap)

        torch.cuda.mem_get_info = torch.cuda.memory.mem_get_info = mem_get_info
        state['total'] = real()[1]
        headroom()
        print(f'[comfy_capped] ComfyUI sees a {cap / GIB:g} GiB card ({os_share / GIB:g} GiB of it kept for the desktop); '
              f'the real card has {state["total"] / GIB:.2f} GiB', flush=True)

    class BeforeMemoryManager:
        """Patches torch just before ComfyUI's memory manager is first imported (it asks the card at once)."""
        done = False

        def find_spec(self, name, path=None, target=None):
            if name == 'comfy.model_management' and not self.done:
                self.done = True  # (not taken out of sys.meta_path here: the import system is walking that list)
                patch_torch()
            return None  # the usual finders load the module

    sys.meta_path.insert(0, BeforeMemoryManager())


def main():
    if len(sys.argv) < 2 or not sys.argv[1].endswith('main.py'):
        raise SystemExit('usage: comfy_capped.py COMFYUI/main.py [ComfyUI options]')
    comfy_main = Path(sys.argv[1]).resolve()
    sys.argv = [str(comfy_main)] + sys.argv[2:]
    sys.path[0] = str(comfy_main.parent)  # as when main.py is run itself (and none of the dance programs' modules)
    cap = os.environ.get('DANCE3D_VRAM_CAP_GB')
    if cap:
        install(float(cap) * GIB, float(os.environ.get('DANCE3D_VRAM_OS_SHARE_GB', '0.6')) * GIB)
    runpy.run_path(str(comfy_main), run_name='__main__')


if __name__ == '__main__':
    main()
