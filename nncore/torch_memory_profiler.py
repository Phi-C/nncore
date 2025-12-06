import torch
import torchvision
from torch.utils._python_dispatch import TorchDispatchMode

import contextlib
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
from pathlib import Path
from collections import defaultdict


class TorchMemorySnapshot:
    """
    Context manager to record and dump a snapshot of the memory usage in PyTorch program.

    Args:
        max_entries (int): Maximum number of entries to keep track of memory usage history. Default is 10000.
        save_path (Optional[str | Path]): Path to save the snapshot file. Default is 'snapshot.pickle'.

    Usage:
    >>> with TorchMemorySnapshot():
    ...     # Your code here
    ...     pass
    """

    def __init__(
        self,
        max_entries: int = 10000,
        save_path: Optional[str | Path] = "snapshot.pickle",
    ):
        self.max_entries = max_entries
        self.save_path = save_path

    def __enter__(self):
        torch.cuda.memory._record_memory_history(max_entries=self.max_entries)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            torch.cuda.memory._dump_snapshot(self.save_path)
        except Exception as e:
            print(f"Error occurred while dumping snapshot: {e}")
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
        return False


# =============== Global States ==================
MB = 1024.0 * 1024.0

operator_names: Dict[str, int] = defaultdict(int)
mem_usage: Dict[int, Tuple[str, float]] = defaultdict(lambda: ("", 0.0))
max_mem_usage: Dict[int, Tuple[str, float]] = defaultdict(lambda: ("", 0.0))
markers: Dict[str, int] = defaultdict(int)
op_id: int = 0
cur_module: str = ""


# =============== Helper Functions ==================
def add_marker(mark_name: str):
    marker_val = len(mem_usage.values())
    markers[mark_name] = marker_val


def print_top_mem_op(topk):
    op_num = len(mem_usage.keys())
    op_diff: Dict[str, float] = defaultdict(float)
    pre_mem = 0.0
    for idx in range(1, op_num):
        name, mem = mem_usage[idx]
        op_diff[name] = mem - pre_mem
        pre_mem = mem

    print("============================================================")
    print(f"Top{topk} operators that comsume most GPU memory are:")
    for k, v in sorted(op_diff.items(), key=lambda item: item[1], reverse=True)[:topk]:
        print(f"{k}: {v:.3f} MB")
    print("============================================================")


def show_graph(filename):
    y = [mb for (name, mb) in mem_usage.values()]
    min_val = min(y)
    max_val = max(y)
    x = [i for i in range(len(y))]
    fig = plt.figure(figsize=(16, 8))
    plt.plot(x, y, label="memory")
    plt.xlabel("# Operator Calls")
    plt.ylabel("Allocated Memory (MB)")

    for mark_name, marker_val in markers.items():
        plt.plot(
            [marker_val, marker_val], [min_val, max_val], "r", lw=2, label=mark_name
        )
    plt.legend()
    plt.savefig(filename)
    try:
        print(f"Saving figure {filename}")
        plt.savefig(filename)
    except Exception as e:
        print(e)
    finally:
        plt.close()
        markers.clear()


def fwd_hook_wrapper(name):
    def fwd_hook(module, args):
        global cur_module
        cur_module = name + ".forward"

    return fwd_hook


def bwd_hook_wrapper(name):
    def bwd_hook(module, grad_input, grad_output):
        global cur_module
        cur_module = name + ".backward"

    return bwd_hook


# NOTE:
# register_forward_pre_hook / register_forward_hook
# register_full_backward_pre_hook / register_full_backward_hook
# If we use register_full_backward_hook here instead of register_backward_hook,
# it will trigger an error related to autograd's logic to handle view+inplace.
# It's a problem deserved further investigation.
@contextlib.contextmanager
def debug_model(model):

    hook_handles = []
    for name, module in model.named_modules():
        hook_handles.append(module.register_forward_pre_hook(fwd_hook_wrapper(name)))
        hook_handles.append(module.register_backward_hook(bwd_hook_wrapper(name)))

    try:
        yield
    finally:
        for handle in hook_handles:
            handle.remove()


# =============== Core Logic ==================
def record_fn(func_name, verbose=False):
    global op_id
    mem: float = torch.cuda.memory_allocated() / MB
    max_mem: float = torch.cuda.max_memory_allocated() / MB
    mem_usage[op_id] = (func_name, mem)
    max_mem_usage[op_id] = (func_name, max_mem)
    torch.cuda.reset_peak_memory_stats()
    if verbose:
        print(f"Memory Usage: {op_id:06d}: [{func_name}]: {mem:.3f}")
    op_id += 1


class MemoryProfileDispatchMode(TorchDispatchMode):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        rs = func(*args, **kwargs)
        global cur_module
        if func == torch.ops.aten.detach.default:
            return rs
        operator_names[func.__name__] += 1
        func_name = (
            cur_module + "." + func.__name__ + "_" + str(operator_names[func.__name__])
        )
        record_fn(func_name, self.verbose)
        return rs


# =============== Demo ==================
def test_memory_profile_dispatch_mode():
    model = torchvision.models.resnet34().cuda()
    input = torch.rand(32, 3, 224, 224, device="cuda")

    with debug_model(model):
        with MemoryProfileDispatchMode(True):
            loss = model(input)
            add_marker("fw_bw_boundary")
            loss.sum().backward()

    print_top_mem_op(10)
    show_graph("memory_profiler.png")


if __name__ == "__main__":
    test_memory_profile_dispatch_mode()

