import unittest

from nncore.memory_profiler import (
    MemoryProfileDispatchMode,
    debug_model,
    add_marker,
    print_top_mem_op,
    show_graph,
)
import torch
import torchvision


class TestMemoryProfileDispatchMode(unittest.TestCase):
    def test_dispatch_mode(self) -> None:
        model = torchvision.models.resnet34().cuda()
        input = torch.rand(32, 3, 224, 224, device="cuda")

        with debug_model(model):
            with MemoryProfileDispatchMode(True):
                loss = model(input)
                add_marker("fw_bw_boundary")
                loss.sum().backward()

        print_top_mem_op(10)
        show_graph("memory_profiler.png")
