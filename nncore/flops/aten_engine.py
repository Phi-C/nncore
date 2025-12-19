"""
Copyright (C) 2024 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
"""

import sys
import traceback
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .utils import flops_to_string
from .aten_ops import ATEN_OPS_MAPPING


class FlopCounterMode(TorchDispatchMode):
    def __init__(
        self,
        module=None,
        verbose=False,
        print_per_layer_stat=False,
        output_params=None,
        custom_hooks={},
        ignored_ops=[],
    ):
        self.verbose = verbose
        if output_params is None:
            output_params = defaultdict(dict)
        self.output_params = output_params
        self.print_fn = partial(print, **self.output_params["print_params"])
        self.all_ops = deepcopy(ATEN_OPS_MAPPING)
        self.all_ops.update(custom_hooks)
        self.ignored_ops = ignored_ops

        self.print_per_layer_stat = print_per_layer_stat
        self.flop_counts = defaultdict(lambda: defaultdict(int))
        self.parents = ["Global"]
        self._total_complexity = None
        if module is not None:
            for name, mod in dict(module.named_children()).items():
                mod.register_forward_pre_hook(self.enter_module(name))
                mod.register_forward_hook(self.exit_module(name))

    @property
    def complexity(self):
        return self._total_complexity

    def enter_module(self, name):
        def f(*args):
            self.parents.append(name)

        return f

    def exit_module(self, name):
        def f(*args):
            assert self.parents[-1] == name
            self.parents.pop()

        return f

    def __enter__(self):
        self.flop_counts.clear()
        super().__enter__()

    def __exit__(self, *args):
        self._total_complexity = sum(self.flop_counts["Global"].values())
        if self.print_per_layer_stat:
            self.print_fn(
                "Total:"
                + flops_to_string(
                    self._total_complexity, **self.output_params["serialize_params"]
                )
            )
            for mod in self.flop_counts.keys():
                self.print_fn("Module: ", mod)
                for k, v in self.flop_counts[mod].items():
                    self.print_fn(
                        f"{k}: "
                        + flops_to_string(v, **self.output_params["serialize_params"])
                    )
                self.print_fn()
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        def normalize_tuple(x):
            if not isinstance(x, tuple):
                return (x,)
            return x

        kwargs = kwargs if kwargs else {}

        out = func(*args, **kwargs)
        func_packet = func._overloadpacket

        if func_packet in self.ignored_ops:
            self.print_fn(f"Warning: {func_packet} operation is ignored")
        elif func_packet in self.all_ops:
            flop_count = self.all_ops[func_packet](args, normalize_tuple(out))
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count
        elif self.verbose:
            self.print_fn(f"Warning: {func_packet} operation is treated as a zero-op")

        return out
