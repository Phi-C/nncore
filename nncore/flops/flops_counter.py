import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union

import torch.nn as nn

from .aten_engine import FlopCounterMode
from .utils import flops_to_string, params_to_string


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def get_model_complexity_info_general(
    fn: Callable,
    *,
    args: tuple[Any, ...] = (),  # fn's positional arguments
    kwargs: dict[str, Any] | None = None,  # fn's keyword arguments
    # counter configuration
    tgt_module: nn.Module = None,
    ignore_ops: List[Any] = [],
    print_per_layer_stat: bool = True,
    ost: TextIO = sys.stdout,
    verbose: bool = False,
    custom_modules_hooks: Dict[Union[nn.Module, Any], Any] = {},
    # output configuration
    output_precision=2,
    flops_units: Optional[str] = "GMac",
    param_units: Optional[str] = "M",
    as_strings: bool = True,
) -> Tuple[Union[str, int, None], Union[str, int, None]]:

    output_params = {
        "serialize_params": {"units": flops_units, "precision": output_precision},
        "print_params": {"file": ost},
    }

    assert isinstance(
        tgt_module, nn.Module
    ), "tgt_module must be an instance of torch.nn.Module"
    assert tgt_module is not None, "tgt_module must be provided"

    # Get parameter sum
    params_count = get_model_parameters_number(tgt_module)

    # Get forward flops count
    tgt_module.eval()

    try:
        counter = FlopCounterMode(
            tgt_module,
            verbose,
            print_per_layer_stat,
            output_params,
            custom_modules_hooks,
            ignore_ops,
        )

        kwargs = kwargs if kwargs is not None else {}
        with counter:
            fn(*args, **kwargs)
        flops_count = counter.complexity
    except Exception as e:
        print(
            "Flops estimation was not finished successfully because of"
            f" the following exception: \n{type(e)}: {e}"
        )
        traceback.print_exc()

        return None, None

    if as_strings and flops_count is not None and params_count is not None:
        flops_string = flops_to_string(
            flops_count, units=flops_units, precision=output_precision
        )
        params_string = params_to_string(
            params_count, units=param_units, precision=output_precision
        )
        return flops_string, params_string

    return flops_count, params_count
