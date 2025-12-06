import sys
import functools
import inspect
import ast
import torch
from typing import Any, Callable, Optional


def print_rank0(*args, **kwargs) -> None:
    """Print only on rank 0 in distributed training."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def tensor_tracer(func: Callable) -> Callable:
    """
    Decorator to trace tensor shape changes during function execution.

    Args:
        func: The function to be traced

    Returns:
        The wrapped function with tracing capability
    
    Example:
        >> from tensor_tracer import tensor_tracer
        >> import torch

        >> @tensor_tracer
        >> def my_function(x, y):
        >>     w = x+y
        >>     z = w*2
        >>     return w, z

        >> x = torch.randn(3, 4)
        >> y = torch.randn(3, 4)
        >> my_function(x, y)

        >> from torchvision.models import resnet18

        >> model = resnet18(pretrained=True)
        >> model.eval()
        >> inp = torch.randn(3, 3, 224, 224)
        >> model.forward = tensor_tracer(model.forward)
        >> model(inp)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Store the original trace function to restore later
        original_trace = sys.gettrace()

        def trace(frame, event: str, arg: Any) -> Optional[Callable]:
            if event == "line":
                local_vars = frame.f_locals
                current_line = inspect.getframeinfo(frame).code_context

                if not current_line:
                    return trace

                current_line = current_line[0].strip()
                # print(f"current line: {current_line}")
                # Parse the current line to find variable names
                # This is a simple way to find variable names, but it may not be perfect
                # For example, it won't handle complex expressions or function calls
                # In a real-world scenario, you might want to use a more robust parser
                try:
                    tree = ast.parse(current_line)
                except Exception as e:
                    return trace
                else:
                    current_line_vars = set()
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name) and isinstance(
                            node.ctx, ast.Load
                        ):
                            current_line_vars.add(node.id)
                    # print(f"Current line {frame.f_lineno} variables: {current_line_vars}")
                    # print(f"Frame local variables: {local_vars.keys()}")

                    if current_line_vars:
                        filename = frame.f_code.co_filename
                        lineno = frame.f_lineno
                        current_line_shape_info = []
                        for var_name in current_line_vars:
                            if var_name in local_vars and torch.is_tensor(
                                local_vars[var_name]
                            ):
                                current_line_shape_info.append(
                                    f"{var_name} {tuple(local_vars[var_name].shape)}"
                                )
                        if current_line_shape_info:
                            print_rank0(
                                f"{filename}:{lineno}: {' | '.join(current_line_shape_info)}"
                            )

            return trace

        # Enable tracing
        sys.settrace(trace)

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print_rank0(f"Error during execution of {func.__name__}: {str(e)}")
            raise
        finally:
            # Always restore original trace function
            sys.settrace(original_trace)

        # Print return values
        if isinstance(result, (tuple, list)):
            for i, value in enumerate(result):
                if torch.is_tensor(value):
                    print_rank0(
                        f"{func.__name__} returned: result[{i}] {tuple(value.shape)}"
                    )
        elif torch.is_tensor(result):
            print_rank0(f"{func.__name__} returned: result = {tuple(result.shape)}")

        return result

    return wrapper
