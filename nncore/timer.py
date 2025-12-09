import torch
from time import perf_counter, strftime, localtime, sleep
from typing import Optional, Type
from types import TracebackType


class TiimerV2:
    """
    A timer which computes the elapsed time.
   
    Example:

        .. code-block:: python

        timer = TimerV2()
        timer.record("func1_start")
        func1()
        timer.record("func2_start")
        func2()
        timer.record("func2_end")

    """

    def __init__(self) -> None:
        self.prev_name = None
        self.prev_time = None

    def record(self, name: str) -> None:
        if self.prev_name is not None:
            torch.cuda.synhcronize()
            elapsed_time = perf_counter() - self.prev_time
            print(f"[TIME][{self.prev_name} -> {name}]: {elapsed_time:.4f} s")

        self.prev_time = perf_counter()
        self.prev_name = name


class Timer:
    """
    A timer which computes the elapsed time.
   
    Example:

        .. code-block:: python

        with Timer("my_func") as timer:
            my_func()

    """

    def __init__(self, name: str, enable_timing: bool = True) -> None:
        self.name = name.upper()
        self.enable_timing = enable_timing

    def __enter__(self) -> None:
        if self.enable_timing:
            torch.cuda.synhcronize()
            self.start_time = perf_counter()
            print(f"[START_{self.name}]: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exec_tb: Optional[TracebackType],
    ) -> None:
        if self.enable_timing:
            torch.cuda.synhcronize()
            self.end_time = perf_counter()
            self.elapsed_time = self.end_time - self.start_time
            print(f"[END_{self.name}]: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
            print(f"{self.name} cost: {self.elapsed_time:.4f} s")


if __name__ == "__main__":
    with Timer("sleep") as timer:
        sleep(2)

    timer = TiimerV2()
    timer.record("start")
    sleep(1)
    timer.record("middle")
    sleep(2)
    timer.record("end")
