from time import perf_counter, strftime, localtime, sleep
from typing import Optional, Type
from types import TracebackType


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
            self.end_time = perf_counter()
            self.elapsed_time = self.end_time - self.start_time
            print(f"[END_{self.name}]: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
            print(f"{self.name} cost: {self.elapsed_time:.4f} s")


if __name__ == "__main__":
    with Timer("sleep") as timer:
        sleep(2)
