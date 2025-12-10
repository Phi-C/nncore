import torch
import csv
from time import perf_counter, strftime, localtime, sleep
from typing import Optional, Type, List, Dict
from types import TracebackType


class TimerV2:
    """
    A timer which computes the elapsed time.

    Example:

        .. code-block:: python

        timer = TimerV2()
        timer.record("func1_start")
        func1()
        timer.record("func2_start")
        func2()
        timer.record("func2_end", stop=True)
        timer.finalize("/path/to/save/timer.csv")

    """

    def __init__(self) -> None:
        self.prev_name = None
        self.prev_time = None
        self.time_records: List[Dict[str, str]] = []

    def record(self, name: str, stop: bool = False) -> None:
        """
        Record a time point.

        Args:
            name: The name of the time point.
            stop: Whether to stop recording after this time point. In loop situation (a->b->c), if set to True, c->a will not be recorded.
        """
        if self.prev_name is not None:
            torch.cuda.synchronize()
            cur_time = perf_counter()
            elapsed_time = cur_time - self.prev_time
            print(f"[TIME][{self.prev_name} -> {name}]: {elapsed_time:.4f} s")
            self.time_records.append(
                {
                    "START_NAME": self.prev_name,
                    "END_NAME": name,
                    "START_TIME": self.prev_time,
                    "END_TIME": cur_time,
                    "ELAPSED_TIME": elapsed_time,
                }
            )

        if stop:
            self.prev_name = None
            self.prev_time = None
        else:
            self.prev_name = name
            self.prev_time = perf_counter()

    def finalize(self, save_path: str = "timer.csv") -> None:
        """
        Finalize the timer, save the recorded time to a csv file.
        """
        torch.cuda.synchronize()

        with open(save_path, "w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "START_NAME",
                "END_NAME",
                "START_TIME",
                "END_TIME",
                "ELAPSED_TIME",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.time_records)


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
            torch.cuda.synchronize()
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
            torch.cuda.synchronize()
            self.end_time = perf_counter()
            self.elapsed_time = self.end_time - self.start_time
            print(f"[END_{self.name}]: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
            print(f"{self.name} cost: {self.elapsed_time:.4f} s")


if __name__ == "__main__":
    with Timer("sleep") as timer:
        sleep(2)

    timer = TimerV2()
    timer.record("start")
    sleep(1)
    timer.record("middle")
    sleep(2)
    timer.record("end")
