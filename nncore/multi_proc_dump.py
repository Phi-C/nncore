import os

def multi_proc_dump(rank: int, dir_path: str) -> int:
    """
    Redirects the current process's stdout and stderr to a log file in the specified directory.

    Args:
        rank (int): Unique identifier for the process (typically the process rank or ID).
        dir_path (str): Directory path where the log file will be stored.
    
    Returns:
        fd (int): File descriptor of the log file.
    """
    os.makedirs(dir_path, exist_ok=True)

    log_file = os.path.join(dir_path, f"{rank}.log")

    fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    return fd