import sys
import os
import subprocess
import distutils.util as util
import shlex
import multiprocessing

def get_ips():
    """Read IP address from hostfile"""

    ip_config = os.path.join(os.path.dirname(__file__), "hostfile")
    if not os.path.exists(ip_config):
        raise RuntimeError("ip_config file does not exist")
    print(f"[INFO] Reading IP list from {ip_config}")

    cond = lambda x: x.strip() and not x.strip().startswith("#")
    with open(ip_config, "r") as f:
        return [line.strip().split(" ")[0] for line in f.readlines() if cond(line)]

def get_self_ip():
    """Get IP address of current host"""

    ip = subprocess.check_output(["hostname", "-i"]).decode().strip()
    return ip


def concat_cmd(cmd):
    """Concatenate commands into one string"""

    return " ".join([shlex.quote(c) for c in cmd])

def execute_cmd(cmd):
    """Execute a shell command and return its exit code"""

    print(f"[EXEC] {cmd}")
    return os.system(cmd)


def is_script_file(file_path, check_extensions=('.sh'), check_executable=False):
    """Check if a file is a script file"""

    # 基础检查：是否为文件
    if not os.path.isfile(file_path):
        return False

    # 检查扩展名（如果指定）
    if check_extensions:
        if not file_path.lower().endswith(check_extensions):
            return False

    # 检查可执行权限（如果启用）
    if check_executable and not os.access(file_path, os.X_OK):
        return False

    return True

def sync_file(path):
    """Synchronize files/directories to all remote hosts via rsync commands
    
    Notes:
    1. 不同步当前机器
    2. 待同步文件在各机器位于同于路径下
    """

    if isinstance(path, (list, tuple)):
        for p in path:
            sync_file(p)
        return

    path = os.path.abspath(path)
    if not os.path.exists(path):
        print(f"[WARN] Path does not exist: {path}")
        return

    self_ip = get_self_ip()
    ips = get_ips()
    cmds = []

    for ip in ips:
        if ip == self_ip:
            continue

        if os.path.isdir(path):
            rsync_cmd = f"rsync -avz --delete {shlex.quote(path)}/ {ip}:{shlex.quote(path)}/"
        else:
            rsync_cmd = (
                f"rsync -avz {shlex.quote(path)} {ip}:{shlex.quote(path)}"
            )

        cmds.append(rsync_cmd)

    if cmds:
        num_process = len(cmds)
        if "MAX_PROCESS" in os.environ:
            num_process = min(num_process, int(os.environ["MAX_PROCESS"]))
        with multiprocessing.Pool(num_process) as pool:
            pool.map(execute_cmd, cmds)


def sync_cmd(cmd):
    """Synchronize commands across multiple machines"""

    if len(cmd) == 1 and is_script_file(cmd[0]):
        cmd_list = []
        cmd_list.append("cd {0}".format(os.path.abspath(".")))
        with open(cmd[0], "r") as f:
            lines = f.readlines()
        
        for line in lines:
            if line.strip() and not line.strip().startswith("#"):
                cmd_list.append(line.strip())

        full_cmd = "; ".join(cmd_list)
    else:
        full_cmd = "cd {0}; {1}".format(os.path.abspath("."), concat_cmd(cmd))


    self_ip = get_self_ip()
    ips = get_ips()
    cmds = []

    for ip in ips:
        is_self = ip == self_ip
        if not is_self:
            # ref: https://github.dev/dmlc/dgl/blob/master/tools/launch.py
            # 先配置SSH密钥免密登录:
            # 1. 在master服务器生成密钥对: ssh-keygen -t rsa
            # 2. 将公钥拷贝到其他服务器: ssh-copy-id username@target_ip
            exe_cmd = concat_cmd(["ssh", "-o", "StrictHostKeyChecking=no", f"root@{ip}", full_cmd])
        else:
            exe_cmd = full_cmd

        if is_self and util.strtobool(os.getenv("SKIP_SELF", "0")):
            print("[INFO] Skip self...")
            continue

        print(f"Execute command on {ip}: {exe_cmd}")
        cmds.append(exe_cmd)

    if cmds:
        num_process = len(cmds)
        if "MAX_PROCESS" in os.environ:
            num_process = min(num_process, int(os.environ["MAX_PROCESS"]))
        with multiprocessing.Pool(num_process) as pool:
            pool.map(execute_cmd, cmds)


if __name__ == "__main__":
    mode = sys.argv[1]
    assert mode in ["file", "cmd"]
    if mode == "file":
        sync_file(sys.argv[2:])
    elif mode == "cmd":
        sync_cmd(sys.argv[2:])
    else:
        raise ValueError("Invalid mode: {}".format(mode))