"""
Usage: 
torchrun                \
    --nnodes=2          \
    --node_rank=0       \
    --nproc_per_node=8  \
    --master_addr=<MASTER_ADDR>     \
    --master_port=<PORT>            \
    --rdzv_id=123                   \
    --rdzv_backend=c10d             \
    --rdzv_endpoint=<MASTER_ADDR>:<PORT> \
    connectivity.py
"""

import torch
import torch.distributed as dist


def connectivity_test(
    backend: str = "gloo",
    distributed_init_method: str = "env://",
    test_pytorch_nccl: bool = True,
    test_pytorch_gloo: bool = True,
):
    print("===== Connectivity Test =====")
    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
        )
    except Exception as e:
        raise f"{e}: can not init process group with method {distributed_init_method} and backend {backend}"

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    print(
        f"world_size = {world_size}, global_rank = {global_rank}, local_rank = {local_rank}"
    )

    if test_pytorch_nccl:
        print("===== PyTorch NCCL Test =====")
        data = torch.FloatTensor(
            [
                1,
            ]
            * 128
        ).to("cuda")
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        value = data.mean().item()
        assert value == world_size, f"Expected {world_size}, got {value}"
        print("PyTorch NCCL is successful")

    if test_pytorch_gloo:
        print("===== PyTorch Gloo Test =====")
        gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        cpu_data = torch.FloatTensor(
            [
                1,
            ]
            * 128
        )
        dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
        value = cpu_data.mean().item()
        assert value == world_size, f"Expected {world_size}, got {value}"
        print("PyTorch Gloo is successful")


if __name__ == "__main__":
    connectivity_test()
