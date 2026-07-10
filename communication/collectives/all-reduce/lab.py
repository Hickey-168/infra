#!/usr/bin/env python3
"""all reduce — 多进程验证

叶子: communication/collectives/all-reduce
启动: torchrun --nproc_per_node=2 lab.py   (见 run.sh)
目标: 观察 rank / collective / shape / 通信量，验证对并行机制的理解。
"""
import os


def main() -> None:
    import torch
    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    # ── 假设 ──────────────────────────────────────────
    # 我预测: ...
    # TODO: 构造张量 -> 执行 collective -> 各 rank 打印 shape/值验证语义
    if rank == 0:
        print(f"world_size={world}, backend={backend}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
