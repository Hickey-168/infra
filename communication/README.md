# Communication

通信目录关注多 GPU/多机间的数据交换。它是分布式训练、MoE、分布式推理和 checkpoint 的共同瓶颈。

## Structure

- `collectives`: broadcast、reduce、all-reduce、all-gather、reduce-scatter、all-to-all。
- `algorithms`: ring、tree、hierarchical collectives。
- `nccl`: NCCL debug、topology、env vars、tests。
- `overlap`: compute/communication overlap、bucket size、streams。
- `network`: PCIe、NVLink、InfiniBand/RDMA 的基本边界。

## Exit Bar

你应该能给出一次 all-reduce 的通信量估算，解释为什么 FSDP 常用 reduce-scatter/all-gather，并能从 NCCL/PyTorch trace 里定位通信瓶颈。

