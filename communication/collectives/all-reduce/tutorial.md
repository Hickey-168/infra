---
artifact: curriculum-tutorial
leaf: communication/collectives/all-reduce
priority: P0
environment: Mac-or-WSL2-GPU
---

# All-Reduce：让每个 rank 得到同一个聚合结果

## 问题

数据并行中每个 rank 用不同 data shard 算出局部 gradient。若各自直接 optimizer step，参数会立即分叉；若只把结果送到 rank 0，又会让其他 rank 没有下一步需要的完整梯度。

## 语义先于算法

设 rank r 有同 shape tensor `x_r`。sum all-reduce 的结果在每个 rank 都是：

```text
y = sum_r x_r
```

average 是 `y / world_size` 的额外约定，可能由 DDP、loss scaling 或调用者完成，不能想当然地认为 collective 自动平均。all-reduce 与 reduce 的差异在结果归属；与 all-gather 的差异在操作是逐元素规约而非拼接。

## 因果链

1. `init_process_group` 建立 ranks、backend 和 rendezvous。
2. 每 rank 创建本地 tensor；collective 是全体参与者协议，所有 rank 必须在兼容顺序、dtype、shape 下参与。
3. backend 执行规约并把结果交回所有 rank；任一 rank 缺席或调用次序不一致会 hang 或报错。
4. 调用者在正确同步点消费结果，例如 DDP 在 optimizer step 前消费已规约 gradient。

## 预测后验证

用 `torchrun --standalone --nproc_per_node=2` 在 CPU/Gloo 或 GPU/NCCL 跑：rank 0 初始化全 1，rank 1 初始化全 2，sum all-reduce 后各自应看到全 3。打印 `rank/world_size/backend/device/before/after`。再故意只改变一个 rank 的 shape（在受控小实验中）观察错误/超时，理解 collective contract；不要在共享生产环境制造 hang。

## 下一跳

语义正确仍不够，下一叶 `ring-allreduce` 解释一个 all-reduce 如何分 chunk、走网络，并给出可计算的通信量与时间模型。
