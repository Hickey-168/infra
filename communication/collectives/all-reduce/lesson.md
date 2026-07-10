---
artifact: course-lesson
leaf: communication/collectives/all-reduce
language: zh-CN
estimated_time: 75min
environment: Mac-or-WSL2-GPU
---

# 课程：先精确定义 All-Reduce，再讨论 NCCL 算法

## 1. 语义

rank r 各持一个同 shape tensor `x_r`。sum all-reduce 后，每个 rank 都拥有同一个 `y = sum_r x_r`。这是“reduce”与“结果分发”两件事的组合；不是只把结果放在 rank 0。平均梯度是调用者或框架对 y 除以 world size 的约定，不能默认 collective 自动完成。

## 2. collective 是协议，不是普通函数调用

所有 ranks 必须以兼容的顺序、shape、dtype 和 group 参与。某个 rank 少调用一次、顺序不同或 tensor shape 不同，其他 ranks 可能等待并 hang。实际排查先打印 rank、world size、backend、device、collective 序号和 tensor metadata，而不是先怀疑网络。

## 3. 放进 DDP 的含义

每个 rank 在自己的 data shard 上算 local gradient；all-reduce 后各 rank 用相同聚合 gradient step，因此参数副本保持一致。all-gather 是拼接数据，reduce-scatter 是规约后各自只留一片；这两个在后续 ring/FSDP 会出现。

## 实验课

用 `torchrun` 启动 2 ranks：rank 0 放全 1，rank 1 放全 2；先预测 sum 和 average 的结果，再执行 all-reduce。打印 before/after 与 device。随后在隔离实验中故意制造 shape mismatch，观察错误或超时的诊断信息，不要在生产集群做。

## 通关

能用一句数学式定义 all-reduce，区分它与 reduce/all-gather/reduce-scatter。下一课学习 ring 如何实现这条语义并估算时间。
