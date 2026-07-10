---
artifact: course-lesson
leaf: distributed-training/data-parallel/ddp-gradient-allreduce
language: zh-CN
estimated_time: 120min
environment: WSL2-GPU
---

# 课程：DDP 如何让 backward 计算和梯度通信相遇

## 1. 正确性基础

各 rank 从相同参数开始，处理不同 data shard，得到 local gradients。若 optimizer step 前将每个 parameter gradient all-reduce（并按约定平均），各 rank 获得同一 gradient、做同一 update，参数副本继续一致。global batch 通常是每 rank batch 乘 world size，这影响 loss scaling 和学习率解释。

## 2. DDP 比手写循环多了什么

backward 并非同时产出全部 gradients；从输出端到输入端逐层 ready。DDP reducer 把 parameters 组织成 buckets，bucket ready 时可异步启动 all-reduce，同时更早层的 backward 继续运行。这给 compute/communication overlap 创造机会；bucket 大小太小增加启动开销，太大又延迟开始通信。

## 3. no_sync 的边界

gradient accumulation 的前几个 microbatches 可在 `no_sync()` 内只累加 local gradients；最后一个 microbatch 必须同步后再 step。若一直 no_sync，各 rank 参数就会分叉。它改变的是“何时 collective”，不是消除同步语义。

## 实验课

两 GPU 使用 tiny model 和不同 data，记录 local gradient、all-reduce 后 gradient、step 后参数差异。加 communication hook 或 PyTorch debug log 看 bucket ready 顺序；对两个 bucket cap 做 profiler trace。报告区分“看到 overlap 候选时间段”和“证明整体 step 更快”。

## 通关

画出 `forward -> backward -> bucket ready -> all-reduce -> optimizer step`，解释 bucket 不改变数学语义。下一课 FSDP2 将进一步改变 parameters/gradients 的存储所有权。
