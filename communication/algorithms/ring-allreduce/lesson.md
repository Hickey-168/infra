---
artifact: course-lesson
leaf: communication/algorithms/ring-allreduce
language: zh-CN
estimated_time: 90min
environment: Mac
---

# 课程：Ring All-Reduce 的两圈数据流与 alpha-beta 成本

## 1. 不要把“环”想成一个 rank 轮流传完整 tensor

把 S-byte tensor 切成 p 个 chunks。第一圈 reduce-scatter 经 `p-1` 轮邻居收发，让每个 rank 最终拥有一个已聚合 chunk；第二圈 all-gather 再经 `p-1` 轮传播这些 chunks，使每个 rank 恢复完整 tensor。pipeline 允许不同 chunks 同时在环上流动。

## 2. 成本推导

每轮每 rank 发送约 S/p bytes，共两圈各 p-1 轮，因此每 rank payload 约 `2*(p-1)/p*S`；若每次通信有启动延迟 alpha、每 byte 成本 beta：

```text
T ≈ 2*(p-1)*alpha + 2*(p-1)/p*S*beta
```

小消息时 alpha 项明显，p 增大可能更慢；大消息时带宽项占主导，ring 的每 rank payload接近 2S。这是模型而非完整 NCCL 性能预测，拓扑和协议会改变常数。

## 实验课

先写四 rank、四 chunks 的 Python simulator。每轮记录谁向左/右发送哪个 chunk、收到后如何累加；最终与直接逐元素求和比对。再扫 S 与 p，计算模型时间，先预测小/大消息的趋势。不要把 simulator 的时间当成网络 benchmark。

## 通关

不看图画出 reduce-scatter 和 all-gather 的目的，推导每 rank bytes/steps。下一课把 all-reduce 放到 DDP autograd bucket 的真实时间线。
