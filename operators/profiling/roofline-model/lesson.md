---
artifact: course-lesson
leaf: operators/profiling/roofline-model
language: zh-CN
estimated_time: 75min
environment: WSL2-GPU
---

# 课程：Roofline 不是图，而是一次上界推理

## 1. 两个硬件天花板

kernel 的时间至少受两种资源限制：执行 FLOPs 的峰值算力 `P_peak`，以及搬运 bytes 的峰值带宽 `BW_peak`。先计算 arithmetic intensity `I=FLOPs/bytes`，得到：

```text
performance <= min(P_peak, I * BW_peak)
```

当 `I*BW_peak` 较低，优先减少 bytes、提高复用或改善访问；当 `P_peak` 较低，才优先考虑 tensor cores、计算管线和指令效率。

## 2. 一次完整推导

FP32 vector add：N 次加法、约 `3*4*N` bytes，`I≈1/12`。GEMM `M*N*K` 的 FLOPs 约为 `2MNK`，而良好 tiled 实现会复用 A/B，I 随问题规模增大。不要省略“bytes 采用什么模型”：有效 payload、DRAM traffic、cache hit 后的实际流量不是同一个量。

## 3. 实测低于上界并不神秘

上界假设资源被理想使用。小 shape 有 launch overhead，坏 layout 有额外 traffic，寄存器压力会限制 occupancy，library/kernel 还可能未选择最优算法。roofline 的作用是排除不可能的优化方向，不是给出自动修复。

## 实验课

为 vector-add 和 matmul 各写一张表：FLOPs、bytes、I、理论 ceiling、实测 GB/s/TFLOP/s、实测/上界比。先预测各自的 bounding resource，再测。只在同一 GPU 上比较；CPU 练习只能验证算术，不能测 GPU ceiling。

## 通关

带单位推导一次 `I*BW`，并说出一个实测低于上界但并非 bug 的原因。下一课将这个思路用于 prefill 与 decode 的不同 workload。
