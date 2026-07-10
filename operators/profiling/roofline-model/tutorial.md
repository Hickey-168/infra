---
artifact: curriculum-tutorial
leaf: operators/profiling/roofline-model
priority: P0
environment: WSL2-GPU
---

# Roofline：用 FLOPs/bytes 决定先优化什么

## 问题

看到一个 kernel 慢时，人很容易直接改 tile、warp 或代码结构。Roofline 先问一个更基本的问题：即使实现完美，它受硬件算力上限还是内存带宽上限限制？若方向判断错，后续微调只是在错误目标上用力。

## 核心公式

定义 arithmetic intensity `I = FLOPs / bytes`。对给定硬件峰值算力 `P_peak` 与峰值带宽 `BW_peak`，性能上界为：

```text
P <= min(P_peak, I * BW_peak)
```

交点 `P_peak / BW_peak` 是 ridge point。I 低于它的 workload 倾向 bandwidth-bound；高于它的 workload 才可能 compute-bound。这里的 bytes 是你选择的 memory-traffic 模型，必须写清是有效 payload、DRAM bytes 还是 profiler 统计。

## 用两个对照建立直觉

- FP32 vector add：每元素约 1 FLOP、读 x/y 加写 z 共约 12 bytes，`I≈1/12`，明显低。
- 大型 GEMM：FLOPs 随 M*N*K 增长，tile reuse 让 bytes 增长较慢，I 通常高得多。

这不是保证 vector add 一定带宽满、GEMM 一定算力满；launch overhead、shape、cache、占用率和 library 实现仍会让实测低于上界。

## 预测后验证

对同一 GPU 选 copy/vector-add 和 matmul：先写 FLOPs、有效 bytes、I、理论 ceiling；再用统一 warmup/同步/median 测量得到 GB/s 或 TFLOP/s。把“实测/理论”的比值写出来，逐项列可能原因，而不是只说“没跑满”。CPU/Mac 只能练计算，不可声称 GPU roofline 结论。

## 下一跳

`prefill-vs-decode` 会把 roofline 思维带进 LLM serving：同一模型在两种阶段拥有不同 shape、cache traffic 和瓶颈。
