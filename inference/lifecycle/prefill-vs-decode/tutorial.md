---
artifact: curriculum-tutorial
leaf: inference/lifecycle/prefill-vs-decode
priority: P0
environment: Mac-or-WSL2-GPU
---

# Prefill 与 Decode：一次“生成”其实是两类 workload

## 问题

把 LLM serving 当成单一 forward 会导向错误优化。用户先等首 token，再等后续 token；服务端先处理整段 prompt，再反复处理单个新 token。两阶段的张量形状和资源瓶颈不同，TTFT 与 TPOT 也必须分开看。

## 因果链

1. prefill 接收长度 T 的 prompt，一次 forward 为每层构造 T 个 query，并写入 T 个 K/V；attention 的计算和矩阵形状较大。
2. decode 第 t 步只产生一个 query，却需读取历史 `1..t-1` 的 K/V cache；每步 FLOPs 小得多，但反复读取不断增长的状态。
3. scheduler 把多个请求的 prefill/decode work 组合进 batch，因此 prefill 会影响 TTFT，decode 排队和每步执行影响 TPOT。
4. 因而“提高 token/s”必须说明是 prompt token、generation token、单请求还是聚合吞吐；否则指标不可解释。

## 追踪表

| 阶段 | Q 长度 | K/V 长度 | 主要新状态 | 用户指标 |
|---|---:|---:|---|---|
| prefill | T | T | 初始化 KV cache | TTFT 主体之一 |
| decode step t | 1 | T+t-1 | append 1 token KV | TPOT 主体之一 |

## 预测后验证

在小型 causal LM 上记录一次 prompt forward 与连续 16 次 decode 的 q/k/v shape、每步时延和 cache 长度；可以先在 Mac 只做 shape trace，再在 GPU 测时。预测：decode 的 query length 固定为 1、cache length 单调增长；若看到相反结果，先检查框架是否做了 fused/generate cache 管理。实验不能推出所有模型或 backend 的绝对 bottleneck。

## 下一跳

要量化 decode 的状态成本，进入 `kv-shape-and-capacity`，把“越来越长的 cache”写成具体 GiB 与并发上限。
