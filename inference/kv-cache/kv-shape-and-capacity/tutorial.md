---
artifact: curriculum-tutorial
leaf: inference/kv-cache/kv-shape-and-capacity
priority: P0
environment: Mac-or-WSL2-GPU
---

# KV Cache 容量：把上下文、并发和显存连成一条公式

## 问题

decode 若每步重新计算历史 token 的 key/value，计算会重复；KV cache 消除重复计算，却把 token 历史变成常驻显存。服务能接多少并发，不是只看权重大小，而是看每个活跃 token 的 K/V 字节成本。

## 形状与公式

对每层、每个 token，通常存一份 K 和一份 V。忽略 allocator/page metadata 时：

```text
KV bytes = 2 * num_layers * total_cached_tokens
           * num_kv_heads * head_dim * dtype_bytes
```

`total_cached_tokens` 是所有并发 sequence 当前长度之和，不是只有单请求 context length。GQA/MQA 改变 `num_kv_heads`；权重量化不自动缩小 KV cache；KV quantization 才会改变 `dtype_bytes`。

## 因果链

1. prefill 为 prompt token 写入 cache。
2. 每次 decode 读取历史 K/V 并追加一个新 K/V，故容量随活跃 token 增长。
3. 当可用 GPU memory 减去 weights/activations/workspace 后不足以容纳更多 blocks，scheduler 必须拒绝、抢占、swap 或降低并发。
4. Paged KV cache 后续会改变分配方式和碎片特性，但不会消除每 token 的基本 payload。

## 预测后验证

用一个模型 config 写 calculator：扫描 context length、concurrency、BF16/FP8、MHA/GQA；先手算一个 case，再输出 GiB。用小模型真实分配 cache 或打印 K/V shape 对照公式。报告应明确包括哪些层、是否包含 batch、是否忽略 page table；它不证明完整 serving 的峰值显存。

## 下一跳

`continuous-batching` 使用这份容量账本决定每一轮能接纳哪些请求，并把显存约束转化为调度策略。
