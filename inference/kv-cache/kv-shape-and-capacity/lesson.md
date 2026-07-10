---
artifact: course-lesson
leaf: inference/kv-cache/kv-shape-and-capacity
language: zh-CN
estimated_time: 90min
---

# 课程：从一层 K/V tensor 推到并发上限

## 1. cache 保存的到底是什么

每层 attention 对每 token 产生 K 与 V。decode 时不必重新投影历史 tokens，而直接读取其 K/V；代价是它们要一直驻留到请求结束。对总缓存 token 数 `S`，近似 payload：

```text
2 * num_layers * S * num_kv_heads * head_dim * dtype_bytes
```

这里的 S 是所有活跃 sequences 的长度和，不是单条 context length。GQA/MQA 的 `num_kv_heads` 小于 query heads，因此会显著改变容量。

## 2. 从单请求到服务容量

若每条请求 prompt 为 P、预计生成 G、并发为 C，最粗估 `S≈C*(P+G)`；真实 serving 的每条长度不同且随时间变化，但这个公式足以先判断数量级。可用显存还要扣 weights、activations、workspace、allocator/page metadata，不能把整卡显存全给 KV。

## 3. 常见误解

权重量化降低 weights payload，不会自动降低 KV payload；只有 KV cache 自己使用更低 dtype/quantization 才会改变公式中的 dtype_bytes。paged KV cache 主要解决动态分配和碎片，不改变每个有效 token 的基本 K/V 数据量。

## 实验课

写 calculator，输入 model config、C/P/G、dtype，输出 bytes/GiB；先用一个小模型真实 K/V shape 对照。分别扫 context、concurrency、MHA/GQA、BF16/FP8，预测哪一项线性增长。报告明确忽略了哪些 metadata，避免把估算写成精确显存。

## 通关

能够解释 total cached tokens 与 batch size 的区别，能估算一个 case 的 KV GiB。下一课将这份容量预算交给 scheduler，学习连续批处理如何做 admission。
