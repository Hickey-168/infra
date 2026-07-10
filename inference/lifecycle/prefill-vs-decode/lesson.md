---
artifact: course-lesson
leaf: inference/lifecycle/prefill-vs-decode
language: zh-CN
estimated_time: 90min
---

# 课程：把 LLM generate 拆成两条不同的资源路径

## 1. prefill 做了什么

用户提交 T 个 prompt tokens 后，prefill 一次处理它们，计算各层 Q/K/V，并把 K/V 写入 cache。Q 的序列维度是 T，attention 与 GEMM 的矩阵形状较大；用户从请求发出到首 token 的等待中，prefill 通常占重要部分，这就是 TTFT 的来源之一。

## 2. decode 为什么不像缩小版 prefill

第 t 次 decode 只产生一个新 query，Q length 为 1；但 attention 仍要读取历史全部 K/V，长度约为 T+t-1。每步计算量小，却携带不断增长的 memory read 和同步/调度成本；用户感受到的是 token-to-token 间隔 TPOT。

## 3. 读 shape 而非背标签

每层记录 `Q:[B,heads,1,D]` 与 `K/V:[B,kv_heads,L,D]`（精确 layout 随实现变）。关键是 L 随 generation 增长、Q 在 decode 固定为 1。由此你才能解释为什么 KV cache、continuous batching 和 paged attention 是 serving 的核心。

## 实验课

在 tiny causal LM 上分别 trace 一次 prompt forward 和连续 decode 16 步：记录 Q/K/V shapes、cache length、TTFT、每步时间。先预测哪项单调增长、哪项恒定。Mac 可做 shape trace，GPU 才可讨论时延瓶颈。

## 通关

能画出 prefill 与第 t 步 decode 的 Q/K/V 长度；能解释为什么“总 tokens/s”不足以说明用户体验。下一课把 K/V 的形状变成显存容量公式。
