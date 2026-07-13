---
artifact: curriculum-tutorial
leaf: inference/lifecycle/transformer-forward-shapes
priority: P0
environment: Mac
---

# 导学：Transformer 前向到底在做什么

在讨论显存、KV cache、Triton matmul 前，必须先能把一批 token 如何变成 logits 说清楚。本叶不要求你从零实现工业 Transformer；它只建立最小的因果链：token ids -> embedding -> 多层 block -> logits -> loss/采样。

学习后你应能追踪 `[B,T]` 如何成为 `[B,T,V]`，知道 attention/MLP 的输入输出在哪些维度变化，并能区分“训练时算 logits 后接 loss”和“推理时只取最后位置 logits”。下一叶 `training-loop-state-machine` 会把这个 forward 放进完整训练循环。
