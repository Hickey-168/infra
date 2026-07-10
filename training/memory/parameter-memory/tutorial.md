---
artifact: curriculum-tutorial
leaf: training/memory/parameter-memory
priority: P0
environment: Mac
---

# 参数显存：先把“模型大小”拆成可核对的字节账本

## 问题

“7B 模型需要多少显存”没有单一答案。参数数量只给出权重 payload；推理与训练是否可行还取决于 dtype、梯度、optimizer state、master weight 和激活。先把每一项分开，后面才看得懂 ZeRO/FSDP 为什么有效。

## 因果链

1. 每个 parameter 是一个 tensor，payload 为 `numel * element_size`。
2. 推理最基本常驻项是权重；训练中 backward 要产生 gradient，optimizer step 还要读写 optimizer states。
3. 混合精度训练可能同时有低精度模型副本和更高精度的更新副本；具体是否存在由 recipe 决定，不能套死“每参数固定 N bytes”。
4. 因此内存表必须按 state、dtype、是否每 rank 完整复制分别列项；FSDP 的本质正是改变这些项的所有权。

## 账本模板

| 项 | 元素数 | dtype | 每元素 bytes | 常驻时机 | 每 rank 是否完整持有 |
|---|---:|---|---:|---|---|
| parameters | | | | forward/inference | |
| gradients | | | | backward 后 | |
| optimizer states | | | | training | |
| master parameters（若有） | | | | update | |

不要把 activation 填进这张表，它有另一条生命周期，下一叶单独处理。

## 预测后验证

用一个 tiny Transformer 遍历 `model.named_parameters()`，输出 name、shape、numel、dtype、bytes；先手算总和，再同 `sum(p.numel()*p.element_size())` 比较。把模型转为 BF16 后重复一次，预测 payload 是否严格减半。若使用 Adam，检查 optimizer 第一次 `step()` 前后 `optimizer.state` 的 entries，记录实际创建了哪些 tensors，不要凭记忆填表。

## 边界与下一跳

这张表解释的是 tensor payload，不能代表 `nvidia-smi`、caching allocator reserved memory 或 CUDA context 的总占用。下一叶 `activation-memory` 会解释为什么同一个参数账本在不同 batch/sequence 下仍可能 OOM。
