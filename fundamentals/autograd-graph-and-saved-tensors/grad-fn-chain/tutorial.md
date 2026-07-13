---
artifact: curriculum-tutorial
leaf: fundamentals/autograd-graph-and-saved-tensors/grad-fn-chain
priority: P0
environment: Mac
---

# 导学：把 backward 看成图上的反向消息，而不是魔法

本叶解释 `grad_fn`、leaf tensor、saved tensor 与 `.grad` 的关系。它是理解 activation memory、checkpointing 和 DDP hook 的最小 autograd 前置。
