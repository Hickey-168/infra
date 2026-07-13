---
artifact: course-lesson
leaf: fundamentals/autograd-graph-and-saved-tensors/grad-fn-chain
language: zh-CN
estimated_time: 90min
environment: Mac
---

# 课程：沿 grad_fn 链追踪一次反向传播

对 `x`（leaf，`requires_grad=True`）执行 `a=x*2; b=a.relu(); loss=b.sum()`，a/b/loss 是由运算得到的非 leaf tensors，通常各自带 `grad_fn`，指向反向所需的 Function 节点。`loss.backward()` 从 loss 节点反向走，按局部导数把梯度传播到 x；默认只有 leaf parameter 的 `.grad` 被保留。

某些 backward 需要 forward 的输入、输出或掩码，因此 autograd 保存相关 state；这正是 activation memory 的根源。不要从 `grad_fn` 名字推断所有实现细节，但要能追“哪一个运算产生此 tensor、backward 需要什么、grad 最终写到哪个 leaf”。

`detach()` 截断 graph；`no_grad`/`inference_mode` 不记录后续运算；in-place modification 若破坏 saved tensor 的版本检查会报错。它们都是图与状态生命周期的操作，不是单纯性能开关。

**实验课：** 构造上述链并打印 `is_leaf/requires_grad/grad_fn`；运行 backward 后检查 x.grad，尝试读取非 leaf grad 并理解为何默认为空；再插入 detach 验证梯度在哪断开。通关时能从 loss 讲到 parameter.grad，而不是只说“PyTorch 自动求导”。
