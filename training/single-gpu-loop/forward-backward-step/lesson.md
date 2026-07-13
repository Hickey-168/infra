---
artifact: course-lesson
leaf: training/single-gpu-loop/forward-backward-step
language: zh-CN
estimated_time: 75min
environment: Mac
---

# 课程：从 scalar loss 到 parameter.grad

设 `w` 是可训练 parameter，`loss=(x*w-y)^2`。forward 计算中间值和 scalar loss；当 `loss.backward()` 被调用，autograd 从 loss 反向应用链式法则，最终将导数累加到 `w.grad`。此时 w 的值完全不变。只有 `optimizer.step()` 根据 grad 和 optimizer state 改 w。

loss 必须是可反传到 parameters 的标量或显式提供梯度的 tensor。语言模型通常先得到 `[B,T,V]` logits，再将 token-level cross entropy 约简为 scalar；这解释了为什么训练循环监控一个 loss 数，但模型输出不是一个数。

PyTorch 默认累加 gradients：连续两次 backward 而不清零，`w.grad` 是两次导数之和。这是 gradient accumulation 的基础，也是忘记 `zero_grad` 的典型 bug。

**实验课：** 用 scalar 参数显式手算导数，与 `w.grad` 对比；连续 backward 两次观察累加；记录 forward 前后 parameter 相等、backward 后仍相等、step 后才不等。再在 tiny classifier 上打印 logits shape 与 scalar loss。通关时能准确回答“哪一行第一次写 grad、哪一行第一次改 parameter”。
