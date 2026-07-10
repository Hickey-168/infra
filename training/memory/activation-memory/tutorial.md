---
artifact: curriculum-tutorial
leaf: training/memory/activation-memory
priority: P0
environment: Mac-or-WSL2-GPU
---

# 激活显存：理解 backward 为什么让 forward 的中间结果活得更久

## 问题

训练时 OOM 常发生在 forward 看似已经结束之后。原因不是“GPU 忘了释放”，而是 autograd 必须保留某些 forward 中间值来计算梯度；这些 tensors 的生命周期跨越 forward 与 backward。

## 因果链

1. forward 每层产出 activation，下一层消费它；若只做 inference，许多中间值可很快释放。
2. training 的 backward 需要特定输入、输出或统计量来求局部梯度，autograd 将它们作为 saved tensors 关联到 graph。
3. 多层、多 token、多 batch 的 saved tensors 在 backward 到达前同时存活，形成 peak memory；峰值往往发生在 forward 尾端或 backward 初期。
4. activation checkpointing 用重算换取少保存：checkpoint 段在 backward 时重跑 forward，因此降低 payload、增加计算与可能的重算调度成本。

## 观察重点

- 必须区分 parameter payload 与 activation payload；前者通常与 batch 无关，后者通常随 batch/sequence 增长。
- `requires_grad=False`、`torch.no_grad()`、`torch.inference_mode()` 改变 graph 是否建立，不能只看最终数值相同。
- “已释放”应由 tensor lifetime 或 allocator 统计验证，不能仅靠 Python 变量名消失判断。

## 预测后验证

对 tiny MLP/Transformer 做三组：`eval()+inference_mode`、普通训练、训练+checkpoint。先预测三者是否建立 grad graph、peak memory 的相对关系、wall time 的相对关系；再用 hooks 记录每层输出 shape，WSL2 上用 `torch.cuda.max_memory_allocated()` 或 profiler 测量。实验报告必须注明 checkpoint 只减少了哪些保存状态，未证明大模型上的吞吐收益。

## 下一跳

你现在应能用“状态何时被谁消费”而非“训练就是更耗显存”来解释问题。下一叶切到 `operators/cuda/execution-model/thread-block-grid`，把 tensor 元素映射到 GPU 的执行单元。
