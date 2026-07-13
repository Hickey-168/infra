---
artifact: discussion-review
leaf: training/single-gpu-loop/training-loop-state-machine
status: verified-conceptual-model
---

# 复习沉淀：训练闭环是 parameter-memory 的必经前置

## 最终结论

不能从 `shape/dtype/device` 直接进入 `parameter-memory`。参数、梯度、optimizer state、activation 分别是什么、何时出现、是否常驻，只有在理解一次训练 step 后才有意义。正确前置顺序是：Transformer forward -> 训练状态机 -> forward/backward -> autograd 图 -> optimizer step -> 超参/学习率调度 -> parameter-memory。

## 一条完整 step

```text
batch input/labels
-> model.train() 下的 forward: input -> logits
-> loss(logits, labels): token 级损失约简成 scalar
-> loss.backward(): 沿 autograd graph 计算并累加 parameter.grad
-> optional unscale / clip / accumulation decision
-> optimizer.step(): 用 grad、lr、weight decay、optimizer state 修改 parameters
-> scheduler.step(): 按 update 计数调整未来 lr（具体调用顺序依 scheduler contract）
-> optimizer.zero_grad(): 清空或置 None，避免下一轮非预期累加
```

**forward 不等于 training。** forward 只是 input 到 output 的计算；training 是 forward + loss + backward + update 的循环。**backward 不等于 update。** backward 填充 `.grad`，`optimizer.step()` 才改变 parameter。**inference 不等于普通 forward。** inference 通常在 `eval()` 与 `inference_mode()` 下运行，不建立 backward 所需 graph，也不会 update parameters；`eval()` 同时影响 dropout/batch norm 等模块行为。

## 状态所有权与显存对应

| 状态 | 何时出现 | 谁拥有 | 为什么后续要计显存 |
|---|---|---|---|
| parameters | model 初始化后 | model | inference/training 的长期权重 payload |
| logits / activations | forward | autograd / 临时 tensors | backward 前可能必须保存，随 B/T/H 增长 |
| gradients | backward | parameter `.grad` | update 前累加并在训练中占空间 |
| optimizer states | 首次 step 后常创建 | optimizer | Adam 类状态是训练显存的重要项 |
| scheduler state | scheduler step 后 | scheduler | 恢复训练必须恢复 lr 时间线 |

## 超参不是孤立配置

- `micro_batch_size` 改单次 activation shape；`gradient_accumulation` 改多少 microbatches 共用一次 update。
- `global_batch ≈ micro_batch * accumulation * world_size`；改变它通常改变优化统计，不能只看显存。
- `sequence_length` 改 token 维 T，影响 attention、activation 与训练成本。
- `learning_rate` 与 `weight_decay` 改 update；warmup/decay 使 lr 随 optimizer update count 变化。

## 已排除的误解

- “要学显存必须先从零实现完整 Transformer”：不需要。先用 scalar parameter、tiny classifier 或 tiny LM 验证同一状态机；Transformer 前向课只要求你读懂 ids -> logits 的 shape 流。
- “loss.backward() 后模型已经更新”：错误，参数此时不变。
- “训练/推理的差别只是有没有 labels”：错误，grad graph、module mode、parameter update 与 cache 的消费方式也不同。
- “超参是跑不动时再调的数字”：错误，它们直接改变状态机的统计、更新与资源边界。

## 复习检查

1. 不看资料画出 step，并标记第一次产生 logits、grad、optimizer state、parameter mutation 的位置。
2. 解释为何忘记 `zero_grad` 会改变下一轮更新。
3. 对一个 `[B,T,V]` logits，说出训练 loss 如何得到 scalar。
4. 说明 `micro_batch=2, accumulation=4, world_size=8` 的 global batch，并说出它不保证什么。

## 证据边界

这里记录的是 PyTorch 常见训练契约和机制模型；具体 scheduler 调用顺序、optimizer state dtype、mixed precision/unscale 行为应以所用版本和 recipe 的官方文档/实际 trace 核对。尚未做 GPU 性能结论。
