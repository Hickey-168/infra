---
artifact: course-lesson
leaf: training/memory/activation-memory
language: zh-CN
estimated_time: 90min
---

# 课程：Autograd 为什么让中间激活活到 backward

## 1. 先分清“值算完了”与“值可以释放”

forward 中某层输出一旦传给下一层，数值计算似乎结束了；但 training 的 backward 还要用它或相关输入计算梯度。autograd graph 记录算子关系，并保存 backward 所需的 tensors。因此 Python 局部变量离开作用域不等于 GPU payload 立刻可释放。

## 2. 一个两层 MLP 的时间线

令 `h = relu(x @ W1)`，`y = h @ W2`，`loss = y.sum()`。计算 `dW2` 需要 h；计算 ReLU 和第一层梯度又需要 x/h 的某些信息。forward 完成时这些状态仍有 future consumer；backward 从 loss 反向经过第二层、激活、第一层后，它们才依次失去必要性。真实 Transformer 只是在层数、shape 和保存状态上扩大了这个故事。

## 3. 为什么 batch 和 sequence 会放大峰值

若 activation shape 是 `[B,T,H]`，payload 随 `B*T*H*dtype_bytes` 增长。参数量固定时，增大 sequence length 仍可能 OOM，原因就在这里。attention 还可能有额外的与 T 相关状态；不能只用参数量解释训练峰值。

## 4. checkpointing 的交换

activation checkpointing 不保存某一段 forward 的全部中间值，而在 backward 需要时重算该段 forward。它降低保存 payload，却增加计算时间，并可能改变 RNG/通信等实现细节。它不是免费压缩，也不会减少 weights 或 optimizer states。

## 实验课

对 tiny model 分别跑 inference_mode、普通 training、training+checkpoint。先预测三者是否建 graph、哪个 peak memory 最大、checkpoint 的 wall time 怎样变化；再用 hooks 记录 layer output shape，并在 GPU 上用 `max_memory_allocated` 核对。报告必须写“测到的是此模型此 batch 的 allocator 指标”，不是泛化定律。

## 通关

不看资料解释：为何 forward 结束后峰值仍可能上升；checkpoint 减少什么、增加什么；为什么减小 sequence length 往往能缓解 OOM。下一课转向 CUDA，学习这些 tensor 元素如何映射成并行执行。
