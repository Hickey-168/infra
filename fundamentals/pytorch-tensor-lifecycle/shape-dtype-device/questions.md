# 盲点自测 — shape dtype device

> 规则：闭卷口述作答，答完再看正解。答错的题记到「漏点」，作为下次复习入口。

## 七维度题目（AI 生成后填入）

### Q1 调用链
- 问题：一个 tensor 的 payload bytes 公式是什么？它为什么不是训练时的完整内存成本？
- 我的初答：
- 正解：`payload_bytes = numel() * element_size()`。它只计算元素数据本身，不包含 metadata、allocator 保留、输出 tensor、临时 buffer、autograd 保存值、梯度或优化器状态。
- 漏在哪里：

### Q2 shape/state/layout
- 问题：`torch.empty(2,1,5) + torch.empty(3,5)` 的输出 shape 是什么？为什么？
- 我的初答：
- 正解：输出是 `(2,3,5)`。右对齐后第二个 tensor 视作 `(1,3,5)`；对应维度相等或一边为 1 都可 broadcast。
- 漏在哪里：

### Q3 shape/state/layout
- 问题：`torch.tensor([1,2,3]).shape` 为什么是 `(3,)`？逗号是什么意思？
- 我的初答：
- 正解：它有 1 个轴，这个轴长度为 3。`(3,)` 是 Python 单元素 tuple，逗号用于区别 tuple 和整数表达式 `(3)`。
- 漏在哪里：

### Q4 反事实
- 问题：broadcasting 中 size 为 1 的轴扩展到其他长度时，是补 0、补 1，还是别的机制？
- 我的初答：
- 正解：都不是。它会沿这个轴重复读取原来的同一个元素值；通常是虚拟展开，不改变原 tensor。
- 漏在哪里：

### Q5 最小实现
- 问题：写一个最小例子说明 `requires_grad` 的作用。
- 我的初答：
- 正解：`x=torch.tensor([1.,2.,3.], requires_grad=True); loss=(x*x).sum(); loss.backward(); x.grad` 得到 `[2,4,6]`，表示 loss 对 x 的偏导数。
- 漏在哪里：

### Q6 数量级估算
- 问题：shape `(8,2048,4096)` 的 BF16 tensor payload 是多少 GiB？
- 我的初答：
- 正解：`8*2048*4096=67,108,864` 个元素；BF16 2 bytes，所以 134,217,728 bytes，约 0.125 GiB。
- 漏在哪里：

### Q7 来源与边界
- 问题：FP32 和 `torch.float32` 是什么关系？它的 `element_size()` 通常是多少？
- 我的初答：
- 正解：FP32 通常对应 PyTorch 的 `torch.float32`，也常写作 `torch.float`；每个元素 4 bytes。
- 漏在哪里：
