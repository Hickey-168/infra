# 复习卡片 — shape、dtype、device

## 一句话

一个 tensor 的基础状态先看 `(shape, dtype, device)`；成本下界看 `numel() * element_size()`；训练额外成本再看 `requires_grad` 和 autograd 保存了什么。

## shape tuple

`len(shape)` 是轴的数量，也叫 rank/ndim；`shape` 里的每个数字是对应轴的长度。标量 `shape=()`，没有轴但有一个值。向量 `torch.tensor([1,2,3])` 的 `shape=(3,)`，表示 1 个轴、长度为 3；逗号只是 Python 单元素 tuple 的语法。

矩阵 `torch.tensor([[1,2,3,4], [5,6,7,8]])` 的 `shape=(2,4)`，表示 2 行、4 列。行长不一致的嵌套列表不能直接形成普通 dense matrix。

## payload、numel、element_size

`numel()` 是元素个数，只由 shape 决定。`element_size()` 是每个元素的字节数，只由 dtype 决定。`payload_bytes = numel * element_size` 是元素数据本身的字节数下界，不包括 tensor metadata、allocator 保留、梯度、优化器状态、输出 tensor 或临时 buffer。

例子：shape `(3,)` 的 FP32 tensor 有 3 个元素，每个 4 bytes，所以 payload 是 12 bytes。shape `()` 是标量 tensor，但 `numel()` 仍是 1。

## 练习 2 标准答案

```text
a = torch.empty(2, 1, 5)
b = torch.empty(3, 5)
a + b -> (2, 3, 5)
```

`b` 右对齐后视作 `(1,3,5)`；`1` 可以 broadcast 到 `2`，`1` 也可以 broadcast 到 `3`。

```text
c = torch.empty(2, 5, 7)
c @ torch.empty(7, 11) -> (2, 5, 11)
```

matmul 匹配最后一维/倒数第二维的内维 `7`。

```text
c + torch.empty(2, 6, 7) -> error
```

倒数第二维 `5` 和 `6` 既不相等，也没有任何一边是 `1`。错误原因不是“5 比 6 小”，而是不满足 broadcasting 规则。

## broadcasting 的 size=1

广播时会从右向左对齐维度。缺失的左侧维度可以概念上补成 `1`，但这不改变原 tensor 自身的 rank，只是本次运算的对齐视角。

size 为 `1` 的轴可以扩展到其他长度，因为运算会沿这个轴重复读取同一个值。它不是补 `1`，也不是补 `0`。

```text
shape (1,3): [[10,20,30]]
shape (2,1): [[1],
              [2]]

相加 -> shape (2,3):
[[11,21,31],
 [12,22,32]]
```

## requires_grad

`requires_grad=True` 表示 autograd 要追踪这个 tensor 参与的浮点运算，使后续 `loss.backward()` 能把 `loss` 对它的偏导数累积到 `.grad`。`requires_grad=False` 表示不为它构建这部分反向图。

这个标志不改变 tensor 的 shape/dtype/device，但会改变训练内存：autograd 可能保存中间值，参数还会有 `.grad` tensor。

## dtype 边界

FP32 通常就是 PyTorch 的 `torch.float32`，也常写作 `torch.float`，每个元素 4 bytes。BF16 是 `torch.bfloat16`，通常每个元素 2 bytes。

混合 dtype 运算可能发生类型提升。例如 FP32 和 BF16 相加，结果通常是 FP32。每次实验都要打印输出 dtype。

## notebook 观察

`nx @ z` 中如果 `nx` 和 `z` 都是一维向量，`@` 是点积，输出是标量 tensor，shape 为 `()`，`numel()` 为 1。
