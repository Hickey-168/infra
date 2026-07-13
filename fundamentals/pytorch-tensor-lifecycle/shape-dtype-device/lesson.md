---
artifact: course-lesson
leaf: fundamentals/pytorch-tensor-lifecycle/shape-dtype-device
language: zh-CN
estimated_time: 90-120min
environment: Mac
---

# 课程：读懂一个 PyTorch Tensor 的 shape、dtype 与 device

> 先读 `tutorial.md` 了解本课位置。本文是课程正文：按顺序阅读，停下来完成纸面练习，最后再进实验。

## 完成本课后能做什么

面对任意 tensor，你应能在运行前回答四件事：它有多少元素和 payload bytes；某个算子是否 shape 合法；报错属于 shape、dtype 还是 device；一次 `.to(...)` 是 no-op 还是会产生 copy。它们是后续 attention、KV cache、DDP 和 CUDA 的共同坐标系。

## 1. 一个 tensor 是三份事实

先不要把 tensor 看作“很多数字”，而把它看作：

```text
Tensor = (logical coordinates, element representation, storage location)
       = (shape/stride, dtype, device)
```

`shape` 描述逻辑坐标轴。例如语言模型 hidden states 常为 `[B, T, H]`：B 是 batch 内 sequence 数，T 是 token 数，H 是 hidden dimension。`shape=(2,3,4)` 的元素数是 `2*3*4=24`，但这还不知道每格多少 bytes、也不知道数据在哪。

注意中文里“维度”容易混用。更精确地说，`len(shape)` 是轴的数量，也叫 rank/ndim；`shape` 里的每个数字是对应轴的长度。例如 `shape=(3,)` 表示 1 个轴，这个轴长度为 3；`shape=(2,4)` 表示 2 个轴，第一轴长度 2、第二轴长度 4。

标量 tensor 的 `shape=()`，意思是没有坐标轴，但仍有一个值，所以 `numel()=1`。一维 tensor `torch.tensor([1,2,3])` 的 `shape=(3,)`，括号里的 `3` 表示这个唯一轴上有 3 个位置；逗号是 Python 里“单元素 tuple”的写法，用来区别 `(3,)` 这个 tuple 和 `(3)` 这个整数表达式。

二维 tensor 需要每一行长度相同，例如 `torch.tensor([[1,2,3,4], [5,6,7,8]])` 的 `shape=(2,4)`：2 行、4 列。`torch.tensor([[1,2], [1,2,3,4]])` 这种行长不齐，不能直接形成普通 dense matrix。

`dtype` 描述每个格子的表示。常用 `float32` 是 4 bytes，`float16` 与 `bfloat16` 是 2 bytes，`int64` 是 8 bytes。dtype 同时改变数值范围、舍入和硬件算子路径，不能把“降精度”理解成只省显存。

`device` 描述 storage 由谁拥有：CPU 主内存、`cuda:0` 的 NVIDIA GPU、或 Mac 上的 MPS。一个算子通常要求参与运算的 tensors 在兼容 device 上；PyTorch 不会为了让 `cpu_tensor + cuda_tensor` 看似方便而悄悄做一份隐式迁移。

### 练习 1：先手算 payload

```text
x = torch.empty((8, 2048, 4096), dtype=torch.bfloat16)
numel(x) = ?
payload bytes(x) = ?
payload GiB(x) ≈ ?
```

先算 `8 * 2048 * 4096 = 67,108,864` 个元素；BF16 每元素 2 bytes，所以 payload 是 134,217,728 bytes，约 0.125 GiB。这里不包含 allocator 保留、storage padding、Python 对象头、tensor metadata、梯度 tensor、优化器状态或 autograd 保存的其他 tensors。

`payload` 可以理解成“张量实际元素数据的字节数下界”，不是一次训练或一次算子的完整成本。完整成本还要看是否需要输出 tensor、临时 buffer、反向图保存、梯度、通信和 device 迁移。

## 2. shape 是算子的输入契约

shape 不只是打印信息。它约束了一个算子如何把输入坐标映射为输出坐标。

### 逐元素与 broadcasting

逐元素加法先从最后一维对齐比较：对应维度必须相等，或一边为 1，或其中一边不存在。

```text
(2, 3, 4) + (4,)    -> (2, 3, 4)   # (4,) 视作 (1, 1, 4)
(2, 3, 4) + (3, 1)  -> (2, 3, 4)   # 视作 (1, 3, 1)
(2, 3, 4) + (3,)    -> 不合法       # 最后维 4 对 3
```

广播不等于一定复制数据，但广播后的 output shape 决定需要做多少 elementwise work。先把“合法性”和“是否实际 materialize”分开，后者留给 stride/layout 课程。

把缺失的左侧维度补成 `1` 只是广播规则里的对齐方式，不表示原 tensor 永久变成了更高维 tensor。例如 `(2,4)` 和 `(3,2,4)` 运算时，可以把 `(2,4)` 概念上看作 `(1,2,4)` 来对齐，输出是 `(3,2,4)`；但原来的 `(2,4)` tensor 自身还是二维。

size 为 `1` 的轴可以 broadcast 到别的长度，是因为这个轴上只有一个可选坐标，计算时沿新长度重复读取同一个值。它不是补 `1`，也不是补 `0`。

```python
x = torch.tensor([[10, 20, 30]])  # shape (1, 3)
y = torch.tensor([[1], [2]])      # shape (2, 1)
x + y
```

输出 shape 是 `(2,3)`，值是：

```text
[[11, 21, 31],
 [12, 22, 32]]
```

这里 `x` 沿第一轴重复，`y` 沿第二轴重复。重复的是原来的元素值，不是补某个常数。

### 矩阵乘法

`matmul` 的最后两维是矩阵维度。例如线性层：

```text
X: [B, T, H]  @  W: [H, O]  ->  Y: [B, T, O]
```

只有内维 H 相同才合法。`W: [O,H]` 即使元素数相同也不是正确 layout，通常需转置。后面的 Triton matmul 会把这条契约展开为 M/N/K 的 pointer formula。

### 练习 2：不运行先写结果

```python
a = torch.empty(2, 1, 5)
b = torch.empty(3, 5)
c = torch.empty(2, 5, 7)
d = torch.empty(2, 6, 7)
```

写出 `a+b`、`c @ torch.empty(7,11)`、`c+d` 的 output shape 或错误原因。先从右向左处理加法；matmul 则先找相邻内维。

答案：

```text
a + b:
  (2, 1, 5)
+    (3, 5)   # 右对齐后视作 (1, 3, 5)
-> (2, 3, 5)

c @ torch.empty(7, 11):
  (2, 5, 7) @ (7, 11)
-> (2, 5, 11)

c + d:
  (2, 5, 7)
+ (2, 6, 7)
-> 不合法，因为倒数第二维 5 和 6 既不相等，也没有任何一边是 1
```

注意 `a+b` 不会报错：第二维是 `1` 对 `3`，`1` 是 broadcast 的通配尺寸，可以扩展到 `3`。`c+d` 报错也不是因为 `5` 比 `6` 小，而是因为 elementwise broadcasting 的规则只认“相等 / 一边为 1 / 缺失维度”，不认大小关系。

## 3. dtype 是成本模型的一部分

每个 tensor payload 的最低估算都是：

```text
payload_bytes = numel * element_size
```

`numel()` 是元素个数，只由 shape 决定；`element_size()` 是每个元素占多少 bytes，只由 dtype 决定。例如 shape `(3,)` 的 FP32 tensor 有 `numel=3`、`element_size=4`，所以 payload 是 12 bytes。标量 tensor 的 shape 是 `()`，但仍有 `numel=1`。

因此 BF16 tensor 的 payload 通常是同 shape FP32 的一半，collective 需要传输的 payload 也随之变化。FP32 通常对应 PyTorch 里的 `torch.float32`，也常写作 `torch.float`；它的 `element_size()` 是 4 bytes。BF16 对应 `torch.bfloat16`，通常是 2 bytes。

cast 不是就地改标签：目标 dtype 不同时通常会生成新的表示；浮点 tensor 转整数还会失去可微语义。遇到 OOM 或通信量异常，先打印 `numel()`、`dtype`、`element_size()`，再猜模型配置。

PyTorch 还会对混合 dtype 运算做类型提升。例如一个 FP32 tensor 和一个 BF16 tensor 相加，结果通常提升为 FP32；所以要观察结果 dtype，而不要只看输入之一。

## 4. device 是数据位置与迁移边界

`x.to(device='cuda')` 的语义是取得目标 device 上的 tensor。若目标 dtype/device 已经相同、也没有强制 copy，PyTorch 可以返回原 tensor；否则它返回转换后的 tensor。

```python
x_cpu = torch.ones(4, device='cpu')
x_same = x_cpu.to('cpu')          # 通常可复用 x_cpu
x_fp16 = x_cpu.to(torch.float16)  # 新的元素表示
x_gpu = x_cpu.to('cuda')          # CUDA 环境中的 host-to-device 迁移
```

device mismatch 常来自临时 tensor：模型已在 GPU，但 `torch.arange(...)` 或 mask 未指定 device，默认留在 CPU。修复不是到处调用 `.cuda()`，而是让相关 tensors 从同一来源取得 device，例如 `device = next(model.parameters()).device`。

## 5. requires_grad 是 autograd 开关

`requires_grad` 表示 PyTorch autograd 是否需要追踪这个 tensor 参与的浮点运算，以便之后对它求梯度。这里的 `grad` 是 gradient，意思是“某个标量目标对这个 tensor 的偏导数”。

最小例子：

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
loss = (x * x).sum()
loss.backward()
print(x.grad)  # tensor([2., 4., 6.])
```

`requires_grad=False` 时，PyTorch 把它当作普通数值参与计算，不为它保存反向传播所需的历史。它不是 shape、dtype、device 的一部分，但会影响训练时的额外内存成本：为了反向传播，autograd 可能保存中间结果，并在参数上累积 `.grad` tensor。

## 6. 固定调试顺序

每次 tensor bug 都先打印：

```python
def describe(name, x):
    print({
        'name': name,
        'shape': tuple(x.shape),
        'dtype': str(x.dtype),
        'device': str(x.device),
        'numel': x.numel(),
        'element_size': x.element_size(),
        'payload_bytes': x.numel() * x.element_size(),
        'requires_grad': x.requires_grad,
    })
```

先检查 shape contract，再检查 dtype/精度预期，最后检查 device；不要一上来猜 CUDA 或盲加 `.contiguous()`。

## 实验课

1. 在 CPU 上建立同 shape 的 FP32、BF16、INT64 tensors，核对纸面 bytes。
2. 对三组 broadcasting 例子先写输出 shape，再运行；保留错误 case 的异常。
3. 对一个小线性层打印 input、weight、output，验证 `[B,T,H] @ [H,O] -> [B,T,O]`。
4. 比较 `x.to('cpu') is x` 与 `x.to(torch.float16) is x`。
5. 在 MPS 或 CUDA 环境故意让 model/input 异 device，读懂异常后用统一 device 来源修复。

在 `lab.*` 中每一步先写预测。实验后在 `note.md` 不看本文写出：shape、dtype、device 各回答什么问题，为什么不能相互替代。

## 通关与核对

通关条件：你能计算 `[4,1024,4096]` BF16 payload；解释 `(2,3,4)+(3,)` 为什么失败；解释一维向量 `x @ y` 为什么输出 shape 是 `()`；解释临时 index tensor 为什么常造成 device mismatch；说明 `.to()` 何时可能不分配新 storage。

本文已包含完成本课所需内容；下列只用于你遇到版本行为差异时核对：

- [PyTorch Tensor Attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html)
- [PyTorch Tensor documentation](https://docs.pytorch.org/docs/stable/tensors.html)
- [Tensor.to contract](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html)
