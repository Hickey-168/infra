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

`dtype` 描述每个格子的表示。常用 `float32` 是 4 bytes，`float16` 与 `bfloat16` 是 2 bytes，`int64` 是 8 bytes。dtype 同时改变数值范围、舍入和硬件算子路径，不能把“降精度”理解成只省显存。

`device` 描述 storage 由谁拥有：CPU 主内存、`cuda:0` 的 NVIDIA GPU、或 Mac 上的 MPS。一个算子通常要求参与运算的 tensors 在兼容 device 上；PyTorch 不会为了让 `cpu_tensor + cuda_tensor` 看似方便而悄悄做一份隐式迁移。

### 练习 1：先手算 payload

```text
x = torch.empty((8, 2048, 4096), dtype=torch.bfloat16)
numel(x) = ?
payload bytes(x) = ?
payload GiB(x) ≈ ?
```

先算 `8 * 2048 * 4096 = 67,108,864` 个元素；BF16 每元素 2 bytes，所以 payload 是 134,217,728 bytes，约 0.125 GiB。这里不包含 allocator 保留、autograd 保存的其他 tensors 或 Python 对象。

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

## 3. dtype 是成本模型的一部分

每个 tensor payload 的最低估算都是：

```text
payload_bytes = numel * element_size
```

因此 BF16 tensor 的 payload 通常是同 shape FP32 的一半，collective 需要传输的 payload 也随之变化。但 cast 不是就地改标签：目标 dtype 不同时通常会生成新的表示；浮点 tensor 转整数还会失去可微语义。遇到 OOM 或通信量异常，先打印 `numel()`、`dtype`、`element_size()`，再猜模型配置。

## 4. device 是数据位置与迁移边界

`x.to(device='cuda')` 的语义是取得目标 device 上的 tensor。若目标 dtype/device 已经相同、也没有强制 copy，PyTorch 可以返回原 tensor；否则它返回转换后的 tensor。

```python
x_cpu = torch.ones(4, device='cpu')
x_same = x_cpu.to('cpu')          # 通常可复用 x_cpu
x_fp16 = x_cpu.to(torch.float16)  # 新的元素表示
x_gpu = x_cpu.to('cuda')          # CUDA 环境中的 host-to-device 迁移
```

device mismatch 常来自临时 tensor：模型已在 GPU，但 `torch.arange(...)` 或 mask 未指定 device，默认留在 CPU。修复不是到处调用 `.cuda()`，而是让相关 tensors 从同一来源取得 device，例如 `device = next(model.parameters()).device`。

## 5. 固定调试顺序

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

通关条件：你能计算 `[4,1024,4096]` BF16 payload；解释 `(2,3,4)+(3,)` 为什么失败；解释临时 index tensor 为什么常造成 device mismatch；说明 `.to()` 何时可能不分配新 storage。

本文已包含完成本课所需内容；下列只用于你遇到版本行为差异时核对：

- [PyTorch Tensor Attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html)
- [PyTorch Tensor documentation](https://docs.pytorch.org/docs/stable/tensors.html)
- [Tensor.to contract](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html)
