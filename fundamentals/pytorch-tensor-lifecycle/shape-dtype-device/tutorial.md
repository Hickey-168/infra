---
artifact: curriculum-tutorial
leaf: fundamentals/pytorch-tensor-lifecycle/shape-dtype-device
priority: P0
environment: Mac
---

# Shape、dtype 与 device：后续所有张量推理的坐标系

## 先建立问题意识

同一句 `y = x + 1`，是否能执行、会占多少内存、结果在哪块设备上，取决于三个独立事实：逻辑形状 `shape`、每个元素的表示 `dtype`、实际存储位置 `device`。AI infra 中很多“算子错了”“显存不够”“通信挂了”最终不是算法问题，而是其中一个事实被默默假设错了。

## 因果链

1. 模型代码先用 shape 描述批、序列、头和隐藏维，shape 决定逐元素操作和矩阵乘的合法性。
2. 同一 shape 用 FP32、BF16、INT8 表示时，元素字节数和数值范围不同；因此参数显存、KV cache、通信量也不同。
3. tensor 真正的数据位于 CPU、CUDA 或 MPS device；跨 device 运算不是隐式免费拷贝，通常直接报错或需要显式迁移。
4. 所以后续每次追调用链，都先写出 `(shape, dtype, device)`，再谈 kernel、通信或性能。

## 必须掌握的硬切片

- `numel() * element_size()` 是 tensor payload 的下界；它不包含 allocator 保留、Python 对象或 autograd 保存的其他 tensor。
- `torch.empty((B, T, H), dtype=..., device=...)` 的 shape 是逻辑坐标，dtype 决定每格 bytes，device 决定谁拥有 storage。
- broadcasting 先检查从右对齐的维度是否相等或为 1；“能 broadcast”不等于“不产生额外计算或内存”。

## 预测后验证

在 Mac 上新建 `lab.ipynb`：构造 `(2, 3, 4)` 的 FP32、BF16、INT64 tensor，先手算 bytes，再打印 `shape/dtype/device/numel/element_size/nbytes`。接着尝试 `(2,3,4) + (4,)` 与 `(2,3,4) + (3,)`，预测哪个会失败。最后将一个 tensor 放到 MPS（若可用）或保留 CPU，观察跨 device 运算的错误信息。

记录格式：**预测 -> 代码 -> 实际 shape/dtype/device -> 解释 -> 此实验不证明什么**。这个实验只验证 PyTorch 契约，不证明 GPU kernel 性能。

## 自检与下一跳

闭眼回答：一个 `(8, 2048, 4096)` 的 BF16 activation payload 有多少 bytes？若报 “expected all tensors on the same device”，你应先打印哪三项？

下一叶是 `training/memory/parameter-memory`：把单个 tensor 的字节公式扩展为模型全部参数与训练状态的账本。
