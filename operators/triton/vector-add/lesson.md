---
artifact: course-lesson
leaf: operators/triton/vector-add
language: zh-CN
estimated_time: 90min
environment: WSL2-GPU
---

# 课程：用 Triton 写出第一个正确、可测的 GPU kernel

## 1. Triton 把 CUDA index mapping 换成 SPMD program

目标仍是 `z[i]=x[i]+y[i]`。不同点是 Triton 不要求你显式写每个 CUDA thread；你编写一个 program，指定它一次处理 `BLOCK_SIZE` 个元素。runtime 启动 `ceil(N/BLOCK_SIZE)` 个 program instances，每个用 `pid=tl.program_id(0)` 得到自己的块编号。

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < N
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
tl.store(z_ptr + offsets, x + y, mask=mask)
```

这里 `offsets` 是一个向量；`x_ptr + offsets` 是一块地址而不是单地址。它正是 CUDA 中“多 threads 分别处理多个 i”的 SPMD 表达。

## 2. 为什么 mask 是核心而非样板

N=1025、BLOCK_SIZE=1024 时 grid 为 2。pid=1 的 offsets 是 1024..2047，mask 只有第一个为真。load/store 必须同时使用这张 mask：只 mask store 会先非法 load；只 mask load 会非法 store。不要删除 mask 来做实验，越界访问是未定义行为；用 tail-size correctness sweep 验证它。

## 3. 先算成本再测性能

FP32 out-of-place add 每元素读 x/y 各 4 bytes，写 z 4 bytes，约 12 bytes；只做一次 add，因此 `I≈1/12 FLOP/byte`。大 N 时它通常接近 bandwidth workload。有效带宽定义为 `3*N*element_size / seconds`，它是可比较的 payload 指标，不是完整硬件 transaction 计数。

## 实验课

按 `lab.ipynb`：先手算 tail mask；对 N=1、1023、1024、1025、1_000_003 与 torch.add 做 exact correctness；再固定输入、把 allocation 放在计时外，使用 CUDA events 和 warmup sweep block size。最后写出：为什么 block size 翻倍不保证两倍速度、为什么不同步会得到错误 latency。

## 通关

能从 wrapper 调到 `pid -> offsets -> mask -> load/add/store`，并估算一次 N 元素 FP32 add 的 bytes。下一课进入 tiled matmul：同一 pointer 思维会扩展为二维 tile 和 K-loop reuse。
