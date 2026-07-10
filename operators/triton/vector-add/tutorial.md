---
artifact: curriculum-tutorial
maintainer: ai-curriculum
learner_edit_policy: annotate-or-copy-to-note
leaf: operators/triton/vector-add
title: Triton vector add
priority: P0
environment: WSL2-GPU
status: ready-to-study
---

# Triton Vector Add Tutorial

> 这是一份课前教程，不是你的学习笔记。先读到能做出预测，再跑 `lab.ipynb`；把真正的理解、错误预测和实验数字写进 `note.md`。

## Position

`x + y` 是自定义 GPU 算子的最小完整链条：PyTorch tensor 传入 wrapper，wrapper 定义 launch grid，Triton kernel 生成一块地址，masked load/store 保证尾部安全，GPU 异步执行，最后用正确性和带宽验证。后续 softmax、RMSNorm、FlashAttention 与 fused epilogue 都会复用这条链，只是 index 和数据复用更复杂。

它不是一个值得手工替代 `torch.add` 的生产优化，而是 SPMD、指针算术和性能测量的低噪声训练场。

## Prerequisites And Route Position

- Required: `fundamentals/pytorch-tensor-lifecycle/shape-dtype-device`。能说出 `numel()`、dtype 的 element size 与 CUDA tensor 的设备归属。
- Required: `operators/cuda/execution-model/thread-block-grid` 与 `operators/cuda/memory-hierarchy/global-memory-coalescing`。已见过 grid 覆盖 index range，理解相邻地址通常比大 stride 更有利。
- This leaf: P0 第 6 步，见 [P0 Learning Map](/Users/youyu/workspace/python/infra/atlas/p0-learning-map.md)。
- Next: `operators/triton/matmul/tiled-matmul`。这里的一维 pointer block 将扩展为 M/N/K 三维 tile、循环和数据复用。
- Defer: warp 数、pipeline stages、layout、cache modifiers、autotuning。没有先掌握 coverage 与 mask，这些参数调优只会变成玄学。

## Primary Reading

1. [Triton official vector-add tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)。读到能解释 `program_id`、`arange`、`load`、`store` 与 launch grid。
2. [Triton language API](https://triton-lang.org/main/python-api/triton.language.html)。只查上述四个 primitive 的契约。
3. 本目录的 `lab.ipynb`。先阅读代码单元和题目，写下预测后再运行。

## Mechanism Walkthrough

目标是长度为 `N` 的 contiguous CUDA vectors：

```text
z[i] = x[i] + y[i], for 0 <= i < N
```

wrapper 选择编译期常量 `BLOCK_SIZE`，并发起 `ceil(N / BLOCK_SIZE)` 个 Triton program instances。第 `pid` 个实例处理连续逻辑块：

```text
block_start = pid * BLOCK_SIZE
offsets     = block_start + [0, 1, ..., BLOCK_SIZE - 1]
mask        = offsets < N
```

它将 `x_ptr + offsets`、`y_ptr + offsets` 解释成地址块，masked `tl.load` 两块值，相加后再 masked `tl.store` 到 `z_ptr + offsets`。

| Stage | State | Why it exists |
|---|---|---|
| Wrapper | `x`, `y`, `z`, `N`, `BLOCK_SIZE` | 检查输入，分配输出，定义全局覆盖范围。 |
| Grid | `(ceil(N / BLOCK_SIZE),)` | 让 program instances 足以覆盖每个有效 index。 |
| `program_id` | `pid` | 每个 instance 知道自己负责哪一段。 |
| Pointer block | `x_ptr + offsets` | 把逻辑 index 映射为 global-memory 地址。 |
| Tail mask | `offsets < N` | 最后一个不完整块不得越界访问。 |
| Synchronization | `torch.cuda.synchronize()` | launch 通常异步，正确计时需要同步边界。 |

Triton program 是逻辑 SPMD 工作单元。它与 CUDA grid/block 思维相似，但不要把一个 program 机械等同于一个 CUDA thread 或固定 CUDA block；具体 lowering 交给编译器。

## Invariants To Prove

1. **Coverage and uniqueness:** 每个有效 `i` 恰由 `pid = floor(i / BLOCK_SIZE)` 的一个 offsets 覆盖。
2. **Memory safety:** 对尾块，`tl.load` 与 `tl.store` 都必须使用同一个 `mask`。
3. **Reference equivalence:** 被测长度上 `triton_add(x, y)` 与 `torch.add(x, y)` 的误差在 dtype 容忍范围内；FP32 随机加法可以要求精确一致。
4. **Measurement boundary:** warmup、同步和是否计入 allocation 都要写清楚。Mac/CPU toy 只能验证语义，不能推出 GPU 带宽结论。

## Memory And Cost Model

FP32 out-of-place add 每元素读 `x` 与 `y` 各 4 bytes，写 `z` 4 bytes，总流量近似 `3 * N * 4` bytes；每元素只做一次加法：

```text
arithmetic intensity = 1 FLOP / 12 bytes = 1 / 12 FLOP per byte
effective_GBps = 3 * N * element_size / median_seconds / 1e9
```

因此大尺寸 vector add 通常受 global-memory bandwidth 限制。effective GB/s 是同一机器、同一方法间的可比指标，不是完整 DRAM transaction 的精确计数；cache 和硬件 transaction 会让真实流量不同。

## Predict Before The Lab

1. `N = 1025`、`BLOCK_SIZE = 1024` 时 launch 几个 programs？`pid=1` 的哪些 offsets 有效？
2. `N = 1`、`1023`、`1024`、`1025` 能否都与 `torch.add` 对齐？哪组最能暴露 mask bug？
3. 为什么 doubling `BLOCK_SIZE` 不应带来接近 2x 的加速？
4. 若不在计时区间同步，测得的 latency 偏大还是偏小？为什么？

## Lab Sequence

1. 在 WSL2 NVIDIA GPU 环境完成 preflight，记录 PyTorch、Triton、CUDA、GPU 名称。
2. 在执行前手算 `N=1025` 的 grid、`pid=1` offsets 与 mask。
3. 对多种长度做 correctness sweep，记录最大误差、dtype、设备和 block size。
4. 仅改变 `BLOCK_SIZE`，固定 N、dtype、warmup 和计时方法，记录 median ms 与 effective GB/s。
5. 对比 `torch.add`，但结论只适用于当前环境和测量方法。
6. 不要删除 mask 来制造失败实验；越界 global-memory access 可能非法或未定义。用 `N = BLOCK_SIZE + 1` 的尾块来验证它的必要性。

## Common Confusions

- **“`BLOCK_SIZE` 就是 CUDA thread 数。”** 不准确，它是本 kernel 一个 program 处理的元素数。
- **“用了 `cdiv` 就不需要 mask。”** 错，最后一个 program 常会覆盖到 `N` 之外。
- **“一个加法不值得 benchmark。”** 它恰好是低 arithmetic intensity 的干净带宽例子。
- **“Python 返回代表 GPU 完成。”** 错，正确性读取和计时都有同步边界。
- **“torch baseline 慢就证明 kernel 好。”** 不成立；融合、allocation、输入规模和同步方式都会改变比较。

## Write Your Own Note

完成 lab 后，在 `note.md` 用自己的话解释 `pid -> offsets -> mask -> load/add/store` 的调用链；指出哪条观察支持“memory-bound”、哪条只是推断；记录一次错误预测，并写下一个延后问题，例如“program 数、warp 数和 occupancy 如何关联”。

## Closed-Book Check And Next Link

完成自己的 `note.md` 后，再回答 `questions.md`。通过标准是不看教程也能推导尾块的 mask、估算字节数、说明计时边界。

下一站是 `operators/triton/matmul/tiled-matmul`：保留 `pid -> offsets -> mask -> load/store`，把一维 block 推广到二维 tile，并学习 K 维循环如何改变数据复用与 arithmetic intensity。
