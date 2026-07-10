---
artifact: curriculum-tutorial
leaf: operators/cuda/execution-model/thread-block-grid
priority: P0
environment: WSL2-GPU
---

# CUDA 执行模型：从一个元素到 thread、block、grid

## 问题

GPU kernel 不是“自动并行”。程序必须给每个 thread 一个可计算的工作坐标，并在元素数不能整除 block size 时保证没有 thread 越界。这个 index mapping 是 CUDA、Triton 和分布式分块的共同原型。

## 因果链

1. 一维 vector add 有 `N` 个独立输出元素，最朴素映射是一 thread 处理一个 `i`。
2. launch 指定 `blockDim.x`，runtime 创建多个 blocks；thread 的全局坐标为 `i = blockIdx.x * blockDim.x + threadIdx.x`。
3. grid size 取 `ceil(N / blockDim.x)`，因而最后一个 block 通常有一部分 thread 的 `i >= N`。
4. kernel 用 `if (i < N)` 保护 load/store。grid 的职责是覆盖，bounds check 的职责是安全；二者不能互相替代。

## 硬切片

- thread 是执行索引，不等于一个完整 SM；block 被调度到某个 SM，多个 blocks 可并发或分时执行。
- warp 通常按 32 threads 调度；block size 常选 warp 的倍数，但“是倍数”不等于一定快。
- `N=1025, block=256` 时 grid 是 5；最后一个 block 的 thread 0 处理 index 1024，其余 255 threads 必须被 bounds check 屏蔽。

## 预测后验证

在 WSL2 上运行一个最小 CUDA vector-add（可先只读代码）。在 host 端打印 N、block、grid；在小 N 时把 `blockIdx/threadIdx/i` 写入诊断 buffer，而不是在大 kernel 里 `printf`。分别测试 `N=255/256/257`，先写出最后一个 block 的有效 thread 数，再检查输出与 PyTorch reference。

## 下一跳

正确 index mapping 只说明语义正确；不同线程如何读地址才决定带宽。继续 `global-memory-coalescing`。
