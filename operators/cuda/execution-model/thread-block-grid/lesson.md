---
artifact: course-lesson
leaf: operators/cuda/execution-model/thread-block-grid
language: zh-CN
estimated_time: 90min
environment: WSL2-GPU
---

# 课程：CUDA 如何给每个元素分配一个执行坐标

## 1. 从串行循环出发

CPU vector add 可以写成 `for i in range(N): z[i]=x[i]+y[i]`。GPU 的第一步不是“把循环搬过去”，而是让许多 threads 同时各自得到一个 i。最常见的一维映射是：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) z[i] = x[i] + y[i];
```

`threadIdx.x` 是 block 内局部坐标，`blockIdx.x` 是 block 在 grid 中的坐标，`blockDim.x` 是每个 block 的宽度。三者相乘加和得到全局逻辑 index。

## 2. 为什么既要 grid 又要 bounds check

host 端常取 `grid = ceil(N/block_size)`。例如 N=1025、block=256，grid=5，共启动 1280 threads；它确保 index 0..1024 全部被覆盖，却必然多出 255 个 threads。`if (i<N)` 负责阻止这些多余 threads 的越界访问。grid 解决 coverage，guard 解决 safety，不能省其中任意一个。

## 3. block、warp、SM 的正确关系

thread 是代码看到的最小执行坐标；threads 被组织成 block，block 在一个 SM 上执行，block 内 threads 可协作/共享 shared memory；硬件通常以 32-thread warp 调度。不要把“一个 block”理解成“一次全 GPU 执行”：grid 中可能有远多于 SM 数量的 blocks，runtime 会分批调度，block 之间没有普通 grid 内全局同步保证。

## 实验课

先手算 N 为 255、256、257 时的 grid 和最后 block 有效 threads。再运行最小 CUDA vector-add，与 torch reference 比较。小 N 时将 `(blockIdx, threadIdx, i)` 写入诊断 array（而不是大量 printf），用结果验证 index formula。记录 block size 改变时语义不变，性能可能改变。

## 通关

解释 N=1025、block=256 时哪一个 thread 写 `z[1024]`，为什么其余最后 block threads 不可读写。下一课说明这些 threads 虽然语义正确，却可能因地址模式浪费带宽。
