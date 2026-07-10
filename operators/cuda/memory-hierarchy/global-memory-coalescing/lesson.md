---
artifact: course-lesson
leaf: operators/cuda/memory-hierarchy/global-memory-coalescing
language: zh-CN
estimated_time: 90min
environment: WSL2-GPU
---

# 课程：从线程 index 到 global-memory transaction

## 1. 语义正确不代表带宽高

同一 warp 的 32 个 threads 都执行 `x[...]`，但硬件关心的是 32 个地址如何分布。相邻 lanes 访问相邻元素时，设备能以较少 transactions 服务这些请求；若 lane k 访问 `x[k*stride]` 且 stride 很大，logical payload 相同，却可能需要更多 cache line/transaction，effective bandwidth 降低。

## 2. 用地址公式判断，而不背口号

对 contiguous FP32 vector，lane l 访问 `base + 4*l`，连续 32 lanes 的逻辑 payload 是 128 bytes。对二维 row-major `A[row,col]`，地址随 col 连续、随 row 跳 `num_cols`。所以让 threadIdx 对应 col 往往比对应 row 更接近连续访问。真正 transaction 粒度会随硬件和 cache policy 变化；学习目标是先能从 index formula 预测“更连续或更分散”。

## 3. 连接 PyTorch stride

transpose 可不复制数据而改变 logical axes；若 kernel 仍按错误维度把 thread 映射到元素，访问会变成大 stride。因而“shape 相同”不足以判断访存，必须同时问 layout/stride。下一阶段 Triton 的 pointer arithmetic 会把同一件事写得更显式。

## 实验课

写两个 kernel，固定总访问元素数：一个 `i=global_tid`，一个 `i=global_tid*stride`，都确保不越界。先预测 stride=1、2、8、32 的相对 GB/s；用 CUDA events warmup 后测 median。再核对 tensor 是否 contiguous、数据规模是否足以越过 launch overhead。实验只说明当前 GPU 与当前 access pattern。

## 通关

能从 `address = base + (row*W + col)*4` 说明为何让相邻 lanes 走 col 通常更好。下一课把一维连续访问迁移到 Triton 的 `program_id/offsets/mask`。
