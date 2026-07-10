---
artifact: course-lesson
leaf: operators/triton/matmul/tiled-matmul
language: zh-CN
estimated_time: 120min
environment: WSL2-GPU
---

# 课程：从点积到 tile，理解 GEMM 的数据复用

## 1. 单元素写法为什么浪费

`C[m,n]=sum_k A[m,k]*B[k,n]`。若每个 C 元素独立从 global memory 读完整行/列，A 的同一元素会被不同 n 重复读取，B 的同一元素会被不同 m 重复读取。GEMM 快的关键不是少做乘加，而是让一次加载服务多个输出。

## 2. 一个 program 计算 C 的矩形

令一个 program 负责 `BLOCK_M x BLOCK_N` 个 C 元素。它构造 `offs_m`、`offs_n`，accumulator 初始为零；沿 K 按 `BLOCK_K` 循环加载 `A[BLOCK_M,BLOCK_K]` 与 `B[BLOCK_K,BLOCK_N]`，使用 `tl.dot` 或等价乘加更新 accumulator。K-loop 结束后才 store C tile。

必须分别画出 M/N/K 三种边界：M/N mask 保护最终 store；K mask 保护最后一次加载。对 `M=5,N=6,K=7`、tile `4x4x4`，手画右下 C tile 与最后 K block，避免把三个 mask 混成一件事。

## 3. 性能取舍

更大 tile 可能提高 reuse、增大 arithmetic intensity；也会提高寄存器/共享资源需求，降低能同时驻留的 programs。layout、num_warps、num_stages 都是这个平衡的参数，而不是越大越好。

## 实验课

先实现 FP32 correctness kernel，测试 M/N/K 均不整除 tile 的 cases；再固定 dtype/shape/计时方法，sweep tile sizes 并报 TFLOP/s。报告中区分“正确性通过”“当前 shape 更快”“能解释全部 shapes 的性能规律”三种不同强度的结论。

## 通关

能写出 A/B/C pointer 的二维索引来源，说明 K-loop 如何减少重复读取。下一课用 roofline 判断一个 kernel 受 bytes 还是 FLOPs 限制。
