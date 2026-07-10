---
artifact: curriculum-tutorial
leaf: operators/triton/matmul/tiled-matmul
priority: P0
environment: WSL2-GPU
---

# Triton Tiled Matmul：把重复读取变成可复用的 tile

## 问题

Transformer 的线性层、attention 投影和 MLP 都由 GEMM 主导。若逐个输出元素计算 `C[m,n]`，同一行 A 和同一列 B 会被反复从 global memory 读取；计算量很大，但数据复用没有被表达出来。

## 因果链

1. 目标是 `C[M,N] = A[M,K] @ B[K,N]`，每个 C 元素沿 K 做 dot product。
2. 一个 Triton program 不计算单元素，而计算一个 `BLOCK_M x BLOCK_N` 的 C tile；`pid` 被映射为 tile 坐标 `(pid_m, pid_n)`。
3. program 沿 K 循环加载 `A[BLOCK_M,BLOCK_K]` 与 `B[BLOCK_K,BLOCK_N]`，累加到寄存器 accumulator；同一加载块服务多个 C 元素，提升 arithmetic intensity。
4. M/N/K 不能整除 tile 时，pointer block 的 mask 保护边界；最终 accumulator 转回目标 dtype 并 store。

## 三个必须画出来的对象

- A pointers：`offs_m[:,None] * stride_am + offs_k[None,:] * stride_ak`。
- B pointers：`offs_k[:,None] * stride_bk + offs_n[None,:] * stride_bn`。
- C pointers：`offs_m[:,None] * stride_cm + offs_n[None,:] * stride_cn`。

不要背公式；对一个 `M=5,N=6,K=7,BLOCK_M=4,BLOCK_N=4,BLOCK_K=4` 的例子，画出 `pid_m=1,pid_n=1` 的三种 mask。只有能画出来，才知道为什么 K tail 与 M/N tail 是不同边界。

## 预测后验证

先只实现 FP32 correctness 版本，比较 `torch.matmul`；测试不整除 M/N/K 和 non-contiguous stride（若不支持，明确 assert contiguous）。再固定 shape，扫 `BLOCK_M/N/K`，记录 median ms、TFLOP/s、寄存器压力/occupancy（可后置用 profiler）。预测：较大 tile 增加复用，却可能增加寄存器资源和降低可并发 program 数；不存在全尺寸通吃的 block size。

## 边界与下一跳

不要拿初版 kernel 和 cuBLAS 的所有 shape 做“谁更强”结论。这里的目标是正确 pointer arithmetic、K-loop reuse 和成本表达。下一叶 `roofline-model` 将把“带宽受限还是算力受限”变成可估算的上界。
