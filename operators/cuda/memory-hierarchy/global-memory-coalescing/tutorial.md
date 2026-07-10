---
artifact: curriculum-tutorial
leaf: operators/cuda/memory-hierarchy/global-memory-coalescing
priority: P0
environment: WSL2-GPU
---

# Global Memory Coalescing：让一个 warp 的访问像一条连续的车流

## 问题

许多 elementwise kernel 只有极少 FLOPs，却要搬运大量 global-memory 数据。此时优化重点不是“多开线程”，而是一个 warp 的地址模式能否被少量 memory transactions 合并服务。

## 因果链

1. warp 中的 lanes 并发发出 load/store；硬件根据实际地址形成 memory transactions。
2. 若 lane 0..31 访问连续或规则相邻元素，所需 transaction 较少，payload 利用率高。
3. 若每 lane 跨很大 stride、随机跳跃，仍可能读取同样数量的有效元素，却需更多 transactions，effective bandwidth 下降。
4. tensor 的 logical shape 不等于访问连续性；要看 stride 和 index formula。transpose 后的视图尤其容易让“看着是二维”的访问变成跨步访问。

## 必须会算

对 contiguous FP32 vector，连续 32 lanes 的逻辑 payload 是 `32 * 4 = 128 bytes`。这不是保证只有一次 transaction 的硬件承诺，但它给出判断方向：让相邻 lane 的地址相邻。对 `x[tid * stride]`，把 stride 从 1 增大，预测有效带宽会如何变。

## 预测后验证

写两个只读/只写 copy kernel：一个 `i = global_tid`，一个 `i = global_tid * stride`（确保分配足够大且都在 bounds 内）。固定总访问次数，改变 stride，使用 CUDA events 测 median time 与 effective GB/s。先预测曲线，再用 Nsight Compute 或至少 timing 验证；结果要注明 cache、访问规模和 GPU 型号，不能把一次曲线当硬件定律。

## 下一跳

`operators/triton/vector-add` 用 Triton 重写同一种连续访问，将 CUDA index formula 换成 `program_id -> offsets -> mask`。
