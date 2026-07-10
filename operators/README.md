# Operators

这个目录对应脑图里的“算子”。它不是只学 CUDA kernel，而是覆盖从 PyTorch op 到 Triton/CUDA/CUTLASS，再到 attention、融合、compiler lowering、profiling 的整条链。

## Structure

- `cuda`: thread/block/grid、memory hierarchy、streams、CUDA Graphs、tensor cores。
- `triton`: program model、mask、reduction、softmax、matmul、persistent matmul、FP8。
- `attention`: standard attention、online softmax、FlashAttention、PagedAttention decode。
- `fusion`: elementwise fusion、bias+gelu、dropout+add+layernorm、GEMM epilogue。
- `libraries`: cuBLAS、cuDNN、CUTLASS、FlashInfer、torchao、Transformer Engine。
- `compiler`: PyTorch dispatcher、Dynamo、AOTAutograd、Inductor、custom op、TensorRT build。
- `profiling`: microbenchmark、roofline、PyTorch Profiler、Nsight Systems、Nsight Compute。

## Exit Bar

你应该能从一个 PyTorch 表达式追到实际 kernel，解释它是 memory-bound 还是 compute-bound，并用 profiler 或 microbenchmark 验证。

