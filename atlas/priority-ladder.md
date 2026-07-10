# Priority Ladder

优先级不是“这个技术厉不厉害”，而是“它是否跨框架、跨公司、跨硬件周期复用，并且是否能帮助解释更多下游现象”。

## P0: Core Mechanisms

必须优先学习。它们是大部分 AI infra 问题的共同语言。

- Tensor shape / dtype / stride / layout / device / ownership
- GPU memory hierarchy, occupancy, bandwidth, arithmetic intensity
- CUDA execution model and streams
- Triton programming model and tiled matmul
- Attention, FlashAttention, KV cache
- Prefill vs decode, TTFT vs TPOT, continuous batching
- Sampling basics: temperature, top-k, top-p, beam search
- PyTorch autograd, optimizer state, activation memory
- DDP all-reduce, NCCL collectives, communication cost model
- FSDP/ZeRO memory accounting
- Profiling with PyTorch Profiler, Nsight Systems, Nsight Compute

## P1: Production-Common Systems

短期很有价值，很多团队都会用到。

- vLLM mechanisms: PagedAttention, block table, scheduler, prefix cache, chunked prefill
- SGLang mechanisms: RadixAttention, structured generation runtime, prefix reuse
- TensorRT-LLM mechanisms: in-flight batching, paged KV cache, quantization, multi-GPU inference
- torch.compile / Dynamo / Inductor / CUDA Graphs
- Megatron parallelisms: TP, PP, SP, EP, CP
- DeepSpeed ZeRO stages and offload
- FP16/BF16/FP8 recipes, Transformer Engine
- Quantization: weight-only INT4/INT8, SmoothQuant, GPTQ/AWQ, KV cache quantization
- Serving SLO: throughput, p50/p95/p99, admission control, backpressure
- GPU scheduling on Kubernetes/Ray/KServe

## P2: Specialized But Worth Knowing

有较高收益，但依赖场景或硬件。

- Speculative decoding variants: draft model, Medusa/EAGLE-style heads, verification kernels
- Multi-LoRA serving and adapter batching
- Disaggregated prefill/decode
- Prefix-aware routing and KV cache locality
- MoE expert parallelism, router load balance, token dispatch
- Pipeline bubble scheduling and interleaving
- Long-context attention variants and paged context attention
- NVMe offload, distributed checkpoint resharding
- Vector database internals and ANN index trade-offs
- Feature store / embedding store consistency for search/recommendation

## P3: Defer Until Needed

先知道名字和边界，后面遇到真实问题再深入。

- Framework-specific edge flags and one-off optimizations
- Rare decoding algorithms not used in your workload
- Hardware-specific micro-optimizations before CUDA/Triton basics solid
- Full compiler backend hacking before you can read generated Triton/CUDA behavior
- Exotic distributed topologies without access to comparable hardware

## Ranking Heuristic

给一个节点打分时看五项：

| Dimension | Question |
|---|---|
| Reuse | 是否跨 vLLM/SGLang/TensorRT-LLM/PyTorch/Megatron 都能复用？ |
| Bottleneck | 是否解释显存、吞吐、延迟、通信、正确性中的一个核心瓶颈？ |
| Observability | 是否能用 profiler、trace、benchmark 或 toy implementation 验证？ |
| Dependency | 是否是理解更多节点的前置知识？ |
| Local Feasibility | 是否能在 Mac CPU、WSL2 + RTX 5080、或小规模 GPU 上完成实验？ |

