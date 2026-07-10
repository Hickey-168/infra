# Infra Roadmap

这个文件负责学习顺序；完整目录树见 [learning-directory.md](/Users/youyu/workspace/python/infra/atlas/learning-directory.md)。

标记说明：

- `P0`: 跨系统核心机制，优先学习。
- `P1`: 生产常见机制，P0 后尽快补。
- `P2`: 场景相关，遇到真实需求时深入。
- `Mac`: 可以在 Mac 上完成概念实验或 toy implementation。
- `WSL2-GPU`: 建议在 Windows + WSL2 + NVIDIA GPU 上跑。

## P0 Route: Build The Spine

这条线把 tensor、GPU、算子、推理、通信、分布式训练串起来。

| Order | Path | Why | Verification |
|---|---|---|---|
| 1 | `fundamentals/pytorch-tensor-lifecycle` | 后面所有 shape/layout/state 分析的前置知识。 | view/reshape/transpose/contiguous stride 实验。Mac |
| 2 | `training/memory/parameter-memory` | 先会手算参数显存。 | 小 transformer 参数量和 dtype 显存计算。Mac |
| 3 | `training/memory/activation-memory` | 理解训练显存为什么远大于推理。 | hooks 或 profiler 观察 saved tensors。Mac/WSL2-GPU |
| 4 | `operators/cuda/execution-model/thread-block-grid` | CUDA 最小执行模型。 | vector add kernel。WSL2-GPU |
| 5 | `operators/cuda/memory-hierarchy/global-memory-coalescing` | 大部分 kernel 优化从内存访问开始。 | stride copy bandwidth benchmark。WSL2-GPU |
| 6 | `operators/triton/vector-add` | 进入 Triton 编程模型。 | Triton vs torch correctness/latency。WSL2-GPU |
| 7 | `operators/triton/matmul/tiled-matmul` | GEMM 是 transformer 主计算。 | matmul benchmark + block size sweep。WSL2-GPU |
| 8 | `operators/profiling/roofline-model` | 判断 compute-bound vs memory-bound。 | copy/matmul roofline 估算与实测对比。WSL2-GPU |
| 9 | `inference/lifecycle/prefill-vs-decode` | 推理系统最核心的阶段差异。 | toy transformer 记录两阶段 shape/FLOPs。Mac/WSL2-GPU |
| 10 | `inference/kv-cache/kv-shape-and-capacity` | KV cache 是 serving 显存核心。 | KV cache calculator + 小模型显存观察。WSL2-GPU |
| 11 | `inference/scheduling/continuous-batching` | vLLM/SGLang/TensorRT-LLM 都绕不开的调度机制。 | waiting/running/finished toy scheduler。Mac |
| 12 | `communication/collectives/all-reduce` | DDP 和分布式训练的通信底座。 | torch.distributed 多进程 all-reduce。Mac/WSL2-GPU |
| 13 | `communication/algorithms/ring-allreduce` | 会算通信量和时间。 | ring simulator + alpha-beta cost model。Mac |
| 14 | `distributed-training/data-parallel/ddp-gradient-allreduce` | 最小分布式训练闭环。 | 2 process DDP hook/bucket trace。WSL2-GPU |
| 15 | `distributed-training/sharded-data-parallel/fsdp2/fully-shard-api` | 大模型训练显存切分核心。 | FSDP2 小模型显存对比。WSL2-GPU |

## P1 Route: Production-Common Mechanisms

| Theme | Paths |
|---|---|
| vLLM-style serving | `inference/kv-cache/paged-kv-cache/block-table`, `inference/kv-cache/paged-kv-cache/block-allocator`, `inference/frameworks/vllm/paged-attention`, `inference/frameworks/vllm/scheduler` |
| Prefix reuse | `inference/kv-cache/prefix-cache/block-hash`, `inference/kv-cache/radix-cache/radix-tree-prefix-sharing`, `inference/frameworks/sglang/radix-attention` |
| Decode algorithms | `inference/decoding/beam-search`, `inference/decoding/speculative-draft-verify`, `inference/decoding/json-schema-constrained-decoding` |
| Attention kernels | `operators/attention/online-softmax`, `operators/attention/flash-attention-v1`, `operators/attention/flash-attention-v2`, `operators/attention/paged-attention-kernel` |
| Compiler/runtime | `operators/compiler/torch-dynamo`, `operators/compiler/torch-inductor`, `operators/compiler/graph-break-debugging`, `inference/frameworks/vllm/torch-compile` |
| Distributed training | `distributed-training/tensor-parallel/column-parallel-linear`, `distributed-training/tensor-parallel/row-parallel-linear`, `distributed-training/pipeline-parallel/one-forward-one-backward`, `distributed-training/composition/megatron-rank-groups` |
| Quantization | `inference/quantization/weight-only-int4`, `inference/quantization/awq`, `inference/quantization/fp8-inference`, `training/precision/fp8-transformer-engine` |
| Deployment | `deployment/containers/nvidia-container-runtime`, `deployment/kubernetes/gpu-operator`, `deployment/serving/ray-serve-dynamic-batching`, `deployment/serving/kserve-inferenceservice` |
| Observability | `observability-benchmarking/metrics/ttft`, `observability-benchmarking/metrics/tpot`, `observability-benchmarking/load-testing/latency-throughput-curve`, `observability-benchmarking/profiling/nsight-systems` |

## P2 Route: Specialized Depth

这些先扫盲，等你遇到真实 workload 或面试/项目需要再深入。

| Theme | Paths |
|---|---|
| Advanced inference | `inference/distributed-inference/disaggregated-prefill-decode`, `inference/distributed-inference/prefix-aware-routing`, `inference/decoding/medusa-eagle-style-decoding`, `inference/multimodal/vision-token-prefill` |
| MoE | `distributed-training/expert-parallel/moe-router-topk`, `distributed-training/expert-parallel/token-dispatch-alltoall`, `operators/fusion/moe-fused-kernels` |
| Long context | `distributed-training/context-parallel/ring-attention`, `distributed-training/context-parallel/ulysses-style-attention`, `inference/attention-backends/mla-attention-deepseek` |
| Storage/platform | `storage/object-store/eventual-consistency`, `storage/cache/kv-cache-storage`, `deployment/reliability/autoscaling-stability` |
| Search/recommendation | `retrieval-recommendation/retrieval/vector-retrieval`, `retrieval-recommendation/retrieval/recall-merge-dedup`, `retrieval-recommendation/reranking/llm-rerank`, `retrieval-recommendation/feature-store/offline-online-consistency` |

## How To Use A Leaf

每个叶子按同一闭环推进：

```text
map -> code path -> prediction -> experiment -> mechanism note -> teach-back -> gap check
```

建议复制 [templates/learning-item.md](/Users/youyu/workspace/python/infra/templates/learning-item.md) 到对应叶子目录，然后再开始学习。

