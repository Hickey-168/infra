# Infra 学习路线图

本文件负责规划**学习顺序**；完整目录树见 [learning-directory.md](/Users/youyu/workspace/python/infra/atlas/learning-directory.md)。

标记说明：

- `P0`：跨系统核心机制，优先学习。
- `P1`：生产常见机制，P0 后尽快补。
- `P2`：场景相关，遇到真实需求时再深入。
- `Mac`：可以在 Mac 上完成概念实验或 toy 实现。
- `WSL2-GPU`：建议在 Windows + WSL2 + NVIDIA GPU 上跑。
- 🆕：本轮新并入的环节（原路线未排入学习顺序，目录中大多已存在）。

> 注：表格 `路径` 列对应磁盘上的真实目录，保持英文原样，请勿翻译。

## P0 主线：搭建脊梁

这条线把 张量、GPU、算子、推理、通信、分布式训练 串起来。

| 顺序 | 路径 | 为什么学 | 验证方式 |
|---|---|---|---|
| 1 | `fundamentals/pytorch-tensor-lifecycle` | 后面所有 shape/layout/state 分析的前置知识。 | view/reshape/transpose/contiguous 的 stride 实验。`Mac` |
| 2 | `fundamentals/memory-accounting-single-gpu` | 🆕 先建立完整显存账本（参数/梯度/优化器/激活四项）与 allocated vs reserved 概念，是排查 OOM 的第一现场。 | 打印四项占用；对比 `torch.cuda.memory_allocated` 与 `memory_reserved`。`WSL2-GPU` |
| 3 | `training/memory/parameter-memory` | 先会手算参数显存。 | 手算小 transformer 参数量与 dtype 显存。`Mac` |
| 4 | `training/memory/activation-memory` | 理解训练显存为什么远大于推理。 | 用 hooks 或 profiler 观察 saved tensors。`Mac / WSL2-GPU` |
| 5 | `operators/cuda/execution-model/thread-block-grid` | CUDA 最小执行模型。 | vector add kernel。`WSL2-GPU` |
| 6 | `operators/cuda/memory-hierarchy/global-memory-coalescing` | 大部分 kernel 优化从内存访问开始。 | stride copy 带宽 benchmark。`WSL2-GPU` |
| 7 | `operators/triton/vector-add` | 进入 Triton 编程模型。 | Triton vs torch 的正确性 / 延迟对比。`WSL2-GPU` |
| 8 | `operators/triton/matmul/tiled-matmul` | GEMM 是 transformer 的主计算。 | matmul benchmark + block size 扫描。`WSL2-GPU` |
| 9 | `operators/profiling/roofline-model` | 判断 compute-bound 还是 memory-bound。 | copy/matmul 的 roofline 估算与实测对比。`WSL2-GPU` |
| 10 | `operators/profiling/pytorch-profiler` | 🆕 在上 Nsight 之前的第一件 profiling 武器，先学会读 timeline 与算子耗时。 | 用 profiler 抓取一次前向，定位耗时 top 算子。`WSL2-GPU` |
| 11 | `inference/lifecycle/prefill-vs-decode` | 推理系统最核心的阶段差异。 | toy transformer 记录两阶段 shape/FLOPs。`Mac / WSL2-GPU` |
| 12 | `inference/kv-cache/kv-shape-and-capacity` | KV cache 是 serving 显存核心。 | KV cache 计算器 + 小模型显存观察。`WSL2-GPU` |
| 13 | `inference/scheduling/continuous-batching` | vLLM/SGLang/TensorRT-LLM 都绕不开的调度机制。 | waiting/running/finished 的 toy scheduler。`Mac` |
| 14 | `communication/collectives/all-reduce` | DDP 和分布式训练的通信底座。 | torch.distributed 多进程 all-reduce。`Mac / WSL2-GPU` |
| 15 | `communication/algorithms/ring-allreduce` | 会算通信量和时间。 | ring simulator + alpha-beta 成本模型。`Mac` |
| 16 | `distributed-training/data-parallel/ddp-gradient-allreduce` | 最小分布式训练闭环。 | 2 进程 DDP hook/bucket trace。`WSL2-GPU` |
| 17 | `distributed-training/sharded-data-parallel/fsdp2/fully-shard-api` | 大模型训练显存切分核心。 | FSDP2 小模型显存对比。`WSL2-GPU` |

## P1 主线：生产常见机制

| 主题 | 路径 |
|---|---|
| vLLM 式服务 | `inference/kv-cache/paged-kv-cache/block-table`、`inference/kv-cache/paged-kv-cache/block-allocator`、`inference/frameworks/vllm/paged-attention`、`inference/frameworks/vllm/scheduler` |
| 前缀复用 | `inference/kv-cache/prefix-cache/block-hash`、`inference/kv-cache/radix-cache/radix-tree-prefix-sharing`、`inference/frameworks/sglang/radix-attention` |
| 🆕 分块预填充 | `inference/scheduling/chunked-prefill`（与 continuous batching 齐名的调度机制，长 prompt prefill 切块与 decode 混排） |
| 解码算法 | `inference/decoding/beam-search`、`inference/decoding/speculative-draft-verify`、`inference/decoding/json-schema-constrained-decoding` |
| Attention kernel | `operators/attention/online-softmax`、`operators/attention/flash-attention-v1`、`operators/attention/flash-attention-v2`、`operators/attention/paged-attention-kernel` |
| 编译器 / 运行时 | `operators/compiler/torch-dynamo`、`operators/compiler/torch-inductor`、`operators/compiler/graph-break-debugging`、`inference/frameworks/vllm/torch-compile` |
| 分布式训练 | `distributed-training/tensor-parallel/column-parallel-linear`、`distributed-training/tensor-parallel/row-parallel-linear`、`distributed-training/pipeline-parallel/one-forward-one-backward`、`distributed-training/composition/megatron-rank-groups` |
| 🆕 NCCL 实操 | `communication/nccl/process-group-nccl`、`communication/nccl/nccl-tests`、`communication/nccl/nccl-env-vars`（分布式一切问题的排查起点） |
| 🆕 计算-通信重叠 | `communication/overlap/ddp-gradient-buckets`、`communication/overlap/fsdp-prefetch`、`communication/overlap/compute-communication-overlap`（分布式性能优化的核心手段） |
| 量化 | `inference/quantization/weight-only-int4`、`inference/quantization/awq`、`inference/quantization/smoothquant`、`inference/quantization/fp8-inference`、`training/precision/fp8-transformer-engine` |
| 🆕 分布式 checkpoint | `distributed-training/checkpointing/distributed-state-dict`、`distributed-training/checkpointing/checkpoint-resharding`（大规模训练的存活能力，几乎必踩） |
| 部署 | `deployment/containers/nvidia-container-runtime`、`deployment/kubernetes/gpu-operator`、`deployment/serving/ray-serve-dynamic-batching`、`deployment/serving/kserve-inferenceservice` |
| 可观测 | `observability-benchmarking/metrics/ttft`、`observability-benchmarking/metrics/tpot`、`observability-benchmarking/load-testing/latency-throughput-curve`、`observability-benchmarking/profiling/nsight-systems` |

## P2 主线：专项深入

这些先扫盲，等你遇到真实 workload 或面试/项目需要时再深入。

| 主题 | 路径 |
|---|---|
| 进阶推理 | `inference/distributed-inference/disaggregated-prefill-decode`、`inference/distributed-inference/prefix-aware-routing`、`inference/decoding/medusa-eagle-style-decoding`、`inference/multimodal/vision-token-prefill` |
| MoE | `distributed-training/expert-parallel/moe-router-topk`、`distributed-training/expert-parallel/token-dispatch-alltoall`、`operators/fusion/moe-fused-kernels` |
| 长上下文 | `distributed-training/context-parallel/ring-attention`、`distributed-training/context-parallel/ulysses-style-attention`、`inference/attention-backends/mla-attention-deepseek` |
| 存储 / 平台 | `storage/object-store/eventual-consistency`、`storage/cache/kv-cache-storage`、`deployment/reliability/autoscaling-stability` |
| 搜索 / 推荐 | `retrieval-recommendation/retrieval/vector-retrieval`、`retrieval-recommendation/retrieval/recall-merge-dedup`、`retrieval-recommendation/reranking/llm-rerank`、`retrieval-recommendation/feature-store/offline-online-consistency` |
| 🆕 profiling 深挖 | `operators/profiling/nsight-compute`（roofline / warp stall 归因）、`operators/profiling/kernel-metrics-occupancy`、`operators/profiling/kernel-metrics-memory-throughput` |
| 🆕 推理经济性 | 每 token 成本与 GPU 单位经济模型、SLA 驱动的容量规划（p99 反推并发）、batch/并发 与 TTFT-TPOT 权衡曲面（暂无对应叶子，待建） |
| 🆕 训练稳定性 | loss spike 诊断与恢复、数值不稳定 / NaN 定位、长跑任务静默错误（SDC）与校验（暂无对应叶子，待建） |
| 🆕 硬件新范式 | Hopper/Blackwell 架构差异（FP8/FP4、TMA、cluster）、GB200/NVL72 与 NVLink domain 扩展、KV over RDMA（暂无对应叶子，待建） |

## 如何推进一个叶子

每个叶子按同一闭环推进：

```text
map -> code path -> prediction -> experiment -> mechanism note -> teach-back -> gap check
```

建议复制 [templates/learning-item.md](/Users/youyu/workspace/python/infra/templates/learning-item.md) 到对应叶子目录，然后再开始学习。
