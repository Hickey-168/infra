# AI Infra Learning Repository

这个仓库用于系统学习 AI infra，重点覆盖大模型推理、训练、分布式、GPU 算子、通信、存储、平台化部署、可观测性，以及和检索/推荐相关的工程分支。

原始脑图保留在 [naixu.png](/Users/youyu/workspace/python/infra/naixu.png)。本仓库把脑图中的 `infra` 分支整理成英文目录，并把叶子节点拆成可以独立学习、实现、验证的工作项。

## How To Navigate

- [atlas/infra-roadmap.md](/Users/youyu/workspace/python/infra/atlas/infra-roadmap.md): 全局知识树，按优先级、学习产出、验证方式组织。
- [atlas/priority-ladder.md](/Users/youyu/workspace/python/infra/atlas/priority-ladder.md): 学习优先级规则，帮助判断先学什么、后学什么。
- [atlas/original-mindmap-review.md](/Users/youyu/workspace/python/infra/atlas/original-mindmap-review.md): 对 `naixu.png` 中 infra 分支的查缺补漏和修正。
- [templates/learning-item.md](/Users/youyu/workspace/python/infra/templates/learning-item.md): 每个叶子节点的统一笔记/实验模板。
- [templates/experiment-note.md](/Users/youyu/workspace/python/infra/templates/experiment-note.md): 单次实验记录模板。

## Learning Domains

| Directory | Role |
|---|---|
| [fundamentals](/Users/youyu/workspace/python/infra/fundamentals/README.md) | Linux/Python/PyTorch/GPU 基础，给后续 infra 学习打地基。 |
| [inference](/Users/youyu/workspace/python/infra/inference/README.md) | LLM serving、decode/prefill、KV cache、调度、采样、推理框架。 |
| [training](/Users/youyu/workspace/python/infra/training/README.md) | 单机训练、优化器、混合精度、激活/显存、checkpoint。 |
| [distributed-training](/Users/youyu/workspace/python/infra/distributed-training/README.md) | DDP/FSDP/ZeRO/TP/PP/SP/EP/CP、Megatron/DeepSpeed。 |
| [operators](/Users/youyu/workspace/python/infra/operators/README.md) | CUDA、Triton、CUTLASS、FlashAttention、算子融合、compiler lowering。 |
| [communication](/Users/youyu/workspace/python/infra/communication/README.md) | NCCL、collectives、拓扑、overlap、通信调试。 |
| [deployment](/Users/youyu/workspace/python/infra/deployment/README.md) | 容器、Kubernetes、Ray Serve/KServe、路由、可靠性、弹性伸缩。 |
| [storage](/Users/youyu/workspace/python/infra/storage/README.md) | 数据集格式、对象存储、缓存、checkpoint、模型权重格式。 |
| [retrieval-recommendation](/Users/youyu/workspace/python/infra/retrieval-recommendation/README.md) | 搜索/推荐/向量召回/排序/重排/LLM4Rec。 |
| [observability-benchmarking](/Users/youyu/workspace/python/infra/observability-benchmarking/README.md) | profiling、benchmark、SLO、日志/指标/链路追踪。 |
| [systems-tooling](/Users/youyu/workspace/python/infra/systems-tooling/README.md) | GitHub/Obsidian/WSL2/远程 GPU/实验工程化。 |

## Directory-First Map

学习目录已经按真实子目录展开，入口在 [atlas/learning-directory.md](/Users/youyu/workspace/python/infra/atlas/learning-directory.md)。

大致形态是：

```text
inference/
  kv-cache/
    paged-kv-cache/
      block-table/
      block-allocator/
      ref-count-and-copy-on-write/
  scheduling/
    continuous-batching/
    chunked-prefill/
operators/
  triton/
    matmul/
      tiled-matmul/
      persistent-matmul/
distributed-training/
  tensor-parallel/
    column-parallel-linear/
    row-parallel-linear/
```

## Leaf Rule

一个叶子节点必须满足：

1. 能用一两句话说明它解决的瓶颈或正确性问题。
2. 能指向至少一个真实系统、论文、官方文档或源码位置。
3. 能设计一个最小实验、toy implementation、trace、benchmark 或 profiler 证据。
4. 能回答一个 call-chain 问题、一个 shape/state 问题、一个 counterfactual 问题。
5. 能明确哪些结论是事实、哪些是实验观察、哪些只是推断。

不满足这些条件的节点，例如只写 `vllm`、`Megatron`、`CUDA`，都不是最终叶子。它们只能作为中间目录或主题簇。

## Suggested First Pass

先按 P0 路线走：

```text
fundamentals/pytorch-tensor-lifecycle
-> operators/cuda/memory-hierarchy/global-memory-coalescing
-> operators/triton/matmul/tiled-matmul
-> inference/lifecycle/prefill-vs-decode
-> inference/kv-cache/kv-shape-and-capacity
-> inference/scheduling/continuous-batching
-> communication/algorithms/ring-allreduce
-> distributed-training/data-parallel/ddp-gradient-allreduce
-> distributed-training/sharded-data-parallel/fsdp2/fully-shard-api
```

这条线会把 LLM 生命周期、GPU 内存、算子、推理调度、通信、分布式训练串起来。
