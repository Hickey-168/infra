# Learning Directory

这个文件和真实目录结构对应。一级目录是学习主干；二级目录是脑图中的大分支；更深层目录是可以独立学习和验证的机制叶子。

叶子目录原则：

- 不用框架名当最终叶子，框架名只作为案例入口。
- 叶子必须能设计实验、toy implementation、trace、benchmark 或 profiler 验证。
- 优先学习跨框架机制，再学习框架实现差异。

## Fundamentals

`fundamentals` 是支撑目录，用来保证后续推理、训练、算子、通信都能落到 shape/state/memory/call chain 上。

### `fundamentals/pytorch-tensor-lifecycle`

- `shape-dtype-device`
- `stride-layout`
- `view-vs-reshape`
- `contiguous-memory`
- `storage-aliasing`

### `fundamentals/autograd-graph-and-saved-tensors`

- `grad-fn-chain`
- `saved-tensors-hooks`
- `no-grad-vs-inference-mode`
- `checkpoint-recompute`

### `fundamentals/module-forward-call-chain`

- `module-call`
- `forward-hooks`
- `backward-hooks`
- `parameter-registration`

### `fundamentals/memory-accounting-single-gpu`

- `parameter-gradient-optimizer-activation`
- `cuda-allocated-vs-reserved`
- `peak-memory-measurement`

### `fundamentals/linux-process-and-file-io`

- `process-env-fd`
- `mmap-page-cache`
- `async-file-io`

### `fundamentals/python-async-and-queueing`

- `asyncio-task-queue`
- `producer-consumer`
- `backpressure-timeout`

## Inference

`inference` 关注 LLM 请求从 API 进入，到 tokenize、prefill、decode、KV cache、调度、采样、返回的全过程。

### `inference/lifecycle`

- `tokenization-and-chat-template`: chat template、special tokens、prompt 组装边界。
- `transformer-forward-shapes`: batch/seq/hidden/head 形状流。
- `prefill-vs-decode`: prefill compute-bound，decode memory/cache-bound 的差异。
- `ttft-vs-tpot`: 首 token 延迟和每 token 延迟的指标拆分。
- `request-state-machine`: waiting/running/swapped/finished 状态迁移。

### `inference/decoding`

- `greedy-decoding`: argmax 路径和确定性输出。
- `sampling-temperature`: temperature 如何改变 logits 分布。
- `sampling-top-k`: top-k 截断和概率重归一。
- `sampling-top-p`: nucleus sampling 的累计概率边界。
- `beam-search`: hypothesis expansion、score、early stop。
- `length-penalty`: beam score 为什么需要长度归一。
- `logits-processor`: repetition、bad words、min length 等处理链。
- `repetition-penalty`: 重复惩罚对 logits 的符号敏感处理。
- `speculative-draft-verify`: draft model + target verify 的接受率机制。
- `medusa-eagle-style-decoding`: 多 token draft head 的加速边界。
- `guided-decoding-fsm`: FSM 约束 token 选择。
- `json-schema-constrained-decoding`: JSON schema 到 token mask 的约束。

### `inference/kv-cache`

- `kv-shape-and-capacity`: KV cache 显存公式。
- `contiguous-kv-cache`: 单块大 tensor 的浪费来源。
- `paged-kv-cache/block-table`: logical token block 到 physical block 的映射。
- `paged-kv-cache/block-allocator`: block 分配/释放/复用。
- `paged-kv-cache/ref-count-and-copy-on-write`: prefix 共享时的可变性和正确性。
- `prefix-cache/block-hash`: prefix block hash key 设计。
- `prefix-cache/cache-hit-rate`: 命中率和 workload 结构的关系。
- `radix-cache/radix-tree-prefix-sharing`: SGLang RadixAttention 风格的 prefix tree。
- `kv-cache-quantization`: KV 从 BF16/FP16 到 FP8/INT8 的容量收益和质量风险。
- `kv-cache-offload`: GPU/CPU/NVMe 之间的 cache 分层。
- `kv-cache-transfer`: P/D disaggregation 时的 KV 迁移。
- `kv-cache-fragmentation`: block/page 分配造成的碎片。

### `inference/scheduling`

- `static-batching`: 固定 batch 的空洞和等待成本。
- `dynamic-batching`: batching window 的吞吐/延迟 trade-off。
- `continuous-batching`: iteration-level batching。
- `chunked-prefill`: 长 prompt prefill 切块和 decode 混排。
- `decode-priority`: decode 优先如何影响 TPOT。
- `preemption-and-swapping`: 请求抢占、swap、恢复。
- `admission-control`: 控制并发保护 p99 和 OOM。
- `fairness-and-starvation`: 长短请求混合下的公平性。
- `queueing-latency-model`: 用排队模型解释吞吐/延迟曲线。

### `inference/attention-backends`

- `flash-attention-prefill`: prefill 阶段的 full attention kernel。
- `paged-attention-decode`: decode 阶段单 token query 对 paged KV 的读取。
- `flashinfer-backend`: serving attention backend 的接口边界。
- `xformers-backend`: 通用 attention backend 对比。
- `rope-and-position-ids`: RoPE、position ids、cache 位置的一致性。
- `mla-attention-deepseek`: MLA 对 KV cache 形状和容量的影响。

### `inference/frameworks`

- `vllm/scheduler`, `vllm/paged-attention`, `vllm/prefix-caching`, `vllm/chunked-prefill`, `vllm/speculative-decoding`, `vllm/torch-compile`, `vllm/disaggregated-prefill-decode`
- `sglang/radix-attention`, `sglang/structured-generation`, `sglang/speculative-decoding`, `sglang/constrained-decoding`
- `tensorrt-llm/inflight-batching`, `tensorrt-llm/paged-kv-cache`, `tensorrt-llm/quantization`, `tensorrt-llm/multi-gpu-inference`
- `tgi/continuous-batching`
- `llama-cpp/gguf-and-cpu-offload`

### `inference/serving-api`

- `openai-compatible-chat-api`: request/response/schema。
- `streaming-sse`: Server-Sent Events 和 token streaming。
- `cancellation-and-timeout`: 客户端断开和服务端清理。
- `tool-calling-protocol`: tool calls 的结构和状态。
- `error-and-retry-semantics`: retry 是否安全、幂等边界。
- `tokenizer-server-boundary`: tokenizer 放客户端/服务端的取舍。

### `inference/distributed-inference`

- `tensor-parallel-inference`
- `pipeline-parallel-inference`
- `data-parallel-replicas`
- `expert-parallel-inference`
- `disaggregated-prefill-decode`
- `prefix-aware-routing`
- `kv-cache-aware-load-balancing`
- `multi-node-serving`

### `inference/quantization`

- `weight-only-int8`, `weight-only-int4`, `gptq`, `awq`, `smoothquant`
- `fp8-inference`, `nvfp4-mxfp4`, `activation-quantization`, `kv-cache-quantization`

### `inference/multimodal`

- `vision-token-prefill`
- `encoder-decoder-cross-attention-cache`
- `image-video-batching`
- `multimodal-projector-cache`

## Training

`training` 关注单机和基础训练机制，分布式拆到 `distributed-training`。

### `training/single-gpu-loop`

- `forward-backward-step`
- `loss-scaling-and-grad-norm`
- `gradient-accumulation`
- `optimizer-step`
- `learning-rate-scheduler`
- `gradient-clipping`
- `training-loop-state-machine`

### `training/memory`

- `parameter-memory`
- `gradient-memory`
- `optimizer-state-memory`
- `activation-memory`
- `activation-checkpointing`
- `cuda-caching-allocator`
- `cpu-nvme-offload`
- `memory-snapshot-analysis`

### `training/precision`

- `fp32-and-tf32`
- `bf16-vs-fp16`
- `loss-scaling`
- `fp8-transformer-engine`
- `delayed-scaling`
- `mxfp8-nvfp4`
- `determinism-and-numerical-drift`

### `training/data-pipeline`

- `tokenization-throughput`
- `sequence-packing`
- `document-shuffling`
- `dataloader-workers`
- `pinned-memory-prefetch`
- `jsonl-parquet-arrow`
- `webdataset-streaming`

### `training/checkpointing`

- `model-state-dict`
- `optimizer-state-dict`
- `rng-and-scheduler-state`
- `resume-loss-continuity`
- `distributed-checkpoint`
- `checkpoint-resharding`
- `checkpoint-conversion-hf-megatron`

### `training/finetuning`

- `sft-data-collator`
- `lora-low-rank-update`
- `qlora-and-nf4`
- `dpo-loss-shapes`
- `ppo-rlhf-memory-map`
- `evaluation-regression-gate`

### `training/compiler-runtime`

- `torch-compile-training`
- `graph-breaks`
- `aot-autograd`
- `inductor-generated-kernels`
- `cuda-graphs-training`
- `dynamic-shapes`

### `training/frameworks`

- `pytorch-native`
- `torchtitan`
- `minitorch-or-tinygrad`
- `deepspeed`
- `megatron-lm`
- `huggingface-accelerate`
- `lightning-fabric`

## Operators

`operators` 对应“算子”，覆盖 CUDA/Triton/attention/fusion/compiler/profiling。

### `operators/cuda`

- `execution-model/thread-block-grid`
- `execution-model/warps-and-schedulers`
- `memory-hierarchy/global-memory-coalescing`
- `memory-hierarchy/shared-memory-tiling`
- `memory-hierarchy/register-pressure`
- `memory-hierarchy/l2-cache`
- `streams-events`
- `cuda-graphs`
- `occupancy`
- `atomics-and-reductions`
- `warp-level-primitives`
- `tensor-cores-and-mma`
- `async-copy-and-tma`

### `operators/triton`

- `programming-model/program-id-and-blocks`
- `programming-model/masks-and-boundary-checks`
- `vector-add`
- `reductions`
- `fused-softmax`
- `matmul/tiled-matmul`
- `matmul/autotune`
- `matmul/persistent-matmul`
- `block-pointers`
- `tl-dot`
- `debugging-and-profiling`
- `fp8-matmul`

### `operators/attention`

- `standard-attention`
- `online-softmax`
- `flash-attention-v1`
- `flash-attention-v2`
- `flash-attention-v3`
- `paged-attention-kernel`
- `decode-attention-with-kv-cache`
- `causal-mask`
- `alibi-and-rope`
- `rmsnorm`
- `layernorm`

### `operators/fusion`

- `elementwise-fusion`
- `bias-gelu-fusion`
- `dropout-add-layernorm`
- `gemm-epilogue-fusion`
- `rmsnorm-fusion`
- `moe-fused-kernels`
- `rope-fusion`

### `operators/libraries`

- `cublas`
- `cudnn`
- `cutlass/gemm-tiling`
- `cutlass/epilogue`
- `flashinfer`
- `torchao`
- `transformer-engine`
- `xformers`

### `operators/compiler`

- `pytorch-dispatcher-aten`
- `torch-dynamo`
- `aot-autograd`
- `torch-inductor`
- `triton-lowering`
- `graph-break-debugging`
- `custom-pytorch-op`
- `onnx-export`
- `tensorrt-engine-build`

### `operators/profiling`

- `microbenchmark-harness`
- `roofline-model`
- `pytorch-profiler`
- `nsight-systems`
- `nsight-compute`
- `kernel-metrics-occupancy`
- `kernel-metrics-memory-throughput`

## Communication

`communication` 是分布式训练、MoE、分布式推理的共同底座。

### `communication/collectives`

- `broadcast`, `reduce`, `all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`, `scatter-gather`, `barrier`

### `communication/algorithms`

- `ring-allreduce`
- `tree-allreduce`
- `hierarchical-allreduce`
- `recursive-doubling`
- `reduce-scatter-allgather`
- `chunking-and-pipelining`

### `communication/nccl`

- `process-group-nccl`
- `nccl-tests`
- `nccl-debug-logs`
- `nccl-env-vars`
- `topology-discovery`
- `nccl-ras`
- `nccl-timeout-debugging`

### `communication/overlap`

- `ddp-gradient-buckets`
- `fsdp-prefetch`
- `async-collectives`
- `compute-communication-overlap`
- `stream-priority`
- `bucket-size-tuning`

### `communication/topology`

- `pcie`
- `nvlink`
- `nvls`
- `infiniband-rdma`
- `ethernet-roce`
- `numa-cpu-affinity`
- `multi-node-network-model`

### `communication/failure-debugging`

- `collective-mismatch`
- `hanging-rank`
- `timeout-retry`
- `shape-mismatch`
- `rank-to-device-mapping`

## Distributed Training

`distributed-training` 关注并行策略、rank group、通信量、显存切分、checkpoint 和容错。

### `distributed-training/data-parallel`

- `ddp-gradient-allreduce`
- `ddp-bucketization`
- `no-sync-gradient-accumulation`
- `gradient-as-bucket-view`
- `static-graph-ddp`

### `distributed-training/sharded-data-parallel`

- `fsdp1`
- `fsdp2/fully-shard-api`
- `fsdp2/dtensor-placement`
- `zero-stage-1`
- `zero-stage-2`
- `zero-stage-3`
- `zero-offload`
- `hybrid-shard`
- `reshard-after-forward`

### `distributed-training/tensor-parallel`

- `column-parallel-linear`
- `row-parallel-linear`
- `vocab-parallel-embedding`
- `vocab-parallel-cross-entropy`
- `sequence-parallel-layernorm`
- `tensor-parallel-rank-groups`

### `distributed-training/pipeline-parallel`

- `gpipe-schedule`
- `one-forward-one-backward`
- `interleaved-pipeline`
- `bubble-ratio`
- `microbatching`
- `activation-send-recv`
- `virtual-pipeline-stages`

### `distributed-training/context-parallel`

- `sequence-splitting`
- `ring-attention`
- `long-context-training`
- `context-parallel-allgather`
- `ulysses-style-attention`

### `distributed-training/expert-parallel`

- `moe-router-topk`
- `capacity-factor`
- `token-dispatch-alltoall`
- `expert-load-balancing`
- `shared-experts`
- `moe-gemm-grouping`

### `distributed-training/composition`

- `dp-tp-pp-3d-parallelism`
- `tp-pp-dp-rank-mapping`
- `fsdp-plus-tp`
- `megatron-rank-groups`
- `deepspeed-megatron-composition`
- `parallelism-cost-model`

### `distributed-training/checkpointing`

- `distributed-state-dict`
- `sharded-optimizer-state`
- `checkpoint-resharding`
- `load-with-different-world-size`
- `fault-tolerant-save`

### `distributed-training/elastic`

- `torchrun-rendezvous`
- `elastic-restart`
- `rank-reassignment`
- `job-preemption-recovery`

## Deployment

`deployment` 关注容器、Kubernetes、服务框架、路由、可靠性和安全。

### `deployment/containers`

- `cuda-base-images`
- `nvidia-container-runtime`
- `dockerfile-for-training`
- `dockerfile-for-serving`
- `image-size-and-cache`
- `reproducible-builds`

### `deployment/kubernetes`

- `pod-gpu-request`
- `device-plugin`
- `gpu-operator`
- `node-selector-and-taints`
- `mig-partitioning`
- `gpu-time-slicing`
- `hpa-and-keda`
- `volumes-and-model-cache`

### `deployment/serving`

- `ray-serve-dynamic-batching`
- `kserve-inferenceservice`
- `triton-inference-server`
- `vllm-production-server`
- `sglang-production-server`
- `model-warmup-and-reload`

### `deployment/routing`

- `load-balancing`
- `session-affinity`
- `prefix-aware-routing`
- `canary-release`
- `ab-testing`
- `shadow-traffic`

### `deployment/reliability`

- `health-checks`
- `readiness-vs-liveness`
- `graceful-shutdown`
- `request-draining`
- `retries-and-circuit-breaker`
- `autoscaling-stability`
- `oom-recovery`

### `deployment/security`

- `secrets-management`
- `authn-authz`
- `network-policy`
- `image-supply-chain`
- `model-artifact-permissions`

## Storage

### `storage/model-artifacts`

- `safetensors`
- `gguf`
- `hf-cache-layout`
- `tokenizer-files`
- `model-config-and-generation-config`
- `weight-sharding`

### `storage/datasets`

- `jsonl`
- `parquet`
- `arrow`
- `mmap`
- `webdataset`
- `streaming-dataset`
- `data-versioning`

### `storage/checkpoint`

- `single-rank-checkpoint`
- `distributed-checkpoint`
- `optimizer-state-layout`
- `rng-state`
- `checkpoint-index`
- `checkpoint-conversion`

### `storage/cache`

- `local-model-cache`
- `dataset-cache`
- `kv-cache-storage`
- `lru-eviction`
- `versioned-cache`
- `cache-consistency`

### `storage/object-store`

- `s3-oss-gcs`
- `multipart-upload`
- `retry-and-idempotency`
- `eventual-consistency`
- `throughput-tuning`

### `storage/vector-store`

- `hnsw`
- `ivf-pq`
- `flat-index`
- `metadata-filtering`
- `hybrid-search-index`

## Retrieval Recommendation

### `retrieval-recommendation/retrieval`

- `i2i`
- `u2i`
- `u2i2i`
- `vector-retrieval`
- `hybrid-retrieval`
- `query-rewrite`
- `recall-merge-dedup`

### `retrieval-recommendation/ranking`

- `pointwise-ranking`
- `pairwise-ranking`
- `listwise-ranking`
- `feature-crossing`
- `calibration`

### `retrieval-recommendation/reranking`

- `cross-encoder-rerank`
- `llm-rerank`
- `late-interaction-colbert`
- `latency-quality-tradeoff`

### `retrieval-recommendation/llm4rec`

- `prompt-based-recommendation`
- `embedding-based-recommendation`
- `session-understanding`
- `explanation-generation`
- `tool-augmented-rec`

### `retrieval-recommendation/feature-store`

- `offline-online-consistency`
- `feature-freshness`
- `backfill`
- `point-in-time-correctness`
- `embedding-store`

### `retrieval-recommendation/online-serving`

- `candidate-generation-service`
- `ranking-service`
- `feature-fetch-service`
- `ab-experimentation`
- `latency-budget`

## Observability Benchmarking

### `observability-benchmarking/metrics`

- `ttft`
- `tpot`
- `token-throughput`
- `qps-concurrency`
- `gpu-utilization`
- `gpu-memory`
- `queue-depth`
- `cost-per-million-tokens`

### `observability-benchmarking/tracing`

- `opentelemetry`
- `request-id-propagation`
- `distributed-trace`
- `trace-sampling`
- `span-design-for-llm-serving`

### `observability-benchmarking/logging`

- `structured-logs`
- `error-taxonomy`
- `slow-request-log`
- `audit-log`

### `observability-benchmarking/load-testing`

- `wrk-hey-basics`
- `locust-scenarios`
- `genai-perf`
- `benchmark-serving-script`
- `latency-throughput-curve`
- `burst-and-soak-test`

### `observability-benchmarking/profiling`

- `pytorch-profiler`
- `nsight-systems`
- `nsight-compute`
- `memory-snapshot`
- `flamegraph`

### `observability-benchmarking/regression`

- `pytest-benchmark`
- `golden-output-correctness`
- `performance-baseline`
- `variance-and-warmup`
- `quality-regression-gate`

## Systems Tooling

`systems-tooling` 是学习工作流支撑目录，不是 AI infra 技术主干，但它保证 Mac/WSL2/GitHub/Obsidian/远程 GPU 能协作。

- `environment/uv-conda-pip`, `environment/cuda-wheel-compatibility`, `environment/requirements-locking`, `environment/pre-commit`
- `wsl2/nvidia-smi-and-driver`, `wsl2/torch-cuda-check`, `wsl2/triton-check`, `wsl2/windows-filesystem-boundary`, `wsl2/remote-wsl-development`
- `github/branch-per-learning-item`, `github/commit-message-style`, `github/pr-learning-log`, `github/issue-backlog`
- `obsidian/wikilinks`, `obsidian/frontmatter`, `obsidian/tags`, `obsidian/backlinks-to-code`
- `remote-gpu/ssh-config`, `remote-gpu/rsync-artifacts`, `remote-gpu/run-log-collection`, `remote-gpu/tmux-long-running-jobs`
- `metadata/experiment-yaml-schema`, `metadata/hardware-inventory`, `metadata/software-version-capture`, `metadata/result-table-schema`
