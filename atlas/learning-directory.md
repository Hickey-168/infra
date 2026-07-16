# Learning Directory 学习目录

这个文件和真实目录结构对应。一级目录是学习主干；二级目录是脑图中的大分支；更深层目录是可以独立学习和验证的机制叶子。

叶子目录原则：

- 不用框架名当最终叶子，框架名只作为案例入口。
- 叶子必须能设计实验、toy implementation、trace、benchmark 或 profiler 验证。
- 优先学习跨框架机制，再学习框架实现差异。

> 注：所有反引号包裹的路径与叶子名（如 `fundamentals/pytorch-tensor-lifecycle`、`stride-layout`）都是磁盘上的真实目录名，保持英文原样，请勿翻译；冒号后为中文说明。

## Fundamentals 基础

`fundamentals` 是支撑目录，用来保证后续推理、训练、算子、通信都能落到 shape/state/memory/call chain 上。

### `fundamentals/pytorch-tensor-lifecycle` 张量生命周期

- `shape-dtype-device`: 张量的形状、数据类型、所在设备三要素。
- `stride-layout`: stride 如何决定内存布局与元素寻址。
- `view-vs-reshape`: view 与 reshape 何时共享存储、何时拷贝。
- `contiguous-memory`: 连续内存的含义与 contiguous 的触发时机。
- `storage-aliasing`: 多个张量共享同一 storage 的别名与副作用。

### `fundamentals/autograd-graph-and-saved-tensors` 自动求导图与保存张量

- `grad-fn-chain`: 反向图中 grad_fn 节点串成的链路。
- `saved-tensors-hooks`: 前向为反向保存哪些张量，如何用 hook 干预。
- `no-grad-vs-inference-mode`: no_grad 与 inference_mode 的差异与适用场景。
- `checkpoint-recompute`: 用重算换显存的 checkpoint 机制。

### `fundamentals/module-forward-call-chain` 模块前向调用链

- `module-call`: nn.Module 的 __call__ 到 forward 的调用路径。
- `forward-hooks`: 前向 hook 的注册与拦截时机。
- `backward-hooks`: 反向 hook 观察/修改梯度。
- `parameter-registration`: 参数与 buffer 的注册和遍历。

### `fundamentals/memory-accounting-single-gpu` 单卡显存核算

- `parameter-gradient-optimizer-activation`: 参数/梯度/优化器状态/激活四项显存账本。
- `cuda-allocated-vs-reserved`: allocated 与 reserved 的区别，解释显存为何“用不掉”。
- `peak-memory-measurement`: 峰值显存的测量方法。

### `fundamentals/linux-process-and-file-io` Linux 进程与文件 IO

- `process-env-fd`: 进程、环境变量、文件描述符基础。
- `mmap-page-cache`: mmap 与 page cache 的读写路径。
- `async-file-io`: 异步文件 IO 的模型。

### `fundamentals/python-async-and-queueing` Python 异步与队列

- `asyncio-task-queue`: asyncio 任务与队列调度。
- `producer-consumer`: 生产者-消费者模型。
- `backpressure-timeout`: 背压与超时控制。

## Inference 推理

`inference` 关注 LLM 请求从 API 进入，到 tokenize、prefill、decode、KV cache、调度、采样、返回的全过程。

### `inference/lifecycle` 请求生命周期

- `tokenization-and-chat-template`: chat template、special tokens、prompt 组装边界。
- `transformer-forward-shapes`: batch/seq/hidden/head 形状流。
- `prefill-vs-decode`: prefill compute-bound，decode memory/cache-bound 的差异。
- `ttft-vs-tpot`: 首 token 延迟和每 token 延迟的指标拆分。
- `request-state-machine`: waiting/running/swapped/finished 状态迁移。

### `inference/decoding` 解码

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

### `inference/kv-cache` KV 缓存

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

### `inference/scheduling` 调度

- `static-batching`: 固定 batch 的空洞和等待成本。
- `dynamic-batching`: batching window 的吞吐/延迟 trade-off。
- `continuous-batching`: iteration-level batching。
- `chunked-prefill`: 长 prompt prefill 切块和 decode 混排。
- `decode-priority`: decode 优先如何影响 TPOT。
- `preemption-and-swapping`: 请求抢占、swap、恢复。
- `admission-control`: 控制并发保护 p99 和 OOM。
- `fairness-and-starvation`: 长短请求混合下的公平性。
- `queueing-latency-model`: 用排队模型解释吞吐/延迟曲线。

### `inference/attention-backends` Attention 后端

- `flash-attention-prefill`: prefill 阶段的 full attention kernel。
- `paged-attention-decode`: decode 阶段单 token query 对 paged KV 的读取。
- `flashinfer-backend`: serving attention backend 的接口边界。
- `xformers-backend`: 通用 attention backend 对比。
- `rope-and-position-ids`: RoPE、position ids、cache 位置的一致性。
- `mla-attention-deepseek`: MLA 对 KV cache 形状和容量的影响。

### `inference/frameworks` 推理框架

- `vllm/scheduler`, `vllm/paged-attention`, `vllm/prefix-caching`, `vllm/chunked-prefill`, `vllm/speculative-decoding`, `vllm/torch-compile`, `vllm/disaggregated-prefill-decode`: vLLM 的调度、分页注意力、前缀缓存、分块预填充、投机解码、torch.compile、PD 分离等实现。
- `sglang/radix-attention`, `sglang/structured-generation`, `sglang/speculative-decoding`, `sglang/constrained-decoding`: SGLang 的 RadixAttention、结构化生成、投机解码、约束解码。
- `tensorrt-llm/inflight-batching`, `tensorrt-llm/paged-kv-cache`, `tensorrt-llm/quantization`, `tensorrt-llm/multi-gpu-inference`: TensorRT-LLM 的 inflight batching、分页 KV、量化、多卡推理。
- `tgi/continuous-batching`: TGI 的连续批处理实现。
- `llama-cpp/gguf-and-cpu-offload`: llama.cpp 的 GGUF 格式与 CPU offload。

### `inference/serving-api` 服务 API

- `openai-compatible-chat-api`: request/response/schema。
- `streaming-sse`: Server-Sent Events 和 token streaming。
- `cancellation-and-timeout`: 客户端断开和服务端清理。
- `tool-calling-protocol`: tool calls 的结构和状态。
- `error-and-retry-semantics`: retry 是否安全、幂等边界。
- `tokenizer-server-boundary`: tokenizer 放客户端/服务端的取舍。

### `inference/distributed-inference` 分布式推理

- `tensor-parallel-inference`: 张量并行推理。
- `pipeline-parallel-inference`: 流水线并行推理。
- `data-parallel-replicas`: 数据并行副本。
- `expert-parallel-inference`: 专家并行推理。
- `disaggregated-prefill-decode`: 预填充-解码分离部署。
- `prefix-aware-routing`: 前缀感知路由。
- `kv-cache-aware-load-balancing`: KV 缓存感知的负载均衡。
- `multi-node-serving`: 多节点服务。

### `inference/quantization` 推理量化

- `weight-only-int8`, `weight-only-int4`, `gptq`, `awq`, `smoothquant`: 权重量化路线（INT8/INT4、GPTQ、AWQ、SmoothQuant）。
- `fp8-inference`, `nvfp4-mxfp4`, `activation-quantization`, `kv-cache-quantization`: FP8 推理、NVFP4/MXFP4、激活量化、KV 量化。

### `inference/multimodal` 多模态

- `vision-token-prefill`: 视觉 token 的预填充。
- `encoder-decoder-cross-attention-cache`: 编解码交叉注意力缓存。
- `image-video-batching`: 图像/视频批处理。
- `multimodal-projector-cache`: 多模态投影层缓存。

## Training 训练

`training` 关注单机和基础训练机制，分布式拆到 `distributed-training`。

### `training/single-gpu-loop` 单卡训练循环

- `forward-backward-step`: 前向-反向-参数更新的一步。
- `loss-scaling-and-grad-norm`: 混合精度 loss scaling 与梯度范数。
- `gradient-accumulation`: 梯度累积等效放大 batch。
- `optimizer-step`: 优化器 step 的状态更新。
- `learning-rate-scheduler`: 学习率调度策略。
- `gradient-clipping`: 梯度裁剪防爆炸。
- `training-loop-state-machine`: 训练循环的状态机。
- `training-hyperparameters`: global batch、sequence length、learning rate、weight decay、warmup 与 accumulation 如何共同改变训练循环、资源和优化轨迹。

### `training/memory` 训练显存

- `parameter-memory`: 参数显存占用。
- `gradient-memory`: 梯度显存占用。
- `optimizer-state-memory`: 优化器状态显存（如 Adam 的一阶/二阶矩）。
- `activation-memory`: 激活显存，训练显存的大头。
- `activation-checkpointing`: 激活重算换显存。
- `cuda-caching-allocator`: CUDA 缓存分配器行为。
- `cpu-nvme-offload`: 显存到 CPU/NVMe 的 offload。
- `memory-snapshot-analysis`: 显存快照分析定位占用。

### `training/precision` 训练精度

- `fp32-and-tf32`: FP32 与 TF32 的精度/性能取舍。
- `bf16-vs-fp16`: BF16 与 FP16 的动态范围与精度差异。
- `loss-scaling`: FP16 训练的 loss scaling。
- `fp8-transformer-engine`: 基于 Transformer Engine 的 FP8 训练。
- `delayed-scaling`: FP8 延迟缩放策略。
- `mxfp8-nvfp4`: MXFP8/NVFP4 新型低精度格式。
- `determinism-and-numerical-drift`: 确定性与数值漂移。

### `training/data-pipeline` 数据管线

- `tokenization-throughput`: tokenize 吞吐瓶颈。
- `sequence-packing`: 序列打包减少 padding 浪费。
- `document-shuffling`: 文档级 shuffle。
- `dataloader-workers`: dataloader 多进程 worker。
- `pinned-memory-prefetch`: pinned memory 与预取。
- `jsonl-parquet-arrow`: JSONL/Parquet/Arrow 数据格式。
- `webdataset-streaming`: webdataset 流式读取。

### `training/checkpointing` 检查点

- `model-state-dict`: 模型 state_dict 结构。
- `optimizer-state-dict`: 优化器 state_dict 结构。
- `rng-and-scheduler-state`: RNG 与调度器状态保存。
- `resume-loss-continuity`: 恢复训练时 loss 的连续性。
- `distributed-checkpoint`: 分布式 checkpoint。
- `checkpoint-resharding`: 不同并行度间的 checkpoint 重切分。
- `checkpoint-conversion-hf-megatron`: HF 与 Megatron 格式互转。

### `training/finetuning` 微调

- `sft-data-collator`: SFT 数据 collator。
- `lora-low-rank-update`: LoRA 低秩更新。
- `qlora-and-nf4`: QLoRA 与 NF4 量化。
- `dpo-loss-shapes`: DPO 损失的张量形状。
- `ppo-rlhf-memory-map`: PPO/RLHF 的显存分布。
- `evaluation-regression-gate`: 评估回归门禁。

### `training/compiler-runtime` 编译器运行时

- `torch-compile-training`: 训练态的 torch.compile。
- `graph-breaks`: graph break 成因与影响。
- `aot-autograd`: AOT autograd 的前反向图捕获。
- `inductor-generated-kernels`: Inductor 生成的 kernel。
- `cuda-graphs-training`: 训练中使用 CUDA Graphs。
- `dynamic-shapes`: 动态 shape 的编译处理。

### `training/frameworks` 训练框架

- `pytorch-native`: PyTorch 原生训练。
- `torchtitan`: torchtitan 训练栈。
- `minitorch-or-tinygrad`: minitorch/tinygrad 教学式实现。
- `deepspeed`: DeepSpeed。
- `megatron-lm`: Megatron-LM。
- `huggingface-accelerate`: HF Accelerate。
- `lightning-fabric`: Lightning Fabric。

## Operators 算子

`operators` 对应“算子”，覆盖 CUDA/Triton/attention/fusion/compiler/profiling。

### `operators/cuda` CUDA

- `execution-model/thread-block-grid`: thread/block/grid 执行模型。
- `execution-model/warps-and-schedulers`: warp 与硬件调度器。
- `memory-hierarchy/global-memory-coalescing`: 全局内存访问合并。
- `memory-hierarchy/shared-memory-tiling`: shared memory 分块。
- `memory-hierarchy/register-pressure`: 寄存器压力对占用率的影响。
- `memory-hierarchy/l2-cache`: L2 cache 行为。
- `streams-events`: stream 与 event 的并发控制。
- `cuda-graphs`: CUDA Graphs 降低 launch 开销。
- `occupancy`: 占用率与性能关系。
- `atomics-and-reductions`: 原子操作与规约。
- `warp-level-primitives`: warp 级原语（shuffle 等）。
- `tensor-cores-and-mma`: Tensor Core 与 MMA 指令。
- `async-copy-and-tma`: 异步拷贝与 TMA。

### `operators/triton` Triton

- `programming-model/program-id-and-blocks`: program id 与 block 的编程模型。
- `programming-model/masks-and-boundary-checks`: mask 与边界检查。
- `vector-add`: 入门 vector add kernel。
- `reductions`: 规约类 kernel。
- `fused-softmax`: 融合 softmax。
- `matmul/tiled-matmul`: 分块矩阵乘。
- `matmul/autotune`: matmul 自动调参。
- `matmul/persistent-matmul`: persistent 风格的 matmul。
- `block-pointers`: block pointer 用法。
- `tl-dot`: tl.dot 的使用与约束。
- `debugging-and-profiling`: Triton 调试与 profiling。
- `fp8-matmul`: FP8 matmul。

### `operators/attention` 注意力算子

- `standard-attention`: 标准 attention 实现。
- `online-softmax`: online softmax，FlashAttention 的核心技巧。
- `flash-attention-v1`: FlashAttention v1。
- `flash-attention-v2`: FlashAttention v2。
- `flash-attention-v3`: FlashAttention v3（Hopper 优化）。
- `paged-attention-kernel`: 分页注意力 kernel。
- `decode-attention-with-kv-cache`: 带 KV cache 的 decode attention。
- `causal-mask`: 因果 mask。
- `alibi-and-rope`: ALiBi 与 RoPE 位置编码。
- `rmsnorm`: RMSNorm 算子。
- `layernorm`: LayerNorm 算子。

### `operators/fusion` 算子融合

- `elementwise-fusion`: 逐元素融合。
- `bias-gelu-fusion`: bias + GELU 融合。
- `dropout-add-layernorm`: dropout-add-layernorm 融合。
- `gemm-epilogue-fusion`: GEMM epilogue 融合。
- `rmsnorm-fusion`: RMSNorm 融合。
- `moe-fused-kernels`: MoE 融合 kernel。
- `rope-fusion`: RoPE 融合。

### `operators/libraries` 算子库

- `cublas`: cuBLAS。
- `cudnn`: cuDNN。
- `cutlass/gemm-tiling`: CUTLASS 的 GEMM 分块。
- `cutlass/epilogue`: CUTLASS epilogue。
- `flashinfer`: FlashInfer 库。
- `torchao`: torchao 量化/优化库。
- `transformer-engine`: Transformer Engine。
- `xformers`: xFormers。

### `operators/compiler` 编译器

- `pytorch-dispatcher-aten`: PyTorch dispatcher 与 ATen。
- `torch-dynamo`: TorchDynamo 图捕获。
- `aot-autograd`: AOT autograd。
- `torch-inductor`: TorchInductor 后端。
- `triton-lowering`: 到 Triton 的 lowering。
- `graph-break-debugging`: graph break 调试。
- `custom-pytorch-op`: 自定义 PyTorch 算子。
- `onnx-export`: ONNX 导出。
- `tensorrt-engine-build`: TensorRT engine 构建。

### `operators/profiling` 算子性能剖析

- `microbenchmark-harness`: 微基准测试框架。
- `roofline-model`: Roofline 模型判断瓶颈类型。
- `pytorch-profiler`: PyTorch profiler。
- `nsight-systems`: Nsight Systems 系统级 timeline。
- `nsight-compute`: Nsight Compute 单 kernel 深剖。
- `kernel-metrics-occupancy`: kernel 占用率指标。
- `kernel-metrics-memory-throughput`: kernel 内存吞吐指标。

## Communication 通信

`communication` 是分布式训练、MoE、分布式推理的共同底座。

### `communication/collectives` 集合通信原语

- `broadcast`, `reduce`, `all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`, `scatter-gather`, `barrier`: 常见集合通信原语（广播、规约、全规约、全收集、规约散射、全交换、散射/收集、栅栏）。

### `communication/algorithms` 通信算法

- `ring-allreduce`: 环形 all-reduce。
- `tree-allreduce`: 树形 all-reduce。
- `hierarchical-allreduce`: 分层 all-reduce。
- `recursive-doubling`: 递归倍增算法。
- `reduce-scatter-allgather`: reduce-scatter + all-gather 组合。
- `chunking-and-pipelining`: 分块与流水线化。

### `communication/nccl` NCCL

- `process-group-nccl`: NCCL process group。
- `nccl-tests`: nccl-tests 基准。
- `nccl-debug-logs`: NCCL debug 日志解读。
- `nccl-env-vars`: NCCL 关键环境变量。
- `topology-discovery`: 拓扑发现。
- `nccl-ras`: NCCL RAS 可靠性特性。
- `nccl-timeout-debugging`: NCCL 超时调试。

### `communication/overlap` 计算通信重叠

- `ddp-gradient-buckets`: DDP 梯度分桶。
- `fsdp-prefetch`: FSDP 参数预取。
- `async-collectives`: 异步集合通信。
- `compute-communication-overlap`: 计算与通信重叠。
- `stream-priority`: stream 优先级调度。
- `bucket-size-tuning`: bucket 大小调优。

### `communication/topology` 硬件拓扑

- `pcie`: PCIe 互连。
- `nvlink`: NVLink 互连。
- `nvls`: NVLink SHARP（NVLS）。
- `infiniband-rdma`: InfiniBand RDMA。
- `ethernet-roce`: 以太网 RoCE。
- `numa-cpu-affinity`: NUMA 与 CPU 亲和性。
- `multi-node-network-model`: 多节点网络性能模型。

### `communication/failure-debugging` 故障调试

- `collective-mismatch`: 集合通信不匹配。
- `hanging-rank`: 挂起的 rank 定位。
- `timeout-retry`: 超时与重试。
- `shape-mismatch`: 通信张量 shape 不匹配。
- `rank-to-device-mapping`: rank 到 device 的映射错误。

## Distributed Training 分布式训练

`distributed-training` 关注并行策略、rank group、通信量、显存切分、checkpoint 和容错。

### `distributed-training/data-parallel` 数据并行

- `ddp-gradient-allreduce`: DDP 梯度 all-reduce。
- `ddp-bucketization`: DDP 梯度分桶。
- `no-sync-gradient-accumulation`: no_sync 下的梯度累积。
- `gradient-as-bucket-view`: gradient-as-bucket-view 优化。
- `static-graph-ddp`: static graph DDP。

### `distributed-training/sharded-data-parallel` 切片数据并行

- `fsdp1`: FSDP1。
- `fsdp2/fully-shard-api`: FSDP2 的 fully_shard API。
- `fsdp2/dtensor-placement`: FSDP2 的 DTensor placement。
- `zero-stage-1`: ZeRO stage 1（切分优化器状态）。
- `zero-stage-2`: ZeRO stage 2（再切分梯度）。
- `zero-stage-3`: ZeRO stage 3（再切分参数）。
- `zero-offload`: ZeRO offload 到 CPU/NVMe。
- `hybrid-shard`: 混合切分策略。
- `reshard-after-forward`: 前向后 reshard。

### `distributed-training/tensor-parallel` 张量并行

- `column-parallel-linear`: 列并行 Linear。
- `row-parallel-linear`: 行并行 Linear。
- `vocab-parallel-embedding`: 词表并行 embedding。
- `vocab-parallel-cross-entropy`: 词表并行交叉熵。
- `sequence-parallel-layernorm`: 序列并行 layernorm。
- `tensor-parallel-rank-groups`: 张量并行的 rank group 划分。

### `distributed-training/pipeline-parallel` 流水线并行

- `gpipe-schedule`: GPipe 调度。
- `one-forward-one-backward`: 1F1B 调度。
- `interleaved-pipeline`: 交错式流水线。
- `bubble-ratio`: 流水线气泡比例。
- `microbatching`: 微批处理。
- `activation-send-recv`: 阶段间激活的 send/recv。
- `virtual-pipeline-stages`: 虚拟流水线阶段。

### `distributed-training/context-parallel` 上下文并行

- `sequence-splitting`: 序列切分。
- `ring-attention`: ring attention。
- `long-context-training`: 长上下文训练。
- `context-parallel-allgather`: 上下文并行的 all-gather。
- `ulysses-style-attention`: Ulysses 式 attention。

### `distributed-training/expert-parallel` 专家并行

- `moe-router-topk`: MoE router 的 top-k 选择。
- `capacity-factor`: 专家容量因子。
- `token-dispatch-alltoall`: token 分发的 all-to-all。
- `expert-load-balancing`: 专家负载均衡。
- `shared-experts`: 共享专家。
- `moe-gemm-grouping`: MoE 的 grouped GEMM。

### `distributed-training/composition` 并行组合

- `dp-tp-pp-3d-parallelism`: DP/TP/PP 三维并行。
- `tp-pp-dp-rank-mapping`: TP/PP/DP 的 rank 映射。
- `fsdp-plus-tp`: FSDP 与 TP 组合。
- `megatron-rank-groups`: Megatron 的 rank group 组织。
- `deepspeed-megatron-composition`: DeepSpeed 与 Megatron 组合。
- `parallelism-cost-model`: 并行策略的成本模型。

### `distributed-training/checkpointing` 分布式检查点

- `distributed-state-dict`: 分布式 state_dict。
- `sharded-optimizer-state`: 切片优化器状态。
- `checkpoint-resharding`: checkpoint 重切分。
- `load-with-different-world-size`: 用不同 world size 加载。
- `fault-tolerant-save`: 容错保存。

### `distributed-training/elastic` 弹性训练

- `torchrun-rendezvous`: torchrun rendezvous 机制。
- `elastic-restart`: 弹性重启。
- `rank-reassignment`: rank 重分配。
- `job-preemption-recovery`: 任务抢占后的恢复。

## Deployment 部署

`deployment` 关注容器、Kubernetes、服务框架、路由、可靠性和安全。

### `deployment/containers` 容器

- `cuda-base-images`: CUDA 基础镜像。
- `nvidia-container-runtime`: NVIDIA container runtime。
- `dockerfile-for-training`: 训练用 Dockerfile。
- `dockerfile-for-serving`: 服务用 Dockerfile。
- `image-size-and-cache`: 镜像大小与构建缓存。
- `reproducible-builds`: 可复现构建。

### `deployment/kubernetes` Kubernetes

- `pod-gpu-request`: Pod 的 GPU 资源请求。
- `device-plugin`: GPU device plugin。
- `gpu-operator`: NVIDIA GPU Operator。
- `node-selector-and-taints`: node selector 与 taints。
- `mig-partitioning`: MIG 多实例切分。
- `gpu-time-slicing`: GPU 时间片共享。
- `hpa-and-keda`: HPA 与 KEDA 自动扩缩。
- `volumes-and-model-cache`: volume 与模型缓存挂载。

### `deployment/serving` 服务框架

- `ray-serve-dynamic-batching`: Ray Serve 动态批处理。
- `kserve-inferenceservice`: KServe InferenceService。
- `triton-inference-server`: Triton Inference Server。
- `vllm-production-server`: vLLM 生产部署。
- `sglang-production-server`: SGLang 生产部署。
- `model-warmup-and-reload`: 模型预热与热重载。

### `deployment/routing` 路由

- `load-balancing`: 负载均衡。
- `session-affinity`: 会话亲和。
- `prefix-aware-routing`: 前缀感知路由。
- `canary-release`: 金丝雀发布。
- `ab-testing`: A/B 测试。
- `shadow-traffic`: 影子流量。

### `deployment/reliability` 可靠性

- `health-checks`: 健康检查。
- `readiness-vs-liveness`: readiness 与 liveness 探针。
- `graceful-shutdown`: 优雅关闭。
- `request-draining`: 请求排空。
- `retries-and-circuit-breaker`: 重试与熔断。
- `autoscaling-stability`: 自动扩缩容稳定性。
- `oom-recovery`: OOM 恢复。

### `deployment/security` 安全

- `secrets-management`: 密钥管理。
- `authn-authz`: 认证与授权。
- `network-policy`: 网络策略。
- `image-supply-chain`: 镜像供应链安全。
- `model-artifact-permissions`: 模型产物权限。

## Storage 存储

### `storage/model-artifacts` 模型产物

- `safetensors`: safetensors 格式。
- `gguf`: GGUF 格式。
- `hf-cache-layout`: HuggingFace cache 布局。
- `tokenizer-files`: tokenizer 相关文件。
- `model-config-and-generation-config`: 模型 config 与生成 config。
- `weight-sharding`: 权重分片。

### `storage/datasets` 数据集

- `jsonl`: JSONL 格式。
- `parquet`: Parquet 格式。
- `arrow`: Arrow 格式。
- `mmap`: mmap 内存映射读取。
- `webdataset`: webdataset 格式。
- `streaming-dataset`: 流式数据集。
- `data-versioning`: 数据版本化。

### `storage/checkpoint` 检查点存储

- `single-rank-checkpoint`: 单 rank checkpoint。
- `distributed-checkpoint`: 分布式 checkpoint。
- `optimizer-state-layout`: 优化器状态布局。
- `rng-state`: RNG 状态。
- `checkpoint-index`: checkpoint 索引。
- `checkpoint-conversion`: checkpoint 格式转换。

### `storage/cache` 缓存

- `local-model-cache`: 本地模型缓存。
- `dataset-cache`: 数据集缓存。
- `kv-cache-storage`: KV cache 落盘存储。
- `lru-eviction`: LRU 淘汰。
- `versioned-cache`: 版本化缓存。
- `cache-consistency`: 缓存一致性。

### `storage/object-store` 对象存储

- `s3-oss-gcs`: S3/OSS/GCS 对象存储。
- `multipart-upload`: 分段上传。
- `retry-and-idempotency`: 重试与幂等。
- `eventual-consistency`: 最终一致性。
- `throughput-tuning`: 吞吐调优。

### `storage/vector-store` 向量存储

- `hnsw`: HNSW 索引。
- `ivf-pq`: IVF-PQ 索引。
- `flat-index`: flat 暴力索引。
- `metadata-filtering`: 元数据过滤。
- `hybrid-search-index`: 混合检索索引。

## Retrieval Recommendation 检索推荐

### `retrieval-recommendation/retrieval` 检索

- `i2i`: item-to-item 检索。
- `u2i`: user-to-item 检索。
- `u2i2i`: user-to-item-to-item 检索。
- `vector-retrieval`: 向量检索。
- `hybrid-retrieval`: 混合检索。
- `query-rewrite`: query 改写。
- `recall-merge-dedup`: 召回合并去重。

### `retrieval-recommendation/ranking` 排序

- `pointwise-ranking`: pointwise 排序。
- `pairwise-ranking`: pairwise 排序。
- `listwise-ranking`: listwise 排序。
- `feature-crossing`: 特征交叉。
- `calibration`: 打分校准。
- `weight-norm-regularization`: 用权重范数约束排序/召回模型的复杂度，连接 L1/L2、weight decay 与 embedding 打分。

### `retrieval-recommendation/reranking` 重排

- `cross-encoder-rerank`: cross-encoder 重排。
- `llm-rerank`: LLM 重排。
- `late-interaction-colbert`: ColBERT 式 late interaction。
- `latency-quality-tradeoff`: 延迟-质量权衡。

### `retrieval-recommendation/llm4rec` LLM 推荐

- `prompt-based-recommendation`: prompt 式推荐。
- `embedding-based-recommendation`: embedding 式推荐。
- `session-understanding`: 会话理解。
- `explanation-generation`: 推荐解释生成。
- `tool-augmented-rec`: 工具增强推荐。

### `retrieval-recommendation/feature-store` 特征存储

- `offline-online-consistency`: 离线在线一致性。
- `feature-freshness`: 特征新鲜度。
- `backfill`: 特征回填。
- `point-in-time-correctness`: point-in-time 正确性。
- `embedding-store`: embedding 存储。

### `retrieval-recommendation/online-serving` 在线服务

- `candidate-generation-service`: 候选生成服务。
- `ranking-service`: 排序服务。
- `feature-fetch-service`: 特征拉取服务。
- `ab-experimentation`: A/B 实验。
- `latency-budget`: 延迟预算。

## Observability Benchmarking 可观测与压测

### `observability-benchmarking/metrics` 指标

- `ttft`: 首 token 延迟。
- `tpot`: 每 token 延迟。
- `token-throughput`: token 吞吐。
- `qps-concurrency`: QPS 与并发。
- `gpu-utilization`: GPU 利用率。
- `gpu-memory`: GPU 显存占用。
- `queue-depth`: 队列深度。
- `cost-per-million-tokens`: 每百万 token 成本。

### `observability-benchmarking/tracing` 链路追踪

- `opentelemetry`: OpenTelemetry。
- `request-id-propagation`: request-id 传播。
- `distributed-trace`: 分布式 trace。
- `trace-sampling`: trace 采样。
- `span-design-for-llm-serving`: LLM serving 的 span 设计。

### `observability-benchmarking/logging` 日志

- `structured-logs`: 结构化日志。
- `error-taxonomy`: 错误分类。
- `slow-request-log`: 慢请求日志。
- `audit-log`: 审计日志。

### `observability-benchmarking/load-testing` 压测

- `wrk-hey-basics`: wrk/hey 基础用法。
- `locust-scenarios`: locust 场景编写。
- `genai-perf`: genai-perf 压测工具。
- `benchmark-serving-script`: benchmark serving 脚本。
- `latency-throughput-curve`: 延迟-吞吐曲线。
- `burst-and-soak-test`: 突发与长稳测试。

### `observability-benchmarking/profiling` 性能剖析

- `pytorch-profiler`: PyTorch profiler。
- `nsight-systems`: Nsight Systems。
- `nsight-compute`: Nsight Compute。
- `memory-snapshot`: 显存快照。
- `flamegraph`: 火焰图。

### `observability-benchmarking/regression` 回归

- `pytest-benchmark`: pytest-benchmark。
- `golden-output-correctness`: 黄金输出正确性。
- `performance-baseline`: 性能基线。
- `variance-and-warmup`: 方差与 warmup。
- `quality-regression-gate`: 质量回归门禁。

## Systems Tooling 系统工具

`systems-tooling` 是学习工作流支撑目录，不是 AI infra 技术主干，但它保证 Mac/WSL2/GitHub/Obsidian/远程 GPU 能协作。

- `environment/uv-conda-pip`, `environment/cuda-wheel-compatibility`, `environment/requirements-locking`, `environment/pre-commit`: 环境管理（uv/conda/pip、CUDA wheel 兼容性、依赖锁定、pre-commit）。
- `wsl2/nvidia-smi-and-driver`, `wsl2/torch-cuda-check`, `wsl2/triton-check`, `wsl2/windows-filesystem-boundary`, `wsl2/remote-wsl-development`: WSL2 相关（驱动/nvidia-smi、torch-cuda 自检、triton 自检、Windows 文件系统边界、Remote-WSL 开发）。
- `github/branch-per-learning-item`, `github/commit-message-style`, `github/pr-learning-log`, `github/issue-backlog`: GitHub 工作流（每学习项一分支、commit 风格、PR 学习日志、issue backlog）。
- `obsidian/wikilinks`, `obsidian/frontmatter`, `obsidian/tags`, `obsidian/backlinks-to-code`: Obsidian 笔记（wikilinks、frontmatter、tags、反链到代码）。
- `remote-gpu/ssh-config`, `remote-gpu/rsync-artifacts`, `remote-gpu/run-log-collection`, `remote-gpu/tmux-long-running-jobs`: 远程 GPU（ssh 配置、rsync 产物、运行日志收集、tmux 长任务）。
- `metadata/experiment-yaml-schema`, `metadata/hardware-inventory`, `metadata/software-version-capture`, `metadata/result-table-schema`: 元数据（实验 YAML schema、硬件清单、软件版本捕获、结果表 schema）。
