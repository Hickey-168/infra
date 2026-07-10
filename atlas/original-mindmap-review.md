# Original Mind Map Review

来源：[naixu.png](/Users/youyu/workspace/python/infra/naixu.png)

## What Is Already Good

- 你已经把 `infra` 从泛泛的 AI 学习里拆出来，这是对的；infra 应该独立成库。
- `推理 / 训练 / 算子 / 通信 / 分布式训练 / 存储` 这些一级方向基本正确。
- 你把 vLLM、SGLang、TGI、Megatron 放进图里，说明已经在从“模型使用者”转向“系统机制学习者”。
- 你意识到 `profile -> 优化 -> 验证` 是闭环，这非常重要。infra 学习不能只停在概念。
- `搜推` 放在硬技能下也合理，但它更像 AI infra 的应用系统分支，应和 LLM infra 主线分开组织。

## Corrections

- `vllm`、`SGLang`、`TGI` 是框架名，不是叶子节点。叶子应该是 `paged-attention-block-table`、`continuous-batching-scheduler`、`prefix-cache-hit-rate`、`chunked-prefill-tradeoff` 这类机制。
- `beam-search` 不是 vLLM 专属技术，而是 decoding/search algorithm。它可以放在 `inference/decoding` 下，优先级低于 KV cache、batching、prefill/decode。
- `triron` 应为 `Triton`。
- `NVIDIA Nsight Systems` 是系统级 timeline profiler；还应该补 `Nsight Compute`，它更适合 kernel-level counters。
- `netron` 更像模型图/权重结构查看器，不是性能 profiler。可以放到工具层，但不能替代 PyTorch Profiler、Nsight Systems、Nsight Compute。
- `压测prof` 建议拆成 `load-generation`、`latency-throughput-curves`、`profiler-trace-reading`、`bottleneck-hypothesis`。
- `tf算子融合` 建议泛化成 `operator-fusion`。当前更常见的学习入口是 PyTorch Inductor/Triton/XLA，而不是只从 TensorFlow 学。
- `Tensor算子` 太宽，需要拆成 PyTorch dispatcher/ATen、自定义 op、layout/stride、autograd、compiler lowering。
- `分布式训练` 不只是训练框架，还包括 memory accounting、parallelism composition、通信 overlap、checkpoint resharding、fault recovery。
- `通信` 不应只记 NCCL 名字，需要拆成 collective semantics、ring/tree all-reduce、topology、bandwidth/latency model、NCCL debug。
- `存储` 需要补：dataset format、dataloader pipeline、object store、model weight format、distributed checkpoint、cache invalidation。

## Missing Branches To Add

- `fundamentals`: Linux/Python/PyTorch/GPU mental model。否则后面 CUDA/Triton/FSDP 会没有抓手。
- `deployment`: K8s/Ray/KServe/GPU Operator/路由/弹性伸缩。推理框架之外还有平台层。
- `observability-benchmarking`: 指标、trace、压测、SLO、成本模型。它贯穿所有分支。
- `quantization-and-precision`: 可放在 inference/training/operators 交叉处，但需要显式列出 FP16/BF16/FP8/INT8/INT4/KV quant。
- `compiler-stack`: torch.compile/Dynamo/AOTAutograd/Inductor/Triton/CUDA Graphs。它连接 PyTorch、算子和服务延迟。
- `evaluation-and-correctness`: 训练/推理优化后要验证数值、质量、稳定性，不只是跑得快。
- `security-and-reliability`: 权限、镜像、供应链、OOM/failure recovery，可后置但不能完全漏掉。

## Suggested Reframe

把“框架”降级为案例，把“机制”升级为叶子。

例如：

```text
inference
-> serving-frameworks
   -> vllm
      -> paged-attention
      -> scheduler
      -> prefix-caching
      -> chunked-prefill
      -> speculative-decoding-integration
```

其中真正要独立学习和验证的是最后几项，而不是 `vllm` 这个名字。
