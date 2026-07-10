# Inference

推理目录关注 LLM serving 的端到端路径：请求进入、tokenize、prefill、decode、KV cache、调度、采样、流式返回、压测和优化。

## Structure

- `decoding`: greedy、sampling、beam search、speculative decoding、guided decoding。
- `kv-cache`: KV cache layout、PagedAttention、prefix cache、chunked prefill。
- `scheduling`: continuous batching、admission control、preemption、fairness。
- `frameworks`: vLLM、SGLang、TensorRT-LLM、TGI，只作为案例入口。
- `api`: OpenAI-compatible API、SSE streaming、cancellation、timeout。

## Priority Notes

先学机制，再学框架。`vLLM` 的优先级来自 PagedAttention、continuous batching、prefix cache 等机制；`SGLang` 的优先级来自 RadixAttention 和结构化程序执行；`beam-search` 是重要 decoding 基础，但不如 KV cache 和 scheduler 通用。

