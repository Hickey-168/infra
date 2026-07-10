# Observability Benchmarking

这个目录贯穿所有分支。infra 学习必须能回答：我怎么知道它真的更快、更省、更稳定，还是只是看起来如此？

## Structure

- `metrics`: TTFT、TPOT、token/s、QPS、GPU utilization、memory、cost。
- `profiling`: PyTorch Profiler、Nsight Systems、Nsight Compute。
- `load-testing`: latency/throughput curves、p50/p95/p99、burst。
- `tracing`: OpenTelemetry、request id、跨服务链路。
- `regression`: benchmark harness、baseline、variance、quality gates。

## Exit Bar

你应该能从一次 profiler trace 中提出瓶颈假设，做一个最小改动验证，并说明实验不能证明什么。

