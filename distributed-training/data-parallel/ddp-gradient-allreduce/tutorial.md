---
artifact: curriculum-tutorial
leaf: distributed-training/data-parallel/ddp-gradient-allreduce
priority: P0
environment: WSL2-GPU
---

# DDP Gradient All-Reduce：把一致性嵌进 backward 的时间线

## 问题

手动在所有 `loss.backward()` 后遍历参数并 all-reduce 虽可得到正确梯度，但会把通信推迟到计算结束，且容易遗漏平均、unused parameter、顺序和同步。DDP 的价值不仅是调用 collective，而是把通信挂到 gradient ready 的事件上。

## 因果链

1. 每 rank 保留一份模型副本并处理不同 data shard，forward 得到本地 loss。
2. backward 从末层向前传播；某个 parameter 的 gradient 一旦 ready，DDP reducer 将它放入 bucket。
3. bucket 满或满足触发条件后启动异步 all-reduce；较前层的 backward 仍可继续计算，形成潜在 overlap。
4. 在 optimizer step 前，所有需要的 buckets 必须完成规约；各 rank 因而用相同平均梯度更新同一初始参数，保持副本一致。

## 关键状态

- local gradient：尚未同步、只代表本 rank data。
- bucket：一组参数梯度的连续通信单元；bucket size 是性能权衡，不改变正确性语义。
- reduced gradient：所有 rank 聚合后可用于一致 update。
- `no_sync()`：延迟同步用于 gradient accumulation；最后一个 accumulation microbatch 必须同步。

## 预测后验证

双 GPU 跑 tiny model，固定 seed 并让两 ranks 使用不同数据。先预测：all-reduce 前后梯度、optimizer step 后 parameters 是否相同；再用 DDP communication hook/debug log 记录 bucket 触发次序。比较不同 bucket cap 的 profiler 时间线，观察“可能 overlap”而非宣称必然更快。报告要写清 world size、global batch 和 reduction 是否平均。

## 下一跳

DDP 让每 rank 保存完整训练 state。`fsdp2/fully-shard-api` 将用 all-gather/reduce-scatter 改变这些 state 的所有权，交换通信以换显存。
