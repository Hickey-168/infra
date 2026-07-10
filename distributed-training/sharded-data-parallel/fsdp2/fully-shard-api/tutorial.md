---
artifact: curriculum-tutorial
leaf: distributed-training/sharded-data-parallel/fsdp2/fully-shard-api
priority: P0
environment: WSL2-GPU
---

# FSDP2 fully_shard：把“完整副本常驻”改成按需 materialize

## 问题

DDP 的每个 rank 都保留完整 parameters、gradients 和 optimizer states。模型一大，单卡首先被复制的训练状态撑爆，即使总集群显存很大也无济于事。FSDP 的核心不是压缩，而是分片所有权与按需通信。

## 因果链

1. `fully_shard` 将参数等训练相关状态分布到 data-parallel ranks；稳态下每 rank 只拥有自己的 shard。
2. 某个 module 要执行 forward 时，runtime all-gather 需要的完整参数，使本模块暂时 materialize full parameter。
3. backward 计算局部 gradients 后，reduce-scatter 将聚合梯度直接分回 shards；根据 reshard 策略，完整参数在不再需要时释放或重新分片。
4. optimizer 在本地 shard 上更新。于是 peak memory 取决于“同时 materialize 了哪些 modules、activations、prefetch 和通信 buffer”，而非只看总参数量。

## 必须追的状态表

| 时刻 | parameter | gradient | optimizer state | collective |
|---|---|---|---|---|
| 空闲/稳态 | shard | 无或 shard | shard | 无 |
| module forward 前 | 正在 all-gather | 无 | shard | all-gather |
| module forward | 暂时 full | 无 | shard | 无 |
| backward 后 | 可 reshard | local -> reduced shard | shard | reduce-scatter |

具体实现和配置的精确时序可能变化；实验应以当前 PyTorch FSDP2 文档/trace 为准，但“谁拥有 full state、何时释放”是不变的学习主线。

## 预测后验证

在相同 tiny model、global batch、dtype 下比较 single-process/DDP/FSDP2：记录 peak allocated memory、每 rank parameter bytes、step time 与通信量。先预测 FSDP 哪一项下降、哪一项不会自动消失（activation），再用 memory snapshot/profiler 验证。world size 为 1 的 FSDP 不能证明分片收益；至少使用两 GPU。不要把一次 tiny-model 速度结果泛化到大模型。

## P0 终点与 P1 分叉

到这里你应能串起 tensor bytes -> activation lifetime -> GPU kernel -> communication -> sharded ownership。下一阶段再进入 tensor parallel、ZeRO、pipeline parallel、checkpoint resharding，以及 vLLM/PagedAttention 等框架实现；它们会复用这条语言，而不是取代它。
