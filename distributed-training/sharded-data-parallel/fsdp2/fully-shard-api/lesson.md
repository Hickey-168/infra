---
artifact: course-lesson
leaf: distributed-training/sharded-data-parallel/fsdp2/fully-shard-api
language: zh-CN
estimated_time: 120min
environment: WSL2-GPU
---

# 课程：FSDP2 如何用通信换取训练状态显存

## 1. DDP 的复制瓶颈

DDP 每 rank 复制完整 parameters、gradients 和 optimizer states。即使集群总显存很多，单 rank 仍可能因完整副本 OOM。FSDP 的基本承诺是让这些 state 分片常驻，而在计算某模块时短暂取得需要的完整参数。

## 2. 一个 module 的生命周期

稳态下 rank 只拥有 parameter shard。module forward 前进行 all-gather，暂时 materialize full parameter；forward 后是否立刻 reshard 由策略决定。backward 得到 local gradient 后，通过 reduce-scatter 聚合并直接返回 gradient shards；optimizer 在本地 shards 上更新。关键问题永远是：此刻哪个 rank 持有 full state、它何时释放、峰值与哪些模块重叠。

## 3. 显存并没有全部消失

FSDP 主要改变 parameter/gradient/optimizer state 的复制；activation 仍随 batch/sequence 存在，all-gather/reduce-scatter 也需要通信 buffer 和时间。prefetch 可改善通信隐藏，也可能增加同时 materialize 的峰值。不要把“参数被 shard”误读成“所有 OOM 自动消失”。

## 实验课

用相同 tiny model、global batch、dtype 比较单进程、DDP、FSDP2（至少 2 GPU）。记录每 rank peak allocated memory、step time、参数 state bytes；先预测 FSDP 降低哪项、不会降低哪项。用 profiler/memory snapshot 核对 all-gather 与 reduce-scatter 的位置。world size=1 不能验证 sharding 收益。

## 通关

能画出 idle、forward、backward、optimizer 四时刻的 parameter/gradient/optimizer ownership，解释 FSDP 为什么以通信交换显存。P0 至此闭环；P1 再进入 tensor parallel、ZeRO、checkpoint resharding 和服务框架实现。
