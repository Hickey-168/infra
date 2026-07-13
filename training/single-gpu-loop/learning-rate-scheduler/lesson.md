---
artifact: course-lesson
leaf: training/single-gpu-loop/learning-rate-scheduler
language: zh-CN
estimated_time: 75min
environment: Mac
---

# 课程：用 step count 驱动学习率曲线

典型 schedule 是 warmup 后 decay：前 W 次 updates 将 lr 从较小值升到峰值，之后按 cosine、linear 或 step rule 降低。横轴必须先说清是 optimizer updates 还是 dataloader batches；gradient accumulation 时两者不同，若 scheduler 每 microbatch 调一次会得到错误曲线。

scheduler 有状态：当前 step、base lr、阶段参数。断点恢复时若没有恢复 scheduler state，模型和 optimizer 虽然加载了，lr 时间线却跳回起点或错位，loss 连续性被破坏。

**实验课：** 写 20 次 update 的 warmup+cosine scheduler，先手画前 10 个 lr，再每次 `optimizer.step()` 后按 chosen convention 调 `scheduler.step()` 并记录。重复一次故意在错误频率调用 scheduler，比较曲线。通关时能解释 scheduler 应按 update 而非 epoch/microbatch 的条件。
