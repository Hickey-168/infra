---
artifact: curriculum-tutorial
leaf: training/single-gpu-loop/learning-rate-scheduler
priority: P0
environment: Mac
---

# 导学：学习率不是常数，scheduler 是训练状态的一部分

本叶把 warmup、decay、step count 与 checkpoint 联系起来。它在理解 optimizer 后学习：optimizer 定义一次如何更新，scheduler 定义每次更新使用多大的 lr。
