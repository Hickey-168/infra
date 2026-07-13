---
artifact: curriculum-tutorial
leaf: training/single-gpu-loop/training-loop-state-machine
priority: P0
environment: Mac
---

# 导学：训练不是一个黑盒 `fit()`，而是可追踪的状态机

这节是 parameter-memory 前的总览课。它回答 forward、inference、loss、backward、optimizer update 和超参各自在何时发生、读写什么状态。先建立全图，后续叶子再钻细节。

完成后，你应能画出一个 batch 从数据到参数更新的闭环，且能说出 inference 从哪里分叉出去。不要先实现完整 Transformer；先用最小线性/语言模型 toy 验证同一条状态机。
