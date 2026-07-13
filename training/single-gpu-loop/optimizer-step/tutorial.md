---
artifact: curriculum-tutorial
leaf: training/single-gpu-loop/optimizer-step
priority: P0
environment: Mac
---

# 导学：gradient 告诉方向，optimizer 决定如何走

backward 只给出当前 loss 对参数的导数。optimizer step 将导数、学习率、weight decay 与内部状态组合成真实更新；这是“模型正在训练”真正发生的时刻。
