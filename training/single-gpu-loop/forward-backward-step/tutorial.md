---
artifact: curriculum-tutorial
leaf: training/single-gpu-loop/forward-backward-step
priority: P0
environment: Mac
---

# 导学：forward 产生值，backward 产生梯度，二者都不更新参数

本叶拆开训练中最常被混淆的三件事：forward 计算预测/loss，backward 根据链式法则填充 `.grad`，optimizer step 才改变 parameters。完成后再看 optimizer state 和显存才不会把“有梯度”误认为“已经训练”。
