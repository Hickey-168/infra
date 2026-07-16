---
artifact: course-tutorial
leaf: retrieval-recommendation/ranking/weight-norm-regularization
language: zh-CN
estimated_time: 45min
environment: Mac
---

# 导学：用范数控制推荐模型的可用复杂度

这节课解决一个很实际的问题：推荐模型为何会把少数训练样本“记死”，以及为什么给权重或 embedding 加一个范数代价能缓解它。它不要求先掌握深度学习；只需会向量点积、知道训练会最小化一个 loss。

学习顺序是：向量范数（长度）→ 矩阵范数（线性变换的大小）→ `数据损失 + 范数惩罚` → 双塔/矩阵分解中的 user、item embedding。先区分三个对象：样本的**特征向量** `x`、模型的**参数** `w/W`、由 ID 查表得到的 **embedding** `e`。常规的权重正则化约束后两者，不是把输入特征的数值直接压小。

读完后，先预测：若一个只有一次点击的 item 为了拟合那一次正样本而获得很大的 embedding，给所有 embedding 加 L2 惩罚后，它的范数与训练分数会如何变化？再运行 `lab.ipynb` 检查这个预测。

下游连接：`pointwise-ranking`、`pairwise-ranking`、`feature-crossing`、双塔向量召回和线上 embedding store。与训练主线的连接是 `training/single-gpu-loop/optimizer-step`：正则化最终必须落到一次 parameter update。
