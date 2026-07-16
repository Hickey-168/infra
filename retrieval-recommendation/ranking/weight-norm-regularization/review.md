---
artifact: course-review
leaf: retrieval-recommendation/ranking/weight-norm-regularization
language: zh-CN
---

# 复习：范数、权重衰减与推荐 embedding

**结论：** 范数给向量/矩阵一个“大小”度量。把参数范数加入训练目标，会使模型只有在显著降低数据损失时才值得使用极大的权重；因此它限制了可选函数的规模/敏感性，并在多个同样能拟合训练数据的解中偏好小范数解。它不是把所有权重压成零，也不是直接处理输入特征。

**机制：** 对 `L = L_data + lambda/2 * ||theta||2^2`，梯度多出 `lambda*theta`。SGD 更新为 `(1-eta*lambda)*theta - eta*grad_data`，即每一步参数向零收缩。L1 更容易产生零坐标；L2 通常是平滑 shrinkage。矩阵的 Frobenius 范数是所有元素的 L2，适合权重总量/embedding 表；谱范数给出最强输入放大上界，二者不是同义词。

**推荐映射：** 样本特征 `x` 是输入；`W` 是排序 MLP 的参数；embedding table 的每一行也是可训练参数。双塔 `u^T v` 可加 `lambda/2 (||U||F^2+||V||F^2)`，防止少量交互靠异常大的 user/item 向量被记住，并解决 `cu` 与 `v/c` 打分不变的尺度歧义。若使用余弦分数，长度不直接进入分数，但训练与数值行为仍会受范数约束影响。

**边界：** λ 需看验证集；太大造成欠拟合。SGD 的 L2 penalty 与 weight decay 形式对应；AdamW 的 decoupled decay 是工程上更明确的收缩规则。bias、LayerNorm 和 embedding 是否 decay 是参数组设计问题。

**事实：** Lp/Frobenius/诱导范数定义、Cauchy–Schwarz 边界和 L2 的梯度是稳定数学结论。  
**推断：** 小范数更难依赖极端权重记住偶然样本，常改善泛化；真实业务收益必须由离线/在线实验验证。  
**观察：** 本叶子的 toy lab 尚未记录环境运行结果。

复述检查：能区分 `x`、`W`、embedding；从 L2 目标推导 `(1-eta*lambda)`；解释为何点积模型会有缩放歧义；说出 Frobenius 与谱范数各约束什么。
