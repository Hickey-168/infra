# 盲点自测 — weight norm regularization

> 规则：闭卷口述作答，答完再看正解。答错的题记到「漏点」，作为下次复习入口。

## 七维度题目（AI 生成后填入）

### Q1 调用链
- 问题：从一个 batch 的 user_id/item_id 到一次 L2-regularized SGD update，参数、embedding lookup、数据 loss、范数项和 optimizer 分别按什么顺序参与？
- 我的初答：
- 正解：ID 查 embedding 与其他特征编码，前向得到 score 和数据 loss；从要正则的参数构造范数项并相加；backward 产生 `grad_data + lambda*theta`；optimizer step 用该梯度更新。必须检查 embedding/bias 是否被放入 decay 参数组。
- 漏在哪里：

### Q2 shape/state/layout
- 问题：item embedding table `V` 的 shape 为 `[N_item, d]`，`||V||F^2` 展开后是什么？它和每个 item 的 L2 范数是什么关系？
- 我的初答：
- 正解：`sum_i sum_j V[i,j]^2`，也等于所有 item 行向量的 `sum_i ||V[i]||2^2`。
- 漏在哪里：

### Q3 shape/state/layout
- 问题：Frobenius 范数和谱范数分别回答矩阵的什么“大小”问题？
- 我的初答：
- 正解：Frobenius 是所有元素的总平方能量；谱范数是任意输入在 L2 下可能得到的最大放大倍数。
- 漏在哪里：

### Q4 反事实
- 问题：若 λ 大到压倒数据 loss，推荐模型的训练/验证表现与 embedding 会怎样？
- 我的初答：
- 正解：权重和 embedding 接近零，模型分数缺少区分度，通常出现欠拟合；范数小不保证泛化好。
- 漏在哪里：

### Q5 最小实现
- 问题：从 `L = L_data + lambda/2*||theta||2^2` 推导一行 SGD update。
- 我的初答：
- 正解：`theta <- theta - eta*(grad_data + lambda*theta) = (1-eta*lambda)*theta - eta*grad_data`。
- 漏在哪里：

### Q6 数量级估算
- 问题：一个 `[10^7, 128]` 的 item table，若平均每行 L2 范数为 2，则 `||V||F^2` 约为多少？
- 我的初答：
- 正解：约 `10^7 * 2^2 = 4e7`。真正正则项还要乘 `lambda/2`；数值大小要与数据 loss 的 reduction 方式一起看。
- 漏在哪里：

### Q7 来源与边界
- 问题：为什么不能不加检查地说“Adam 的 L2 regularization 等于 AdamW weight decay”？
- 我的初答：
- 正解：Adam 会按坐标自适应缩放梯度，耦合的 L2 梯度也会被该缩放；AdamW 将 shrinkage 从 adaptive update 中解耦，二者一般不等价。
- 漏在哪里：
