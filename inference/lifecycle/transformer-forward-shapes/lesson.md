---
artifact: course-lesson
leaf: inference/lifecycle/transformer-forward-shapes
language: zh-CN
estimated_time: 150-210min
environment: Mac
---

# 课程：从 token ids 到 logits，读懂最小 Decoder Transformer 的前向

> 本文是训练循环前的模型知识输入。目标不是让你手写高性能 Transformer，而是让你能看懂训练代码中的输入、每层主要 shape、loss 为什么这样对齐，以及生成时为什么只取最后一个位置。

## 1. 模型收到的不是文字

文本先经 tokenizer 变成 vocabulary 中的整数 id。一个 batch 常为：

```text
input_ids: [B, T], dtype=int64
```

其中 B 是 batch 内序列数，T 是当前每条序列的 token 数。整数 id 只是查表索引，不包含可供矩阵乘使用的连续特征。embedding table 是可训练参数 `E:[V,H]`：V 为 vocabulary size，H 为 hidden size。lookup 后：

```text
X = E[input_ids]     # [B, T, H]
```

同一 token id 在不同位置先得到相同的 embedding，因此模型还需要位置信息。绝对 position embedding 是把 `P:[T,H]` 加到 X；RoPE 则在 attention 的 Q/K 中编码位置。无论具体方案，位置的核心问题是“第 t 个 token 与第 t+1 个 token 不能被当作无序集合”。

## 2. Decoder block 为什么维持 `[B,T,H]`

一个 decoder-only Transformer 通常重复 L 个相同外形的 block。每个 block 接收 `[B,T,H]`，输出仍为 `[B,T,H]`；这使 block 可堆叠。内部有两条主要子层：self-attention 让每个位置读取历史 token，MLP 对每个位置独立做非线性变换。

以 pre-norm 结构为例：

```text
R0 = X
U  = LayerNorm(R0)
A  = SelfAttention(U)
R1 = R0 + A                 # residual connection
M  = MLP(LayerNorm(R1))
R2 = R1 + M                 # output, still [B,T,H]
```

residual 的意义不只是“加法”：它提供一条直接信息/梯度路径，使 block 学习对已有表示的增量修改。因为相加两边必须 shape 一致，attention 和 MLP 最终都投回 H。

## 3. Self-attention 的 shape 与计算

先忽略 batch。对 `X:[T,H]`，三组可训练投影把 H 维表示变成 query/key/value：

```text
Q = X @ Wq, K = X @ Wk, V = X @ Wv
```

多头 attention 将 H 分为 `heads * D`，所以常 reshape 为：

```text
Q, K, V: [B, heads, T, D]
H = heads * D
```

对一个 head，位置 i 的 query 与所有 key 做点积：

```text
scores[i,j] = dot(Q[i], K[j]) / sqrt(D)
weights[i,:] = softmax(scores[i,:])
output[i] = sum_j weights[i,j] * V[j]
```

矩阵写法为 `softmax(Q @ K^T / sqrt(D)) @ V`。`Q @ K^T` 的最后两维是 `[T,T]`：第 i 行是“位置 i 对所有位置 j 的注意力分数”。这就是序列长度为什么会强烈影响 attention 的工作量和内存。

### Causal mask：为什么语言模型不能偷看答案

next-token prediction 中，位置 i 只应看到不晚于 i 的 tokens。causal mask 将 `j>i` 的 score 设为极小值（概念上 `-inf`），softmax 后这些位置权重为 0：

```text
allowed: j <= i
blocked: j > i
```

没有它，训练时位置 i 可看到未来正确 token，loss 会虚假地变低，生成时却无法复现同样信息。这是训练/推理一致性的关键约束。

## 4. MLP 与 logits：从表示到 vocabulary 分数

attention 后，MLP 对每个 token 位置独立执行：

```text
[B,T,H] -> Linear(H,F) -> activation -> Linear(F,H) -> [B,T,H]
```

F 常称 intermediate/FFN size，通常大于 H。attention 负责跨位置混合信息，MLP 负责在每个位置的 feature space 做非线性变换；两者配合才形成 Transformer block。

L 个 blocks 后得到 hidden states `Z:[B,T,H]`。language-model head 是最后一个线性投影：

```text
logits = Z @ W_vocab       # W_vocab:[H,V], logits:[B,T,V]
```

每个 `[V]` 向量是“当前位置下一个 token 是 vocabulary 每个候选项”的未归一化分数。logit 最大的 token 是 greedy decoding 的选择；temperature/top-k/top-p 会改变如何从这些 logits 选 token。

## 5. 为何训练要 shift labels

给定 token 序列 `[A,B,C,D]`，模型在看到 A 后应预测 B，看到 A,B 后预测 C。概念上的监督对齐：

```text
model input positions:  A      B      C
desired next tokens:    B      C      D
```

若 logits 是 `[B,T,V]`，常用 `logits[:, :-1, :]` 对应 `labels[:, 1:]` 计算 cross entropy。框架也可能把 shift 包在 model/loss 内部；读代码时只需找到“哪个 logits position 对哪个 label id”的对应关系，不要因为 API 位置不同误以为机制不同。

cross entropy 对每个有效 token 产生一个 loss，随后通常对 B 和 T 做 mean，得到 scalar。这个 scalar 才进入 `backward()`；logits 不直接进入 optimizer。

## 6. 同一 forward 如何服务 training 与 generation

training 时通常一次处理完整 `[B,T]`，建立 autograd graph，计算所有有效位置的 loss。generation 时也调用相同 block，但当前上下文已经有 T 个 tokens，只需要最后位置的 `logits[:, -1, :]` 来选择新 token。选出 id 后 append，下一轮上下文长度变为 T+1。

朴素 generation 每步重算所有历史 K/V；实际 serving 使用 KV cache 保存每层历史 K/V。于是 prefill 处理 prompt 的全部 T 个位置，decode 每步新增一个 query 并读取历史 cache。这是后续 `prefill-vs-decode`、`kv-shape-and-capacity` 的直接前置。

## 7. 用一个极小配置手算 shapes

令：

```text
B=2, T=4, V=32, H=8, heads=2, D=4, F=16
```

按顺序写出：

| 阶段 | shape | 含义 |
|---|---|---|
| input ids | `[2,4]` | 8 个整数 token ids |
| embedding | `[2,4,8]` | 每 token 一个 8 维向量 |
| Q/K/V | `[2,2,4,4]` | 2 heads，每 head 4 维 |
| attention scores | `[2,2,4,4]` | 每位置看 4 个位置，受 causal mask 限制 |
| attention output | `[2,4,8]` | 合并 heads 后回到 H |
| MLP hidden | `[2,4,16]` | 每位置扩张到 F |
| block output | `[2,4,8]` | residual 保持主 shape |
| logits | `[2,4,32]` | 每位置对 32 个 token 打分 |

能亲手走完这张表，比记住“Transformer 有 attention”更有用；之后所有显存、kernel、KV cache 公式都从这些轴开始。

## 8. 实验课：从小模块走到 block

按顺序运行，不要一开始加载大模型：

1. 实现 `Embedding(V,H) -> Linear(H,V)`，确认 ids/logits/loss shapes。
2. 用随机小 tensor 手写单 head attention 的 `QK^T -> mask -> softmax -> V`，打印 causal mask 前后权重，验证未来位置权重为 0。
3. 把 H reshape 为 heads*D，确认 split/merge 前后 element count 不变。
4. 在一个 tiny Transformer block 上用 hooks 打印每个模块的 input/output shape；检查 residual 两侧为什么可加。
5. 用固定序列比较训练时所有位置 logits 与 generation 时最后位置 logits，写下二者如何消费。

每步先预测 shape 和一个关键性质。这个 lab 验证语义与 shape，不验证 FlashAttention、KV cache 或大模型吞吐。

## 9. 常见错误模型

- **“embedding 把 token 变成概率。”** 错，它是可训练查表，输出 hidden vectors。
- **“attention 输出就是 logits。”** 错，attention/MLP 生成 hidden states，LM head 才投影到 V。
- **“causal mask 只在推理时需要。”** 错，训练也必须防止看未来。
- **“训练与生成使用不同模型。”** 多数情况下是同一 forward 的不同调用/消费方式。
- **“T 只影响输入长度。”** 错，attention score、activation、KV cache 和计算量都依赖 T。

## 通关

不看本文回答：为什么 QK^T 是 `[T,T]`；为什么 residual 必须回到 H；logits 与 labels 如何对齐；generation 为什么取最后位置 logits；causal mask 若缺失会发生什么。

## 延伸与核对

本文已给出完成本课所需机制。之后可核对 PyTorch attention API、模型源码或原始 Transformer 论文；把它们当作版本/实现差异的证据，而非本课的前置阅读。
