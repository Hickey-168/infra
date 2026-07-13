---
artifact: course-lesson
leaf: training/single-gpu-loop/training-loop-state-machine
language: zh-CN
estimated_time: 150-210min
environment: Mac
---

# 课程：从一条样本到一次参数更新，完整理解训练循环

> 本文是主要知识输入。读完本文，你不需要先会实现完整 Transformer；你应该能读懂一个最小 PyTorch 训练脚本，知道每一行在读什么状态、产生什么状态、什么时候消耗显存，以及它与后续大模型训练有什么关系。

## 学习目标

完成后你应能独立回答：

1. forward、training、inference 三个词分别指什么，哪里重叠、哪里不同。
2. 一批 token 如何变成 logits、loss、gradient，最后如何改变 parameter。
3. 为什么 `backward()` 后参数还没有变，为什么忘记 `zero_grad()` 会出错。
4. batch size、sequence length、learning rate、weight decay、accumulation、scheduler 分别插在循环的哪一处。
5. 为什么理解这条循环是学习 parameter/activation/optimizer memory 的前提。

---

## 0. 先建立最小世界：训练到底想改变什么

机器学习模型可以先抽象成一个带参数的函数：

```text
prediction = model(input; parameters)
```

参数是模型内部可被训练改变的数，例如线性层的 weight/bias，或 Transformer 的 embedding、attention projection、MLP weight。训练数据同时给出 input 与“希望模型给出的答案” target/label。loss 是一个标量，用来量化本次 prediction 距离 target 有多远。

训练不是“调用模型一次”。训练反复做下面的事：用当前参数产生 prediction，计算 loss，问“每个参数往哪个方向微调会让 loss 下降”，然后真的修改参数。推理则只使用已经训练好的参数产生 output，不再修改它们。

### 一个可手算的模型

先不用神经网络。令模型只有一个参数 `w`，输入 `x=2`、目标 `y=10`：

```text
prediction = x * w
loss = (prediction - y)^2
```

若当前 `w=3`，则 prediction=6，loss=16。模型并不知道应把 w 改成多少；这正是 backward 和 optimizer 的分工：backward 计算 loss 对 w 的导数，optimizer 用 learning rate 决定实际走多远。

---

## 1. 数据进入训练循环前已经是什么形状

对文本模型，原始字符串先被 tokenizer 编为 token ids。例如 vocabulary 中 `"我" -> 17`、`"爱" -> 42`。一个 batch 的输入常是：

```text
input_ids: [B, T]  int64
labels:    [B, T]  int64
```

B 是本次同时处理的序列数，T 是每条序列参与训练的 token 数。它们不是模型参数；它们是每次循环从 dataloader 取出的 batch。

因果语言模型的目标是“根据当前位置之前的 token 预测下一个 token”。所以常见对齐是：

```text
tokens: [  A,   B,   C,   D]
input:  [  A,   B,   C]
label:  [  B,   C,   D]
```

真实框架可能把完整 `[B,T]` input/labels 交给 model 后在内部 shift；不论 API 放在哪里，监督信号的含义不变：位置 t 的 logits 应预测位置 t+1 的真实 id。

### 停下来检查

若 B=2、T=4，则一个 batch 有几个“下一个 token”训练目标？若最后 token 没有下一个目标，答案通常是 `B*(T-1)=6`。这不是死记：它决定 cross entropy 在哪些位置计算，也会影响 token-level loss 的约简方式。

---

## 2. forward：用当前参数把输入变成 logits

**forward** 是一次从 input 到 output 的函数计算。它不天然等于训练，也不天然等于推理。

最小语言模型可以写成：

```text
Embedding table E: [V, H]
input_ids:           [B, T]
hidden = E[input_ids]             -> [B, T, H]
lm_head W: [H, V]
logits = hidden @ W               -> [B, T, V]
```

V 是 vocabulary size，H 是 hidden size。embedding lookup 将每个整数 id 变成 H 维浮点向量；最后的线性层为每个位置、每个 vocabulary token 给出一个分数。这个分数叫 **logit**，还不是概率。

若某位置 logits 为 `[2.0, 0.0, -1.0]`，softmax 才会把它转换成三个非负、和为 1 的概率。训练通常把 logits 直接交给 `cross_entropy`，因为它内部以数值稳定方式组合 log-softmax 与 negative log likelihood；不要先手动 softmax 再喂给 cross entropy。

### forward 产生的状态

一次可求梯度的 forward 除了 logits，还可能产生 autograd graph 和 saved tensors。它们记录“这个输出由哪些运算/参数得到，backward 需要哪些中间值”。这就是为什么 training forward 的内存行为不同于 inference forward；activation-memory 课程会继续追这件事。

---

## 3. loss：把一大块 logits 变成一个可优化的标量

一个 batch 的 logits shape 是 `[B,T,V]`，labels 是 `[B,T]`。对每个有效位置，cross entropy 取该位置 label 对应的 logit，比较模型分配给正确 token 的概率；模型越确信正确 token，loss 越低。随后会对 batch 和 token 位置做 mean 或 sum，得到 scalar loss。

为什么必须得到 scalar？因为它提供一个统一目标函数，autograd 可以问“若每个 parameter 增加一点，这个总 loss 怎样变化”。并非所有训练 loss 必须是单一 scalar tensor，但最常见的 `.backward()` 调用以 scalar loss 为起点。

注意三个不同对象：

| 名称 | 典型 shape | 含义 |
|---|---|---|
| `logits` | `[B,T,V]` | 模型对所有候选 token 的未归一化分数 |
| `labels` | `[B,T]` | 正确 token 的整数 id |
| `loss` | `[]`（scalar） | 本 batch 的优化目标 |

把 logits 当 prediction、把 loss 当模型输出、或把 labels 当浮点向量，都会让后续 shape/梯度推理混乱。

---

## 4. backward：计算“往哪边改”，但不改参数

回到标量例子：

```text
loss = (x*w-y)^2
d(loss)/dw = 2 * (x*w-y) * x
```

当 `x=2,w=3,y=10`，导数是 `2*(-4)*2=-16`。负导数表示增加 w 会降低当前 loss。PyTorch 不要求你手写导数：在 forward 中记录运算图后，执行：

```python
loss.backward()
```

autograd 从 loss 沿图反向应用链式法则，将每个 leaf parameter 的导数累加到 `parameter.grad`。注意这句话中的“累加”：默认情况下 `.grad` 不是被覆盖，而是 `old_grad + new_grad`。这允许 gradient accumulation，也意味着如果你忘记清零，上一 batch 的 gradient 会意外参与下一次 update。

backward 后的三个事实：

```text
loss 已经存在
parameter.grad 已经存在（对参与计算的可训练参数）
parameter.data 仍然没有改变
```

这第三点极重要。`backward()` 是微分，不是学习规则。

### `grad_fn` 与 leaf 的直觉

model parameters 通常是 leaf tensors：它们由用户/Module 创建，并持有 `.grad`。logits/loss 是运算结果，常有 `grad_fn` 指向产生它们的反向节点。你不需要背每个 `grad_fn` 名称，但要能沿着 `loss -> logits -> hidden -> parameter` 说明梯度为什么能回去。下一节 `grad-fn-chain` 会把这条图拆开观察。

---

## 5. optimizer.step：参数第一次真的改变

有了 gradient，optimizer 才执行更新。最简单的 SGD：

```text
w_new = w_old - learning_rate * w.grad
```

继续标量例子。若 lr=0.1、w=3、grad=-16：

```text
w_new = 3 - 0.1 * (-16) = 4.6
```

这一步才是“模型从数据学到东西”的参数 mutation。真实训练常用 AdamW：它不只读当前 gradient，还维护每个 parameter 的一阶/二阶动量 state，并可施加 weight decay。因此 optimizer 本身持有状态和显存；parameter-memory 课程中的 optimizer-state 一项由此而来。

执行顺序的核心不是某个框架样板，而是依赖关系：必须先有 grad 才能 step；step 后要在下一轮 backward 前处理旧 grad。一个典型顺序是：

```python
optimizer.zero_grad(set_to_none=True)
logits = model(input_ids)
loss = loss_fn(logits, labels)
loss.backward()
optimizer.step()
```

有些脚本把 `zero_grad` 放在上轮 step 后，语义同样可行；关键是每次要计算新 grad 前，旧 grad 不得意外残留。scheduler、mixed precision unscale、gradient clipping 会插入这个骨架，但不会改变“backward 先于 step”的因果关系。

---

## 6. training、evaluation、inference：同一个模型的三种运行语境

### Training

训练使用 `model.train()`，运行可求梯度的 forward，计算 loss、backward 和 step。`train()` 不等于“打开梯度”，它还让 dropout 随机丢弃激活、让 batch norm 更新/使用训练统计（Transformer 常无 batch norm，但 dropout 很常见）。

### Evaluation

验证通常不更新参数，但仍有 labels/loss/metrics；常见组合是 `model.eval()` 和 `torch.no_grad()` 或 `torch.inference_mode()`。`eval()` 改 module 行为；no_grad/inference_mode 改 autograd 是否记录。两者解决不同问题，不能互相替代。

### Inference / generation

推理只关心输出 token 或 scores，不计算用于更新的 loss，也不 backward/step。LLM generate 反复调用 forward，但每步只消费最后位置 logits，并可能维护 KV cache；这就是 prefill/decode 课程的入口。

| 模式 | `train/eval` | autograd graph | loss | parameter update |
|---|---|---|---|---|
| training | `train()` | 通常记录 | 是 | 是 |
| evaluation | `eval()` | 通常关闭 | 可计算 | 否 |
| inference | `eval()` | 关闭 | 通常否 | 否 |

---

## 7. 超参在哪里改变这条循环

超参不是训练脚本最后的一组数字；它们直接改写前面各阶段的输入、状态或更新。

- **micro batch size**：改变一次 forward/backward 的 B，影响 activation memory 和 gradient 的样本统计。
- **sequence length**：改变 T，影响 token 数、attention 工作量与 activation memory。
- **learning rate**：改变 optimizer step 的距离，过大可导致 loss 振荡/发散，过小则学习慢。
- **weight decay**：改变 update rule 中参数收缩部分，和“训练后再缩小参数”不是一回事。
- **gradient accumulation**：连续处理多个 microbatches、累加 grad，最后再 step；近似 global batch 为 `micro_batch * accumulation * world_size`。
- **scheduler/warmup**：让 lr 随 optimizer update 次数改变；accumulation 时要分清 microbatch 数和实际 update 数。

这些项目会在独立课程详细讨论。现在只需能把它们放回状态机：B/T 改 forward/backward 的 shape 和成本；lr/decay 改 step；accumulation/scheduler 改 step 的时机和参数。

---

## 8. 一段最小可读代码

先读每行的因果，不急着运行：

```python
model = TinyLanguageModel(vocab_size=32, hidden_size=16)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for input_ids, labels in dataloader:
    optimizer.zero_grad(set_to_none=True)  # 防止旧 grad 非预期累加
    logits = model(input_ids)              # [B, T] -> [B, T, V]
    loss = cross_entropy_for_causal_lm(logits, labels)  # -> scalar
    loss.backward()                        # 填充 parameter.grad
    optimizer.step()                       # 第一次修改 parameters
```

阅读这段时，对每行问五个问题：输入 shape 是什么？输出 shape 是什么？谁拥有新 state？是否分配/保留 tensor？是否修改已有 state？这比背训练 loop 模板更能迁移到 DDP、FSDP 和 mixed precision。

---

## 9. 实验课：不实现完整 Transformer，也能验证所有关键点

按以下顺序在 `lab.ipynb` 做。每一步先写预测，再运行。

1. **标量 SGD：** 创建 `w=torch.nn.Parameter(torch.tensor(3.0))`、x=2、y=10；打印 forward 前 w、loss、backward 后 grad、step 后 w。核对手算 -16 与更新到 4.6。
2. **gradient accumulation：** 不清零连续 backward 两次，检查 `.grad` 是否翻倍；再调用 `zero_grad(set_to_none=True)`，检查 grad 状态。
3. **token classifier：** 用 `Embedding(V,H)` 加 `Linear(H,V)`，输入 `[B,T]` ids，打印 logits `[B,T,V]` 和 scalar cross entropy loss。
4. **模式对比：** 在 training forward 与 `model.eval()+torch.inference_mode()` 下检查输出的 `requires_grad/grad_fn`；若含 dropout，重复 forward 观察 train/eval 输出差异。
5. **超参小实验：** 固定数据和 seed，只改 lr；记录 loss 曲线。不要从 toy 曲线推出任何大模型最佳 lr。

## 10. 常见错误模型

- **“forward 就是在训练。”** forward 只是函数计算；没有 loss/backward/step 就没有参数学习。
- **“loss.backward 会更新模型。”** backward 只写 grad；step 才写 parameter。
- **“eval 等于 no_grad。”** eval 改 module behavior；no_grad/inference_mode 改 autograd recording。
- **“zero_grad 是清空 loss。”** 清的是 parameter gradients；loss 是本轮计算结果。
- **“超参只影响最终精度。”** B/T/lr/accumulation 直接改变每步 shape、显存、统计与更新时间线。

## 11. 进入下一课前的通关问题

不看本文回答：

1. logits、labels、loss 的 shape/语义分别是什么？
2. 在哪一行第一次出现 grad，在哪一行第一次改变 parameter？
3. 为什么 gradient accumulation 需要控制 zero_grad 和 step 的位置？
4. `model.eval()` 与 `torch.inference_mode()` 各解决什么问题？
5. 为什么 parameter-memory 必须放在本课之后？

## 延伸与核对

本文足以开始本课；以下只用于核对 API 版本行为或继续深入：PyTorch `autograd`、`torch.optim`、`nn.Module.train/eval` 官方文档，以及本仓库紧随其后的 `forward-backward-step`、`grad-fn-chain`、`optimizer-step` 叶子。
