---
artifact: course-lesson
leaf: training/memory/parameter-memory
language: zh-CN
estimated_time: 75min
---

# 课程：从参数量到训练显存账本

`tutorial.md` 解释了为什么要分账；本课教你实际分什么账、如何用代码核验。

## 1. 参数量只是一张账的第一行

一个 parameter tensor 的 payload 是 `p.numel() * p.element_size()`。把所有 parameters 相加得到 weights payload。若一个模型有 N 个 BF16 parameters，权重 payload 近似 `2N` bytes；这只回答推理时最基础的权重成本，尚未回答训练是否放得下。

训练新增的 state 不能用一个魔法常数概括。gradient 通常与 parameter 有相同元素数，但 dtype 取决于训练 recipe；optimizer state 在第一次 step 后才可能被懒创建；混合精度可能维护更高精度 master weights。正确方法是逐项列出“元素数、dtype、bytes、在哪个时刻存在、每 rank 是否完整拥有”。

## 2. 用 Adam 看见状态何时出现

Adam 的直觉是：除参数外，它为每个被优化的 parameter 维护一阶和二阶动量估计。不要先背“Adam 一定占多少倍参数内存”。建立 optimizer 后、第一次 backward 后、第一次 `optimizer.step()` 后分别检查 `optimizer.state[p]`：你会看到 state 是何时真实 materialize 的，且其 dtype 是当前实现/recipe 的事实。

## 3. 手算例子

某线性层 weight shape 为 `[4096, 11008]`，BF16。先算参数数：`4096*11008`；再乘 2 bytes。若另一份同 shape gradient 也是 BF16，它又加同样 payload；若有两个 FP32 optimizer state，每份按 4 bytes 另算。把每一项写成表，而不是立即相加，这样 FSDP 时才知道哪一项被 shard。

## 实验课

创建 tiny Transformer，遍历 `named_parameters()`，输出 name、shape、numel、dtype、bytes；将手算总和与程序总和对比。随后跑一次 backward 和 step，在三个时间点输出 optimizer state。预测 BF16 model weights 是否减半、optimizer state 是否也一定减半，再让运行结果纠正你。

## 通关

你能解释“模型有 7B 参数”为什么不能推出训练显存；并能画出 parameters、gradients、optimizer states、master weights（若存在）的独立列。下一课是 `activation-memory`：它解释 batch/sequence 为什么又给账本加上一项随工作负载变化的峰值。
