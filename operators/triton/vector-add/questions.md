# 盲点自测 - Triton vector add

> 规则：先闭卷口述或写下“我的初答”，再打开 `tutorial.md`、官方资料或实验输出纠正。正解不要提前填；你的漏点才是下一次复习的入口。

## Q1 调用链

- 问题：从 `triton_add(x, y)` 到一个元素 `z[1024]` 被写入，依次经过哪些 Python 和 Triton 状态？`N=1025`、`BLOCK_SIZE=1024` 时，谁写它？
- 我的初答：
- 正解：
- 漏在哪里：

## Q2 shape/state/layout

- 问题：`pid=1` 时 `offsets` 和 `mask` 分别是什么 shape、包含哪些有效值？为什么 `x_ptr + offsets` 不是一个标量地址？
- 我的初答：
- 正解：
- 漏在哪里：

## Q3 shape/state/layout

- 问题：若 `x` 是 contiguous FP32 且长度为 `N`，一次 out-of-place add 至少涉及多少有效 bytes？这个估算隐含了哪些 layout 与 cache 假设？
- 我的初答：
- 正解：
- 漏在哪里：

## Q4 反事实

- 问题：保持 grid 为 `ceil(N / BLOCK_SIZE)`，但在最后一个 block 去掉 load/store mask，会破坏哪条安全不变量？为什么不能把“没有报错”当成正确性证据？
- 我的初答：
- 正解：
- 漏在哪里：

## Q5 最小实现

- 问题：不用 Triton，怎样用 Python 伪代码写一个 CPU toy，显式遍历 `pid`、构造 offsets，并只处理 `offset < N` 的元素？它保留了 Triton 的哪条核心不变量？
- 我的初答：
- 正解：
- 漏在哪里：

## Q6 数量级估算

- 问题：FP32、`N = 2**26` 时，按 3 个 tensor 的有效流量估算要搬运多少 GiB？若 median kernel time 为 1.0 ms，effective GB/s 是多少？写出单位。
- 我的初答：
- 正解：
- 漏在哪里：

## Q7 来源与边界

- 问题：哪些结论来自 Triton 官方文档，哪些来自你的本机测量，哪些只是“它大概率 memory-bound”的推断？一次单卡 benchmark 不能说明什么？
- 我的初答：
- 正解：
- 漏在哪里：
