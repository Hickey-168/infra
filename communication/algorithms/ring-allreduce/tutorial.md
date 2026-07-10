---
artifact: curriculum-tutorial
leaf: communication/algorithms/ring-allreduce
priority: P0
environment: Mac
---

# Ring All-Reduce：把全局求和拆成两圈局部收发

## 问题

all-reduce 的 API 只给出语义，无法告诉你通信量为何随 world size、payload 和链路延迟变化。ring 算法把 tensor 切为 p 份，在环上分阶段流动，适合建立第一个成本模型。

## 两阶段机制

1. **reduce-scatter**：p 个 ranks 各持有完整 S-byte tensor 的分块。经过 `p-1` 轮邻居 send/recv，每个 rank 得到一个已被所有 rank 相加的 chunk。
2. **all-gather**：再经过 `p-1` 轮，每个 rank 收集其他已规约 chunks，恢复完整结果。
3. 所以每 rank 总传输 payload 近似 `2 * (p-1)/p * S`，总步数 `2*(p-1)`；带宽项接近常量倍 S，延迟项随 p 增长。

一个常见 alpha-beta 近似是：

```text
T ≈ 2*(p-1)*alpha + 2*(p-1)/p * S * beta
```

它忽略拓扑竞争、protocol、chunk pipeline 和 compute overlap，价值在于揭示小消息主要受 alpha、大消息主要受 bytes/beta 影响。

## 预测后验证

写 p=4、每 tensor 4 chunks 的 Python simulator：每轮打印“哪个 rank 把哪个 chunk 发给谁、接收后累加到哪里”。先不用并发；先保证最终每 rank 都得到逐元素和。再用不同 S、p 画 alpha-beta 估算，预测增大 p 对小/大 tensor 的不同影响。这个 simulator 不测 NCCL 性能，只验证算法状态机。

## 下一跳

`ddp-gradient-allreduce` 会将这一 collective 放入 autograd 的真实时间线，问题不再只是总通信量，还包括 bucket ready 与 compute/communication overlap。
