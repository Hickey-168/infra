---
artifact: curriculum-tutorial
leaf: inference/scheduling/continuous-batching
priority: P0
environment: Mac
---

# Continuous Batching：每轮重新决定谁占用模型一步

## 问题

static batch 要等一整批请求全完成才换人：短请求会空等，长请求占住槽位，且新请求只能等待。LLM decode 恰好按 token 一步一步推进，因此更自然的调度单位不是“一个请求完成”，而是“下一轮 model iteration”。

## 因果链

1. 请求进入 waiting，经过 prefill 后进入 running；每个 running request 每轮通常生成一个 token，并增长自己的 KV cache。
2. 一轮结束时，完成的请求释放 slots/blocks，新到请求可能被 admission；scheduler 再依据 token budget、KV 容量、prefill chunk 与优先级选择下一批。
3. 因为 batch 组成在 iteration 边界变化，短请求可提前离开、新请求可提前进入，提升资源利用率。
4. 但吞吐、TTFT、TPOT 和公平性互相牵制：总是优先 decode 可改善活跃用户 TPOT，却可能饿死长 prompt；总是优先 prefill 又会拖慢已有 generation。

## 最小状态机

```text
waiting --admit/prefill--> running --one decode step--> running
running --eos/max_tokens/cancel--> finished --release KV--> waiting can enter
running --memory pressure--> preempted/swapped (后续专题)
```

状态所有权比队列名字重要：谁占用 KV blocks，何时释放，哪个事件允许请求重新进入。

## 预测后验证

写纯 Python simulator，不调用模型。输入 5 个请求的 arrival time、prompt length、generation length 与 token budget；每轮打印 waiting/running/finished、发出的 prefill/decode tokens、queue depth。比较 static batch 和 continuous batch，先预测短请求 completion time 与长 prompt TTFT 的变化。该 toy 证明调度语义，不证明真实 GPU throughput。

## 下一跳

训练侧也有“多个参与者必须对齐状态”的问题，但通过 collective 解决。进入 `communication/collectives/all-reduce`。
