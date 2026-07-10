---
artifact: course-lesson
leaf: inference/scheduling/continuous-batching
language: zh-CN
estimated_time: 90min
environment: Mac
---

# 课程：用 iteration 而不是“整批请求”做 LLM 调度

## 1. static batch 的浪费从哪里来

若一批请求必须等全部生成结束才能换下一批，短请求完成后留下的槽位无法立刻给新请求，长请求决定整批完成时间。LLM decode 天然按 token step 前进，所以更合适的调度边界是每一次 model iteration。

## 2. 最小状态机

请求先在 `waiting`，被 admission 后进行 prefill，进入 `running`；每轮 running 请求各做一个 decode step；遇到 EOS、max_tokens 或 cancel 后进入 `finished` 并释放 KV blocks。显存压力下还可能 preempt/swap，但第一版 simulator 先不加入它。理解状态转移时，始终问：此刻谁拥有 KV memory，何时能释放。

## 3. 每轮 scheduler 要解决什么

它在 token budget、KV capacity、已有 decode 的 TPOT、新 prompt 的 TTFT 和公平性之间取舍。优先 decode 常改善已开始生成的体验，却可能让长 prompt 饥饿；优先 prefill 可改善新请求 TTFT，却使 running 请求等更久。continuous batching 不是“必然更快”，而是允许在更细粒度上表达策略。

## 实验课

写纯 Python simulator：为 5 个请求设 arrival、prompt length、generation length；每轮打印 waiting/running/finished、执行的 prefill/decode tokens、KV token 总数。先预测 static 和 continuous 下短请求完成时间与长请求 TTFT，再验证。这个实验验证调度语义，不能代表真实 GPU token/s。

## 通关

能画出请求的一次完整状态迁移，给出一个“吞吐变好但 p99 变差”的 workload。下一课进入 all-reduce：训练侧的多个 ranks 也要协调状态，只是工具是 collective。
