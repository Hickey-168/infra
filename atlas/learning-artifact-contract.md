# Learning Artifact Contract

一个叶子目录有四类材料，它们刻意不混写。目标不是多写几份 Markdown，而是保留从“外部解释”到“自己的证据”的学习轨迹。

| 文件 | 所有者 | 什么时候看/写 | 允许回答的问题 |
|---|---|---|---|
| `tutorial.md` | AI curriculum | 课程正文前阅读 | 导学：这个机制为何存在、依赖什么、应预测什么。 |
| `lesson.md` | AI curriculum | 导学后、实验前完整阅读 | 课程正文：概念推导、例子、纸面练习、实验课与通关条件。 |
| `lab.ipynb` / `lab.py` / `bench.py` | 你主导，AI 可辅助 | 阅读后运行和修改 | 预测是否成立；边界条件和量纲是什么。 |
| `note.md` | **你** | 看到实验结果后写 | 我现在如何因果地解释它；我先前哪里错了；结论的适用边界是什么。 |
| `questions.md` | AI 出题、你闭卷作答 | 笔记后 | 不看资料能否迁移到调用链、shape/state、反事实和数量级。 |
| `review.md` | AI curriculum | 已学过后复习 | 讨论后的完整正确结论、误区、边界和证据，不要求重走初学流程。 |

## Read Order

```text
tutorial.md
  -> lesson.md
  -> write a prediction outside the answer key
  -> lab.*
  -> note.md
  -> questions.md
  -> review.md (when a discussion creates a durable conclusion)
  -> update atlas link / choose the next leaf
```

`tutorial.md` 是导学，`lesson.md` 是可直接学习的课程正文；两者都由 AI 根据官方文档、论文和源码维护。`note.md` 是不可替代的个人理解：脚手架和 AI 不应以长篇正文覆盖它。

## What Belongs Where

- 把定位、依赖、预习问题放到 `tutorial.md`；把机制解释、推导、例子和实验课放到 `lesson.md`。
- 把运行命令、版本、张量形状、数值和 profiler 截图对应的结论放到 `lab.*` 或 `templates/experiment-note.md`。
- 把错误预测、自己的比喻、费曼解释、仍然不懂的点放到 `note.md`。
- 把闭卷答案和漏点放到 `questions.md`；先回答，再展开正解。
- 把一次讨论最终确认的完整结论、反例、误区和证据边界放到 `review.md`；它必须可脱离聊天记录复习。

## Completion Rule

不要以“看完教程”算完成。完成一个叶子的最低证据是：一次可复跑的实验、一段不看教程的因果解释、一次闭卷检查，以及一个明确的下游链接。CPU/Mac 的 toy experiment 只能验证语义；Triton、CUDA、通信带宽和训练显存的性能结论要在 WSL2-GPU 或远程 GPU 上验证。
