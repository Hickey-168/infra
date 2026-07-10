# Leaf Study Plan（叶子学习方案）

这个文件定义**每个叶子目录该有哪些文件、如何学习**。它是 skill `infra-leaf-study` 的人类可读版说明。

## 为什么这样设计

490 个叶子目录不预先铺满文件（会污染 git、制造大量空模板），而是**按需 scaffold**：真正开始学某个叶子时，用脚本生成该叶子的标准文件布局。这与「只展开重点区」的学习哲学一致。

## 每个叶子的标准文件布局

| 文件 | 谁写 | 作用 |
|---|---|---|
| `README.md` | 脚本生成 | 叶子索引卡（路径、代码形态、完成标准） |
| `tutorial.md` | AI curriculum | 预先由 AI 编写的课程：机制、依赖、资料入口、预测和实验顺序；脚手架不会生成它 |
| `lesson.md` | AI curriculum | 课程正文：自包含机制讲解、例子、推导、实验课和通关条件；脚手架不会生成它 |
| `note.md` | **我亲手写** | 费曼式笔记，实例化 `templates/learning-item.md`。唯一必须我自己总结的文件 |
| `questions.md` | AI 出题、我作答 | 盲点自测五维题（边界/实现/性能/跨概念/数量级） |
| `review.md` | AI curriculum | 讨论后按复习视角写的完整沉淀；只有产生稳定讨论结论时由 AI 创建 |
| 代码文件 | AI + 我 | 领域自适应，见下 |

## 代码文件形态（按领域自适应）

不搞「一刀切全用 notebook」，因为分布式要多进程、CUDA 要编译、部署是声明式。脚本 `tools/scaffold_leaf.py` 的 `DOMAIN_CODE_KIND` 表决定形态：

| 领域                                                                                             | 形态            | 文件                                 | 理由                  |
| ---------------------------------------------------------------------------------------------- | ------------- | ---------------------------------- | ------------------- |
| fundamentals / triton / attention / fusion / 多数 inference / training 概念 / rec                  | `notebook`    | `lab.ipynb`                        | 交互探索、图文混排           |
| cuda / compiler / data-pipeline / checkpointing / frameworks / serving-api / storage / tooling | `script`      | `lab.py` + `run.sh`                | 需编译/多进程/可 diff      |
| distributed-training / communication / distributed-inference                                   | `distributed` | `lab.py`(torchrun) + `run.sh`      | 多进程 rank/collective |
| deployment                                                                                     | `config`      | `manifest.yaml` + `run.sh`         | 声明式 K8s/serving     |
| profiling / metrics / load-testing / regression                                                | `bench`       | `bench.py` + `run.sh` + `results/` | 强制 warmup/多次测量/单位   |

预览某领域形态：`python3 tools/scaffold_leaf.py --list-domain operators/triton`

## 用法

```bash
cd /Users/youyu/workspace/python/infra
# 开始学一个叶子（按需生成文件，已存在则跳过）
python3 tools/scaffold_leaf.py operators/triton/vector-add
# 覆盖重建
python3 tools/scaffold_leaf.py <leaf> --force
```

学习顺序固定为：`tutorial.md`（导学）→ `lesson.md`（课程正文）→ 写下预测 → `lab.*` → `note.md` → `questions.md`。
完整所有权规则见 [learning-artifact-contract.md](/Users/youyu/workspace/python/infra/atlas/learning-artifact-contract.md)。

## 完成标准

一个叶子算「学完」= note.md 的 Verification Checklist 全勾 + questions.md 正确率 ≥ 80%，
然后把 note.md frontmatter 的 `status` 改为 `verified`。

## 学习顺序

按 `infra-roadmap.md` 的 P0 → P1 → P2。P0 spine 15 个叶子先打通（tensor-lifecycle 起，到 fsdp2 止）。
每 5 个叶子做一次横向整合，合并 Knowledge Map 找反复出现的核心概念。

## 已示范的叶子

以下 3 个叶子已 scaffold 作为三种代码形态的样例：

- `fundamentals/pytorch-tensor-lifecycle/view-vs-reshape`（notebook）
- `communication/collectives/all-reduce`（distributed / torchrun）
- `operators/profiling/roofline-model`（bench）
