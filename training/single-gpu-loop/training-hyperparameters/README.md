# training hyperparameters

> 叶子路径: `training/single-gpu-loop/training-hyperparameters` · 代码形态: `notebook`

本目录是一个可独立学习与验证的机制叶子。文件约定：

- `note.md` — 我自己写的费曼式笔记（唯一需要我亲手总结的文件）。
- `lab.ipynb` — 交互式验证（失败模式实验 + 数量级估算），带完整中文注释。
- `questions.md` — 盲点自测题（AI 出题，我闭卷作答后回填正解与漏点）。

若该叶子已预置 AI 编写的 `tutorial.md` 和 `lesson.md`，学习流程为：导学 → 课程正文 → 写下预测 → 实践(跑代码验证) → 填 `note.md` → 闭卷作答 `questions.md`。
脚手架不生成导学或课程正文；它们由课程作者按优先级显式写入。文件所有权见 `atlas/learning-artifact-contract.md`。

完成标准：note.md 的 Verification Checklist 全勾 + questions.md 正确率 ≥ 80%，然后把 frontmatter 的 `status` 改为 `verified`。
