# roofline model

> 叶子路径: `operators/profiling/roofline-model` · 代码形态: `bench`

本目录是一个可独立学习与验证的机制叶子。文件约定：

- `note.md` — 我自己写的费曼式笔记（唯一需要我亲手总结的文件）。
- `bench.py` — benchmark/profiling 脚本（含 warmup、多次测量、单位）。
- `run.sh` — 采集命令。
- `results/` — csv/json/trace 摘要（大文件走 .gitignore）。
- `questions.md` — 盲点自测题（AI 出题，我闭卷作答后回填正解与漏点）。

学习流程见 skill `infra-leaf-study`：诊断 → 精读 → 重构(填 note.md) → 实践(跑代码) → 盲点扫描(questions.md)。

完成标准：note.md 的 Verification Checklist 全勾 + questions.md 正确率 ≥ 80%，然后把 frontmatter 的 `status` 改为 `verified`。
