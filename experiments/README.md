# Experiments

这里存放跨主题实验记录、原始 benchmark 输出、trace 摘要和复现实验索引。

建议每个实验目录至少包含：

- `README.md`: 实验目的、机器、命令、关键结论。
- `run.sh` 或命令记录。
- `results.*`: csv/json/markdown。
- `notes.md`: 观察、解释、后续问题。

不要把大型 trace、checkpoint、模型权重直接提交到 Git；用 `.gitignore` 或外部 artifact 存储管理。

