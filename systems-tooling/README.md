# Systems Tooling

工具目录服务于你的学习工作流：Mac 本地开发、Windows + WSL2 + RTX 5080 跑 GPU 实验、GitHub 管理历史、Obsidian 管理知识图谱。

## Structure

- `environment`: Python env、CUDA wheel、driver、Triton/PyTorch version。
- `wsl2`: Windows/WSL2/NVIDIA GPU 开发记录。
- `github`: branch、commit、PR、学习日志。
- `obsidian`: note links、backlinks、frontmatter。
- `remote-gpu`: rsync/git sync、run log、artifact collection。
- `metadata`: 实验 schema、硬件/版本/命令/结果。

## Suggested Workflow

```text
choose leaf from atlas
-> create branch
-> copy learning-item template into domain folder
-> write prediction
-> run toy code / benchmark / profiler
-> record experiment
-> summarize mechanism and gaps
-> commit
```

