# 盲点自测 — uv conda pip

> 规则：闭卷口述作答，答完再看正解。答错的题记到「漏点」，作为下次复习入口。

## 七维度题目（AI 生成后填入）

### Q1 调用链
- 问题：`python -m pip install torch` 中的 `python` 和最后 `import torch` 的 Python 必须满足什么关系？
- 我的初答：
- 正解：它们应该是同一个解释器，或至少指向同一个环境的 site-packages；否则 torch 可能被安装到 A 环境，却在 B 环境里 import。
- 漏在哪里：

### Q2 shape/state/layout
- 问题：诊断 notebook 找不到 torch 时，最先打印哪个状态？
- 我的初答：
- 正解：在 notebook cell 里打印 `sys.executable`，确认 kernel 实际运行的 Python 路径。
- 漏在哪里：

### Q3 shape/state/layout
- 问题：为什么“另一个项目的 venv 已经有 torch”不能证明当前项目可用？
- 我的初答：
- 正解：venv 隔离 `site-packages`；当前项目解释器不会自动读取另一个 venv 的包目录。
- 漏在哪里：

### Q4 反事实
- 问题：如果 PyCharm 切换解释器后选择 detach 当前 Managed Server，会发生什么？
- 我的初答：
- 正解：旧 server 继续运行并使用旧解释器；新解释器要等 stop/restart server 或重启 kernel 后才生效。
- 漏在哪里：

### Q5 最小实现
- 问题：写一行命令，把 torch 安装到当前 notebook kernel 对应的 Python。
- 我的初答：
- 正解：在 notebook 中运行 `import sys` 后执行 `!{sys.executable} -m pip install torch`。
- 漏在哪里：

### Q6 数量级估算
- 问题：为什么不推荐把 torch 直接装到全局 Python？
- 我的初答：
- 正解：torch 是重型依赖，全局安装容易污染其他项目，也容易被 Python 版本、权限、系统包管理策略影响；项目级 venv 更可复现。
- 漏在哪里：

### Q7 来源与边界
- 问题：当前 PyTorch 稳定版对 Python 版本的基本边界是什么？
- 我的初答：
- 正解：官方安装页给出的边界是 Python 3.10 或更新版本；具体 torch wheel 还要按操作系统和 CPU/CUDA/ROCm 平台选择。
- 漏在哪里：
