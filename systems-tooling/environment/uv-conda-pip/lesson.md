# uv / conda / pip 环境边界

## 核心结论

事实: Python 包安装到某一个解释器对应的 `site-packages` 目录里。另一个 virtualenv、conda env、PyCharm interpreter 或 Jupyter kernel 即使在同一台机器上，也不会自动看见这个包。

这解释了一个常见现象: A 项目的 venv 里已经安装了 `torch`，但 B 项目的 notebook 仍然 `ModuleNotFoundError: No module named 'torch'`。notebook 使用的是它自己的 kernel 解释器，不是“机器上任意一个装过 torch 的 Python”。

## 三个名字，不要混在一起

1. 解释器: `sys.executable` 指向的 `python` 二进制，例如 `.venv/bin/python`。
2. 包安装器: `python -m pip` 会把包装进这个 `python` 对应的环境；裸 `pip` 可能来自另一个环境。
3. Notebook kernel: Jupyter 运行 cell 的后台 Python。它必须和项目解释器一致，或者至少安装了项目需要的包。

推断: 对本仓库这类 notebook + 脚本混合学习项目，最稳妥的默认环境是仓库根目录下一个项目级 `.venv`，并让 PyCharm Project Interpreter、PyCharm Managed Server/Jupyter kernel、终端命令全部指向它。

## PyCharm Managed Server 提示的含义

当 PyCharm 提示 “The interpreter for the current project Managed Server has changed. To use the updated interpreter, a new server launch is required”，意思是当前 notebook/server 仍在旧解释器里运行。选择 stop/restart 会让新 server 使用新解释器；选择 detach 会保留旧 server，它仍然使用旧解释器，因此刚切换的 venv 不会立刻生效。

这不是 torch 安装失败，也不是 Python 环境损坏；它是 IDE 在提醒“后台进程要重启才能吃到解释器变更”。

## 安装命令的安全写法

始终把 pip 绑定到目标 Python:

```bash
/path/to/python -m pip install torch
```

在 notebook 里也可以用同一原则:

```python
import sys
!{sys.executable} -m pip install torch
```

这样安装目标就是当前 notebook kernel，而不是 shell 里碰巧排在 PATH 前面的 `pip`。

事实: PyTorch 官方安装页建议用 pip 安装预编译包；当前稳定版要求 Python 3.10 或更新版本。macOS 上 Python 3.x 的基本安装命令是 `pip3 install torch torchvision`，实际项目可按需要只安装 `torch`。

## 边界

不使用 venv 也可以安装 torch，但包会进入全局 Python 或用户级 site-packages，容易和其他项目互相污染。macOS 上 Homebrew/Python.org/system Python 的权限和 externally-managed 策略也可能让全局安装变脆。只有在一次性实验、容器环境、或明确可丢弃的机器环境里，才适合把重型 ML 包直接装到全局解释器。
