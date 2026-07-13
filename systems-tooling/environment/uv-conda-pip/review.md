# 复习卡片 — uv / conda / pip 环境边界

## 一句话

`torch` 是否可 import，取决于当前运行代码的 `sys.executable` 对应环境里有没有安装它，而不是这台机器或另一个项目的 venv 里有没有安装。

## 快速诊断

运行:

```bash
python -c "import sys; print(sys.executable)"
python -m pip show torch
```

在 notebook 里运行:

```python
import sys
print(sys.executable)
```

如果这两个路径不同，就说明终端和 notebook 不是同一个 Python。用 `{sys.executable} -m pip install ...` 可以把包装到 notebook 当前 kernel。

## PyCharm 提示

“Managed Server interpreter changed，需要 new server launch” 表示后台 notebook/server 还在旧解释器里。要让新 venv 生效，应停止并重新启动 server；detach 只会保留旧 server，旧 server 继续使用旧解释器。

## 推荐实践

对本仓库，优先使用项目根目录 `.venv`。让 PyCharm 项目解释器、Jupyter kernel、终端都指向同一个 `.venv/bin/python`。安装包时写成:

```bash
.venv/bin/python -m pip install torch
```

## 常见误区

- 误区: “我在别的项目 venv 装过 torch，所以这里也能用。”
- 更正: venv 的目的就是隔离包；另一个 venv 的 `site-packages` 不会自动进入当前环境。

- 误区: “切换 PyCharm interpreter 后，当前 notebook cell 会立刻用新环境。”
- 更正: notebook/server 是已经启动的进程，必须重启 kernel/server 才能换解释器。

- 误区: “裸 `pip install torch` 一定装到当前项目。”
- 更正: 裸 `pip` 取决于 PATH；`python -m pip` 才绑定到指定解释器。

## 回忆检查

- 我能说出当前 notebook 的 `sys.executable` 吗？
- 我安装包时是否使用了同一个解释器的 `python -m pip`？
- PyCharm 改解释器后，我是否重启了 Managed Server 或 notebook kernel？
