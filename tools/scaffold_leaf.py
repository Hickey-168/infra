#!/usr/bin/env python3
"""
scaffold_leaf.py — 按需为 infra 知识库的叶子目录生成标准学习文件布局。

设计原则（对齐 atlas/learning-directory.md 与 templates/）：
  - 只在你真正开始学某个叶子时才 scaffold，不预先污染 490 个空目录。
  - 每个叶子的文件布局固定，方便 3 个月后扫读与复习。
  - 代码文件形态按叶子所属领域自适应（notebook / py+run.sh / yaml+sh）。
  - note.md 只放骨架，正文由你自己（费曼式）填写；AI 只辅助精读与出题。

用法:
  python scaffold_leaf.py operators/triton/vector-add
  python scaffold_leaf.py fundamentals/pytorch-tensor-lifecycle/view-vs-reshape --force
  python scaffold_leaf.py --list-domain operators/triton   # 预览该领域会生成什么
"""
from __future__ import annotations
import argparse
import datetime as _dt
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]  # 脚本放在 tools/ 下
TEMPLATE_LEARNING = REPO_ROOT / "templates" / "learning-item.md"

# ── 领域 → 代码形态映射 ─────────────────────────────────────────
# code_kind:
#   "notebook"    : lab.ipynb（交互探索，CPU/单 GPU 可跑）
#   "script"      : lab.py + run.sh（可版本控制/可 diff，含多进程 torchrun 场景）
#   "distributed" : lab.py + run.sh（torchrun 多进程模板）
#   "config"      : manifest.yaml + run.sh（部署/K8s/serving 类）
#   "bench"       : bench.py + run.sh（profiling/benchmark 类，产出 results/）
DOMAIN_CODE_KIND = {
    "fundamentals": "notebook",
    "operators/triton": "notebook",
    "operators/attention": "notebook",
    "operators/fusion": "notebook",
    "operators/cuda": "script",          # 需 nvcc/nsight，脚本更合适
    "operators/compiler": "script",
    "operators/profiling": "bench",
    "operators/libraries": "notebook",
    "training/single-gpu-loop": "notebook",
    "training/memory": "notebook",
    "training/precision": "notebook",
    "training/data-pipeline": "script",
    "training/checkpointing": "script",
    "training/finetuning": "notebook",
    "training/compiler-runtime": "script",
    "training/frameworks": "script",
    "distributed-training": "distributed",
    "communication": "distributed",
    "inference/lifecycle": "notebook",
    "inference/decoding": "notebook",
    "inference/kv-cache": "notebook",
    "inference/scheduling": "notebook",
    "inference/attention-backends": "notebook",
    "inference/quantization": "notebook",
    "inference/multimodal": "notebook",
    "inference/frameworks": "script",
    "inference/serving-api": "script",
    "inference/distributed-inference": "distributed",
    "deployment": "config",
    "storage": "script",
    "retrieval-recommendation": "notebook",
    "observability-benchmarking/metrics": "bench",
    "observability-benchmarking/load-testing": "bench",
    "observability-benchmarking/profiling": "bench",
    "observability-benchmarking/tracing": "script",
    "observability-benchmarking/logging": "script",
    "observability-benchmarking/regression": "bench",
    "systems-tooling": "script",
}


def resolve_code_kind(rel: str) -> str:
    """最长前缀匹配领域，回退到顶层目录，再回退到 notebook。"""
    parts = rel.split("/")
    for depth in range(len(parts), 0, -1):
        prefix = "/".join(parts[:depth])
        if prefix in DOMAIN_CODE_KIND:
            return DOMAIN_CODE_KIND[prefix]
    return "notebook"


def title_from(rel: str) -> str:
    leaf = rel.rstrip("/").split("/")[-1]
    return leaf.replace("-", " ")


def note_md(rel: str) -> str:
    """基于 templates/learning-item.md 生成，预填 frontmatter 与定位。"""
    today = _dt.date.today().isoformat()
    leaf = rel.split("/")[-1]
    domain = rel.split("/")[0]
    base = TEMPLATE_LEARNING.read_text(encoding="utf-8") if TEMPLATE_LEARNING.exists() else ""
    frontmatter = (
        "---\n"
        f"leaf: {rel}\n"
        f"title: {title_from(rel)}\n"
        "status: in_progress   # planned | in_progress | verified\n"
        "priority: P?          # P0/P1/P2/P3, 见 atlas/priority-ladder.md\n"
        f"domain: {domain}\n"
        "environment: Mac       # Mac | WSL2-GPU | remote-GPU\n"
        f"date_started: {today}\n"
        "prereq: []            # 依赖的其它叶子路径\n"
        "unlocks: []           # 本叶子解锁的下游叶子\n"
        "---\n\n"
    )
    # 把模板里的 <Topic> 占位替换成叶子名
    body = base.replace("# <Topic>", f"# {title_from(rel)}")
    return frontmatter + body


def readme_md(rel: str, code_kind: str) -> str:
    leaf = title_from(rel)
    code_files = {
        "notebook": "- `lab.ipynb` — 交互式验证（失败模式实验 + 数量级估算），带完整中文注释。",
        "script": "- `lab.py` — 可运行验证脚本（含失败模式触发与修复），完整注释。\n- `run.sh` — 一键运行命令记录。",
        "distributed": "- `lab.py` — 多进程验证脚本（rank/collective/shape trace）。\n- `run.sh` — `torchrun --nproc_per_node=N lab.py` 启动记录。",
        "config": "- `manifest.yaml` — 声明式配置（K8s/serving/容器）。\n- `run.sh` — 部署/验证命令记录。",
        "bench": "- `bench.py` — benchmark/profiling 脚本（含 warmup、多次测量、单位）。\n- `run.sh` — 采集命令。\n- `results/` — csv/json/trace 摘要（大文件走 .gitignore）。",
    }[code_kind]
    return (
        f"# {leaf}\n\n"
        f"> 叶子路径: `{rel}` · 代码形态: `{code_kind}`\n\n"
        "本目录是一个可独立学习与验证的机制叶子。文件约定：\n\n"
        "- `note.md` — 我自己写的费曼式笔记（唯一需要我亲手总结的文件）。\n"
        f"{code_files}\n"
        "- `questions.md` — 盲点自测题（AI 出题，我闭卷作答后回填正解与漏点）。\n\n"
        "学习流程见 skill `infra-leaf-study`：诊断 → 精读 → 重构(填 note.md) → 实践(跑代码) → 盲点扫描(questions.md)。\n\n"
        "完成标准：note.md 的 Verification Checklist 全勾 + questions.md 正确率 ≥ 80%，"
        "然后把 frontmatter 的 `status` 改为 `verified`。\n"
    )


def questions_md(rel: str) -> str:
    return (
        f"# 盲点自测 — {title_from(rel)}\n\n"
        "> 规则：闭卷口述作答，答完再看正解。答错的题记到「漏点」，作为下次复习入口。\n\n"
        "## 五维度题目（AI 生成后填入）\n\n"
        "### Q1 边界/反例\n- 问题：\n- 我的初答：\n- 正解：\n- 漏在哪里：\n\n"
        "### Q2 底层实现\n- 问题：\n- 我的初答：\n- 正解：\n- 漏在哪里：\n\n"
        "### Q3 性能权衡\n- 问题：\n- 我的初答：\n- 正解：\n- 漏在哪里：\n\n"
        "### Q4 跨概念串联\n- 问题：\n- 我的初答：\n- 正解：\n- 漏在哪里：\n\n"
        "### Q5 数量级感知\n- 问题：\n- 我的初答：\n- 正解：\n- 漏在哪里：\n"
    )


NOTEBOOK_STUB = """{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# LEAF_TITLE — 验证 lab\\n", "\\n", "对应叶子: `LEAF_REL`\\n", "\\n", "本 notebook 用于**用代码检验理解**，不是复现教程。至少包含一个失败模式实验或一个数量级估算。"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["# 环境自检\\n", "import sys\\n", "print(sys.version)\\n", "try:\\n", "    import torch\\n", "    print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())\\n", "except ImportError:\\n", "    print('torch 未安装')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 假设 (Hypothesis)\\n", "\\n", "我预测：..."]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["# 失败模式实验 / 数量级估算\\n", "# TODO: 写一段会触发预期现象的代码，观察是否与假设一致\\n"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 观察与解释\\n", "\\n", "- 观察（保留数字与单位）：\\n", "- 解释：\\n", "- **不能证明**：\\n"]}
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
 "nbformat": 4, "nbformat_minor": 5
}
"""

SCRIPT_STUB = '''#!/usr/bin/env python3
"""LEAF_TITLE — 验证脚本

叶子: LEAF_REL
目标: 用代码检验对本机制的理解，至少包含一个失败模式或一个可量化观察。
"""
import sys


def main() -> None:
    print("Python", sys.version)
    # ── 假设 ──────────────────────────────────────────
    # 我预测: ...
    # ── 失败模式 / 观察 ───────────────────────────────
    # TODO: 触发预期现象，打印关键数字与单位
    # ── 解释 ──────────────────────────────────────────
    # 证明了什么 / 不能证明什么
    pass


if __name__ == "__main__":
    main()
'''

DISTRIBUTED_STUB = '''#!/usr/bin/env python3
"""LEAF_TITLE — 多进程验证

叶子: LEAF_REL
启动: torchrun --nproc_per_node=2 lab.py   (见 run.sh)
目标: 观察 rank / collective / shape / 通信量，验证对并行机制的理解。
"""
import os


def main() -> None:
    import torch
    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    # ── 假设 ──────────────────────────────────────────
    # 我预测: ...
    # TODO: 构造张量 -> 执行 collective -> 各 rank 打印 shape/值验证语义
    if rank == 0:
        print(f"world_size={world}, backend={backend}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
'''

BENCH_STUB = '''#!/usr/bin/env python3
"""LEAF_TITLE — benchmark / profiling

叶子: LEAF_REL
产出: results/ 下的 csv/json；报告 warmup、多次测量、均值与方差、单位。
"""
import time
import statistics


def timed(fn, iters: int = 50, warmup: int = 10):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    return statistics.mean(samples), statistics.pstdev(samples)


def main() -> None:
    # ── 假设 ──────────────────────────────────────────
    # roofline 判断: 这个 workload 应该是 compute-bound / memory-bound
    # TODO: 定义 workload, 用 timed() 测量, 对比理论上界
    mean, std = timed(lambda: sum(range(10000)))
    print(f"latency: {mean:.3f} ± {std:.3f} ms")


if __name__ == "__main__":
    main()
'''

RUN_SH = {
    "script": "#!/usr/bin/env bash\nset -euo pipefail\npython lab.py\n",
    "distributed": "#!/usr/bin/env bash\nset -euo pipefail\n# 单机多进程，按本机可用 GPU/CPU 调整 nproc\ntorchrun --standalone --nproc_per_node=2 lab.py\n",
    "config": "#!/usr/bin/env bash\nset -euo pipefail\n# 部署/验证命令，例如:\n# kubectl apply -f manifest.yaml\n# 或本地 dry-run:\n# docker run ... / vllm serve ...\necho 'fill in deploy/verify commands'\n",
    "bench": "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p results\npython bench.py | tee results/run.log\n",
}

MANIFEST_YAML = "# LEAF_TITLE — 声明式配置骨架\n# 叶子: LEAF_REL\n# TODO: 填入最小可验证的 K8s/serving/容器配置\napiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: leaf-placeholder\ndata:\n  note: \"replace with real manifest\"\n"


def write_if_absent(path: pathlib.Path, content: str, force: bool) -> str:
    if path.exists() and not force:
        return f"skip (exists): {path.relative_to(REPO_ROOT)}"
    path.write_text(content, encoding="utf-8")
    return f"write: {path.relative_to(REPO_ROOT)}"


def scaffold(rel: str, force: bool) -> list[str]:
    rel = rel.strip().strip("/")
    leaf_dir = REPO_ROOT / rel
    if not leaf_dir.exists():
        sys.exit(f"叶子目录不存在: {leaf_dir}\n请确认路径与 atlas/learning-directory.md 一致。")
    code_kind = resolve_code_kind(rel)
    title = title_from(rel)
    logs = []

    logs.append(write_if_absent(leaf_dir / "README.md", readme_md(rel, code_kind), force))
    logs.append(write_if_absent(leaf_dir / "note.md", note_md(rel), force))
    logs.append(write_if_absent(leaf_dir / "questions.md", questions_md(rel), force))

    def sub(stub: str) -> str:
        return stub.replace("LEAF_TITLE", title).replace("LEAF_REL", rel)

    if code_kind == "notebook":
        logs.append(write_if_absent(leaf_dir / "lab.ipynb", sub(NOTEBOOK_STUB), force))
    elif code_kind == "script":
        logs.append(write_if_absent(leaf_dir / "lab.py", sub(SCRIPT_STUB), force))
        logs.append(write_if_absent(leaf_dir / "run.sh", RUN_SH["script"], force))
    elif code_kind == "distributed":
        logs.append(write_if_absent(leaf_dir / "lab.py", sub(DISTRIBUTED_STUB), force))
        logs.append(write_if_absent(leaf_dir / "run.sh", RUN_SH["distributed"], force))
    elif code_kind == "config":
        logs.append(write_if_absent(leaf_dir / "manifest.yaml", sub(MANIFEST_YAML), force))
        logs.append(write_if_absent(leaf_dir / "run.sh", RUN_SH["config"], force))
    elif code_kind == "bench":
        logs.append(write_if_absent(leaf_dir / "bench.py", sub(BENCH_STUB), force))
        logs.append(write_if_absent(leaf_dir / "run.sh", RUN_SH["bench"], force))
        (leaf_dir / "results").mkdir(exist_ok=True)
        logs.append(f"mkdir: {(leaf_dir / 'results').relative_to(REPO_ROOT)}")

    # 保留 .gitkeep 语义: 有真实文件后 .gitkeep 可留可删，这里不动它
    return logs


def main() -> None:
    ap = argparse.ArgumentParser(description="按需为 infra 叶子目录生成标准学习文件布局")
    ap.add_argument("leaf", nargs="?", help="叶子相对路径, 如 operators/triton/vector-add")
    ap.add_argument("--force", action="store_true", help="覆盖已存在文件")
    ap.add_argument("--list-domain", metavar="DOMAIN", help="预览某领域的代码形态")
    args = ap.parse_args()

    if args.list_domain:
        print(f"{args.list_domain} -> code_kind = {resolve_code_kind(args.list_domain)}")
        return
    if not args.leaf:
        ap.error("需要提供叶子路径，或用 --list-domain 预览")
    for line in scaffold(args.leaf, args.force):
        print(line)


if __name__ == "__main__":
    main()
