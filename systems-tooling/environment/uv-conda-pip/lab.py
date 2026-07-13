#!/usr/bin/env python3
"""uv conda pip — 验证脚本

叶子: systems-tooling/environment/uv-conda-pip
目标: 验证 pip、Python 解释器、notebook kernel 不是同一个抽象层。
"""
import importlib.util
import site
import sys
import sysconfig


def main() -> None:
    print("python executable:", sys.executable)
    print("python version:", sys.version.split()[0])
    print("purelib:", sysconfig.get_paths()["purelib"])
    print("user site:", site.getusersitepackages())

    spec = importlib.util.find_spec("torch")
    if spec is None:
        print("torch import target: NOT FOUND in this interpreter")
    else:
        print("torch import target:", spec.origin)

    print()
    print("Install into this exact interpreter with:")
    print(f"{sys.executable} -m pip install torch")


if __name__ == "__main__":
    main()
