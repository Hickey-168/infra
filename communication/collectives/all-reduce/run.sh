#!/usr/bin/env bash
set -euo pipefail
# 单机多进程，按本机可用 GPU/CPU 调整 nproc
torchrun --standalone --nproc_per_node=2 lab.py
