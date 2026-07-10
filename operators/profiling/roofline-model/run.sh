#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
python bench.py | tee results/run.log
