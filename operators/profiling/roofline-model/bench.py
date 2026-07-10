#!/usr/bin/env python3
"""roofline model — benchmark / profiling

叶子: operators/profiling/roofline-model
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
