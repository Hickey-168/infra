# Fundamentals

这个目录是 AI infra 的地基，目标不是补“通用 CS 全家桶”，而是补那些会直接影响推理、训练、GPU、分布式、可观测性的基础概念。

## Must-Know Leaves

- `pytorch-tensor-lifecycle`: shape/dtype/device/stride/layout/ownership。
- `autograd-graph-and-saved-tensors`: 训练显存为什么大。
- `module-forward-call-chain`: 从 module call 到 hooks/autograd。
- `memory-accounting-single-gpu`: 参数、梯度、优化器状态、activation 的显存估算。
- `python-async-and-queueing`: serving scheduler 的队列基础。
- `linux-process-and-file-io`: 数据加载、mmap、page cache、进程隔离。

## Exit Bar

学完这里后，你应该能看懂一个训练 step 或推理 request 中 tensor 在 CPU/GPU/内存/计算图之间怎么流动，并能写脚本验证自己的判断。

