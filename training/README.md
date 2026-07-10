# Training

训练目录关注单机训练机制：forward/backward、优化器、混合精度、activation 显存、checkpoint、数据加载和质量回归。

## Structure

- `memory`: activation、optimizer state、gradient accumulation、checkpointing。
- `precision`: FP32、TF32、FP16、BF16、FP8。
- `optimizer`: SGD、AdamW、fused optimizer、state sharding 的前置知识。
- `data`: dataloader、prefetch、pin memory、数据格式。
- `checkpoint`: save/load/resume、rng、scheduler、optimizer state。
- `evaluation`: 训练优化后的质量回归验证。

## Exit Bar

你应该能对一个小 transformer 手算显存组成，预测 activation checkpointing 或 mixed precision 的效果，并用 profiler/实验验证。

