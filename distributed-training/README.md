# Distributed Training

分布式训练目录关注多卡/多机训练的状态切分、通信和调度组合。

## Structure

- `data-parallel`: DDP、gradient buckets、overlap。
- `sharding`: FSDP、ZeRO-1/2/3、offload。
- `model-parallel`: tensor parallel、pipeline parallel、sequence/context parallel。
- `expert-parallel`: MoE router、token dispatch、all-to-all。
- `frameworks`: Megatron-LM、DeepSpeed、PyTorch native distributed。
- `checkpoint`: distributed checkpoint、resharding、resume。

## Priority Notes

先理解 DDP all-reduce 和 FSDP/ZeRO memory accounting，再深入 Megatron 的 TP/PP/SP/EP/CP 组合。框架配置只有在你能解释 rank group 和通信 volume 时才有意义。

