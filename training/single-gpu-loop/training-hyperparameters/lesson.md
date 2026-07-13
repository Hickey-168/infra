---
artifact: course-lesson
leaf: training/single-gpu-loop/training-hyperparameters
language: zh-CN
estimated_time: 90min
environment: Mac
---

# 课程：把训练超参映射回状态机

**micro batch size** 决定单次 forward/backward 的 B，直接影响 activation memory；**gradient accumulation steps** 让多次 microbatch grad 累加后再 step，近似扩大 global batch，却不消除每次的 activation 峰值。分布式下常用近似：`global_batch = micro_batch * accumulation * world_size`。

**sequence length** 改变 T，影响 token 数、attention compute、activation 和 KV 相关工作；它不是普通 batch 参数。**learning rate** 改 optimizer update 幅度；**weight decay** 改更新规则；**warmup** 让早期 lr 从小值增长以降低训练初期不稳定风险；**epochs/steps** 决定数据被看几遍或更新几次。

超参联动是常态：增大 global batch 却不调整 lr/训练步数会改变优化轨迹；增大 T 可能先触发显存而非数据吞吐瓶颈。没有跨模型通用的最佳值，课程目标是形成可测试假设。

**实验课：** 固定数据和 seed，比较两个 micro batch/accumulation 组合但保持 global batch 相同；记录 loss 曲线、step 数、peak memory。再只改 lr 观察一个稳定/不稳定边界。报告注明这只是 toy workload 的观察。通关时能说出每个超参影响状态机哪一箭头。
