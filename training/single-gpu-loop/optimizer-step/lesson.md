---
artifact: course-lesson
leaf: training/single-gpu-loop/optimizer-step
language: zh-CN
estimated_time: 90min
environment: Mac
---

# 课程：从 SGD 到 AdamW 的参数更新

最小 SGD 为 `theta <- theta - lr * grad`。lr 是更新步长；过大可能震荡/发散，过小则收敛缓慢。weight decay 不是“把 loss 后处理”，而是 update 规则中的参数收缩项；AdamW 将其与 adaptive gradient 更新解耦。

Adam 类 optimizer 为每个 parameter 维护动量/方差等 state，因此首次 step 后显存账本会多出 tensors。state 属于 optimizer，不在 parameter `.grad` 里；checkpoint 若只保存 model state 而丢 optimizer/scheduler state，恢复训练的轨迹可能改变。

正确顺序通常是 backward -> 可选 unscale/clip -> step -> scheduler（依约定）-> zero_grad。对同一 gradient 调两次 step 会更新两次；在 grad 为 None 时某些 parameter 会被跳过，需按当前 PyTorch/optimizer contract 判断。

**实验课：** 对标量参数分别用手写 SGD 与 `torch.optim.SGD`，比较每步值；再用 AdamW 打印 `optimizer.state` 在第一步后出现的字段。记录 parameter、grad、lr、state 的变化时刻。通关时能解释 optimizer state 为什么属于 parameter-memory。
