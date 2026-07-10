# Deployment

这个目录对应 infra 中的部署和平台化能力：容器、Kubernetes、GPU 调度、模型服务框架、路由、可靠性、安全和弹性伸缩。

## Structure

- `containers`: CUDA 镜像、nvidia-container-runtime、训练/推理 Dockerfile、可复现构建。
- `kubernetes`: GPU request、device plugin、GPU Operator、MIG、time-slicing、HPA/KEDA。
- `serving`: Ray Serve、KServe、Triton Inference Server、vLLM/SGLang production server。
- `routing`: load balancing、session affinity、prefix-aware routing、canary、shadow traffic。
- `reliability`: health checks、graceful shutdown、request draining、OOM recovery。
- `security`: secrets、auth、network policy、镜像供应链、模型权限。

## Exit Bar

你应该能把一个 mock LLM 服务容器化，部署到本地或远端 K8s/Ray 环境，压测并解释 p95/p99、GPU 利用率和扩缩容行为。

