# 2026-07-21 模型切换与自助路由比较协议

## 目标

1. 对 Proposed、llama-swap、ServerlessLLM、SwapServeLLM、vLLM Sleep 测量模型生命周期转换。
2. 统一定义单模型一次转换时间为 `sleep/evict/swap-out + wake/restore/swap-in`；只有两个阶段都完成物理/功能 post-condition 且恢复后推理正确，才生成时间。
3. 对能根据 OpenAI request 的 `model` 字段自主选择/切换后端的系统运行同一双模型端到端 trace。

## 模型与公共条件

- Qwen2.5-1.5B-Instruct、Qwen2.5-3B-Instruct，FP16。
- `max_model_len=1024`、`gpu_memory_utilization=0.70`、`enforce_eager`、单 RTX 3080 10 GiB。
- 生命周期每 model/system 目标 5 次 steady-state cycles；若系统只能通过更小模型的机制 gate，则只报告 gate，不跨模型推断。
- 页面缓存不做全局 drop；涉及 reload 的结果标为 warm-page-cache operational result。

## 方法语义

- Proposed：同一 model process 的优化 L1，第一次 sleep/wake 为 backup miss；后续 cycle 使用 retained clean backup，使下一次 sleep 跳过 D2H。报告 steady cycles 的 `sleep+wake`，并披露 host backup 成本。
- vLLM Sleep：stock/upstream-compatible L1 sleep/wake。若当前机器无法构建/运行与 Proposed 同版本栈兼容的 stock checkout，保留 exact blocker；不得用 Proposed 的实现冒充 stock vLLM。
- llama-swap：机制为 terminate/start，没有显式 sleep+wake API。公平表中把 terminate 记为 sleep、目标 process ready 记为 wake；这是 process-switch proxy，不是 CPU-backed sleep。
- ServerlessLLM：delete/scale-to-zero 只有在 model metadata 消失、backend actor/process 消失、logical GPU 归还且物理 GPU idle 后才算 sleep；register + first successful inference 算 wake。若 post-condition 失败，不生成 latency。
- SwapServeLLM：`/api/swapout` + `/api/swapin`；必须验证 CUDA PID checkpointed、GPU residency 下降、pause/restore 和恢复后 inference。1.5B/3B 分别运行。

## 端到端 workload

只纳入 request-driven autonomous routers：Proposed、llama-swap；ServerlessLLM 只有在当前源码成功 register/infer/delete/re-register 且能按请求自动 scale/load 时才纳入；SwapServeLLM 当前 controller 也可按 model request 唤醒，但单容器/模型配置不能公平执行同一双模型 trace，除非建立两模型配置并过 gate；vLLM sleep API 本身无多模型 router，不纳入 E2E。

使用完全相同的 20-request frozen traces：

- alternating：A/B 交替；
- burst/locality：A×5/B×5/A×5/B×5；
- 同一 absolute scheduled offsets、prompt、32 output tokens、temperature=0、seed、streaming。

成功：HTTP 2xx、无 SSE error、非空 semantic token、完整 `[DONE]`、目标输出非空。主要报告 trace makespan（第一个 scheduled arrival 到最后一个成功 completion）及 semantic TTFT/E2E；失败保留在分母。

## 证据等级

- 可比数字：目标模型、公共参数、完整 post-condition、重复与 raw provenance 均通过。
- 机制 smoke：机制真实执行但模型/路由/重复不完整，不进入排名。
- Blocker：保留 source/image、exact command、error/post-condition failure；不估算数字。
