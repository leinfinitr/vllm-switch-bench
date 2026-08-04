# 单卡模型切换与自助路由系统对比（2026-07-21）

## 1. 摘要

在同一台单 NVIDIA RTX 3080 10 GiB 主机上，我们实测 Proposed、stock vLLM L1 sleep、llama-swap、ServerlessLLM 和 SwapServeLLM。统一生命周期指标定义为：

> **一次模型切换时间 = sleep / evict / swap-out 阶段 + wake / restore / swap-in 阶段。**

对具有双模型自主路由能力并通过本机双模型 gate 的 Proposed 与 llama-swap，进一步使用完全相同的 20-request Qwen2.5-1.5B/3B 流式请求 trace 比较端到端 makespan。

主要结论：

- Proposed 稳态切换中位数为 **0.402 s（1.5B）/ 0.606 s（3B）**，是本轮最快方案。
- stock vLLM L1 为 **0.501 s / 0.803 s**；Proposed 分别快约 **19.9% / 24.5%**。收益主要来自重复 sleep 时复用已存在的 pinned CPU clean backup，省去 D2H 复制。
- SwapServeLLM 为 **1.317 s / 1.930 s**，可真实释放 GPU residency 并恢复推理，但整进程 CUDA checkpoint 的双向复制开销更高。
- llama-swap 请求可见切换为 **12.658 s / 13.644 s**。它没有显式 sleep/wake API，因此该数值是“终止当前进程 + 启动目标进程 + 首次完整流式推理”，不是严格 phase decomposition。
- ServerlessLLM 1.5B delete/register 切换为 **16.182 s**；3B 因一次启动失败后逻辑 GPU reservation 未回收而未通过 runnable gate，不能报告 latency。
- 在 20-request 自主路由 trace 上，Proposed alternating makespan **29.464 s**，llama-swap **53.707 s**；Proposed 缩短 **45.1%**。Burst/locality 场景为 **28.852 s vs 53.781 s**，缩短 **46.4%**。

这些是单机 exploratory 结果，不是跨机器或顶会 confirmatory 结论。生命周期每格 4–5 个稳态周期；E2E Proposed 只有 2 个 alternating 和 1 个 strict-success burst run，llama-swap 每场景 3 个 run。

## 2. 实验环境与固定边界

- GPU：NVIDIA GeForce RTX 3080，10,240 MiB
- 模型：Qwen2.5-1.5B-Instruct、Qwen2.5-3B-Instruct
- 精度：FP16/half（SwapServeLLM vLLM image 使用 `dtype=auto`，Qwen checkpoint 仍为 FP16）
- `max_model_len=1024`
- 生命周期主要统计为稳态中位数；Proposed/vLLM 排除第一次建立 CPU backup 的 iteration 0
- 请求严格成功：HTTP 2xx、无 error、完整 SSE `[DONE]`、semantic TTFT 存在、输出非空
- 本机实测与论文/README 数字不混排

代码/版本：

| 系统 | 固定版本与运行方式 |
|---|---|
| Proposed | modified vLLM `b2057ef` + controller 分支；本次 in-process lifecycle 直接使用 modified vLLM |
| stock vLLM L1 | upstream tree `0decac0d...`，使用相同已编译扩展，仅 Python source 切回 upstream tree |
| llama-swap | `c6adf57`，本机 Go binary，按需启动两个 vLLM 子进程 |
| ServerlessLLM | source `2618762...` 只读 overlay 到旧官方 image；验证当前 delete cleanup 路径 |
| SwapServeLLM | `69f8aec...`，rootless Podman + NVIDIA CDI + `cuda-checkpoint` `00d5cce...` |

## 3. 生命周期定义与公平性

统一报告 `sleep + wake`，但不同系统保持的状态不同，因此这是**操作口径统一**，不是机制完全等价：

| 系统 | sleep/evict 终点 | wake/restore 终点 |
|---|---|---|
| Proposed | L1 sleep 完成；权重已释放到 pinned CPU backup；重复周期复用 clean backup | `/wake_up` 完成且后续推理正确 |
| stock vLLM L1 | L1 sleep 完成；每轮重新做 GPU→CPU backup | `/wake_up` 完成且后续推理正确 |
| SwapServeLLM | vLLM unload + CUDA checkpoint + GPU residency 消失 + 容器 pause | unpause + CUDA restore + vLLM load；后续推理正确 |
| ServerlessLLM | delete 完成、模型从列表消失、GPU 回落到 idle threshold | register 后首个成功推理完成 |
| llama-swap | 没有显式 phase API | 请求可见的旧进程终止、目标进程启动和首次完整推理合并计时 |

llama-swap 的数字含一次实际推理；ServerlessLLM wake 也以首个成功推理为终点；Proposed/vLLM/SwapServe 的 wake API 后另做正确性推理但不计入 wake。因而 llama-swap 和 ServerlessLLM 的生命周期数字是更保守的 request-visible 口径，应避免将微小差异解释为机制本身的精确倍数。

## 4. 不同模型的 sleep + wake 结果

### 4.1 稳态中位数

| 系统 | 模型 | 成功/尝试 | sleep 中位数 | wake 中位数 | **sleep+wake 中位数** |
|---|---:|---:|---:|---:|---:|
| **Proposed** | 1.5B | 5/5 | 0.107 s | 0.294 s | **0.402 s** |
| **Proposed** | 3B | 5/5 | 0.138 s | 0.468 s | **0.606 s** |
| stock vLLM L1 | 1.5B | 5/5 | 0.208 s | 0.294 s | **0.501 s** |
| stock vLLM L1 | 3B | 5/5 | 0.339 s | 0.464 s | **0.803 s** |
| SwapServeLLM | 1.5B | 5/5 | 0.661 s | 0.654 s | **1.317 s** |
| SwapServeLLM | 3B | 5/5 | 0.926 s | 1.003 s | **1.930 s** |
| llama-swap* | 1.5B | 4/4 | — | — | **12.658 s** |
| llama-swap* | 3B | 5/5 | — | — | **13.644 s** |
| ServerlessLLM | 1.5B | 5/5 | 0.830 s | 15.356 s | **16.182 s** |
| ServerlessLLM | 3B | gate failed | — | — | **不报告** |

![生命周期结果](../../results/model_switch_eval/latest/lifecycle-switch-time.png)

### 4.2 Proposed 相对 stock vLLM

- 1.5B：`0.501 → 0.402 s`，总切换时间降低约 **19.9%**。
- 3B：`0.803 → 0.606 s`，降低约 **24.5%**。
- wake 阶段几乎相同：1.5B `0.294 vs 0.294 s`，3B `0.468 vs 0.464 s`。
- 差异集中在 sleep：1.5B `0.208 → 0.107 s`；3B `0.339 → 0.138 s`。
- Modified allocator 的 profile 显示重复周期 `cpu_backup_reused_bytes` 非零且 D2H copy time 为 0；因此收益与“保留 clean pinned backup，重复 sleep 不再复制权重回 CPU”机制一致。

## 5. 自主路由端到端场景

### 5.1 场景设计

只有 Proposed 与 llama-swap 在本机实际通过了双模型自主路由 gate，故进入 E2E 排名。ServerlessLLM 当前单卡 scheduler 在双模型失败路径存在 reservation leak；SwapServeLLM 本轮仅验证单模型 lifecycle，未把多个启动时全被 swap-out 的容器包装成公平双模型路由结果；stock vLLM sleep 本身没有多模型 router。

统一 trace：

- 每 run 20 个请求；Qwen 1.5B/3B；相同 prompt、`max_tokens=32`、`temperature=0`、`seed=1`、streaming。
- **Alternating**：A/B 高频交替，1.5 s 到达间隔。
- **Burst/locality**：局部性 burst 后切换模型，检验 switch coalescing/locality。
- Makespan：从 trace 时刻 0 到最后一个请求完成。

### 5.2 端到端结果

| 场景 | Proposed | llama-swap | Proposed 缩短 |
|---|---:|---:|---:|
| Alternating | **29.464 s**（2 runs, 40/40 requests） | **53.707 s**（3 runs, 60/60） | **45.1%** |
| Burst/locality | **28.852 s**（1 run, 20/20 requests） | **53.781 s**（3 runs, 60/60） | **46.4%** |

![端到端 makespan](../../results/model_switch_eval/latest/e2e-makespan.png)

补充观察：

- Proposed alternating run-median semantic TTFT 约 **536 ms**；llama-swap 约 **12.807 s**。
- Proposed burst strict-success run 的 semantic TTFT 中位数约 **20 ms**；llama-swap 约 **15.273 s**。
- llama-swap 在启动阶段会排队并合并相同目标请求，因此 E2E 测量的是完整 control-plane 行为，不等于“每个名义模型变化都单独冷启动”。
- Proposed 的后续第二个 burst run 出现 3 个 deadline failure，并在下一个 cell 进入 `sleeping_in_progress` hang；成功表只使用 strict-success run。原始失败 JSONL、未完成 matrix、controller events 和终止日志未被保留，因此 `3/20` 及 hang 只能视为本机运行时观察，不能作为可独立审计的失败分母或 deadline-penalty 结果。这揭示当前 controller 在高并发/长尾路径仍需强化 deadline recovery 和 lifecycle reconciliation。

## 6. 外部系统具体发现

### 6.1 SwapServeLLM

- 1.5B 和 3B 均实际完成 5 次 swap-out/swap-in，后续推理正确。
- 1.5B awake GPU 约 7.1 GiB，swap-out 后约 4 MiB；3B 使用 `gpu_memory_utilization=0.90` 才能为默认 32K max length 留足 KV cache。
- 3B 在 `0.75` 下首先失败：vLLM 报告默认 32K 至少需要 1.12 GiB KV cache，而只剩 0.34 GiB；调整到 0.90 后通过。由于 SwapServeLLM 配置未暴露 `max_model_len`，这与其他系统的 1024 context 不是完全相同的资源配置，结果需标注为 operational comparison。

### 6.2 ServerlessLLM

- 1.5B current-source overlay 完成 5/5 delete/register 周期；delete 后目标模型从 API 列表消失，总 GPU 从约 6171 MiB 回落到 233 MiB，且 5 个后续 register/infer 周期均成功。当前 retained raw 未直接记录 Ray actor/process 消失或 scheduler logical GPU reservation，因此这里只声明 model-absent + aggregate-GPU-idle operational lifecycle，不声明完整 actor cleanup post-condition。
- 3B 在 `gpu_memory_utilization=0.75` 初次 engine 初始化失败；该失败 actor 被人工 kill 后，物理 GPU 已释放，但 scheduler 的 `free_gpu` 仍保持 0，后续 `0.80` 重试持续 `No available node`。
- 这属于真实 lifecycle correctness blocker；不能只重启服务后取一个成功数字，也不能将其记为超时 latency。

## 7. 对当前实现的总结

### 优点

1. **切换路径快**：在同模型/同 engine 的稳态生命周期对照中，比 stock vLLM L1 快约 20–25%。
2. **优势机制明确**：主要优化 sleep 阶段，通过复用 pinned clean backup 避免重复 D2H，而不是改变推理计算或隐藏 wake 时间。
3. **双模型请求驱动**：一个 OpenAI-compatible endpoint 根据 `model` 自动 sleep 当前模型、wake 目标模型并转发请求。
4. **流式正确性门禁较强**：严格要求 SSE 完整结束和非空 semantic output；reservation 在终止路径释放。
5. **单卡实用性优于进程冷切换**：20-request trace 的 makespan 比 llama-swap 低约 45–46%，semantic TTFT 差距更大。

### 当前限制与下一步

1. **高并发失败恢复仍有问题**：本轮 Proposed 的第二个 burst run 出现 deadline failure，随后 lifecycle 停在 `sleeping_in_progress`。需要把 shared transition timeout 后的 backend state reconciliation、强制 rollback 和 request draining 再做一轮故障注入。
2. **CPU pinned backup 占 host memory**：速度换取常驻 pinned 内存；必须继续维护 pressure-triggered release、释放确认和重建策略。
3. **进程池启动成本未计入稳态 switch**：两个 engine 预初始化后测量；适用于长期服务，不代表首次部署冷启动。
4. **统计强度有限**：本轮为 exploratory，E2E 样本尤其少；正式论文结论应执行至少 12 个独立初始化 paired blocks，并对失败计 deadline penalty。
5. **运行时 provenance 不完整**：Proposed lifecycle raw 绑定了 benchmark/vLLM commit，但 E2E 与外部系统没有在 run start 统一保存 binary/import path、配置 hash、容器 image digest 和 dirty state。报告中的外部 commit 是后置版本标签，不是与每个 run 的密码学绑定；后续 runner 必须自动写入这些字段以及 PID tree、GPU sampler。

## 8. Artifact 与复现

主要产物：

- `results/model_switch_eval/latest/summary.json`
- `results/model_switch_eval/latest/checksums.json`
- `results/model_switch_eval/raw/`
- `scripts/analyze_model_switch_eval.py`
- `scripts/plot_model_switch_eval.py`
- `docs/plans/2026-07-21-user-requested-switch-routing-evaluation.md`

checksum manifest 覆盖用于统计的 raw、结构化 summary 和图表；飞书 Markdown 副本与本报告不纳入 manifest。报告只引用本机真实执行结果；失败 gate 不生成性能数字。
