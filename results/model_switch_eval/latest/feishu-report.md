# LLM Model Switch

## 1. 结论摘要

测试环境为单张 NVIDIA RTX 3080 10 GiB；模型为 Qwen2.5-1.5B-Instruct 和 Qwen2.5-3B-Instruct。统一生命周期指标定义为：

> **一次模型切换时间 = sleep / evict / swap-out + wake / restore / swap-in。**

主要结果：

- **Proposed 最快**：1.5B 为 **0.402 s**，3B 为 **0.606 s**。
- stock vLLM L1 分别为 **0.501 s / 0.803 s**；Proposed 降低约 **19.9% / 24.5%**。
- SwapServeLLM 为 **1.317 s / 1.930 s**。
- llama-swap 请求可见切换为 **12.658 s / 13.644 s**。
- ServerlessLLM 1.5B 为 **16.182 s**；3B 因 startup failure 后 scheduler GPU reservation 未回收，没有报告 latency。
- 20-request 自主路由 trace：Proposed alternating **29.464 s**，llama-swap **53.707 s**；burst/locality **28.852 s vs 53.781 s**。

> 结果为单机 exploratory evidence，不是独立 block 的 confirmatory 论文结论。生命周期每格 4–5 个稳态周期；E2E Proposed 仅 2 个 alternating、1 个 strict-success burst run，llama-swap 每场景 3 个 run。

## 2. 测试口径

### 2.1 环境

| 项目 | 设置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080，10,240 MiB |
| 模型 | Qwen2.5-1.5B-Instruct、Qwen2.5-3B-Instruct |
| 精度 | FP16/half；SwapServe vLLM image 使用 dtype=auto |
| Context | 主实验 max_model_len=1024；SwapServe 配置未暴露该参数，3B 使用默认 32K |
| 请求 | Streaming；max_tokens=32；temperature=0；seed=1 |
| 成功条件 | HTTP 2xx、无 error、SSE 完整 [DONE]、semantic TTFT 存在、输出非空 |

### 2.2 生命周期 post-condition

| 系统 | sleep / evict 终点 | wake / restore 终点 |
|---|---|---|
| Proposed | vLLM L1 sleep 完成，权重释放到 pinned CPU clean backup | wake API 完成，后续推理正确 |
| stock vLLM L1 | L1 sleep 完成，每轮重新执行 GPU→CPU backup | wake API 完成，后续推理正确 |
| SwapServeLLM | vLLM unload + CUDA checkpoint + GPU residency 消失 + container pause | unpause + CUDA restore + vLLM load，后续推理正确 |
| ServerlessLLM | delete 完成、模型从列表消失、GPU 回落到 idle threshold | register 后首次成功推理 |
| llama-swap | 无显式 phase API | terminate current + start target + 首次完整流式推理合并计时 |

不同系统保留的进程、CUDA context 和 CPU state 不同，所以 `sleep+wake` 是操作口径统一，不代表机制完全等价。llama-swap 和 ServerlessLLM 的数字包含实际推理，是更保守的 request-visible 口径。

## 3. 不同模型的切换时间

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

### Proposed 对 stock vLLM 的机制差异

- 1.5B：`0.501 → 0.402 s`，降低 **19.9%**。
- 3B：`0.803 → 0.606 s`，降低 **24.5%**。
- wake 时间几乎相同；收益集中在 sleep。
- Modified allocator 的 profile 显示重复周期 `cpu_backup_reused_bytes` 非零且 D2H copy time 为 0：我们的实现复用已存在的 pinned clean backup，不需要每次 sleep 都把权重从 GPU 重新复制到 CPU。

## 4. 自主路由端到端对比

只有 Proposed 与 llama-swap 在当前机器实际通过双模型自主路由 gate，因此进入 E2E 排名。ServerlessLLM 的双模型失败路径有 scheduler reservation leak；SwapServeLLM 本轮只验证单模型 lifecycle；stock vLLM sleep 本身没有多模型 router。

### 4.1 Workload

- 每 run 20 个请求；1.5B/3B 使用相同 prompt 和 generation 参数。
- **Alternating**：A/B 高频交替，1.5 s 到达间隔。
- **Burst/locality**：同模型 burst 后再切换，模拟具有局部性的流量。
- 指标：从 trace 时刻 0 到最后请求完成的 makespan。

### 4.2 结果

| 场景 | Proposed | llama-swap | Proposed 缩短 |
|---|---:|---:|---:|
| Alternating | **29.464 s**（2 runs, 40/40） | **53.707 s**（3 runs, 60/60） | **45.1%** |
| Burst/locality | **28.852 s**（1 run, 20/20） | **53.781 s**（3 runs, 60/60） | **46.4%** |

- Proposed alternating run-median semantic TTFT 约 **536 ms**；llama-swap 约 **12.807 s**。
- Proposed burst strict-success run semantic TTFT 中位数约 **20 ms**；llama-swap 约 **15.273 s**。
- llama-swap 可在启动期间排队并 coalesce 相同目标请求，因此这里测量完整 control-plane 行为，不等于每个名义模型变化都独立冷启动。

## 5. 外部系统发现

### 5.1 SwapServeLLM

- 1.5B 与 3B 都真实完成 5 次 CUDA checkpoint/restore，GPU residency 释放且恢复后推理正确。
- 1.5B `sleep+wake=1.317 s`；3B `1.930 s`。
- 3B 在 `gpu_memory_utilization=0.75` 先失败：默认 32K context 至少需要 1.12 GiB KV cache，只剩 0.34 GiB；调到 0.90 后通过。
- 因其配置没有暴露 max_model_len，3B 与其余系统的 1024 context 不是完全相同资源条件，结果仅作 operational comparison。

### 5.2 ServerlessLLM

- 当前源码 overlay 的 1.5B 完成 5/5 delete/register 周期；delete 后 GPU 从约 6171 MiB 回落到 233 MiB。
- 3B 首次 engine 初始化失败；即使人工 kill failed actor，scheduler 的 `free_gpu` 仍为 0，重试持续 `No available node`。
- 这是真实 lifecycle correctness blocker；不把失败包装成性能数字。

## 6. 当前实现总结

### 优点

1. 同 engine 稳态切换比 stock vLLM L1 快约 20–25%。
2. 优化来源明确：复用 pinned clean backup，减少重复 D2H；没有隐藏 wake 时间。
3. 提供单一 OpenAI-compatible endpoint，根据 request 的 model 自动切换和路由。
4. 流式成功契约严格，要求完整 SSE 和非空语义输出。
5. 在 20-request trace 上比 llama-swap 减少约 45–46% makespan。

### 局限和待修复项

1. 本轮 Proposed 的后续第二个 burst run 出现 3 个 deadline failures，随后 lifecycle 卡在 `sleeping_in_progress`。成功表只使用 strict-success run，但这一失败意味着当前实现还不能做 reliability claim；需加强 timeout 后 state reconciliation、rollback 和 request draining。
2. Pinned backup 会占 host memory，必须维持 pressure-triggered release、release ack 和重建机制。
3. 切换结果不含两个 engine 首次进程启动成本，适用于长期服务，不代表部署冷启动。
4. E2E 样本量有限；论文级结论至少应执行 12 个独立初始化 paired blocks，失败计 deadline penalty。
5. 后续 runner 应自动写入 controller/vLLM executable hash、配置 hash、PID tree 和 GPU sampler。

## 7. Artifact

本地可审计产物：

- `results/model_switch_eval/latest/summary.json`
- `results/model_switch_eval/latest/checksums.json`
- `results/model_switch_eval/raw/`
- `docs/reports/model-switch-and-routing-evaluation-2026-07-21.md`

checksum manifest 覆盖用于统计的 raw、结构化 summary 和图表；飞书 Markdown 副本不纳入 manifest。所有进入报告的本机结果均来自实际运行；失败 gate 不生成 latency。
