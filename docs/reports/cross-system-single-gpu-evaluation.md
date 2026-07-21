# 单 RTX 3080 上的跨系统多模型切换评测

日期：2026-07-21  
状态：**可复现的 exploratory artifact**；不是完整的 12-block confirmatory paper claim。

## 1. 结论摘要

在单张 RTX 3080 10 GiB 上，Qwen2.5-1.5B 与 Qwen2.5-3B 不能按本实验配置同时常驻 HBM。我们将 proposed controller 的 pinned CPU clean-backup reuse 与 vLLM cold/L1/L2 lifecycle 以及开源 `llama-swap` 进程级切换作真实本机比较。ServerlessLLM、SwapServeLLM 和 kvcached 经过可运行性门禁，但未满足可比的双模型 weight-switch contract，因此只报告 blocker，不生成或推断性能数字。

主要观测：

- **单模型 lifecycle（5 次独立进程 run / cell）**：
  - 1.5B：cold `13.382 s`，vLLM L1 `1.333 s`，vLLM L2 `0.859 s`；
  - 3B：cold `14.331 s`，vLLM L1 `2.868 s`，vLLM L2 `1.773 s`。
- **统一 request trace**：
  - proposed，alternating：run-median semantic TTFT `546.8 ms`（5 runs，bootstrap 95% CI `[545.1, 558.1]`）；
  - proposed，burst/locality：`19.7 ms`（5 runs，`[19.5, 21.1]`）；
  - llama-swap，alternating：`14.846 s`（3 runs，`[14.224, 46.077] s`）；
  - llama-swap，burst/locality：`15.290 s`（3 runs，`[13.851, 15.293] s`）。
- 对这批 exploratory runs，proposed 的 run-median semantic TTFT 相对 llama-swap 为 alternating `27.2×`、burst/locality `776.4×` 更低；两系统请求均严格成功（proposed `200/200`，llama-swap `120/120`）。

这些差距描述的是**当前单卡、当前软件版本和给定 traces**，不能外推到 H100、多 GPU 或生产 SLO。llama-swap 是控制面真实 external baseline，但其机制本质上是进程 cold start，不能被解读为与 pinned-backup reuse 的同机制消融。

## 2. 系统与版本

| 系统 | 版本/commit | 机制 | 本机门禁 | 是否有性能数据 |
|---|---|---|---|---|
| Proposed | controller `d78155f`；vLLM `b2057ef` | 双 vLLM 进程 + L1 pinned clean backup reuse + request-driven controller | 通过 strict SSE/drain/资源回收验证 | 是 |
| vLLM cold/L1/L2 | vLLM `b2057ef` | terminate/start；sleep level 1；sleep level 2 | 30/30 lifecycle run 成功 | 是 |
| llama-swap | `c6adf57` | OpenAI proxy 管理 vLLM terminate/start | 120/120 strict requests 成功 | 是，secondary |
| ServerlessLLM | `2618762` | serverless fast loading / scale-to-zero | 排除：DELETE 后模型列表为空，但 30 s 后仍有 `6171 MiB` GPU use，其中 `ray::VllmBackend` `5918 MiB` | 否 |
| SwapServeLLM | `69f8aec` | CUDA process/container checkpoint | 排除：缺 `cuda-checkpoint`/rootless podman 路径；pinned Go build 还缺 gpgme/btrfs dev headers | 否 |
| kvcached | 本地 `d78649d`；历史 artifact `623dbf2` | elastic KV/memory manager | 单模型 OpenAI smoke 曾通过，但无本评测需要的双模型 weight lifecycle contract | 否；只作相关系统 |

结构化记录：`results/cross_system/latest/external-systems.json`。

## 3. 实验条件

- GPU：NVIDIA GeForce RTX 3080，10,240 MiB；driver `580.95.05`。
- 模型：`Qwen2.5-1.5B-Instruct`、`Qwen2.5-3B-Instruct`，FP16。
- 公共 vLLM 参数：`--enforce-eager --gpu-memory-utilization 0.70 --max-model-len 1024 --dtype half`。
- 请求：OpenAI chat completions，streaming，`temperature=0`，固定 seed，最多 32 output tokens。
- 成功定义：HTTP 2xx、`error` 为空、semantic token 非空、完整 SSE `[DONE]`、`stream_done is True`。
- 原始 response first byte 没有被误称 semantic TTFT。
- 矩阵 runner 对 frozen manifest 的 `request_id/model/scheduled_offset_s` 逐行绑定，并保存 manifest/output SHA256。
- 统计：请求 trace 以 run 为重复单位；报告 run-median 的中位数以及固定 seed `20260721`、10,000 次 nonparametric bootstrap percentile CI。由于只有 3–5 runs，CI 只反映当前小样本，不宣称高功效 confirmatory inference。

## 4. Workload

### 4.1 Controlled lifecycle microbenchmark

每个模型、每个方法运行 5 次；每次重新启动 vLLM 并完成：ready → correctness inference → evict → restore → post-restore inference。activation latency 定义为 `evict + restore`，包括模型切换必须支付的 HBM release 和恢复阶段。

### 4.2 Request-visible traces

- **Alternating**：20 个请求按 A/B 交替，模拟低 locality、多租户模型轮转。
- **Burst/locality**：A×5/B×5/A×5/B×5，模拟同一模型短 burst 后切换。
- proposed：每 trace 5 runs，共 200 requests；
- llama-swap：每 trace 3 runs，共 120 requests。

这两个系统使用完全相同的 frozen manifests。llama-swap 由 upstream proxy 自动停止当前 vLLM 并启动目标模型；proposed 在 request path 内执行 sleep/wake 和 pinned backup reuse。

## 5. 结果

### 5.1 Lifecycle activation latency

| 模型 | 方法 | n | median | p95 | bootstrap median 95% CI | ready GPU | evicted GPU |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.5B | cold process | 5 | 13,382 ms | 13,770 ms | [12,879, 13,830] | 7,267 MiB | 1 MiB |
| 1.5B | vLLM L1 | 5 | 1,333 ms | 1,510 ms | [1,220, 1,534] | 7,289 MiB | 417 MiB |
| 1.5B | vLLM L2 | 5 | 859 ms | 964 ms | [793, 969] | 7,289 MiB | 411 MiB |
| 3B | cold process | 5 | 14,331 ms | 15,146 ms | [13,833, 15,337] | 7,277 MiB | 1 MiB |
| 3B | vLLM L1 | 5 | 2,868 ms | 3,004 ms | [2,723, 3,028] | 7,297 MiB | 403 MiB |
| 3B | vLLM L2 | 5 | 1,773 ms | 1,838 ms | [1,658, 1,843] | 7,297 MiB | 385 MiB |

![Lifecycle latency](../../../results/cross_system/latest/lifecycle-latency.png)

图中误差棒为 10,000 次 bootstrap 得到的 median 95% percentile CI；纵轴为对数尺度。

这里 L2 比 L1 更快并不矛盾：该 harness 的 level-1 `sleep` 要把 live allocations 复制到 CPU，然后 wake H2D；level-2 丢弃权重并用本地 page cache 重新加载。10 GiB HBM、64 GiB host RAM 和热文件缓存下，L2 的总路径可能更短。它不代表所有存储层次都如此。

### 5.2 Request-visible semantic TTFT

| 系统 | trace | runs | requests | run-median TTFT | 95% CI | pooled request TTFT p95 | run-median E2E | failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed | alternating | 5 | 100 | 546.8 ms | [545.1, 558.1] | 684.1 ms | 800.3 ms | 0/100 |
| Proposed | burst/locality | 5 | 100 | 19.7 ms | [19.5, 21.1] | 645.2 ms | 350.1 ms | 0/100 |
| llama-swap | alternating | 3 | 60 | 14,845.7 ms | [14,223.8, 46,076.5] | 56,918.1 ms | 15,169.5 ms | 0/60 |
| llama-swap | burst/locality | 3 | 60 | 15,289.9 ms | [13,851.0, 15,293.0] | 25,889.1 ms | 15,672.8 ms | 0/60 |

![Trace TTFT](../../../results/cross_system/latest/trace-ttft.png)

Alternating 下 proposed 几乎每个请求都触发真实模型切换，因此 run median 在约 0.55 s。Burst/locality 中大部分请求命中当前 awake model，run median 降至约 20 ms；pooled p95 仍约 645 ms，准确暴露了 burst 边界的切换请求。llama-swap 每次模型改变都要执行完整进程启动；首 run 的 alternating median 达 46.1 s，导致 3-run bootstrap CI 很宽，这个异常没有删除。

## 6. 机制和资源证据

本报告沿用最终 controller commit 上已经真实执行并交叉验证的机制实验：

- strict active-stream smoke：6/6；目标首 SSE event 与首 semantic token 都晚于源 stream drain；
- P1 pressure release：released `3,250,585,600` bytes；
- vLLM worker process-tree RSS 下降 `3,944,714,240` bytes；
- OS `MemAvailable` 增加 `4,044,066,816` bytes；
- pending release 清零，worker PID 仍存活。

因此 proposed 的 host backup 成本不是逻辑计数：它能够在压力下真实归还给 OS。另一方面，保留 clean backup 的普通 steady 状态会占用数 GiB host memory；延迟结果必须和这一成本共同阅读。

## 7. 外部系统门禁与 blocker

### ServerlessLLM

当前 local image 能完成 1.5B inference，但 `DELETE /v1/models/qwen-1.5b` 的控制面 200 不代表物理 scale-to-zero。新 harness 同时要求模型从 `/v1/models` 消失且 GPU 回落到 idle threshold；实测前者满足、后者 30 s 未满足。将这个 200 响应当作成功会严重低估其资源占用，因此不进入排名。

### SwapServeLLM

本地 checkout 是为 rootless podman 和 `cuda-checkpoint` 定制的旧 artifact。当前环境缺少关键 runtime 和 build headers；继续安装会修改共享系统环境，且仍不能保证 driver/process-checkpoint 兼容。按预注册规则记录 blocker，不拿历史 0.5B/1.5B lifecycle 数字拼入本次统一 trace。

### kvcached

kvcached 重点是 KV cache 和共享内存管理，不是与本文相同的模型权重 sleep/wake。历史 official image 在此 RTX 3080 完成过单模型 OpenAI smoke，但未保留可审计 raw artifact，也没有当前双模型 weight-switch contract。因此只列为 related system；不把它的 KV-cache 能力伪装成 weight-switch baseline。

## 8. Validity threats

1. **统计独立性**：proposed traces 在一个 long-lived pool 中重复，llama-swap router 也长驻；5/3 runs 是 runtime repeats，不是完整重新初始化的 12 个 independent blocks。因此结果标为 exploratory。
2. **样本量**：lifecycle n=5，external trace n=3；bootstrap 不会凭空创造信息。特别是 llama-swap alternating CI 很宽。
3. **公平性层级**：Proposed、cold/L1/L2 共享 vLLM 栈；llama-swap 是 external control-plane baseline。其结果适合回答“现成进程切换系统在本机表现如何”，不适合纯机制因果归因。
4. **文件缓存**：没有全局 drop page cache，以避免在共享服务器上破坏其他任务；cold/L2 数字是 warm-cache operational result，不是磁盘 cold boot。
5. **模型与硬件**：只有两个 Qwen 模型、FP16、单 RTX 3080；不覆盖 LoRA、多 GPU、NVLink、H100、量化模型或 NVMe checkpoint。
6. **能耗和磁盘 I/O**：本轮未形成统一高频功耗/块设备采样，不能声称 energy/I/O Pareto 最优。
7. **不完整的 confirmatory matrix**：预注册文档提出 12-block Tier-A W0/W1/W2 paired confirmatory 设计；当前尚未实现 stock L1-no-reuse 双模型独立 block。因此没有 Wilcoxon/Holm confirmatory p-value，也不声称达到最终投稿 artifact 的全部统计要求。

## 9. Reproduction and artifact map

- 协议：`docs/plans/2026-07-21-cross-system-top-tier-evaluation.md`
- 更严格的后续预注册：`docs/plans/2026-07-21-executable-core-matrix.md`
- lifecycle raw：`results/cross_system/raw/vllm/`
- proposed trace raw：`results/cross_system/raw/proposed/request-traces-final/`
- llama-swap trace raw：`results/cross_system/raw/llama-swap/request-traces-final/`
- raw input：所有 request JSONL、lifecycle event JSONL 和 sleep-profile JSONL 均已 force-track；server logs 含绝对路径/环境噪声，保留在本机但不纳入 curated artifact；
- curated summary：`results/cross_system/latest/summary.json`；
- external blockers：`results/cross_system/latest/external-systems.json`
- Provenance 限制：matrix metadata 固定了 benchmark commit/tree、manifest hash 和运行命令；proposed controller/vLLM 与 llama-swap 的 commit 记录在本报告及 `external-systems.json`，但 runner 当时以 external URL 连接服务，未自动把这些 service commit/embed config 写入同一 metadata，因此不是单文件 self-contained provenance bundle；
- analyzer：`src/tool/analyze_cross_system.py`
- plotter：`src/tool/plot_cross_system.py`
- unified matrix runner：`scripts/run_cross_system_matrix.py`

重建命令：

```bash
.venv/bin/python src/tool/analyze_cross_system.py \
  --lifecycle results/cross_system/raw/vllm/qwen-1.5b/20260721_122418/summary.json \
  --lifecycle results/cross_system/raw/vllm/qwen-3b/20260721_122952/summary.json \
  --trace-dir results/cross_system/raw/proposed/request-traces-final \
  --trace-dir results/cross_system/raw/llama-swap/request-traces-final \
  --external-systems results/cross_system/latest/external-systems.json \
  --bootstrap-samples 10000 \
  --output results/cross_system/latest/summary.json

.venv/bin/python src/tool/plot_cross_system.py \
  --summary results/cross_system/latest/summary.json \
  --output-dir results/cross_system/latest
```

## 10. 可声称与不可声称

**当前 artifact 支持：**

- 当前 RTX 3080 上，vLLM sleep/reload 显著快于进程 cold start；
- temporal locality 对 proposed request-visible TTFT 极其关键；
- upstream llama-swap 在相同 frozen traces 上可运行，但进程冷启动使模型改变代价在十秒量级；
- ServerlessLLM/SwapServeLLM/kvcached 在当前条件下有明确、可审计的纳入 blocker。

**当前 artifact 不支持：**

- “达到顶会正式投稿所需的全部 confirmatory evidence”；
- 对所有模型/硬件的普遍结论；
- proposed 在能耗、吞吐或所有内存维度 Pareto 最优；
- kvcached 比本文方案更快/更慢；
- 利用不满足物理 post-condition 的 ServerlessLLM 200 响应生成性能排名。
