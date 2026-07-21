# 单 RTX 3080 上的跨系统多模型切换评测

日期：2026-07-21  
状态：**可复现的 exploratory artifact**；不是完整的 12-block confirmatory paper claim。

## 1. 结论摘要

在单张 RTX 3080 10 GiB 上，Qwen2.5-1.5B 与 Qwen2.5-3B 不能按本实验配置同时常驻 HBM。我们将 proposed controller 的 pinned CPU clean-backup reuse 与 vLLM cold/L1/L2 lifecycle 以及开源 `llama-swap` 进程级切换作真实本机比较。ServerlessLLM、SwapServeLLM 和 kvcached 尚未通过可比的双模型 weight-switch 门禁，因此只报告精确限定的 runtime/环境/语义边界，不生成或推断性能数字。

主要观测：

- **单模型 lifecycle（5 次独立进程 run / cell）**：
  - 1.5B：cold `13.382 s`，vLLM L1 `1.333 s`，vLLM L2 `0.859 s`；
  - 3B：cold `14.331 s`，vLLM L1 `2.868 s`，vLLM L2 `1.773 s`。
- **统一 request trace（当前有序 warm-state repeats 的描述性 bootstrap 区间，不作总体 95% coverage claim）**：
  - proposed，alternating：run-median semantic TTFT `546.8 ms`（5 runs，描述性 bootstrap interval `[545.1, 558.1]`）；
  - proposed，burst/locality：`19.7 ms`（5 runs，`[19.5, 21.1]`）；
  - llama-swap，alternating：`14.846 s`（3 runs，`[14.224, 46.077] s`）；
  - llama-swap，burst/locality：`15.290 s`（3 runs，`[13.851, 15.293] s`）。
- 对这批 exploratory runs，proposed 的 run-median semantic TTFT 相对 llama-swap 为 alternating `27.2×`、burst/locality `776.4×` 更低；两系统请求均严格成功（proposed `200/200`，llama-swap `120/120`）。

这些差距描述的是**当前单卡、当前软件版本和给定 traces**，不能外推到 H100、多 GPU 或生产 SLO。llama-swap 是控制面真实 external baseline，但其机制本质上是进程 cold start，不能被解读为与 pinned-backup reuse 的同机制消融。

## 2. 系统与版本

| 系统 | 版本/commit | 机制 | 本机门禁 | 是否有性能数据 |
|---|---|---|---|---|
| Proposed | controller `d78155f`；vLLM `b2057ef`（service provenance 来自运行记录，未嵌入 matrix metadata） | 双 vLLM 进程 + L1 pinned clean backup reuse + request-driven controller | 本轮 request trace 200/200 strict success；历史 drain/资源验证未随本 artifact 保留 final3 raw | 是 |
| vLLM cold/L1/L2 | vLLM `b2057ef` | terminate/start；sleep level 1；sleep level 2 | 30/30 lifecycle run 成功 | 是 |
| llama-swap | `c6adf57` | OpenAI proxy 管理 vLLM terminate/start | 120/120 strict requests 成功 | 是，secondary |
| ServerlessLLM | 源码 `2618762`；实际镜像 `6a7fb919`（2025-11-03） | serverless fast loading / scale-to-zero | 旧镜像 delete 泄漏 `6171 MiB`；当前源码 overlay gate 只部分通过，并暴露 failed-start backend actor/逻辑 GPU reservation 未清理路径；尚无成功 register/infer/delete/re-register 门禁 | 否 |
| SwapServeLLM | `69f8aec` | CUDA process/container checkpoint | 0.5B 单模型真机机制 smoke 通过：2 次 checkpoint/pause/restore、post-restore inference 与 cleanup 均通过；未执行可比双模型性能矩阵 | 否；仅机制证据 |
| kvcached | 本地 `d78649d`；官方 v0.1.5 `2513db1` | elastic KV/memory manager；controller 可调用 vLLM L1 | 核心机制不是权重切换；官方 controller 缺冲突 victim selection 和单卡 staged startup，未执行“kvcached controller + vLLM L1 + 外部互斥 harness”组合 | 否；只作相关/组合候选 |

结构化记录：`results/cross_system/latest/external-systems.json`。

## 3. 实验条件

- GPU：NVIDIA GeForce RTX 3080，10,240 MiB；driver `580.95.05`。
- 模型：`Qwen2.5-1.5B-Instruct`、`Qwen2.5-3B-Instruct`，FP16。
- 公共 vLLM 参数：`--enforce-eager --gpu-memory-utilization 0.70 --max-model-len 1024 --dtype half`。
- 请求：OpenAI chat completions，streaming，`temperature=0`，固定 seed，最多 32 output tokens。
- 成功定义：HTTP 2xx、`error` 为空、semantic token 非空、完整 SSE `[DONE]`、`stream_done is True`。
- 原始 response first byte 没有被误称 semantic TTFT。
- 矩阵 runner 对本轮 frozen manifest 的旧 raw 格式逐行绑定 `request_id/model/scheduled_offset_s` 并保存 manifest/output SHA256；审查后 runner 已升级为新 run 记录并验证 endpoint/prompt/max_tokens/temperature/seed/stream 全字段。
- 统计：请求 trace 以 run 为重复单位；报告 run-median 的中位数以及固定 seed `20260721`、10,000 次 nonparametric bootstrap percentile **描述区间**。由于服务在同一 long-lived session 内、只有 3–5 个有序 warm-state repeats，这些区间不作总体 95% coverage claim，也不是 independent-block confirmatory CI。

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

### 5.1 Same-model warm-page-cache process-restart/sleep activation reference

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

这里的 lifecycle 表是**同一 checkpoint 的 warm-page-cache operational reference**，每个 cell 固定执行 5 次独立进程 run，不是 A→B/B→A 双模型 confirmatory switch。L2 比 L1 更快并不矛盾：该 harness 的 level-1 `sleep` 要把 live allocations 复制到 CPU，然后 wake H2D；level-2 丢弃权重并用本地 page cache 重新加载。10 GiB HBM、64 GiB host RAM 和热文件缓存下，L2 的总路径可能更短。它不代表所有存储层次都如此。

### 5.2 Request-visible semantic TTFT

| 系统 | trace | runs | requests | run-median TTFT | 描述性 bootstrap interval | pooled request TTFT p95 | run-median E2E | failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed | alternating | 5 | 100 | 546.8 ms | [545.1, 558.1] | 684.1 ms | 800.3 ms | 0/100 |
| Proposed | burst/locality | 5 | 100 | 19.7 ms | [19.5, 21.1] | 645.2 ms | 350.1 ms | 0/100 |
| llama-swap | alternating | 3 | 60 | 14,845.7 ms | [14,223.8, 46,076.5] | 56,918.1 ms | 15,169.5 ms | 0/60 |
| llama-swap | burst/locality | 3 | 60 | 15,289.9 ms | [13,851.0, 15,293.0] | 25,889.1 ms | 15,672.8 ms | 0/60 |

![Trace TTFT](../../../results/cross_system/latest/trace-ttft.png)

Alternating 下 proposed 几乎每个请求都要求真实模型改变，因此 run median 在约 0.55 s。Burst/locality 中大部分请求命中当前 awake model，run median 降至约 20 ms；pooled p95 仍约 645 ms，准确暴露了 burst 边界的切换请求。llama-swap 的 open-loop requests 在十秒级启动期间会排队，并可能合并等待同一目标模型的请求；因此该数字测量的是“控制面排队 + cold process starts + switch coalescing”的完整 trace 行为，**不是每个名义 A/B transition 都对应一次独立进程启动**。首 run 的 alternating median 达 46.1 s，导致 3-run 描述区间很宽，这个异常没有删除。

## 6. 历史机制与资源证据（不作为本轮 final-commit raw artifact）

下列数字不纳入 curated artifact，也不进入 `checksums.json`。它们来自本机后续 final3 验证的会话记录，但**没有在本 benchmark commit 中保留可从 clone 独立审计的 final3 raw files**；仓库 `results/request_switch/latest/` 的 retained artifact 来自更早的 controller commit，不能替代它。因此这些数值只作历史背景，不进入本轮 cross-system headline claim：

- strict active-stream smoke：6/6；目标首 SSE event 与首 semantic token 都晚于源 stream drain；
- P1 pressure release：released `3,250,585,600` bytes；
- vLLM worker process-tree RSS 下降 `3,944,714,240` bytes；
- OS `MemAvailable` 增加 `4,044,066,816` bytes；
- pending release 清零，worker PID 仍存活。

这些历史观测提示 backup 可被真实归还给 OS，但由于 final3 raw 缺失，本 cross-system artifact **不据此提出可独立审计的物理回收结论**。普通 steady 状态的 clean backup 仍会占用数 GiB host memory；延迟结果必须和这一设计成本共同阅读。

## 7. 外部系统门禁与 blocker

### ServerlessLLM

实际运行的 `serverlessllm/sllm:latest` 镜像创建于 2025-11-03，image ID 为 `6a7fb919…`，早于当前源码 `2618762`。它能完成 1.5B inference，但 delete 路径只删除 metadata/actor handle，没有显式 shutdown/kill detached router/backend；因此模型从 `/v1/models` 消失后 GPU 仍在 30 s 内保持 `6171 MiB`。当前源码已包含 `await router.shutdown.remote()` 和 `ray.kill(router)`，所以旧镜像失败**不能外推为当前 ServerlessLLM 的固有缺陷**。

为避免完整重建 36 GB 镜像，后续用只读 bind-mount 把当前 `controller.py`/`roundrobin_router.py` 覆盖到旧镜像，现场验证 live container 确实导入新 cleanup。该低成本 gate 只得到部分结果：错误 host-only model path 导致 backend 在进入 `ready_inference_instances` 前初始化失败；delete 能 kill router，但 router shutdown 只枚举 ready instances，未清理这个 failed-start named backend actor/逻辑 GPU reservation。随后即使改用正确 `/host-models/...` 路径，scheduler 仍显示 `0` free GPU，直到人工 kill orphan actor 才恢复。由于没有完成成功的 register→infer→delete→物理回收→re-register，当前源码仍不进入排名；而且正式门禁必须新增 failed-start/startup-race cleanup。结构化证据：`results/cross_system/external/serverless-current-source-overlay/summary.json`。

### SwapServeLLM

本地 `fix/local-rootless-swapserve` checkout 已通过 rootless Podman socket、NVIDIA CDI、固定 vLLM image、Qwen2.5-0.5B 模型和 vendored Go build 门禁。随后使用固定 commit/hash 的临时 `cuda-checkpoint` 真实执行完整 controller path：两次均得到非空 GPU PID 文件，外部观察到 `checkpointed`，PID 从 GPU compute list 消失、容器 paused；恢复后 CUDA state/running residency 返回，真实 chat inference 输出非空 `OK`。显存从 awake `4490 MiB` 降至 checkpointed `4 MiB`，结束后无 GPU process、8000/8001 无监听。该结果只证明 0.5B **单模型机制可运行**；没有用 Qwen 1.5B/3B frozen traces、独立 repeats 和同一成功协议形成性能结果，因此不进入排名。Raw artifact：`results/cross_system/external/swapserve-smoke/`。

### kvcached

kvcached 核心是 KV cache 的 CUDA VMM/共享内存管理，不是权重切换。v0.1.5 的 controller 确实能在启用 vLLM server-dev/sleep mode 后调用后端 level-1 sleep/wake，因此不能再写成“没有 lifecycle endpoints”。但 controller 不会在唤醒目标模型前选择并 sleep 当前冲突模型，launcher 也会直接启动所有 backends，缺少 10 GiB 单卡所需的 staged initialization。故合理命名应是 **“kvcached controller + vLLM L1 + 外部互斥/staged-start harness”**；该组合本轮未执行。历史单模型 smoke 没有 retained raw，因此仍只列为 related/composite candidate，不产生性能数字。

## 8. Validity threats

1. **统计独立性**：proposed traces 在一个 long-lived pool 中重复，llama-swap router 也长驻；5/3 runs 是 runtime repeats，不是完整重新初始化的 12 个 independent blocks。因此结果标为 exploratory。
2. **样本量**：lifecycle n=5，external trace n=3；bootstrap 不会凭空创造信息。特别是 llama-swap alternating 描述区间很宽。
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
- artifact checksum manifest：`results/cross_system/latest/checksums.json`；
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

- 当前 RTX 3080 上，当前同模型 warm-page-cache reference 中 vLLM sleep/reload 显著快于进程 restart；
- temporal locality 对 proposed request-visible TTFT 极其关键；
- upstream llama-swap 在相同 frozen traces 上可运行，但进程冷启动使模型改变代价在十秒量级；
- ServerlessLLM/SwapServeLLM/kvcached 在当前条件下有明确的本地纳入 blocker/范围判断，但证据强度低于 retained request-trace artifacts。

**当前 artifact 不支持：**

- “达到顶会正式投稿所需的全部 confirmatory evidence”；
- 对所有模型/硬件的普遍结论；
- proposed 在能耗、吞吐或所有内存维度 Pareto 最优；
- kvcached 比本文方案更快/更慢；
- 利用不满足物理 post-condition 的 ServerlessLLM 200 响应生成性能排名。
