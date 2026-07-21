# 12–24 GPU 小时核心实验矩阵（预注册 v1）

日期：2026-07-21  
目标：把原“顶会完整协议”压缩为单张 RTX 3080 10 GiB 上 **12 GPU 小时必达、19 GPU 小时期望、24 GPU 小时硬封顶**的可执行核心；仍保留同栈公平性、独立重复、随机 block、请求级原始数据、资源代价和诚实的 artifact 门禁。

## 0. 一页执行结论

### 必须完成（Core，预计 12 GPUh）

1. **主系统只放同栈 Tier A**：Proposed、vLLM Cold、vLLM L1-no-reuse、vLLM L2。
2. **主 trace**：冻结 W0/W1/W2；20 请求/trace；12 个独立 block/system/workload，共 `4×3×12=144 traces=2,880 requests`。
3. **主 micro**：上述 4 系统，A→B/B→A；3 次 warm-up/direction（不计），12 次 measured/direction，共 96 个方向级 measured observations。
4. **唯一 confirmatory endpoint**：W1 的 run-level semantic TTFT p95；主要 estimand 是 Proposed 相对各 Tier-A baseline 的配对比值 `p95_proposed / p95_baseline`。
5. **资源约束共同报告**：idle GPU MiB + host RSS GiB；不把只报 latency 的方法称为 Pareto 最优。

### 有余量才做（Extension，预计再 7 GPUh）

- llama-swap、ServerlessLLM、SwapServeLLM 各先过门禁；通过者执行 W1/W2×6 block，并做 3 次 W0 steady sanity，不进入同栈 confirmatory test。
- kvcached 只在官方双模型 unified API 成功后单独执行 W0/W1/W2×3；不进入 weight-switch 排名。
- 计划总计 19 GPUh；剩余 5 GPUh 只用于基础设施失败的补跑或把 Core 从 8 补足 12 个 block，禁止结果导向的新 workload/新 baseline。

## 1. 固定研究主张与边界

**主张 C1（confirmatory）**：在同一模型、vLLM backend 与冻结 alternating trace 下，Proposed 降低 request-visible W1 semantic TTFT p95，同时显式付出 CPU pinned backup 资源。

**支持性主张 C2**：Proposed 的切换时延优势来自 clean CPU backup reuse；证据为双向 switch latency、sleep/wake breakdown 和 `D2H=0` 的 reuse 事件。

**不主张**：饱和吞吐、生产 SLO、多 GPU 泛化、不同 engine 的纯机制因果、kvcached 的 weight switching 优越性。单卡/双模型结论不得外推至 H100/NVLink。

## 2. 冻结对象与公平性

- 模型：Qwen2.5-1.5B-Instruct（A）与 Qwen2.5-3B-Instruct（B），相同 revision/checksum、FP16。
- API：OpenAI chat completions，stream=true，temperature=0，seed=1，max_tokens=32。
- engine 参数：`max_model_len=1024`、eager mode；能统一的系统固定相同 vLLM commit/image、tokenizer、CUDA graph/compile 设置和 GPU memory policy。
- traces：
  - W0：A×20，0.5 s 间隔（steady 校准）；
  - W1：A/B alternating×20，1.5 s 间隔（主 workload）；
  - W2：A×5/B×5/A×5/B×5，1.5 s 间隔（locality 对照）。
- 冻结 checksum、机器信息、git SHA/container digest、模型 SHA256、命令和随机化表后才看新矩阵结果。
- `l1_no_reuse` 必须有可验证禁用 clean-reuse 的真实路径；若没有，改用固定 upstream/stock-like commit并明确命名。**禁止用 sleep+手工等待模拟 upstream。**

## 3. 系统纳入矩阵

| tier | 系统 | 机制 | 进入主排名 | 执行条件 |
|---|---|---|---|---|
| A | Proposed | controller + L1 persistent pinned backup clean reuse | 是 | 必须 |
| A | vLLM Cold | terminate/start + health ready | 是 | 必须 |
| A | vLLM L1-no-reuse | CPU-backed sleep/wake，无跨轮 clean reuse | 是 | 可验证开关/独立 commit |
| A | vLLM L2 | discard + wake weights/reload/wake KV | 是 | 必须 |
| B | llama-swap+vLLM | proxy 管理 terminate/start | 否，外部系统图 | 过 G1/G2 |
| B | ServerlessLLM | scale-to-zero/restore；只保留该主方法 | 否，外部系统图 | 过 G1/G2；不用 restore 估算值做主数 |
| B | SwapServeLLM | CUDA process/container checkpoint | 否，外部系统图 | 过 G1/G2 |
| C | kvcached | elastic KV / multi-LLM 邻接机制 | 否，单独图 | 官方双模型路径过 G1/G2 |

`delete/register` 只作 ServerlessLLM sensitivity，只有主方法提前完成且仍有预算才做；不同时把同一系统两个近似路径都塞入核心表。

## 4. 可执行矩阵

### M0：门禁 smoke（不进性能 CI）

每个系统执行：

- S1：A、A、B、B、A、B 共 6 个短请求；全部 HTTP 200、非空 semantic token、完整 `[DONE]`、模型 ID 正确。
- S2：A 的 160-token stream 与 B 的短请求重叠；不得断流、silent drop 或互相污染；允许排队但必须有 controller lifecycle 记录。
- 每种恢复后额外做 tolerance-based deterministic correctness canary。

Tier A 任一系统不过 smoke：暂停整个主比较并修复，不能静默少一根 bar。Tier B/C 不过：保存 blocker，停止该系统，不生成性能数字。

### M1：Tier-A 双向 lifecycle micro（支持性主结果）

| 维度 | 固定值 |
|---|---|
| systems | Proposed、Cold、L1-no-reuse、L2 |
| directions | A→B、B→A |
| warm-up | 每 system×direction 3 次，不计 |
| measured | 每 system×direction 12 次 |
| 统计单位 | direction-level transition；按 block 与方向配对 |
| 指标 | evict、restore、ready、首请求 semantic TTFT/E2E；阶段 breakdown；idle/peak GPU、process-tree RSS |
| 输出 | 双向分别报告，不先平均；另给预注册等权宏平均 |

现有 `bench_vllm_lifecycle.py` 每个 `run_one` 都重启单模型，不等于双模型交替独立 block；主 micro 应由 model-pool/controller 路径产出。现有 harness 可复用 HTTP、sampling 和 L2 breakdown，但不能把旧同模型结果并入新 CI。

### M2：Tier-A request trace（confirmatory 主矩阵）

| 维度 | 固定值 |
|---|---|
| systems | 4 个 Tier A |
| workloads | W0/W1/W2 |
| independent blocks | 12/system/workload |
| requests | 20/trace；总 2,880 |
| session | 每个 block 重新初始化该 system 的双模型状态；block 内 warm-up 后按冻结顺序跑 3 traces |
| order | `configs/core-evaluation-v1.schedule.json` |
| timeout | 每请求单调时钟总 deadline 600 s；超时保留在失败分母 |

一个 block 是独立统计单位。当前同一 long-lived session 的 3 次旧结果只用于 runtime pilot，不纳入新 confirmatory CI，也不能把 20 个请求当 20 个独立重复。

### M3：Tier-B 外部系统（secondary）

对每个通过门禁的系统：

- W1、W2 各 6 个独立 block（共 12 traces/system）；
- W0 只做 3 个独立 steady sanity blocks；
- 共 15 traces=300 requests/system，三个全过时为 900 requests；
- 同样报告 request-visible TTFT/E2E/failure + switch latency + GPU/RSS/disk/checkpoint bytes；
- 可与 Proposed 作描述性 paired ratio/CI，但不进入 confirmatory multiplicity family，不声称 engine 控制完全相同。

### M4：kvcached（conditional secondary）

仅当官方 artifact 能用同一统一 API 管理 A/B 且完成 S1/S2：W0/W1/W2 各 3 block，共 180 requests。标题必须写 “hybrid KV/multi-LLM”；与 weight switching 分图、分表。否则只发表 blocker。

## 5. 预注册 endpoints 与分析

### 5.1 Confirmatory（只检验这一个 endpoint family）

对每个 system×block 的 W1：

1. 用该 trace 内**所有 20 个调度请求**计算 empirical p95 semantic TTFT；量化规则固定为 nearest-rank `x_(ceil(0.95n))`（n=20 时取第 19 个顺序统计量），不用库默认的线性插值。失败/超时不丢弃，而按 600 s deadline 计入该 SLO endpoint；同时另报 raw failure rate。这样避免“最快系统靠失败降低 tail”。
2. 对每个 baseline，在相同 block 内计算 `log(p95_proposed / p95_baseline)`。
3. 点估计：12 个配对 log-ratio 的均值取指数，即 geometric mean ratio；同时给 paired median difference（ms）作为可解释 effect size。
4. CI：对 12 个 block 做 paired nonparametric bootstrap，固定 seed=20260721，10,000 resamples，报告 percentile 95% CI。小样本不宣称 BCa 能弥补信息不足。
5. 三个 primary baseline comparison 对 block-level log-ratio 与 0 做 Wilcoxon signed-rank，并用 Holm 校正（双侧 α=0.05）；方向性结论要求 ratio CI 上界 <1，且 Holm-adjusted p<0.05。若检验离散、ties 太多，则以 exact/permutation 实现并记录方法。
6. **不得**对 W0/W2、p50/p99、micro 中偶然显著的格子升级为主结论。

### 5.2 Secondary / descriptive

- W0/W1/W2：run-level TTFT/E2E median、p95；pooled ECDF 仅作可视化。
- tail uncertainty：cluster bootstrap（先采 block，再保留该 block 的整条 trace；不把同 run 请求独立重采当主 CI）。
- failure、HTTP error、incomplete SSE、deadline miss（1/2/5/600 s）、achieved throughput。
- M1：A→B/B→A 的 switch latency median、paired ratio CI、evict/restore breakdown。
- 资源：idle/peak GPU MiB、worker process-tree RSS GiB、host MemAvailable、disk read/checkpoint bytes。用 latency–memory Pareto 图防止 L1 的 host-copy 成本被隐藏。
- 首次 backup miss 单独显示；steady clean reuse 的 M1 统计不与 first miss 混合。

### 5.3 缺失、outlier 与补跑

- 应用失败、timeout、OOM、断流都是系统结果，不删；失败请求留在分母。
- 只有机器重启、其他 GPU job、采集器崩溃、端口被外部进程占用、artifact 文件损坏属于“基础设施失败”。整 block 标记 invalid，并用冻结 schedule 中相同 block ID 补跑；原始失败 artifact 保留。
- 运行开始后不因 effect size 大/小而提前停；固定 12 blocks。唯一例外是安全/资源硬门禁。

## 6. 随机化与运行控制

- Tier A 使用冻结 schedule：12 个 block 中，每个 system 在 4 个位置各出现 3 次；每个 system 的 6 种 W0/W1/W2 顺序各出现 2 次。
- block 0–3、4–7、8–11 分三批执行；批间做 GPU process/port/磁盘空间检查，不查看 comparative summary，只看门禁状态。
- 每个 system/block 开始前：GPU 无其他 process、端口空闲、温度 `<80°C` 且连续 30 s 无上升；否则等待降温，不换顺序。
- 记录 wall-clock、GPU 温度/功耗/显存（100 ms）、CPU RSS/MemAvailable、disk bytes、major faults；若 100 ms 采样开销在 pilot 中使 TTFT median 改变 >2%，降到 250 ms并全系统统一。
- Tier B 也用 seed=20260721 生成通过系统的随机 block；不得按“先快后慢”固定顺序。

## 7. 运行预算与降级顺序

以下是 GPU 占用墙钟预算，不是纯 kernel time；基于现有 pilot（W1/W2 各 28.5 s；Cold restore 约 16.5–19.5 s）留出启动与清理余量。

| 阶段 | GPUh 预算 | 累计 | 到点动作 |
|---|---:|---:|---|
| 环境冻结 + Tier-A S1/S2 smoke | 1.5 | 1.5 | 不过则修复，禁止进主测 |
| M1 Tier-A 双向 micro | 2.0 | 3.5 | 固定 12/direction |
| M2 Tier-A 144 traces | 5.0 | 8.5 | 这是最高优先级 |
| 资源/正确性审计 | 1.0 | 9.5 | 必须完成 |
| 分析重建 + Core 补跑余量 | 2.5 | **12.0** | 形成最小可投稿 core |
| Tier-B smoke/适配门禁 | 3.0 | 15.0 | 单系统 1 GPUh 上限 |
| 通过者 Tier-B traces | 3.0 | 18.0 | 按 llama-swap→SwapServe→ServerlessLLM；到点即停未开始者 |
| kvcached 条件实验 | 1.0 | **19.0** | 未过门禁则 0 GPUh performance |
| 未分配故障余量 | 5.0 | **24.0 hard cap** | 只补基础设施失败/Core，不加新假设 |

**预算降级顺序**（由低到高保护）：ServerlessLLM sensitivity delete/register → kvcached performance → Tier-B traces → Tier-B 新适配。永不削减 Tier-A 12 blocks 来保外部系统。如果 12 GPUh 前只完成 8 个完整 Tier-A blocks，则先以 8 blocks 标为 exploratory snapshot，并把剩余预算全部用于补到 12；不得以显著性为理由停在 8。

## 8. 停止/继续门禁

### G0 环境门禁（任何测量前）

- commit/image/model checksums 冻结；工作树 patch diff 保存；GPU/driver/CUDA/CPU/RAM/NVMe 记录；raw 目录可写且至少 100 GiB 空闲。
- 任一失败：不启动计时矩阵。

### G1 单系统功能门禁

- S1 6/6 成功、完整 `[DONE]`、输出非空、模型路由正确；恢复 post-condition 明确完成而非仅接受命令。
- 失败：Tier A 修复；Tier B/C 生成 blocker 并停止。

### G2 active-stream 安全门禁

- S2 长流必须完整结束；B 请求可等待但不能导致 A 断流；controller event 可按 request ID 审计。
- 失败同 G1。

### G3 资源安全门禁

- 发生 GPU Xid/ECC fatal、host MemAvailable <4 GiB 持续 10 s、swap thrash >30 s、磁盘余量 <20 GiB：立即终止该 block 并安全清理。
- OOM 若由系统设计导致，记系统失败；若由无关进程导致，记基础设施 invalid 并补跑。

### G4 批次完整性门禁

- 每 4 blocks 校验：trace SHA、20 rows/run、无重复/缺失 block、采样时间覆盖、返回码和 manifest identity。
- 只允许补跑 invalid block；禁止挑选“漂亮”run。

### G5 预算门禁

- 12 GPUh：必须能重建 Tier-A 主表/CI/失败表；不能则停止 secondary 并修 Core。
- 19 GPUh：停止一切新 secondary。
- 24 GPUh：硬停止；未完成项报告 blocker/missing，不估算数值。

## 9. 论文最小图表

1. **主表**：W1 Proposed vs Cold/L1-no-reuse/L2 的 p95 TTFT ratio、paired difference、95% CI、failure。
2. **图 1**：W0/W1/W2 的 run-level TTFT p50/p95 + cluster CI（locality story）。
3. **图 2**：A→B/B→A lifecycle breakdown（evict/restore/ready）。
4. **图 3**：switch latency–idle resource Pareto（GPU + host RSS）。
5. **外部表/图**：只列通过 G1/G2 的 Tier B；kvcached 单独。
6. **artifact 表**：commit/image、机制、保留状态、patch、门禁结果与 blocker。

## 10. 执行前必须补齐的代码差距

现有 runner 固定按 `repeat→W0→W1→W2` 且不重启独立 block，直接运行会重复旧顺序偏差。正式执行前必须：

1. 让 runner 读取 `configs/core-evaluation-v1.schedule.json`；
2. 支持 `system/block/workload` 标识和 block setup/teardown hook；
3. analyzer 以 block 为独立单位，生成 W1 p95 paired ratio bootstrap/Holm 输出；
4. 对失败请求实现预注册的 600 s deadline penalty endpoint，同时保留成功条件延迟；
5. 增加 schedule validator、重复/缺失 block fail-closed 测试；
6. 为 llama-swap/ServerlessLLM/SwapServeLLM 做统一 trace adapter；没有适配就只跑 lifecycle secondary，不伪装 request-trace 对比。

执行脚本和 analyzer 测试全部通过、dry-run 能完整枚举 144 个 Tier-A traces 后，才开始消耗 GPU 预算。
