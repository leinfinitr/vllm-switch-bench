# 单机单 GPU 多模型切换：跨系统性能对比实验协议

日期：2026-07-21  
硬件边界：单 NVIDIA RTX 3080 10 GiB；不把多 GPU/NVLink/H100 论文数字与本机绝对值混排。

## 1. 研究问题

- RQ1：在两个模型不能同时常驻 HBM 的情况下，各系统从请求到目标模型可服务的切换延迟是多少？
- RQ2：不同 temporal locality、到达率和 burst 下，request TTFT、完成延迟、失败率、队列等待如何变化？
- RQ3：低延迟换来的 CPU RAM、磁盘、GPU idle footprint 和功耗成本是多少？
- RQ4：切换期间活跃请求是否被保护，系统是否产生错误输出、断流或饥饿？
- RQ5：各机制的收益来自何处：进程初始化、权重读盘、CPU→GPU、CUDA context/graph、KV 管理或 checkpoint restore？

## 2. 系统矩阵和纳入门禁

### Tier A：必须执行、同引擎/同模型可直接比较

1. Proposed：研究版 vLLM L1 + persistent pinned CPU backup reuse + controller。
2. vLLM Cold：每次切换停止旧 engine、启动目标 engine、等待 health 后服务。
3. vLLM L1 stock-like：CPU backup 保留但不使用跨轮 clean-reuse 优化；若研究分支没有可验证的禁用开关，则使用固定 upstream commit，不能把模拟结果标为 upstream。
4. vLLM L2：丢弃权重后 `wake weights → reload_weights → wake KV`。

### Tier B：外部系统；通过 smoke 门禁后执行

5. ServerlessLLM：scale-to-zero/delete-register + fast loader；使用同一 vLLM backend、同模型和参数。
6. SwapServeLLM：CUDA checkpoint container swapout/swapin；使用同一 vLLM image、同模型和参数。
7. llama-swap + vLLM：进程级 terminate/start，作为成熟通用 OpenAI model-switch proxy；它与 cold restart 机制接近，但量化了真实控制面开销。

### Tier C：相邻问题，不进入“权重切换延迟”主排名

8. kvcached：主要优化 elastic KV、GPU memory ballooning 和并发多模型；只有其官方 controller 在本机完成两个目标模型、统一 API、完整 trace 后，报告“hybrid KV/multi-LLM workload”。结果单独成图，不与 pure temporal weight switching 合成一个排名。

### 外部系统运行门禁

每个系统必须同时满足：

- 固定 upstream commit/tag、许可证与本机 patch diff；
- 当前 GPU 上完成两个目标模型的启动或按需恢复；
- 统一 OpenAI request body `model` 路由，或有不改变机制的薄适配层；
- 6-request S1 smoke 和 active-stream S2 drain smoke 全部成功；
- 每条成功 stream 有 semantic token、非空输出和完整 `[DONE]`；
- 切换前后 GPU/CPU memory 与进程状态可观测；
- 无法满足时发布 blocker 和 artifact smoke，不生成性能数字。

## 3. 模型与统一配置

主模型：

- Qwen2.5-1.5B-Instruct，FP16；
- Qwen2.5-3B-Instruct，FP16。

统一参数：`max_model_len=1024`、`max_tokens∈{32,160}`、temperature 0、固定 seed、eager mode。vLLM 系统尽量统一 backend commit/image；外部系统强制依赖其他版本时记录版本差异并做 sensitivity run。

扩展规模实验（若磁盘和时间允许）：Qwen2.5-0.5B/1.5B/3B，只报告 switch micro，不替代双模型 trace。

## 4. Workload

### 4.1 Controlled switch micro

- A→B 和 B→A 分开；
- 每个方向 3 个 warmup，不计入；
- 每个方向至少 30 个 measured cycles；
- 分解 evict、restore、readiness、request semantic TTFT、request completion；
- L1 分 first backup miss 与 clean reuse；L2 分 wake weights、reload weights、wake KV；
- 冷启动分别报告 page-cache-warm 和“自然磁盘状态”；不以 root 权限 drop page cache。若要 cold-cache，复制模型到独立文件或使用 `posix_fadvise` helper，并单独标注。

### 4.2 Open-loop traces

固定 frozen JSONL，三个 workload 都使用同一组 arrival offsets和请求长度，只改变模型序列：

- W0/steady：单模型校准；
- W1/alternating：A,B,A,B，最差 temporal locality；
- W2/bursty-local：相同 arrival，4-request model-local bursts；
- W3/Zipf：80/20 model popularity，每 60 s 交换热模型；
- W4/interactive：Poisson arrivals，短输入/短输出；
- W5/RAG-like：长输入/短输出；
- W6/generation：短输入/长输出；
- W7/S2 overlap：长 stream 与目标模型短请求重叠，验证 drain/queue。

提供低、中、高三档 offered load。高档定义为 dedicated steady-state capacity 的 30–50%，而不是把单卡压到饱和后混淆切换机制。

每个 system×workload×load：至少 10 个独立重复；每个重复 60–120 requests。执行顺序使用预先生成的 Latin-square/随机 block，避免温度、page cache 与时间漂移总是偏向同一系统。

## 5. 指标

请求层：

- semantic TTFT、E2E latency、TPOT；
- p50/p90/p95/p99 与 ECDF；
- deadline miss ratio（1 s/2 s/5 s）；
- failure、incomplete SSE、HTTP error；
- achieved throughput、queueing delay、drain delay；
- switch-amplified latency = switch request − same-model steady median。

系统层：

- switch latency及阶段 breakdown；
- GPU memory used/free、worker process-tree RSS、host MemAvailable；
- CPU utilization、disk read bytes、major/minor faults；
- GPU utilization、power、SM clock、memory clock，以 100 ms 采样；
- energy/request 和 energy/switch（对功率曲线积分）；
- idle footprint 和恢复后的资源残留；
- 启动/切换期间峰值 memory。

正确性：request ID 端到端关联；manifest identity；完整 `[DONE]`；输出非空；active request drain；失败样本保留在分母。

## 6. 统计

- 重复 run 是统计独立单位，不把同一 run 内请求伪装为独立样本；
- 报告 run-level median 的中位数和 BCa bootstrap 95% CI（至少 10,000 resamples）；
- tail latency同时报告 pooled ECDF，但 CI 使用 cluster bootstrap（先采样 run，再采样 run 内请求）；
- paired comparison 使用相同 trace/seed/run-index，报告 paired median difference 和 ratio CI；
- 多系统多 workload 的显著性检验使用 Wilcoxon signed-rank，并用 Holm 校正；
- 同时报 effect size，不只给 p-value；
- 预先定义 outlier：除基础设施故障外不删除；基础设施故障需整轮重跑并保留失败记录。

## 7. 运行控制

- 开始前记录 GPU/CPU/内核/驱动/CUDA、git SHA、container digest、模型文件 SHA256、文件系统和空闲空间；
- GPU persistence mode/clock 若无权限修改则保持默认并记录；
- 每轮前要求 GPU process 清空、端口清空、温度低于固定门槛；
- 每系统先独立 correctness smoke，再进入测量；
- 所有 timeout 使用单一总 deadline；
- 运行期间禁止其他 GPU workload；
- 每轮保存命令、stdout/stderr、环境白名单、返回码、起止时间和采样原始 CSV/JSONL。

## 8. Artifact 结构

`results/cross_system/latest/` 只保存最新 curated 结果：

- `protocol.md` 和 frozen manifests；
- `systems.json`（commit/tag/image digest/license/patch SHA）；
- `machine.json`；
- 每个 run 的 raw request JSONL、lifecycle events、resource samples和日志；
- `matrix.json` 绑定 system/workload/repeat 到文件 SHA256；
- analyzer 输出、bootstrap samples或seed、表格和图；
- blocker 目录，含命令、错误、环境与最小复现；
- 一条命令重建 summary/figures，且 CI 测试检查 committed summary 可重建。

## 9. 论文图表

- Figure 1：机制和资源位置示意；
- Figure 2：双向 switch micro + 95% CI；
- Figure 3：三个 locality trace 的 semantic TTFT ECDF/p95；
- Figure 4：latency–memory Pareto；
- Figure 5：阶段 breakdown；
- Figure 6：energy/switch；
- Figure 7：offered load sweep；
- Table 1：系统、机制、资源保留、版本和支持状态；
- Table 2：主结果与 paired speedup CI；
- Table 3：failure、deadline miss和正确性。

## 10. 解释边界

- RTX 3080 的 PCIe、内存容量和缺少 NVLink 只支持单机单卡结论；
- patched external artifact 与 stock upstream 分开命名；
- kvcached 的 KV ballooning、SwapServe 的完整 CUDA process checkpoint、ServerlessLLM 的 storage loader、vLLM sleep 是不同资源层，主结果必须同时展示 latency 和保留资源；
- 不把旧运行、论文数字或其他 GPU 的数字合并进本机置信区间；
- 任何无法运行的系统只做 blocker，不插值、不估算、不生成虚构 bar。
