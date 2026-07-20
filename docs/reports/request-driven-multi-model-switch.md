# 请求驱动的多模型快速切换：阶段实验报告

## 1. 结论

本阶段完成了一个最小、可运行的研究原型：用户向统一的 OpenAI-compatible API 发送请求，通过请求体的 `model` 字段选择模型；controller 保护活跃请求，按需 sleep 当前 vLLM engine、wake 目标 engine，再转发请求。权重备份、clean reuse 与释放仍由研究版 vLLM 的 CPU pinned backup pool 负责。

在单张 RTX 3080 10 GiB 上，Qwen2.5-1.5B-Instruct 与 Qwen2.5-3B-Instruct 的真实双模型池运行成功。最终结果由 controller commit `46a7806`、benchmark commit `13123b7` 和 vLLM commit `b2057ef` 生成；三个 tracked 工作树在运行时均干净。三条冻结 workload 每条运行三轮，共 180 个请求，所有请求均收到完整 `[DONE]` SSE 终止事件：

| workload | 语义 TTFT median / p95 | E2E median / p95 | 解释 |
|---|---:|---:|---|
| W0 steady，A×20 | 17.2 / 44.0 ms | 190.8 / 235.9 ms | 无切换 fast path |
| W1 alternating，A/B×10 | 537.7 / 634.7 ms | 791.1 / 965.2 ms | 每请求切换的低负载 alternating 场景 |
| W2 burst，A×5/B×5/A×5/B×5 | 19.9 / 632.9 ms | 349.4 / 962.6 ms | steady hit 为主，组边界形成长尾 |

W1 与 W2 使用完全相同的绝对 arrival offsets：20 个请求、1.5 秒间隔、28.5 秒 scheduled duration、0.667 req/s。二者只改变模型访问顺序，因此这里可以把差异归因于访问 locality；它仍不是饱和负载、thrashing 或生产延迟上限。

## 2. 最终实现

### Controller

- lifecycle POST 与 `/is_sleeping` post-condition 共用一个完整 monotonic deadline；
- lifecycle outcome 不确定时保留 `ERROR` barrier，后续请求 fail closed，不会将 `active_model=None` 当成可安全唤醒另一模型；
- launcher 部分失败时 terminate/wait 本次启动的全部进程，成功 PID 文件原子写入；
- readiness 与 request reservation 在同一 `switch_lock` 临界区内；
- streaming 正常完成、异常、取消和客户端断连均释放 reservation；
- 接受或生成 `X-Request-Id`，同一 ID 写入 controller event 并转发 backend；
- controller 的首字节字段明确命名为 response-body first byte，不冒充 semantic TTFT；
- pressure validator 自动拒绝客户端消失、进程退出、逻辑 release 未增长、RSS 未下降、`MemAvailable` 未上升或 pending 未清零。

### Benchmark

- 冻结 JSONL manifest，校验有限 offset、非空 ID、endpoint、stream、prompt 与 token 参数；
- 绝对 monotonic arrival 的 open-loop runner和共享 `httpx.AsyncClient`；
- 每请求总 deadline，不会被持续 heartbeat 绕过；
- SSE event boundary、多行 data、Content-Type、malformed JSON、EOF remainder 和 `[DONE]` 完整性验证；
- transport/stream failure 保留 partial output；
- semantic TTFT 跳过 role-only event；失败样本保留在失败分母，但不进入成功延迟分位数；
- metadata 保存 benchmark commit、tracked dirty、manifest SHA256 与 prompt schema SHA256；
- analyzer 对缺失、重复、多余或非零退出的 workload/repeat fail closed。

## 3. 请求级结果

| workload | requests | success | offered rate | achieved rate median | TTFT median / p95 | E2E median / p95 |
|---|---:|---:|---:|---:|---:|---:|
| W0 | 60 | 60 | 2.00 req/s | 1.96 req/s | 17.2 / 44.0 ms | 190.8 / 235.9 ms |
| W1 | 60 | 60 | 0.667 req/s | 0.645 req/s | 537.7 / 634.7 ms | 791.1 / 965.2 ms |
| W2 | 60 | 60 | 0.667 req/s | 0.659 req/s | 19.9 / 632.9 ms | 349.4 / 962.6 ms |

矩阵运行窗口内恰好有 180 条 controller OpenAI event：72 次 switch、108 次 steady hit。switch path 分解为：

- sleep median：128.4 ms；
- wake median：392.9 ms；
- request drain median：0.005 ms；
- 完整 switch path median：527.4 ms。

W1 的 TTFT 中位数与完整 switch path 同量级。W2 的模型组边界仍出现切换长尾，但组内请求走 steady path，因此中位数显著更低。

![W0/W1/W2 request-visible latency](../../results/request_switch/latest/request-workloads.png)

![Switch breakdown](../../results/request_switch/latest/switch-breakdown.png)

## 4. CPU pinned backup 机制

最终进程 profile 的 first miss 与 clean reuse：

| model | first sleep miss | D2H | pinned alloc | clean reuse count / median | wake median / H2D |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 925.5 ms | 169.0 ms | 707.8 ms | 43 / 107.7 ms | 273.5 / 176.5 ms |
| Qwen2.5-3B | 2069.6 ms | 326.8 ms | 1717.0 ms | 44 / 143.2 ms | 497.6 / 384.3 ms |

clean reuse 的 `copy_d2h_s=0`。该优化消除的是后续 sleep 的 pinned allocation 与 D2H，不是 wake 的 H2D。

![Backup ablation](../../results/request_switch/latest/backup-ablation.png)

历史、同模型的 cold-reload lifecycle 数据仅作为机制背景：1.5B/3B restore median 分别约 16.52/19.53 秒。它们不是本次 frozen trace 的请求级结果，不参与严格排名。

## 5. P0/P1 物理内存验证

### P0：正常状态保留

最终 P0 的 `A→B→A→B→A` 后：

- 1.5B cache-only backup：3,250,585,600 bytes；
- `requested_release_bytes_total=0`、`released_bytes_total=0`；
- 3 秒观察窗口 worker RSS 不变；
- memory-pressure monitor 为 `normal`。

### P1：受控手工 release

对同一最终进程的 1.5B cache-only backup 发出精确 release request：

- queued/released：3,250,585,600 bytes；
- worker RSS：5,905,862,656 → 1,961,148,416 bytes，下降 3,944,714,240 bytes（3.67 GiB）；
- host `MemAvailable`：38,000,541,696 → 42,025,168,896 bytes，上升 4,024,627,200 bytes（3.75 GiB）；
- pending release：0，worker PID 保持存活；
- release 后首个切出 1.5B 的 sleep 为 927.7 ms，其中 pinned allocation 715.8 ms、D2H 168.4 ms；
- 随后 sleep 重新进入 clean reuse，107.6 ms、D2H=0；
- release 后 `3B→1.5B→3B→1.5B` 四次请求均 HTTP 200。

这证明手工受控 release 的逻辑计数、进程 RSS 和 host memory 方向一致；不等同于证明自动 memory-pressure policy 在真实竞争负载中的长期收益。

![Physical reclaim](../../results/request_switch/latest/physical-reclaim.png)

## 6. 外部 artifact 边界

- kvcached 官方 image 在 RTX 3080 上完成单模型 OpenAI inference smoke，但缺少本 controller 所需的 `/sleep` 与 `/is_sleeping` contract；没有生成不公平的双模型性能排名。
- upstream vLLM L1 被共享 venv 中 flash-attention 二进制扩展不匹配阻塞；没有兼容性移植或伪造数值。
- Prism 依赖旧定制 SGLang、`prism/shm` CUDA extension、Redis 及多 GPU/NVLink 条件；单卡只完成环境/import smoke，不引用 H100 论文绝对数值。
- ServerlessLLM/SwapServeLLM 的既有 lifecycle 数据只作背景，不与本次 request trace 合并排名。

`external-artifacts.json` 是本机观察记录，不能替代上游可复现 artifact；其中明确标注了 raw artifact 可用性与 blocker。

## 7. 有效性边界

1. 单 GPU、两个小模型、低 offered load，不是生产 serving scheduler。
2. W1/W2 只控制了 arrival schedule 与访问顺序；没有证明饱和吞吐、生产 SLO 或统计显著性。
3. 每种 workload 三轮，但在同一个 long-lived session 中按 W0→W1→W2 固定顺序运行，可能受 warm-state 与顺序影响。
4. temporal sharing 与 Prism 的空间+时间共享、placement、KV ballooning 不可直接比较。
5. P1 是手工受控 release，不是外部 RAM 竞争触发的自动 policy 评估。

## 8. 可审计产物与复现

`results/request_switch/latest/` 保存：

- 9 个请求级 raw JSONL（W0/W1/W2 × 3）；
- 同一矩阵窗口的 180 条 controller events；
- metadata、三个 frozen manifest 及校验和；
- 两个最终 worker 的 allocator profile；
- P0、P1、post-reclaim 正确性证据；
- summary、图和外部 artifact 边界记录。

```bash
# Controller repo
uv run python -m controller.main --config configs/models.request_switch.local.yaml
uv run python -m scripts.launch_vllm_pool \
  --config configs/models.request_switch.local.yaml \
  --pid-file results/tmp/request-switch/pids.json

# Benchmark repo
.venv/bin/python scripts/run_request_switch_matrix.py \
  --base-url http://127.0.0.1:9000 \
  --repeats 3 \
  --out-dir results/tmp/request-switch/final-rerun

# 完整 curated summary（包含 request/controller/profile/pressure/provenance）
.venv/bin/python src/tool/build_request_switch_artifact.py \
  --input-dir results/request_switch/latest \
  --provenance results/request_switch/latest/provenance.json \
  --output /tmp/rebuilt-final-summary.json

# request/controller 核心汇总也可单独生成
.venv/bin/python src/tool/analyze_request_switch.py \
  --input-dir results/request_switch/latest \
  --controller-events results/request_switch/latest/controller-events.jsonl \
  --output /tmp/rebuilt-core-summary.json
```

机器路径仅在 gitignored `configs/*.local.yaml` 中；可提交模板为 `configs/models.request_switch.example.yaml`。
