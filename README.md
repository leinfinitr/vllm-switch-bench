# LLM Switch Bench

本仓库提供 LLM serving 生命周期与模型切换的可复用 benchmark harness、microbenchmark、后处理工具和 curated 结果。实现仓库保持独立；本仓库不保存模型、临时日志或所有本地运行。

## 主要入口

- `src/bench_vllm_lifecycle.py`：vLLM cold reload、sleep/wake benchmark。
- `src/bench_vllm_pin_compare.py`：模型无关的 pinned/pageable profiling 矩阵。
- `src/bench_vllm_repeated_sleep_l1.py`：重复 sleep/wake、backup reuse、动态回收及 OS memory 观测。
- `src/bench_request_driven_switch.py`：对统一 OpenAI endpoint 重放冻结的多模型 open-loop trace，记录 transport first byte、semantic TTFT、完成时延和失败。
- `src/bench_baseline3.py`：显式配置的跨系统结果聚合与运行。
- `scripts/run_profiling.sh`、`scripts/run_baseline3.sh`：可复用 shell 入口。
- `src/microbench/`：PCIe copy、CuMemAllocator synthetic copy、safetensors allocation 粒度实验。
- `src/tool/`：只读取已有结果的分析、绘图和合并工具。

## 环境

```bash
uv venv --python 3.12 .venv
uv pip install pytest psutil requests pandas matplotlib pyyaml
```

vLLM 实验需要该环境能够 import 被测源码 checkout；不要依赖全局 wheel。每个可引用 run 应记录 vLLM/bench commit、dirty state、模型、参数和 GPU/host 信息。

测试与静态检查：

```bash
.venv/bin/python -m pytest tests -q
uv run --with ruff ruff check src tests
```

## 模型无关 profiling

pin/no-pin 矩阵必须显式提供模型，不在脚本名或默认参数中编码模型：

```bash
MODEL_SPECS='small=/models/small,0.45 large=/models/large,0.55' \
METHOD=sleep_l1 REPEATS=3 scripts/run_profiling.sh
```

重复 sleep/wake：

```bash
.venv/bin/python src/bench_vllm_repeated_sleep_l1.py \
  --models small=/models/small large=/models/large \
  --out-dir results/profiling/repeated_sleep_l1 \
  --iterations 5
```

带 coordinator 的压力实验应声明期望并自动检查物理回收：

```bash
.venv/bin/python src/bench_vllm_repeated_sleep_l1.py \
  --models model=/models/model \
  --coordinator-url http://127.0.0.1:19090 \
  --iterations 2 \
  --post-wake-observation-s 3 \
  --expect-release \
  --min-worker-rss-reclaim-bytes 1073741824
```

无压力对照使用 `--no-expect-release`。当前输出是：

```text
repeated_sleep_l1_summary.json
repeated_sleep_l1_steps.csv
```

旧 `phase1_two_model_*` 文件只属于历史 run，不是当前 schema。

## Baseline3

复制 example 并设置本地路径：

```bash
cp configs/baseline3.example.yaml configs/baseline3.local.yaml
$EDITOR configs/baseline3.local.yaml
scripts/run_baseline3.sh
```

配置必须显式指定：

- `systems.vllm.result_dir`：与当前模型/工作负载对应且包含 `summary.json`；
- external system repo；
- ServerlessLLM 的 container model path。

聚合器不会猜测“最新 Qwen run”或 host-to-container mount，避免跨模型污染。localhost control traffic 不继承环境 HTTP proxy。

## 结果与报告

请求驱动切换：

```bash
BASE_URL=http://127.0.0.1:9000 \
TRACE=configs/traces/request-switch-alternating.jsonl \
OUTPUT=results/tmp/request-switch/w1.jsonl \
scripts/run_request_switch.sh
```

manifest 使用绝对 `scheduled_offset_s`，每个请求独立调度且共享一个 async HTTP client；失败和 timeout 仍写入输出。仓库提供 alternating 与 burst-locality 两条小 trace。

- artifact policy：`results/README.md`；
- physical reclaim curated summary：`results/profiling/physical_reclaim_validation.json`；
- Baseline3：`docs/reports/baseline3-qwen2p5-*.md`；
- pin/no-pin：`docs/reports/vllm-pin-compare.md`；
- historical repeated-sleep runs：`docs/reports/phase1-two-model-pool.md`；
- copy microbench：`docs/reports/cumem-copy-microbench.md`。
- request-driven multi-model switching：`docs/reports/request-driven-multi-model-switch.md`，最新 curated artifact 位于 `results/request_switch/latest/`。

历史报告保留当时的路径、字段和结论以便审计，不代表当前 CLI。当前复现命令以本 README、`src/README.md` 和 `--help` 为准。

## 论文实验最低要求

1. 成功和失败样本都保留，禁止只筛选成功 run。
2. 比较组使用相同模型、prompt、dtype、GPU budget 和软件版本。
3. 报告 raw samples、重复次数和聚合方法，不只给均值。
4. logical release 用 application accounting；physical reclaim 同时用 worker RSS 和 host `MemAvailable`。
5. pressure/no-pressure 成对运行，并由 harness assertion 判定 release、reuse 和最小物理恢复。
6. 不通过耗尽共享机器 RAM 触发压力；使用提高水位的安全受控测试。
