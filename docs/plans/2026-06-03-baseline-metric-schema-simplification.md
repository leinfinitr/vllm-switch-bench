# Baseline Metric Schema Simplification Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Simplify the baseline result schema so vLLM Sleep, SwapServeLLM, and ServerlessLLM can be compared on model-switching overhead, memory footprint, and post-switch inference performance without introducing an overly complex state-machine taxonomy.

**Architecture:** Keep one flat summary row per `(system, method, model, prompt_name, repeat_index)`. Rename ambiguous fields, add ready/evicted memory fields, compute TPOT from TTFT and output tokens, and make ServerlessLLM `latency_after_s` comparable by measuring the second active request after restore. Preserve existing benchmark harness structure and only add the minimum fields needed for the current research baseline.

**Tech Stack:** Python 3.12, pytest, requests, matplotlib, existing `benchlib.schema.write_summary_csv`, vLLM OpenAI API streaming, Docker/ServerlessLLM, Podman/SwapServeLLM.

---

## Background and motivation

The current baseline3 results are useful, but some fields have inconsistent semantics across systems:

- `startup_to_health_s` is ambiguous because `/health` or `/v1/models` means different things for vLLM, SwapServeLLM, and ServerlessLLM.
- `latency_after_s` for ServerlessLLM currently includes restore + inference, while vLLM and SwapServeLLM measure inference after an explicit restore.
- `tokens_per_s_before/after` is an end-to-end effective throughput, not TPOT.
- SwapServeLLM can expose TTFT through streaming, but the current adapter uses non-streaming requests.
- ServerlessLLM currently has no client-observable streaming TTFT because its vLLM backend has `# TODO stream results`.

This plan implements a pragmatic schema aligned with the current research goal: compare model switching cost, memory footprint, and inference performance across three baseline methods.

## Target flat schema

Keep these existing identifiers:

```text
system
method
model
prompt_name
repeat_index
ok
error
```

Rename:

```text
startup_to_health_s -> startup_latency_s
```

Add memory fields:

```text
memory_gpu_used_ready_mib
memory_cpu_used_ready_mib
memory_gpu_used_evict_mib
memory_cpu_used_evict_mib
```

Keep model switch fields:

```text
evict_latency_s
restore_latency_s
```

Keep request latency fields:

```text
ttft_before_s
ttft_after_s
latency_before_s
latency_after_s
```

Replace throughput fields with TPOT fields:

```text
tokens_per_s_before -> tpot_before_s
tokens_per_s_after  -> tpot_after_s
```

Add token-count and availability metadata:

```text
output_tokens_before
output_tokens_after
restore_latency_estimated
ttft_available
tpot_available
```

Optional compatibility fields may be kept temporarily if needed:

```text
effective_tokens_per_s_before
effective_tokens_per_s_after
```

## Field definitions

### `startup_latency_s`

Definition:

```text
Time from system/model startup, registration, or service launch until the system can process inference requests.
```

System-specific guidance:

- vLLM: process start -> model load -> CUDA initialization -> API ready.
- SwapServeLLM: SwapServeLLM launch -> vLLM backend container initialized -> router ready. If the adapter attaches to an already-running router, leave `startup_latency_s = null`.
- ServerlessLLM: model register / warmup / first readiness path. Do not use controller `/health` alone as startup latency.

### Memory fields

Definitions:

```text
memory_gpu_used_ready_mib: device-level GPU memory after system/model is ready.
memory_cpu_used_ready_mib: process/container CPU RSS after system/model is ready.
memory_gpu_used_evict_mib: device-level GPU memory after model eviction.
memory_cpu_used_evict_mib: process/container CPU RSS after model eviction.
```

GPU memory can initially use global `nvidia-smi memory.used`; this is acceptable for controlled runs with no unrelated compute apps. CPU memory should be process-tree/container RSS and must be documented per adapter.

### `evict_latency_s` and `restore_latency_s`

Definitions:

- vLLM Sleep: `/sleep` latency and `/wake_up` latency.
- SwapServeLLM: `/api/swapout` latency and `/api/swapin` latency.
- ServerlessLLM: `evict_latency_s` is time to scale to zero / idle resource state. `restore_latency_s` should be estimated as:

```text
first_post_evict_request_latency_s - second_active_request_latency_s
```

Set:

```text
restore_latency_estimated = true
```

for ServerlessLLM rows using this estimate.

### `latency_before_s` and `latency_after_s`

Definitions:

```text
latency_before_s: end-to-end inference request latency before eviction.
latency_after_s: end-to-end inference request latency after restore, with model active.
```

Important ServerlessLLM rule:

- The first post-idle request should be used to estimate restore latency.
- The second request after restore should become `latency_after_s`.

This makes `latency_after_s` comparable with vLLM and SwapServeLLM.

### `ttft_before_s` and `ttft_after_s`

Definitions:

```text
Client-observed request start -> first streamed token/chunk.
```

- vLLM: already available through streaming.
- SwapServeLLM: available after changing its adapter to request `stream: true`.
- ServerlessLLM: unavailable through current external API; keep null unless ServerlessLLM streaming or backend-internal instrumentation is implemented.

### `tpot_before_s` and `tpot_after_s`

Definitions:

```text
tpot_s = (latency_s - ttft_s) / max(output_tokens - 1, 1)
```

Only set TPOT when TTFT and output token count are available. Otherwise leave null and set:

```text
tpot_available = false
```

Do not compute TPOT for ServerlessLLM from `latency / tokens` unless the field is explicitly named `effective_tpot_*`.

---

## Task 1: Add schema tests for the new flattened fields

**Objective:** Lock down the target CSV field names and derived TPOT behavior before implementation.

**Files:**
- Modify: `tests/test_benchlib_schema.py`
- Modify: `src/benchlib/schema.py`

**Step 1: Write failing test**

Add a test similar to:

```python
def test_summary_csv_uses_baseline_metric_schema(tmp_path):
    out = tmp_path / "summary.csv"
    write_summary_csv(
        out,
        [
            {
                "system": "vllm",
                "method": "sleep_l1",
                "model": "qwen",
                "prompt_name": "short_short",
                "repeat_index": 0,
                "ok": True,
                "startup_latency_s": 1.5,
                "memory_gpu_used_ready_mib": 4000,
                "memory_cpu_used_ready_mib": 12000,
                "evict": {"latency_s": 0.2},
                "restore": {"latency_s": 0.1},
                "memory_gpu_used_evict_mib": 900,
                "memory_cpu_used_evict_mib": 13000,
                "infer_before": {
                    "ttft_s": 0.05,
                    "client_latency_s": 0.25,
                    "completion_tokens": 11,
                },
                "infer_after": {
                    "ttft_s": 0.04,
                    "client_latency_s": 0.24,
                    "completion_tokens": 11,
                },
            }
        ],
    )
    text = out.read_text()
    assert "startup_latency_s" in text
    assert "memory_gpu_used_ready_mib" in text
    assert "memory_gpu_used_evict_mib" in text
    assert "tpot_before_s" in text
    assert "tokens_per_s_before" not in text
```

**Step 2: Run test to verify failure**

```bash
cd /home/ljl/research-systems/llm-switch-bench
. .venv/bin/activate
python -m pytest tests/test_benchlib_schema.py::test_summary_csv_uses_baseline_metric_schema -q
```

Expected: FAIL because the current CSV writer still emits old fields.

**Step 3: Implement minimal schema support**

Update `src/benchlib/schema.py`:

- Replace `startup_to_health_s` with `startup_latency_s`.
- Add memory fields.
- Add `output_tokens_before/after`.
- Compute `tpot_before_s/after_s` only when TTFT, latency, and output tokens exist.
- Add `restore_latency_estimated`, `ttft_available`, `tpot_available`.
- Optionally keep `effective_tokens_per_s_before/after` for backwards visibility.

**Step 4: Run tests**

```bash
python -m pytest tests/test_benchlib_schema.py -q
python -m pytest tests -q
```

**Step 5: Commit**

```bash
git add src/benchlib/schema.py tests/test_benchlib_schema.py
git commit -m "schema: add baseline switching metric fields"
```

---

## Task 2: Add reusable resource sampling helpers

**Objective:** Provide one helper layer for GPU memory and process/container CPU RSS so adapters do not duplicate logic.

**Files:**
- Create: `src/benchlib/resources.py`
- Create or modify: `tests/test_benchlib_resources.py`

**Step 1: Write failing tests**

Test at least:

```python
def test_parse_nvidia_smi_memory_used():
    assert parse_gpu_memory_used_mib("123\n") == 123
```

and process-tree RSS aggregation with monkeypatch/fake psutil objects.

**Step 2: Run test to verify failure**

```bash
python -m pytest tests/test_benchlib_resources.py -q
```

Expected: FAIL because module does not exist.

**Step 3: Implement helpers**

Implement:

```python
def query_gpu_memory_used_mib() -> int | None:
    ...

def process_tree_rss_mib(pid: int | None) -> float | None:
    ...

def docker_container_rss_mib(container_names: list[str]) -> float | None:
    ...  # optional first version may return None if unavailable

def podman_container_rss_mib(container_ids_or_names: list[str]) -> float | None:
    ...  # optional first version may return None if unavailable
```

Keep first version conservative. If CPU RSS is hard for a system, return `None` rather than fabricating.

**Step 4: Run tests**

```bash
python -m pytest tests/test_benchlib_resources.py -q
python -m pytest tests -q
```

**Step 5: Commit**

```bash
git add src/benchlib/resources.py tests/test_benchlib_resources.py
git commit -m "benchlib: add resource sampling helpers"
```

---

## Task 3: Update vLLM lifecycle harness to new schema

**Objective:** Emit `startup_latency_s`, ready/evict memory fields, output tokens, and TPOT-ready inference fields for vLLM cold reload and sleep methods.

**Files:**
- Modify: `src/bench_vllm_lifecycle.py`
- Modify: `tests/test_bench_vllm_lifecycle.py`
- Modify: `tests/test_bench_vllm_lifecycle_refactor.py`

**Step 1: Write failing tests**

Add assertions that vLLM rows include:

```text
startup_latency_s
memory_gpu_used_ready_mib
memory_gpu_used_evict_mib
completion_tokens / output_tokens in infer_before and infer_after
```

**Step 2: Run targeted tests**

```bash
python -m pytest tests/test_bench_vllm_lifecycle.py tests/test_bench_vllm_lifecycle_refactor.py -q
```

Expected: FAIL until fields are emitted.

**Step 3: Implement**

In `bench_vllm_lifecycle.py`:

- Rename summary assignment from `startup_to_health_s` to `startup_latency_s`.
- After server ready and before `infer_before`, sample memory into `memory_gpu_used_ready_mib` / `memory_cpu_used_ready_mib`.
- After sleep/evict completes, sample memory into `memory_gpu_used_evict_mib` / `memory_cpu_used_evict_mib`.
- Ensure `infer()` returns `completion_tokens` from streaming parser if present; fall back to approximate count only if needed.

**Step 4: Run tests**

```bash
python -m pytest tests/test_bench_vllm_lifecycle.py tests/test_bench_vllm_lifecycle_refactor.py -q
python -m pytest tests -q
```

**Step 5: Commit**

```bash
git add src/bench_vllm_lifecycle.py tests/test_bench_vllm_lifecycle.py tests/test_bench_vllm_lifecycle_refactor.py
git commit -m "bench: emit new metrics for vllm lifecycle"
```

---

## Task 4: Update SwapServeLLM adapter for streaming TTFT and memory fields

**Objective:** Make SwapServeLLM rows comparable with vLLM by collecting client TTFT, output tokens, memory-ready, and memory-evicted fields.

**Files:**
- Modify: `src/bench_swapserve_llm.py`
- Modify: `tests/test_bench_swapserve_llm.py`

**Step 1: Write failing test for streaming inference**

Add a fake streaming response and assert:

```python
row = infer(...)
assert row["ttft_s"] == pytest.approx(...)
assert row["completion_tokens"] == ...
```

**Step 2: Run test to verify failure**

```bash
python -m pytest tests/test_bench_swapserve_llm.py -q
```

Expected: FAIL because current adapter uses non-streaming request and returns `ttft_s=None`.

**Step 3: Implement streaming infer**

In `bench_swapserve_llm.py`:

- Add `"stream": True` to payload.
- Use `requests.post(..., stream=True)`.
- Reuse `benchlib.http.parse_openai_stream_response()`.
- Return `ttft_s`, `client_latency_s`, `completion_tokens`, and output prefix.

**Step 4: Add memory sampling**

- Sample memory after model is ready and `GET /v1/models` succeeds.
- Sample memory after `/api/swapout` succeeds.
- Fill:

```text
memory_gpu_used_ready_mib
memory_cpu_used_ready_mib
memory_gpu_used_evict_mib
memory_cpu_used_evict_mib
```

If CPU RSS cannot be reliably attributed in the first version, set it to `None` and document why.

**Step 5: Run tests**

```bash
python -m pytest tests/test_bench_swapserve_llm.py -q
python -m pytest tests -q
```

**Step 6: Commit**

```bash
git add src/bench_swapserve_llm.py tests/test_bench_swapserve_llm.py
git commit -m "bench: collect swapserve ttft and memory metrics"
```

---

## Task 5: Update ServerlessLLM adapter for comparable restore and post-restore inference latency

**Objective:** Stop mixing ServerlessLLM restore and inference in `latency_after_s`; estimate restore latency and measure post-restore inference with a second active request.

**Files:**
- Modify: `src/bench_serverless_llm.py`
- Modify: `tests/test_bench_serverless_llm.py`

**Step 1: Write failing test for two post-evict requests**

In the `scale_to_zero_restore` test, fake three inference calls:

1. active before eviction
2. first post-evict request: restore + inference
3. second active request: inference after restore

Assert:

```python
assert row["restore"]["latency_s"] == pytest.approx(first_post_evict_latency - second_active_latency)
assert row["infer_after"]["client_latency_s"] == pytest.approx(second_active_latency)
assert row["restore_latency_estimated"] is True
```

**Step 2: Run test to verify failure**

```bash
python -m pytest tests/test_bench_serverless_llm.py::test_scale_to_zero_restore_sequence_records_wait_and_restore -q
```

Expected: FAIL because current code uses first post-idle request as both restore and `infer_after`.

**Step 3: Implement**

In `run_scale_to_zero_restore()`:

- Keep `infer_before` as the active-state before request.
- Wait for scale-to-zero as current `evict`.
- Send first post-evict request and store it internally as `restore_trigger_request` or `stage_breakdown.first_post_evict_request_s`.
- Send second request and set it as `infer_after`.
- Compute:

```python
restore_latency_s = max(0, first_post_evict_request["client_latency_s"] - infer_after["client_latency_s"])
```

- Set:

```python
row["restore_latency_estimated"] = True
row["ttft_available"] = False
row["tpot_available"] = False
```

**Step 4: Add memory sampling**

- Sample ready memory after register / active before request.
- Sample evicted memory after scale-to-zero.

**Step 5: Run tests**

```bash
python -m pytest tests/test_bench_serverless_llm.py -q
python -m pytest tests -q
```

**Step 6: Commit**

```bash
git add src/bench_serverless_llm.py tests/test_bench_serverless_llm.py
git commit -m "bench: separate serverless restore from active inference"
```

---

## Task 6: Update baseline3 merge and analysis scripts

**Objective:** Make merged baseline3 results and plots use the new fields and avoid old `tokens_per_s` / `startup_to_health` labels.

**Files:**
- Modify: `src/bench_baseline3.py`
- Modify: `src/analyze_baseline3.py`
- Modify: `src/plot_baseline3.py`
- Modify: `tests/test_bench_baseline3.py`
- Modify: `tests/test_analyze_baseline3.py`
- Modify: `tests/test_plot_baseline3.py`

**Step 1: Write failing tests**

Add tests that check:

- Aggregated report displays `startup_latency_s`.
- Plot script uses `tpot_before_s` / `tpot_after_s` instead of tokens/s.
- ServerlessLLM restore rows can be marked estimated.

**Step 2: Run tests**

```bash
python -m pytest tests/test_bench_baseline3.py tests/test_analyze_baseline3.py tests/test_plot_baseline3.py -q
```

Expected: FAIL until scripts are updated.

**Step 3: Implement updates**

- Update imported vLLM result fields if needed.
- Update report table headings.
- Update plotting subplots:
  - switch overhead: evict / restore
  - memory footprint: ready vs evicted GPU memory
  - inference latency: before / after
  - TPOT: before / after, skipping unavailable rows

**Step 4: Run tests and regenerate figure**

```bash
python -m pytest tests -q
python src/plot_baseline3.py \
  results/baselines/baseline3/qwen2p5_0p5b/<new_run> \
  --out docs/reports/figures/baseline3-qwen2p5-0p5b-comparison.png \
  --title 'Qwen2.5-0.5B switch comparison'
```

**Step 5: Commit**

```bash
git add src/bench_baseline3.py src/analyze_baseline3.py src/plot_baseline3.py tests docs/reports/figures
git commit -m "analysis: update baseline3 reports for simplified metrics"
```

---

## Task 7: Smoke-run the new schema before full baseline rerun

**Objective:** Validate the new fields on a short, cheap run before spending time on the full baseline matrix.

**Files:**
- Runtime outputs under `results/tmp/` only.
- No commit unless new bugs require code changes.

**Step 1: Run tests**

```bash
cd /home/ljl/research-systems/llm-switch-bench
. .venv/bin/activate
python -m pytest tests -q
```

Expected: all tests pass.

**Step 2: Run vLLM smoke if GPU is available**

```bash
python src/bench_vllm_lifecycle.py \
  --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct \
  --python .venv/bin/python \
  --workdir /home/ljl/research-systems/llm-switch-bench \
  --methods sleep_l1 \
  --prompts short_short \
  --repeats 1 \
  --ready-timeout-s 360 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 1024 \
  --port 0 \
  --out-dir results/tmp/vllm_schema_smoke
```

**Step 3: Run SwapServeLLM smoke if router/backend are up**

```bash
python src/bench_swapserve_llm.py \
  --repo /home/ljl/research-systems/SwapServeLLM \
  --base-url http://127.0.0.1:8000 \
  --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct \
  --api-key dummy \
  --log-dir /tmp/swapserve-exp/logs \
  --prompts short_short \
  --repeats 1 \
  --out-dir results/tmp/swapserve_schema_smoke
```

**Step 4: Run ServerlessLLM smoke if controller is up**

```bash
python src/bench_serverless_llm.py \
  --repo /home/ljl/research-systems/ServerlessLLM \
  --model /host-models/hf/Qwen2.5-0.5B-Instruct \
  --registered-model-name qwen2p5-0p5b \
  --base-url http://127.0.0.1:8343 \
  --prompts short_short \
  --repeats 1 \
  --methods scale_to_zero_restore \
  --out-dir results/tmp/serverless_schema_smoke
```

**Step 5: Verify output fields**

For each smoke output:

```bash
python - <<'PY'
import csv
from pathlib import Path
for p in Path('results/tmp').glob('**/summary.csv'):
    with p.open() as f:
        row = next(csv.DictReader(f))
    print(p)
    for key in [
        'startup_latency_s',
        'memory_gpu_used_ready_mib',
        'memory_gpu_used_evict_mib',
        'evict_latency_s',
        'restore_latency_s',
        'latency_before_s',
        'latency_after_s',
        'tpot_before_s',
        'tpot_after_s',
    ]:
        print(' ', key, row.get(key))
PY
```

Expected: fields exist. Some ServerlessLLM TTFT/TPOT fields may be blank by design.

---

## Task 8: Full baseline3 rerun and report update

**Objective:** Produce final comparable baseline rows using the simplified schema.

**Files:**
- Create: `results/baselines/baseline3/qwen2p5_0p5b/<new_timestamp>/...`
- Modify: `docs/reports/baseline3-qwen2p5-0p5b.md`
- Modify: `docs/reports/figures/baseline3-qwen2p5-0p5b-comparison.png`

**Step 1: Ensure runtimes are ready**

Follow:

- `docs/baselines/baseline1-vllm-cold-reload.md`
- `docs/baselines/baseline2-vllm-sleep-mode.md`
- `docs/baselines/baseline3-engine-checkpoint-hotswap.md`
- `docs/systems/serverlessllm.md`
- `docs/systems/swapservellm.md`

**Step 2: Run full baseline3**

```bash
python src/bench_baseline3.py \
  --config configs/baseline3.local.yaml \
  --systems vllm serverless_llm swapserve_llm \
  --prompts short_short long_short short_long \
  --repeats 3 \
  --out-dir results/baselines/baseline3/qwen2p5_0p5b
```

**Step 3: Generate report and figure**

```bash
python src/analyze_baseline3.py \
  results/baselines/baseline3/qwen2p5_0p5b/<new_timestamp> \
  --out docs/reports/baseline3-qwen2p5-0p5b.md

python src/plot_baseline3.py \
  results/baselines/baseline3/qwen2p5_0p5b/<new_timestamp> \
  --out docs/reports/figures/baseline3-qwen2p5-0p5b-comparison.png \
  --title 'Qwen2.5-0.5B switch comparison'
```

**Step 4: Verify**

```bash
python -m pytest tests -q
file docs/reports/figures/baseline3-qwen2p5-0p5b-comparison.png
```

Expected:

- tests pass
- figure exists and is readable
- ServerlessLLM TPOT may be unavailable unless streaming/instrumentation was added
- report explicitly notes `restore_latency_estimated=True` for ServerlessLLM

**Step 5: Commit**

```bash
git add results/baselines docs/reports
 git commit -m "results: refresh baseline3 simplified metrics"
```

---

## Non-goals for this phase

Do not implement a full state-machine benchmark framework in this phase.

Do not implement ServerlessLLM streaming unless it becomes necessary for the paper's final TTFT comparison. For this phase, leaving ServerlessLLM TTFT/TPOT empty is acceptable if documented.

Do not compare raw `latency_after_s` values unless ServerlessLLM has been changed to use the second active request after restore.

Do not use controller `/health` alone as startup latency.

---

## Acceptance criteria

The implementation is complete when:

1. `summary.csv` uses the simplified schema and no longer emits `startup_to_health_s` or `tokens_per_s_before/after` as primary fields.
2. vLLM and SwapServeLLM rows include client TTFT and TPOT when streaming succeeds.
3. ServerlessLLM rows use second active request as `latency_after_s` and mark restore latency as estimated.
4. Ready and evicted GPU memory fields are present for all systems where measurement is possible.
5. CPU memory fields are either measured or left blank with documented limitations.
6. Tests pass:

```bash
python -m pytest tests -q
```

7. A smoke run confirms the new fields are present before the full baseline is rerun.

---

## Current recommendation

Start with Tasks 1-2, then implement one system adapter at a time. Do not run the full baseline matrix until the smoke run validates that all new fields land correctly in CSV/JSON.
