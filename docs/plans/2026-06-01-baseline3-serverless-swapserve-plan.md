# Baseline3 ServerlessLLM and SwapServeLLM Comparison Plan

> **For Hermes:** Do not execute this plan until the user approves it. After approval, implement task-by-task with TDD, smoke tests, and commits after each stable slice.

**Goal:** Extend `llm-switch-bench` so Baseline3 compares existing vLLM `cold_reload` / `sleep_l1` / `sleep_l2` results with ServerlessLLM and SwapServeLLM model-switch lifecycle performance.

**Architecture:** Keep `llm-switch-bench` as the orchestration and analysis repository. Treat `/home/ljl/research-systems/ServerlessLLM` and `/home/ljl/research-systems/SwapServeLLM` as external checked-out systems, referenced by path in configs/metadata rather than copied into this repo by default. Add a small adapter layer that normalizes lifecycle operations (`start`, `infer_before`, `evict`, `restore`, `infer_after`) and writes the same `summary.json` / `summary.csv` / `events.jsonl` schema as the current vLLM harness, with an additional `system` field.

**Tech Stack:** Python 3.12, uv venv in `llm-switch-bench`, pytest, requests, psutil, nvidia-smi; external systems: ServerlessLLM Python/Ray/FastAPI API, SwapServeLLM Go/Podman/OpenAI-compatible router.

---

## Context discovered so far

- Current benchmark repo: `/home/ljl/research-systems/llm-switch-bench`.
- Current vLLM harness: `src/bench_vllm_lifecycle.py`.
- Current vLLM result schema already records startup, evict, restore, TTFT, latency, token rate, GPU/CPU samples, server logs.
- ServerlessLLM repo: `/home/ljl/research-systems/ServerlessLLM`.
  - API server default port: `8343`.
  - Health: `GET /health`.
  - Register/deploy: `POST /register`; CLI `sllm deploy` sends model config here.
  - Delete: `POST /delete`.
  - Inference: `POST /v1/chat/completions`.
  - vLLM backend uses `load_format="serverless_llm"` if no explicit `pretrained_model_name_or_path` is passed, and can use raw HF path through `backend_config.pretrained_model_name_or_path`.
  - Default config has `min_instances=0`, `keep_alive=0`, which is useful for serverless scale-to-zero restore experiments.
- SwapServeLLM repo: `/home/ljl/research-systems/SwapServeLLM`.
  - Router default port: `8000`.
  - Inference: `POST /v1/chat/completions`.
  - Manual lifecycle endpoints: `POST /api/swapout`, `POST /api/swapin`, body includes `model`.
  - Swap-out implementation logs stages: GPU PID lookup, model unload, CUDA PID toggle/checkpoint, container pause.
  - Swap-in implementation logs stages: container unpause, CUDA restore, server wait, model load.

## Baseline3 scope and fairness rules

### What to compare

Use a single small model first:

- `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`

Use the same prompt set as vLLM:

- `short_short`
- `long_short`
- `short_long`

Use at least 3 repeats per method after smoke tests pass.

### Systems / methods

| system | methods | lifecycle meaning |
|---|---|---|
| `vllm` | `cold_reload`, `sleep_l1`, `sleep_l2` | existing server harness |
| `serverless_llm` | `delete_register`, `scale_to_zero_restore` if feasible | deregister or idle/scale-to-zero then first request/deploy restores model |
| `swapserve_llm` | `swapout_swapin`, optionally `request_triggered_swapin` | `/api/swapout` then `/api/swapin`, and maybe first request when swapped out |

### Metrics to report

Per run:

- `system`, `method`, `model`, `prompt_name`, `repeat_index`, `ok`
- `startup_to_health_s` or equivalent system-start readiness
- `evict_latency_s`
- `restore_latency_s`
- `infer_before.ttft_s`, `infer_before.client_latency_s`, output token rate
- `infer_after.ttft_s`, `infer_after.client_latency_s`, output token rate
- GPU HBM phase snapshots and sampled min/avg/max
- CPU RAM phase snapshots and sampled min/avg/max
- system-specific stage breakdown parsed from logs when available:
  - ServerlessLLM: register/scheduler/backend-init/store-load if logs expose it
  - SwapServeLLM: SwapOut stages and SwapIn stages from logs

### Important fairness caveat

These systems do not implement the same state transition:

- vLLM Sleep is in-process engine-level sleep/wake.
- ServerlessLLM is a serverless scheduling + checkpoint-loading path.
- SwapServeLLM is container/process hot-swap with CUDA state checkpoint/restore.

The final report should avoid claiming “A is faster than B” without qualifying the state being restored. The useful guidance is: what latency/memory each layer buys us, and which bottleneck our future system should attack.

---

## Task 1: Add shared benchmark schema utilities

**Objective:** Extract reusable prompt catalog, event logging, memory sampling, inference parsing, and summary CSV writing so vLLM/ServerlessLLM/SwapServeLLM adapters share one schema.

**Files:**
- Create: `src/benchlib/__init__.py`
- Create: `src/benchlib/schema.py`
- Create: `src/benchlib/http.py`
- Create: `src/benchlib/sampling.py`
- Modify: `src/bench_vllm_lifecycle.py`
- Test: `tests/test_benchlib_schema.py`

**Step 1: Write failing tests**

Add tests for:

```python
def test_summary_csv_includes_system_and_method(tmp_path): ...
def test_prompt_catalog_has_three_shapes(): ...
def test_openai_stream_parser_extracts_ttft_and_text(): ...
def test_event_logger_writes_jsonl(tmp_path): ...
```

Expected first run: fail because `benchlib` does not exist.

Run:

```bash
cd /home/ljl/research-systems/llm-switch-bench
.venv/bin/python -m pytest tests/test_benchlib_schema.py -q
```

**Step 2: Implement minimal benchlib**

Move shared code from `bench_vllm_lifecycle.py` without changing behavior:

- `PROMPTS`
- `Event`
- `JsonlLogger`
- `run_cmd`
- `query_gpu`
- `query_cpu`
- `make_event`
- `Sampler`
- OpenAI streaming request helper currently embedded in `infer`
- `write_summary_csv`, extended with `system`

**Step 3: Update vLLM harness**

Keep CLI compatibility for existing vLLM runs. Add `system="vllm"` to every row. Ensure old tests still pass.

**Verification:**

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python src/bench_vllm_lifecycle.py --model dummy --dry-run --out-dir results/tmp/vllm-dry-run
```

**Commit:**

```bash
git add src/benchlib src/bench_vllm_lifecycle.py tests
git commit -m "bench: factor shared lifecycle benchmark utilities"
```

---

## Task 2: Add repository discovery and baseline config

**Objective:** Make external system paths explicit and reproducible without immediately using git submodules.

**Files:**
- Create: `configs/baseline3.local.example.yaml`
- Create: `src/benchlib/config.py`
- Test: `tests/test_baseline3_config.py`
- Modify: `README.md`

**Decision:** Prefer path references first, not submodules. Submodules can be added later only if we need exact commit pinning inside this repo. For now, metadata will record each external repo path, remote URL, branch, and commit SHA.

**Config shape:**

```yaml
model: /home/ljl/models/hf/Qwen2.5-0.5B-Instruct
prompts: [short_short, long_short, short_long]
repeats: 3
cuda_home: /home/ljl/cuda-13.0
systems:
  serverless_llm:
    repo: /home/ljl/research-systems/ServerlessLLM
    host: 127.0.0.1
    port: 8343
    python: /home/ljl/research-systems/llm-switch-bench/.venv/bin/python
  swapserve_llm:
    repo: /home/ljl/research-systems/SwapServeLLM
    host: 127.0.0.1
    port: 8000
```

**Tests:**

- Config loader expands `~` and validates repo paths exist.
- Metadata collector returns commit SHA for existing repos.
- Missing repo gives actionable error.

**Verification:**

```bash
.venv/bin/python -m pytest tests/test_baseline3_config.py -q
```

**Commit:**

```bash
git add configs src/benchlib/config.py tests/test_baseline3_config.py README.md
git commit -m "bench: add baseline3 external system config"
```

---

## Task 3: Add a ServerlessLLM lifecycle adapter

**Objective:** Add a harness that can start or connect to ServerlessLLM, deploy/register a model, run inference before/after eviction, and record normalized lifecycle metrics.

**Files:**
- Create: `src/bench_serverless_llm.py`
- Create: `tests/test_bench_serverless_llm.py`
- Create: `docs/baseline3_serverless_llm_notes.md`

**Adapter modes:**

1. `--connect-existing`: assume an SLLM server is already running at `--host:--port`; do not start/stop it.
2. `--start-server`: start `sllm start --host ... --port ...` as a subprocess; stop at cleanup.

Start with `--connect-existing` because it is easier and safer for smoke tests. Add `--start-server` after the API path works.

**ServerlessLLM methods:**

- `delete_register`:
  1. ensure server health
  2. register/deploy model
  3. infer before
  4. delete model
  5. register/deploy model again
  6. infer after

- `idle_restore` / `scale_to_zero_restore`:
  1. register with `min_instances=0`, `keep_alive=0`, `target=1`
  2. infer before
  3. idle long enough for scale-to-zero if supported
  4. infer after and treat first request as restore

If scale-to-zero behavior is not easy to trigger deterministically, keep it as “attempted/unsupported” and use `delete_register` as the main ServerlessLLM baseline.

**TDD tests:**

Mock HTTP calls; do not start Ray or vLLM in tests.

```python
def test_build_register_payload_uses_vllm_backend_and_local_model(): ...
def test_serverless_delete_register_sequence_records_evict_restore(): ...
def test_non_200_register_is_failed_with_body_snippet(): ...
def test_summary_row_has_system_serverless_llm(): ...
```

**Implementation details:**

Register payload should set:

```json
{
  "model": "Qwen2.5-0.5B-Instruct or full path alias",
  "backend": "vllm",
  "num_gpus": 1,
  "auto_scaling_config": {"metric": "concurrency", "target": 1, "min_instances": 0, "max_instances": 1, "keep_alive": 0},
  "backend_config": {
    "pretrained_model_name_or_path": "/home/ljl/models/hf/Qwen2.5-0.5B-Instruct",
    "torch_dtype": "float16",
    "max_model_len": 1024,
    "gpu_memory_utilization": 0.45,
    "enforce_eager": false
  }
}
```

Need to verify whether ServerlessLLM’s current vLLM backend accepts `max_model_len` and `gpu_memory_utilization` through `AsyncEngineArgs`; code suggests it filters backend_config by `AsyncEngineArgs` fields, so these should pass through if names match the installed vLLM.

**Smoke command after implementation:**

```bash
# Terminal 1 or subprocess mode:
cd /home/ljl/research-systems/llm-switch-bench
PATH=$PWD/.venv/bin:/home/ljl/cuda-13.0/bin:$PATH \
CUDA_HOME=/home/ljl/cuda-13.0 \
PYTHONPATH=/home/ljl/research-systems/ServerlessLLM:$PYTHONPATH \
.venv/bin/python -m sllm.cli.clic start --host 127.0.0.1 --port 8343

# Benchmark:
.venv/bin/python src/bench_serverless_llm.py \
  --repo /home/ljl/research-systems/ServerlessLLM \
  --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct \
  --methods delete_register \
  --prompts short_short \
  --repeats 1 \
  --out-dir results/baseline3_smoke/serverless_llm
```

**Commit:**

```bash
git add src/bench_serverless_llm.py tests/test_bench_serverless_llm.py docs/baseline3_serverless_llm_notes.md
git commit -m "bench: add serverlessllm lifecycle adapter"
```

---

## Task 4: Add a SwapServeLLM lifecycle adapter

**Objective:** Add a harness that can connect to SwapServeLLM, explicitly call swapout/swapin, and normalize its timings to the shared schema.

**Files:**
- Create: `src/bench_swapserve_llm.py`
- Create: `tests/test_bench_swapserve_llm.py`
- Create: `docs/baseline3_swapserve_llm_notes.md`

**Adapter modes:**

1. `--connect-existing`: assume SwapServeLLM router is running at `127.0.0.1:8000`.
2. `--start-router`: build/start the Go router only after manual config is confirmed.

Start with `--connect-existing` because SwapServeLLM likely needs podman containers, sudo, model container config, and custom CUDA checkpoint tooling. Starting it automatically before understanding its config is risky on the shared server.

**SwapServeLLM methods:**

- `swapout_swapin`:
  1. wait for `/v1/models`
  2. ensure model is available/configured
  3. infer before through `/v1/chat/completions`
  4. `POST /api/swapout {"model": ...}` and time it
  5. sample idle memory
  6. `POST /api/swapin {"model": ...}` and time it
  7. infer after

- `request_triggered_swapin` optional:
  1. infer before
  2. explicit swapout
  3. send inference while swapped out
  4. measure end-to-end first request latency as restore+inference

**TDD tests:**

Mock HTTP calls.

```python
def test_swapserve_api_paths(): ...
def test_swapout_swapin_sequence_records_evict_restore(): ...
def test_non_200_swapin_marks_failure(): ...
def test_summary_row_has_system_swapserve_llm(): ...
```

**Stage parsing:**

SwapServeLLM logs useful stage markers:

- `[🔃 SwapOut Stage] Get GPU PIDs took ...`
- `[🔃 SwapOut Stage] Unload model took ...`
- `[🔃 SwapOut Stage] Checkpoint GPU threads took ...`
- `[🔃 SwapOut Stage] Pause container took ...`
- `[🔄 SwapIn Stage] resumeContainer completed in ...`
- `[🔄 SwapIn Stage] cuda.RestorePID ...`
- `[🔄 SwapIn Stage] WaitForServer completed ...`
- `[🔄 SwapIn Stage] LoadModel completed ...`

Implement `parse_swapserve_stage_logs(text: str) -> dict[str, float]` in the adapter or `benchlib/parsers.py` so final reports can show stage breakdown.

**Smoke command after implementation:**

```bash
.venv/bin/python src/bench_swapserve_llm.py \
  --repo /home/ljl/research-systems/SwapServeLLM \
  --base-url http://127.0.0.1:8000 \
  --model <configured-model-name> \
  --methods swapout_swapin \
  --prompts short_short \
  --repeats 1 \
  --out-dir results/baseline3_smoke/swapserve_llm
```

Need one manual discovery step before smoke: inspect SwapServeLLM config to find the configured model name and backend container image. If no config exists, add an example config under `configs/swapserve_qwen2p5_0p5b.example.yaml` but do not modify SwapServeLLM itself without approval.

**Commit:**

```bash
git add src/bench_swapserve_llm.py tests/test_bench_swapserve_llm.py docs/baseline3_swapserve_llm_notes.md
git commit -m "bench: add swapservellm lifecycle adapter"
```

---

## Task 5: Add Baseline3 orchestrator

**Objective:** Provide one command that runs the selected adapters and writes a single comparable output directory.

**Files:**
- Create: `src/bench_baseline3.py`
- Create: `tests/test_bench_baseline3.py`
- Modify: `README.md`

**CLI shape:**

```bash
.venv/bin/python src/bench_baseline3.py \
  --config configs/baseline3.local.yaml \
  --systems vllm serverless_llm swapserve_llm \
  --prompts short_short long_short short_long \
  --repeats 3 \
  --out-dir results/baseline3/qwen2p5_0p5b
```

**Implementation:**

- Load config.
- Record external repo metadata for vLLM, ServerlessLLM, SwapServeLLM.
- Dispatch each adapter in subprocess mode or import-call mode.
- Merge each adapter’s `summary.json` into one `summary.json` with `system` labels.
- Write `summary.csv` using shared writer.
- Preserve per-system logs under subdirectories.

**TDD tests:**

```python
def test_orchestrator_builds_adapter_commands(): ...
def test_merge_rows_requires_system_field(): ...
def test_metadata_records_external_repo_commits(): ...
```

**Commit:**

```bash
git add src/bench_baseline3.py tests/test_bench_baseline3.py README.md
git commit -m "bench: add baseline3 orchestrator"
```

---

## Task 6: Add comparative analysis and report generation

**Objective:** Produce a report that directly answers Baseline3: how vLLM reload/sleep compares with ServerlessLLM and SwapServeLLM, and what this implies for the future implementation.

**Files:**
- Create: `src/analyze_baseline3.py`
- Create: `tests/test_analyze_baseline3.py`
- Create: `docs/baseline3_report_template.md`

**Report sections:**

1. Environment and commit metadata.
2. Fairness caveats and state-transition definitions.
3. Timing table by system/method/prompt.
4. Phase memory table.
5. Stage breakdown:
   - vLLM sleep_l2 staged wake if present
   - SwapServeLLM swapout/swapin stage logs
   - ServerlessLLM deploy/restore stages if extractable
6. Interpretation:
   - What vLLM Sleep tells us about engine-level state retention
   - What ServerlessLLM tells us about checkpoint-loader/storage bottlenecks
   - What SwapServeLLM tells us about container/CUDA state checkpoint bottlenecks
   - Which bottlenecks are promising future optimization targets

**TDD tests:**

```python
def test_baseline3_report_groups_by_system_method_prompt(): ...
def test_report_marks_unsupported_rows_separately(): ...
def test_stage_breakdown_parser_handles_missing_logs(): ...
```

**Commit:**

```bash
git add src/analyze_baseline3.py tests/test_analyze_baseline3.py docs/baseline3_report_template.md
git commit -m "bench: add baseline3 comparative analysis"
```

---

## Task 7: Run smoke tests and resolve environment blockers

**Objective:** Prove each adapter can produce at least one valid row before running full Baseline3.

**Smoke matrix:**

| system | method | prompt | repeats |
|---|---|---|---:|
| vLLM | cold_reload | short_short | 1 |
| vLLM | sleep_l1 | short_short | 1 |
| ServerlessLLM | delete_register | short_short | 1 |
| SwapServeLLM | swapout_swapin | short_short | 1 |

**Expected blockers to handle explicitly:**

- ServerlessLLM may require installing its package into `.venv` or setting `PYTHONPATH=/home/ljl/research-systems/ServerlessLLM`.
- ServerlessLLM may require Ray resources and a working local storage path.
- ServerlessLLM vLLM backend may need a compatible vLLM version with `serverless_llm` load format; if incompatible, fall back to `pretrained_model_name_or_path` raw HF mode but label it as “SLLM scheduler + vLLM backend, not SLLM fast-loader format”.
- SwapServeLLM may require Podman, sudo, NVIDIA container runtime, and a valid config. If these are unavailable, implement and report a “config/runtime blocker” row rather than fabricating results.

**Verification:**

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python src/bench_baseline3.py --config configs/baseline3.local.yaml --systems vllm serverless_llm swapserve_llm --prompts short_short --repeats 1 --out-dir results/baseline3_smoke/qwen2p5_0p5b
```

**Commit smoke artifacts only if useful and not huge:**

```bash
git add docs results/baseline3_smoke/*/summary.csv results/baseline3_smoke/*/summary.json
git commit -m "bench: record baseline3 smoke results"
```

---

## Task 8: Run full Baseline3 and write final report

**Objective:** Produce the final reviewed Baseline3 result set for the Feishu document’s baseline3 goal.

**Full matrix:**

- systems: `vllm`, `serverless_llm`, `swapserve_llm`
- prompts: `short_short`, `long_short`, `short_long`
- repeats: `3`
- model: `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`

**Command:**

```bash
.venv/bin/python src/bench_baseline3.py \
  --config configs/baseline3.local.yaml \
  --systems vllm serverless_llm swapserve_llm \
  --prompts short_short long_short short_long \
  --repeats 3 \
  --out-dir results/baseline3/qwen2p5_0p5b

.venv/bin/python src/analyze_baseline3.py \
  results/baseline3/qwen2p5_0p5b/<timestamp> \
  --out docs/baseline3_qwen2p5_0p5b_report.md
```

**Final verification:**

```bash
.venv/bin/python -m pytest tests -q
git status --short
```

**Commit:**

```bash
git add src tests configs docs results/baseline3/qwen2p5_0p5b
git commit -m "bench: add baseline3 serverless and swapserve comparison"
```

---

## Open questions for review before execution

1. Should `ServerlessLLM` and `SwapServeLLM` be added as git submodules now, or should this repo record their external path + commit SHA in result metadata? My recommendation: path + SHA first, submodules later only if we need exact snapshot vendoring.
2. For ServerlessLLM, should the primary baseline be `delete_register` or true scale-to-zero restore? My recommendation: implement both if feasible, but report `delete_register` as deterministic and `scale_to_zero_restore` only if we can reliably observe scale-to-zero.
3. For SwapServeLLM, do you already have a working config/container image for Qwen2.5-0.5B? If yes, provide or point me to it before smoke. If no, the first implementation should include config discovery and may initially report a runtime blocker rather than performance numbers.
4. Should the full Baseline3 commit include raw event/server logs, or only `summary.json`, `summary.csv`, phase-memory CSV, and markdown report? My recommendation: commit summaries and docs; keep bulky logs untracked or archived separately unless a bug needs evidence.

