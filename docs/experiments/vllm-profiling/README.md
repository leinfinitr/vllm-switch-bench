# vLLM profiling

## Research question and metric

Which activation phases dominate stock vLLM L1/L2 and Proposed CPU/exact-disk restoration?
Activation latency is seconds from process start or wake invocation through readiness or
completed wake; inference after readiness is excluded. Six observations are collected and
index zero is discarded. The result reports the median and min/max of the remaining five.

The stacked bar is one real sample nearest the median, with sample index as the tie-breaker.
Its disjoint phase durations must close to total latency. Exact-disk read, hashing, and H2D
copy overlap, so the plot uses pipeline wall time rather than summing concurrent work.

The [frozen campaign](../../../results/vllm-profiling/config/campaign.json) is
Qwen2.5-0.5B-Instruct, float16, max model length 1024, eager execution, 0.80 GPU memory
utilization, and one RTX 3080. Cold/L1/L2 launch fresh service processes; Proposed CPU and
disk measurements use same-process cycles. Engine revisions differ, so this is a descriptive
mechanism profile, not a release-matched ranking.

## Retained result

The 2026-08-13 median activation times are `11.534 s` cold load, `0.211 s` stock L1,
`0.355 s` stock L2, `0.299 s` Proposed CPU backup, and `0.742 s` Proposed exact disk.

- [PNG figure](../../../results/vllm-profiling/figures/vllm-profiling.png)
- [PDF figure](../../../results/vllm-profiling/figures/vllm-profiling.pdf)
- [JSON summary](../../../results/vllm-profiling/summary.json)
- [Compact retained samples](../../../results/vllm-profiling/raw/profile-samples.json)

## Reproduce the measurement

Run from the repository root on an idle GPU. Install the package in each runtime environment.
Use a different ignored output directory for every source run.

```bash
uv sync --frozen --group dev

BENCH_ROOT=$PWD
RUN_ROOT="$BENCH_ROOT/results/tmp/vllm-profiling/run-001"
MODEL=/path/to/Qwen2.5-0.5B-Instruct
STOCK_REPO=/path/to/vllm-upstream
STOCK_PYTHON="$STOCK_REPO/.venv/bin/python"
PROPOSED_REPO=/path/to/vllm-switch
PROPOSED_PYTHON="$PROPOSED_REPO/.venv/bin/python"

"$STOCK_PYTHON" -m pip install -e . --no-deps
"$PROPOSED_PYTHON" -m pip install -e . --no-deps
```

Collect cold load and stock L1/L2 separately so each summary has an unambiguous method set:

```bash
scripts/vllm-profiling.sh \
  --model "$MODEL" \
  --served-model-name qwen-0.5b \
  --python "$STOCK_PYTHON" \
  --workdir "$STOCK_REPO" \
  --methods cold_reload \
  --prompts short_short \
  --repeats 6 \
  --ready-timeout-s 360 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1024 \
  --dtype float16 \
  --enforce-eager \
  --out-dir "$RUN_ROOT/cold"

scripts/vllm-profiling.sh \
  --model "$MODEL" \
  --served-model-name qwen-0.5b \
  --python "$STOCK_PYTHON" \
  --workdir "$STOCK_REPO" \
  --methods sleep_l1 sleep_l2 \
  --prompts short_short \
  --repeats 6 \
  --ready-timeout-s 360 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1024 \
  --dtype float16 \
  --enforce-eager \
  --out-dir "$RUN_ROOT/stock"
```

Collect Proposed CPU backup. The generated timestamp directory contains
`repeated_sleep_l1_summary.json`.

```bash
scripts/backup-reuse-reclaim.sh \
  --python "$PROPOSED_PYTHON" \
  --vllm-repo "$PROPOSED_REPO" \
  --models qwen-0.5b="$MODEL" \
  --iterations 6 \
  --no-expect-release \
  --expect-reuse \
  --enforce-eager \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1024 \
  --dtype float16 \
  --out-dir "$RUN_ROOT/cpu"
```

Run the complete [exact-disk producer](../exact-disk/README.md) with six cycles and place its
run directory at `$RUN_ROOT/exact`. Do not use a latency-only disk trace: promotion requires
payload, manifest, chunk, release, and output-equality evidence.

Inspect every source summary. All source rows must be successful, eager mode must be true,
sample/cycle index zero must exist for warm-up removal, and runtime commits/import paths must
match the intended checkouts.

## Update `results/`

Set the timestamp directories produced above, then stage a dry candidate:

```bash
COLD_SUMMARY="$RUN_ROOT/cold/<timestamp>/summary.json"
STOCK_SUMMARY="$RUN_ROOT/stock/<timestamp>/summary.json"
CPU_SUMMARY="$RUN_ROOT/cpu/<timestamp>/repeated_sleep_l1_summary.json"
EXACT_RUN="$RUN_ROOT/exact"

scripts/promote.sh vllm-profiling \
  --candidate-root "$RUN_ROOT/candidate-dry" \
  --collected-at YYYY-MM-DD \
  --cold-summary "$COLD_SUMMARY" \
  --stock-summary "$STOCK_SUMMARY" \
  --cpu-summary "$CPU_SUMMARY" \
  --exact-run "$EXACT_RUN"
```

Promotion compiles machine-local sources into stable compact samples and validates sample
count, warm-up removal, eager scope, phase closure, and source identity. Review the candidate,
then repeat with a new root and `--apply`:

```bash
scripts/promote.sh vllm-profiling \
  --apply \
  --candidate-root "$RUN_ROOT/candidate-apply" \
  --collected-at YYYY-MM-DD \
  --cold-summary "$COLD_SUMMARY" \
  --stock-summary "$STOCK_SUMMARY" \
  --cpu-summary "$CPU_SUMMARY" \
  --exact-run "$EXACT_RUN"

scripts/build_all.sh vllm-profiling
uv run python -m llm_switch_bench.validation.vllm_profiling.validate
git diff -- results/vllm-profiling
```

Repeat build/validation and require no second-pass diff. The previous family is under
`$RUN_ROOT/candidate-apply/previous/`.

## Threats and limitations

This is one model, host, GPU, and five post-warm-up samples per method. Page cache, allocator
history, filesystem behavior, JIT state, and process reuse affect totals and phase shares.
Different engine commits implement the mechanisms. The result does not establish throughput,
capacity, tail latency, or system superiority.
