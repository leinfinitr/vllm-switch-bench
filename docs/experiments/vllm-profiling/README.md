# vLLM profiling experiment

## Question

Under one retained local model/GPU scope, which activation phases dominate stock vLLM L1/L2
and Proposed vLLM CPU/exact-disk backup restoration? A cold process load is retained only as
a scale reference.

## Metric

The primary metric is activation latency in seconds. The boundary starts immediately before
process start or a wake API call and ends when health readiness or the complete public wake
transaction returns. Request generation after readiness is excluded.

Each method reports the median and minimum/maximum range over five successful post-warm-up
samples. The stacked profile uses the real sample nearest the median and requires its
non-overlapping phases to sum to the measured total. `Control overhead` is the non-negative
residual. For exact disk, read, hash, and H2D overlap, so only pipeline wall time is stacked.
Success requires a complete five-sample method cell, finite positive total latency, valid
phase accounting, and the correctness/post-condition checks of the source runner.

## Method

The retained scope is Qwen2.5-0.5B-Instruct, float16, maximum model length 1024,
`gpu_memory_utilization=0.80`, eager execution, and one NVIDIA GeForce RTX 3080. Six samples
or cycles were run and sample/cycle zero was discarded, leaving five observations per method.

- Cold load used upstream vLLM `0decac0d96` and measured fresh process start through health.
- vLLM L1/L2 used profiling checkout `03e5ae2571`; each retained sample used a fresh server.
- Proposed CPU backup used vLLM Switch `e45036767f` and repeated same-process cycles.
- Proposed exact disk used vLLM Switch `e45036767f` and repeated same-process disk restores.

The generic server runner and profile tools live in
`llm_switch_bench.experiments.vllm_profiling`. CPU-backup reuse and exact-disk integrity keep
their specialized producers because those producers enforce resource and correctness
post-conditions beyond latency profiling. The tracked `raw/profile-samples.json` is the
reviewed, byte-preserved compact profiling input. Its producer labels `CPU backup` and
`Exact disk` are rendered as `Proposed CPU backup` and `Proposed exact disk` through
`config/campaign.json`; its `source` fields document machine-local producer files but are not
fresh-checkout dependencies.

## Retained result

The retained medians are approximately 11.532 s for cold load, 0.187 s for vLLM L1,
0.329 s for vLLM L2, 0.296 s for Proposed CPU backup, and 0.747 s for Proposed exact disk.
The stacked representative samples distinguish CPU-to-GPU copy, GPU remap, checkpoint load,
KV-cache remap, and the exact-disk pipeline.

- [vLLM profiling figure (PNG)](../../../results/vllm-profiling/figures/vllm-profiling.png)
- [vLLM profiling figure (PDF)](../../../results/vllm-profiling/figures/vllm-profiling.pdf)
- [Machine-readable summary](../../../results/vllm-profiling/summary.json)
- [Retained profile samples](../../../results/vllm-profiling/raw/profile-samples.json)
- [Result-family notes](../../../results/vllm-profiling/README.md)

## Threats to validity

- The observation covers one model, one GPU, one host, and five retained samples per method.
- Engine commits differ because the mechanisms are not all available in one release-matched
  checkout.
- Cold/L1/L2 use fresh processes while CPU/exact-disk use repeated same-process cycles.
- Page cache, allocator history, pinned-memory state, filesystem behavior, and background
  load can change both totals and phase shares.
- Source-path annotations refer to ignored machine-local evidence and cannot independently
  reproduce the compact retained input from a fresh checkout.

## Limitations

This is a descriptive local mechanism profile, not a fair system ranking or a throughput,
capacity, tail-latency, or multi-model result. It does not replace the separate lifecycle,
backup-reclaim, or exact-disk correctness claims. This reorganization generated no new
measurements. The canonical GPU rerun is not complete, and the source benchmark checkout was
dirty during the retained profiling runs.

## Reproduce

### Deterministic CPU rebuild and validation

From the repository root:

```bash
uv sync --frozen --group dev
scripts/vllm-profiling-build.sh
scripts/vllm-profiling-validate.sh
scripts/validate_all.sh
git diff --exit-code -- results/vllm-profiling
```

Repeat the build/validation/diff sequence once more to verify deterministic output. This
rebuild consumes retained profile samples and performs no GPU measurement.

### Live vLLM profiling measurement

Use an idle GPU and unique directories below `results/tmp/`. Replace every placeholder and
record the actual imported package paths, benchmark and engine commits/dirty states, model
revision/config digest, complete arguments, CUDA/PyTorch/driver/GPU, and storage/page-cache
conditions.

Cold load and stock vLLM L1/L2 must be separate runs because the retained mechanisms use
different engine commits. Use six fresh-process repetitions in each run so sample zero can
be discarded. First collect the upstream cold-load reference:

```bash
MODEL_CONFIG_SHA256=$(sha256sum /path/to/Qwen2.5-0.5B-Instruct/config.json | awk '{print $1}')

scripts/vllm-profiling.sh \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen-0.5b \
  --model-revision /replace/with/model/revision \
  --model-config-sha256 "$MODEL_CONFIG_SHA256" \
  --python /path/to/vllm-upstream/.venv/bin/python \
  --workdir /path/to/vllm-upstream \
  --methods cold_reload \
  --prompts short_short \
  --repeats 6 \
  --port 0 \
  --idle-s 0.2 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1024 \
  --dtype float16 \
  --enforce-eager \
  --out-dir results/tmp/vllm-profiling/cold-load
```

Then collect L1/L2 with the checkout that exposes the phase profiler:

```bash
scripts/vllm-profiling.sh \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen-0.5b \
  --model-revision /replace/with/model/revision \
  --model-config-sha256 "$MODEL_CONFIG_SHA256" \
  --python /path/to/vllm-profiled/.venv/bin/python \
  --workdir /path/to/vllm-profiled \
  --methods sleep_l1 sleep_l2 \
  --prompts short_short \
  --repeats 6 \
  --port 0 \
  --idle-s 0.2 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1024 \
  --dtype float16 \
  --enforce-eager \
  --out-dir results/tmp/vllm-profiling/vllm-l1-l2
```

For Proposed CPU backup, use the same-process runner with six iterations:

```bash
/path/to/vllm-switch/.venv/bin/python \
  -m llm_switch_bench.experiments.backup_reuse_reclaim.run \
  --models qwen-0.5b=/path/to/Qwen2.5-0.5B-Instruct \
  --iterations 6 \
  --no-expect-release \
  --expect-reuse \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1024 \
  --dtype float16 \
  --out-dir results/tmp/vllm-profiling/cpu-backup
```

For Proposed exact disk, follow the live capture in the
[exact-disk protocol](../exact-disk/README.md), set `--cycles 6` on its lifecycle driver, and
retain the profile, output-equality observation, runtime manifest, and physical footprint.
Failed or semantically invalid attempts stay under `results/tmp/` and are never promoted.

After every run, stop all launched servers, confirm their ports and GPU processes are gone,
and review the five post-warm-up samples before replacing `raw/profile-samples.json`. The
plotting-only command for a candidate retained input is:

```bash
scripts/vllm-profiling-plot.sh \
  --input results/vllm-profiling/raw/profile-samples.json \
  --output-dir results/vllm-profiling
```
