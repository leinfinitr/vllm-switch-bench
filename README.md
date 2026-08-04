# LLM Switch Bench

LLM Switch Bench is a research harness and evidence repository for measuring large-language-model lifecycle operations and request-driven model switching. It separates reusable runners, machine-local execution, immutable raw evidence, and deterministic publication artifacts.

> **v0.1 research preview:** the checked-in `results/release-v0.1/` bundle is
> the final single-node, single-GPU release campaign. It contains five lifecycle
> cycles per model/system, strict request traces, stock-vLLM instrumentation,
> and an exact-disk GPU restore run. It is not a production or cluster-scale
> evaluation.

## Scope

The repository provides:

- vLLM cold reload and L1/L2 sleep/wake lifecycle runners;
- repeated L1 sleep/wake profiling for pinned CPU backup reuse and physical reclaim;
- an open-loop, multi-model OpenAI-compatible request-trace runner;
- adapters and lifecycle drivers for ServerlessLLM, **SwapServeLLM**, and **llama-swap**;
- exact runtime disk-backup profiling and evidence collection;
- deterministic analysis, plotting, and checksum tooling.

Canonical baseline names are **SwapServeLLM** and **llama-swap**. Historical machine-readable slugs such as `swapserve_llm` are retained inside immutable artifacts and legacy code paths; do not rename old raw data.

llama-swap is an automatic request router: the request's `model` field selects the target, and the proxy stops the current model process and starts the target as needed. It has no explicit sleep/wake phase API. Request-trace results therefore include queueing and automatic process switching, while lifecycle profiling uses separately instrumented process-state intervals. See [`docs/systems/llama-swap.md`](docs/systems/llama-swap.md).

## Repository layout

```text
configs/                    Portable examples and frozen traces
src/                        Benchmark entry points and shared libraries
src/tool/                   Pure analysis and artifact tools
src/microbench/             CUDA/allocator microbenchmarks
scripts/                    Orchestration, release builders, and checks
 tests/                     CPU-only unit and schema tests
docs/                       Current English runbooks and artifact policy
docs/archive/reports/       Historical reports; not current instructions
results/release-v0.1/       Canonical v0.1 release artifact bundle
results/tmp/                Ignored local output
runtime/                    Ignored external-runtime and backup state
```

[`docs/README.md`](docs/README.md) classifies current and archived documentation. [`results/README.md`](results/README.md) classifies current, historical, blocked, and superseded result families.

## Reproducible CPU development environment

Install [uv](https://docs.astral.sh/uv/), then use the locked Python 3.12 environment:

```bash
uv sync --frozen --group dev
uv run pytest tests -q
uv run ruff check src scripts tests
scripts/check_bash.sh
uv run python scripts/check_docs.py
uv run python scripts/verify_release_artifact.py
```

This repository is intentionally a **non-package project** (`tool.uv.package = false`). Source entry points are run from the repository root. The lockfile covers CPU development, tests, analysis, and plotting. CUDA, vLLM, model checkpoints, container runtimes, and external baseline repositories are experiment inputs and must be frozen in each run's metadata rather than hidden in the development lock.

## Portable configuration

Copy examples to ignored `*.local.yaml` files and replace placeholders:

```bash
cp configs/baseline3.example.yaml configs/baseline3.local.yaml
$EDITOR configs/baseline3.local.yaml
```

Current code and runbooks do not assume a maintainer home directory. Use paths such as `/path/to/model`, repository-relative output paths, or environment variables. Immutable historical raw evidence may still contain producer-machine paths; those bytes are retained for provenance and protected by checksums.

## Main entry points

### vLLM lifecycle

```bash
uv run python src/bench_vllm_lifecycle.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --python .venv/bin/python \
  --workdir /path/to/vllm \
  --methods sleep_l1 sleep_l2 \
  --prompts short_short \
  --repeats 5 \
  --out-dir results/tmp/vllm-lifecycle
```

### Repeated L1 sleep/wake

```bash
uv run python src/bench_vllm_repeated_sleep_l1.py \
  --models small=/path/to/small-model large=/path/to/large-model \
  --out-dir results/tmp/repeated-sleep-l1 \
  --iterations 5
```

Use `--expect-release` with physical RSS/`MemAvailable` thresholds for pressure runs and `--no-expect-release --expect-reuse` for controls.

### Request-driven switching

```bash
BASE_URL=http://127.0.0.1:9000 \
TRACE=configs/traces/request-switch-alternating.jsonl \
OUTPUT=results/tmp/request-switch/alternating.jsonl \
scripts/run_request_switch.sh
```

The trace uses absolute monotonic arrival offsets. A strict success requires HTTP 2xx, no transport/protocol error, a complete SSE stream, semantic first-token timing, and non-empty semantic output. Failed requests remain in raw output.

### Exact disk tier

```bash
uv run python scripts/run_exact_disk_profile.py \
  --model model=/path/to/model \
  --backup-root runtime/exact-disk-backups \
  --out-dir results/tmp/exact-disk/model/RUN_ID \
  -- /path/to/python /path/to/model_agnostic_driver.py
```

The wrapper records disk I/O, source/fallback, process RSS, host `MemAvailable`, footprint growth, and output equality. It keeps raw and curated output separate and refuses to overwrite an existing destination.

## Release artifact

`results/release-v0.1/` is the only canonical release artifact root. Rebuild and verify it from retained inputs with:

```bash
uv run python scripts/build_release_artifact.py
uv run python scripts/build_release_checksums.py
uv run python scripts/verify_release_artifact.py
```

The bundle was atomically replaced after the final GPU campaign. Exact-disk
measurements are included under `raw/exact-disk/`; ServerlessLLM remains a
structured blocker and is excluded from numeric plots.

## Result rules

1. Preserve successful, failed, timed-out, and blocked attempts with explicit status.
2. Never publish failed or semantically invalid samples as numeric baselines.
3. Define sleep and wake endpoints separately; only sum them when the metric is explicitly switch time.
4. Require application accounting plus OS/GPU-visible evidence for physical release claims.
5. Freeze model, prompt, dtype, context, GPU budget, source/image identity, and workload semantics.
6. Build derived files deterministically from tracked inputs and verify both checksum manifests in a fresh checkout.

See [`docs/release-artifact.md`](docs/release-artifact.md), [`results/README.md`](results/README.md), and [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License and citation

The code is released under the [Apache License 2.0](LICENSE). Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
