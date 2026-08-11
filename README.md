# LLM Switch Bench

LLM Switch Bench is an experiment-oriented Python package for studying LLM lifecycle
latency, request-driven model switching, reusable CPU weight backups, physical host-memory
reclaim, and an exact-runtime-byte disk tier. The current checkout installs
`llm-switch-bench==0.2.0.dev0` from `src/llm_switch_bench`; Python implementations live in
the package, while `scripts/` contains only thin shell entry points.

> **Evidence status.** The refactor reorganizes retained measurements into four current
> result families. No new data was generated, and the canonical GPU rerun is not complete.
> In particular, the v0.1 E2E producer did not runtime-bind the engine/controller commits,
> imported package path, or configuration hash. Those numbers are a historical local observation,
> not an exact fresh-checkout runtime reproduction.

The published `v0.1.8` Git tag remains immutable. The default branch no longer carries its
old monolithic release tree or internal Git-file digest lists. Required external executable
digests remain in lifecycle metadata, and exact-disk retains its runtime payload and chunk
checksums because those digests are part of the experiment itself.

## Current result families

| Family | Question | Primary metric | Current figure |
|---|---|---|---|
| [Lifecycle latency](docs/experiments/lifecycle-latency/README.md) | How long do sleep and wake boundaries take? | Median and IQR seconds per phase | [PNG](results/lifecycle-latency/figures/lifecycle-latency.png) · [PDF](results/lifecycle-latency/figures/lifecycle-latency.pdf) |
| [Request-driven switch](docs/experiments/request-driven-switch/README.md) | What latency does an alternating model trace observe? | Per-request completion latency and failures | [PNG](results/request-driven-switch/figures/request-timeline.png) · [PDF](results/request-driven-switch/figures/request-timeline.pdf) |
| [Backup reuse and reclaim](docs/experiments/backup-reuse-reclaim/README.md) | Are exact CPU backups reused, and can pressure reclaim them physically? | Reused/released bytes, D2H time, RSS and `MemAvailable` | [PNG](results/backup-reuse-reclaim/figures/backup-reuse.png) · [PDF](results/backup-reuse-reclaim/figures/backup-reuse.pdf) |
| [Exact disk](docs/experiments/exact-disk/README.md) | Can exact runtime bytes survive CPU-backup release and restore from disk? | Spill/read/release bytes, payload integrity, output equality | [PNG](results/exact-disk/figures/exact-disk.png) · [PDF](results/exact-disk/figures/exact-disk.pdf) |

Each experiment document states its question, metric, method, retained result, threats,
limitations, and reproduction path. [`results/README.md`](results/README.md) defines the
artifact policy.

## Repository layout

```text
configs/                    Frozen request traces and schedules
src/llm_switch_bench/       Installed runners, adapters, builders, and validators
scripts/                    Thin shell wrappers around package modules
resources/                  Optional local runtime inputs (when present)
docs/experiments/           Current experiment protocols and interpretations
results/                    Exactly four current evidence families
results/tmp/                Ignored output for live local runs
tests/                      CPU unit, integration, and semantic-validator tests
runtime/                    Ignored machine-local runtime state
```

See [`src/README.md`](src/README.md), [`scripts/README.md`](scripts/README.md), and
[`docs/README.md`](docs/README.md) for the detailed indexes.

## Locked development and artifact workflow

Install [uv](https://docs.astral.sh/uv/) and Python 3.12, then run from the repository root:

```bash
uv sync --frozen --group dev
uv run python -c 'import llm_switch_bench; print(llm_switch_bench.__version__)'
uv run pytest tests -q
uv run ruff check src tests
uv run ruff format --check src tests
scripts/check_bash.sh
scripts/docs.sh
scripts/build_all.sh
scripts/validate_all.sh
scripts/tracked-ignore.sh
```

The first command installs the source-layout package into the locked environment. The
lockfile covers CPU development, validation, analysis, and plotting. CUDA, vLLM, model
checkpoints, external controllers, and external systems are experiment inputs; a live run
must capture their identity rather than treating the development lock as runtime provenance.

### Deterministic rebuild from a fresh checkout

The retained raw inputs are sufficient to rebuild the current summaries and figures without
a GPU:

```bash
scripts/build_all.sh
scripts/validate_all.sh
git diff --exit-code -- results

scripts/build_all.sh
scripts/validate_all.sh
git diff --exit-code -- results
```

Both passes must leave the tracked result bytes unchanged. Validators do more than compare
files: they recompute aggregates and enforce family-specific sample, success, sequence,
release, payload, and output-equality semantics. `scripts/tracked-ignore.sh` additionally
requires `git ls-files -ci --exclude-standard` to be empty.

## Experiment entry points

The shell commands below forward arguments to installed package modules. They may launch or
interact with GPU runtimes; they are not part of the CPU rebuild above.

### Lifecycle latency

```bash
scripts/lifecycle-latency.sh \
  --model /path/to/model \
  --workdir /path/to/vllm \
  --python /path/to/vllm/.venv/bin/python \
  --methods sleep_l1 sleep_l2 \
  --prompts short_short \
  --repeats 5 \
  --out-dir results/tmp/lifecycle-latency
```

### Request-driven switching

Replay one frozen trace against an already running OpenAI-compatible endpoint:

```bash
scripts/request-driven-switch.sh \
  --base-url http://127.0.0.1:9000 \
  --manifest configs/traces/request-switch-alternating.jsonl \
  --output results/tmp/request-driven-switch/alternating.jsonl
```

For a repeated endpoint-local matrix:

```bash
scripts/request-driven-switch-matrix.sh \
  --base-url http://127.0.0.1:9000 \
  --repeats 3 \
  --out-dir results/tmp/request-driven-switch/matrix
```

### Backup reuse and reclaim

```bash
scripts/backup-reuse-reclaim.sh \
  --models small=/path/to/small-model large=/path/to/large-model \
  --iterations 5 \
  --expect-reuse \
  --out-dir results/tmp/backup-reuse-reclaim
```

Use `--expect-release`, coordinator settings, and explicit RSS thresholds for a pressure
case; use `--no-expect-release --expect-reuse` for its control.

### Exact disk

```bash
scripts/exact-disk-run.sh \
  --model model=/path/to/model \
  --backup-root runtime/exact-disk-backups \
  --out-dir results/tmp/exact-disk/run-001 \
  -- /path/to/python -m llm_switch_bench.experiments.exact_disk.lifecycle_driver
```

A compatible instrumented vLLM runtime is required. The runner records the command outcome,
resource observations, disk footprint, runtime checksums, and before/after output.

## Evidence rules

1. Keep measurement output out of the four current roots until its protocol and evidence are
   reviewed; write live runs below `results/tmp/`.
2. Do not turn failed, timed-out, incomplete, or semantically invalid samples into numeric
   results. Preserve structured diagnostics in local or review artifacts.
3. State phase boundaries and success predicates. Sleep, wake, request completion, logical
   release, and physical reclaim are different measurements.
4. Runtime claims must bind the model and workload plus the actual engine/controller commit,
   imported path, behavior-affecting configuration, executable or image digest, and hardware.
5. Physical reclaim requires application accounting and OS/GPU-visible evidence.
6. The default branch does not maintain checksum lists over tracked Git files. Preserve
   digests only where they identify an external artifact or verify bytes handled at runtime.
7. Do not describe the retained observations as a canonical rerun; this refactor generated
   no measurements.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing code, protocols, or evidence.

## License and citation

Code is released under the [Apache License 2.0](LICENSE). Citation metadata is in
[`CITATION.cff`](CITATION.cff); use the immutable `v0.1.8` tag when citing that published
snapshot and the commit identity when citing current development work.
