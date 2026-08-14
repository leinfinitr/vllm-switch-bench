# vllm-switch-bench

vllm-switch-bench is an experiment-oriented Python package for studying

- LLM lifecycle latency
- vLLM activation profiling
- request-driven model switching

and evaluating the following runtime capabilities with [vllm-switch](https://github.com/leinfinitr/vllm-switch):

- reusable CPU weight backups
- physical host-memory reclaim
- exact-runtime-byte disk tier.

## Current result families

| Family | Question | Primary metric | Current figure |
|---|---|---|---|
| [Lifecycle latency](docs/experiments/lifecycle-latency/README.md) | How long do sleep and wake boundaries take? | Median and IQR seconds per phase | [PNG](results/lifecycle-latency/figures/lifecycle-latency.png) · [PDF](results/lifecycle-latency/figures/lifecycle-latency.pdf) |
| [vLLM profiling](docs/experiments/vllm-profiling/README.md) | Which phases dominate vLLM L1/L2 and vllm-switch CPU/disk backup activation? | Median activation seconds, range, and phase breakdown | [PNG](results/vllm-profiling/figures/vllm-profiling.png) · [PDF](results/vllm-profiling/figures/vllm-profiling.pdf) |
| [Request-driven switch](docs/experiments/request-driven-switch/README.md) | What latency does an alternating model trace observe? | Per-request completion latency and failures | [PNG](results/request-driven-switch/figures/request-timeline.png) · [PDF](results/request-driven-switch/figures/request-timeline.pdf) |
| [Backup reuse and reclaim](docs/experiments/backup-reuse-reclaim/README.md) | Are exact CPU backups reused, and can pressure reclaim them physically? | Reused/released bytes, D2H time, RSS and `MemAvailable` | [PNG](results/backup-reuse-reclaim/figures/backup-reuse.png) · [PDF](results/backup-reuse-reclaim/figures/backup-reuse.pdf) |
| [Exact disk](docs/experiments/exact-disk/README.md) | Can exact runtime bytes survive CPU-backup release and restore from disk? | Spill/read/release bytes, payload integrity, output equality | [PNG](results/exact-disk/figures/exact-disk.png) · [PDF](results/exact-disk/figures/exact-disk.pdf) |

Each experiment document states its question, metric, method, retained result, threats,
limitations, and an ordered measurement-to-promotion runbook.

## Repository layout

```text
configs/                    Experiment and runtime configuration templates
docs/                       Documentation and experiment writeups
results/                    Exactly five current evidence families
scripts/                    Thin shell wrappers around package modules
src/vllm_switch_bench/       Installed runners, adapters, builders, and validators
tests/                      CPU unit, integration, and semantic-validator tests
```

## Locked development and artifact workflow

Install [uv](https://docs.astral.sh/uv/) and Python 3.12, then run from the repository root:

```bash
uv sync --frozen --group dev
uv run python -c 'import vllm_switch_bench'
uv run pytest tests -q
uv run ruff check src tests
uv run ruff format --check src tests
# Check the Bash syntax of every top-level shell wrapper.
scripts/check_bash.sh
# Check the five-family documentation topology and local links.
scripts/docs.sh
```

### Deterministic rebuild from a fresh checkout

The retained raw inputs are sufficient to rebuild the current summaries and figures without
a GPU:

```bash
# Rebuild summaries, metadata, result READMEs, and PNG/PDF figures from retained raw evidence.
scripts/build_all.sh
# Validate the result tree, metadata, and family-specific raw-to-summary semantics.
scripts/validate_all.sh
# Fail if a Git-tracked file is also matched by the repository's ignore rules.
scripts/tracked-ignore.sh
```

Both passes must leave the tracked result bytes unchanged. Validators do more than compare
files: they recompute aggregates and enforce family-specific sample, success, sequence,
release, payload, and output-equality semantics. `scripts/tracked-ignore.sh` additionally
requires `git ls-files -ci --exclude-standard` to be empty.

## License and citation

Code is licensed under the [Apache License 2.0](LICENSE). Citation metadata is in
[`CITATION.cff`](CITATION.cff).
