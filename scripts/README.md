# Shell entry points

`scripts/` contains only thin executable Bash wrappers and repository gates. Each wrapper uses `set -euo pipefail`, changes to the repository root, invokes an installed `llm_switch_bench` module through uv, and forwards all arguments. Reusable logic lives under `src/llm_switch_bench/`.

## Build and validation

```bash
scripts/build_all.sh
scripts/validate_all.sh
scripts/docs.sh
scripts/check_bash.sh
scripts/tracked-ignore.sh
```

## Experiment runners

```bash
scripts/lifecycle-latency.sh [runner arguments]
scripts/request-driven-switch.sh [trace-runner arguments]
scripts/request-driven-switch-matrix.sh [matrix arguments]
scripts/backup-reuse-reclaim.sh [repeated-sleep arguments]
scripts/exact-disk-run.sh [exact-disk wrapper arguments]
```

## Family-specific derived operations

```bash
scripts/exact-disk-build.sh
scripts/exact-disk-validate.sh
```

`build_all.sh` is the canonical builder for all summaries and figures. Individual analysis/plot modules remain available through `uv run python -m llm_switch_bench...` when developing a new run, but publication uses the aggregate builder and validators.

GPU runners require explicitly frozen model/runtime inputs and write to ignored staging output. The checked-in v0.1 families are historical evidence; invoking a wrapper does not reproduce the old GPU run without the original runtime provenance, and the canonical GPU rerun is still incomplete.
