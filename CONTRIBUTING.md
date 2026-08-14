# Contributing

Thank you for improving vllm-switch-bench. Contributions should keep reusable package code,
GPU/runtime measurement, retained evidence, and deterministic publication transforms
separate.

## Development setup

vllm-switch-bench requires Python 3.12 and is installed from `src/vllm_switch_bench`.

```bash
uv sync --frozen --group dev
uv run python -c 'import vllm_switch_bench'
```

The lockfile is authoritative for CPU development, tests, validation, analysis, plotting,
and packaging. CUDA, vLLM, external systems, model checkpoints, and controller processes
are experiment inputs and must be identified by run metadata.

## Choose the right home

- Put reusable Python behavior in `src/vllm_switch_bench/` and add focused tests in `tests/`.
- Keep `scripts/` to thin `.sh` wrappers that execute package modules. Do not add Python
  implementations or embed benchmark logic in shell.
- Put the current protocol for a family in `docs/experiments/<family>/README.md`.
- Write unreviewed live output below ignored `results/tmp/`; do not add ad hoc top-level
  result directories.
- The five current publication families are documented in [`results/README.md`](results/README.md).

## Metric and experiment changes

Open an issue before changing a metric boundary, success predicate, frozen workload, system
matrix, or evidence requirement. A reviewable experiment change must state:

1. the question and hypothesis;
2. the metric, unit, start/end boundary, and valid denominator;
3. the method, controls, and frozen input identities;
4. expected correctness and physical post-conditions;
5. threats, limitations, and whether a new GPU run is necessary;
6. how raw evidence is promoted and how the semantic validator will fail on invalid data.

A live run must bind the benchmark commit/dirty state, engine and controller commits,
actually imported package path and repository state, behavior-affecting configuration or digest, model
revision, workload, external executable or image digest, and hardware/software environment.
Do not infer runtime identity from a nearby checkout or from `uv.lock`.

Failed, timed-out, blocked, or semantically invalid attempts are diagnostics, not numeric
baseline rows. Keep logical release separate from process/host/GPU-visible reclaim, and keep
sleep, wake, queueing, request completion, and combined switch time distinct.

## Retained artifact policy

The default branch relies on Git object identity for tracked repository files and does not
add internal digest manifests over the result tree. Digests remain required when they bind
an external binary/image or verify runtime payload bytes. In particular, do not remove the
exact-disk payload and per-chunk checksums.

Builders must deterministically regenerate summaries, metadata, and figures from the
tracked raw inputs. A documentation or layout refactor must not claim regenerated output as
a new measurement. The five current result families contain the 2026-08-13 local reruns and
bind their runtime inputs in retained raw evidence.

## Required verification

Run all CPU gates from the repository root:

```bash
uv sync --frozen --group dev
uv run pytest tests -q
uv run ruff check src tests
uv run ruff format --check src tests
scripts/check_bash.sh
scripts/docs.sh
scripts/build_all.sh
scripts/validate_all.sh
git diff --exit-code -- results
scripts/build_all.sh
scripts/validate_all.sh
git diff --exit-code -- results
scripts/tracked-ignore.sh
uv build
```

Install the resulting wheel in a fresh temporary environment and smoke every declared
console command with `--help` from the checkout root. If your change needs GPU verification,
list the exact command and result. If no GPU run was made, say so; do not substitute a dry
run or deterministic rebuild for a measurement.

## Commits and pull requests

Use focused English commits. Before committing:

- inspect `git status`, unstaged and staged diffs;
- stage only an explicit allowlist;
- run `git diff --check --cached`;
- verify no credentials, model files, caches, machine-local config, or unrelated output are
  included.

A pull request must describe the question/policy impact, files and raw bytes changed,
verification commands and outcomes, provenance limitations, and GPU runs completed or
still required.
