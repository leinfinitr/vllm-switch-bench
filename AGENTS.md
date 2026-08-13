# Project Context

LLM Switch Bench is an installed `src`-layout Python package. The default branch presents
five experiment families and the deterministic CPU workflow that rebuilds and validates
their retained artifacts. The immutable `v0.1.8` tag remains the reference for the published
v0.1 snapshot; do not recreate its former monolithic layout on the default branch.

## Ownership by directory

- `src/llm_switch_bench/`: all reusable Python runners, adapters, analysis, plotting,
  artifact builders, validators, provenance helpers, and CLI implementations.
- `scripts/`: thin executable `.sh` wrappers only. A wrapper may locate the repository,
  select the locked `uv` environment, and `exec` a package module; it must not contain
  benchmark or analysis logic.
- `docs/experiments/<family>/README.md`: the current scientific contract for one family.
- `results/<family>/`: the retained raw evidence, deterministic summary, metadata, and
  figures for one current family.
- `results/tmp/` and `runtime/`: ignored machine-local measurement and runtime state.
- `tests/`: CPU tests for runners, builders, and semantic validators.

The only current result families are:

1. `lifecycle-latency`
2. `vllm-profiling`
3. `request-driven-switch`
4. `backup-reuse-reclaim`
5. `exact-disk`

Do not add another top-level result directory without changing the scientific scope,
documentation, builder, validator, tests, and result index together.

## Documentation contract

Every experiment document must include:

- the research question;
- metric names, units, boundaries, and success predicate;
- method and frozen workload/model/system scope;
- the retained result and links to both PNG and PDF figures;
- threats to validity;
- limitations and provenance gaps;
- exact rebuild/validation commands and a separately labeled live-run command when one
  exists.

Current documentation must use repository-relative paths or explicit placeholders, never a
maintainer home directory. It must disclose that this refactor generated no new
measurements, that no canonical GPU rerun is complete, and that v0.1 E2E values are a
historical local observation because the producer did not runtime-bind engine/controller
commits, imported package path, or configuration hash.

## Result and provenance policy

- Treat retained raw evidence as measurement input. Builders may regenerate summaries,
  metadata, and figures but must not silently reinterpret or normalize producer bytes.
- The default branch does not maintain digest lists for files already versioned by Git.
  Keep cryptographic digests when they identify external executables/images or validate
  bytes handled by the runtime; exact-disk payload and chunk digests are required evidence.
- A live run must capture the benchmark commit and dirty state, actual engine/controller
  commits, imported module path, behavior-affecting configuration or its digest, model
  identity, executable/image identity, workload, and hardware/software environment.
- Preserve failed, timed-out, blocked, and semantically invalid attempts as diagnostics, but
  never aggregate them into successful numeric results.
- Logical allocator/accounting release is not physical reclaim. A physical claim requires
  process/host/GPU observations appropriate to the resource.
- Keep sleep, wake, request-visible completion, queueing, and combined switch time as
  separate metrics unless an experiment explicitly defines their composition.
- A refactor or deterministic rebuild is not a measurement run. Never present regenerated
  summaries or figures as new data.

## Required CPU verification

From a fresh checkout or clean worktree, run:

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

For package smoke testing, install the built wheel into an isolated environment and invoke
all declared console commands with `--help` from the repository root. CPU CI must not launch
a model server or require a GPU.

## Change discipline

- Read `git status --short --branch` before editing. Treat all pre-existing changes as
  another contributor's work.
- Stage an explicit path allowlist; never use `git add -A` in a shared worktree.
- Keep protocol/metric changes separate from retained-data changes and state whether a GPU
  rerun is required.
- Do not amend, force-push, rewrite the immutable tag, or push without explicit direction.
