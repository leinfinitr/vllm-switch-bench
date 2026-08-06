# Installed source package

`src/llm_switch_bench/` is the single home for reusable Python code. The project uses a
standard installed `src` layout and currently reports version `0.2.0.dev0`:

```bash
uv sync --frozen --group dev
uv run python -c 'import llm_switch_bench; print(llm_switch_bench.__version__)'
```

Do not run source files by path or depend on implicit `sys.path` behavior. Use installed
console commands, `python -m llm_switch_bench...`, or the thin wrappers in `scripts/`.

## Package map

| Package | Responsibility |
|---|---|
| `adapters/` | External-system lifecycle and request adapters for vLLM, llama-swap, SwapServeLLM, and ServerlessLLM |
| `common/` | HTTP, frozen traces, schemas, sampling, process/host resources, merge/analysis helpers, and provenance |
| `experiments/lifecycle_latency/` | vLLM lifecycle runner plus lifecycle analysis and plotting |
| `experiments/request_driven_switch/` | Strict open-loop trace replay, repeated/randomized matrices, analysis, and plotting |
| `experiments/backup_reuse_reclaim/` | Same-process backup reuse/reclaim runner and focused CUDA/pinning microbenchmarks |
| `experiments/exact_disk/` | Runtime wrapper, allocator/lifecycle drivers, evidence collection, and runtime digest handling |
| `plotting/` | Deterministic shared figure style and save behavior |
| `validation/` | Family-specific semantic validators and exact top-level result policy |
| `artifacts.py` | Deterministic builders for all four retained result families |
| `build_all.py` | No-argument all-family build command |
| `check_docs.py` | Current documentation disclosure/reference policy |
| `tracked_ignore.py` | Tracked-versus-ignore consistency gate |

CUDA and vLLM imports are kept inside GPU-specific modules so the core package, builders,
validators, and CLI help paths remain usable in CPU development and packaging environments.

## Installed console commands

`pyproject.toml` exposes:

```text
llm-switch-build-all
llm-switch-validate-all
llm-switch-check-docs
llm-switch-lifecycle
llm-switch-trace-matrix
llm-switch-backup
llm-switch-exact-disk
```

The first three are CPU policy/publication commands. The experiment commands can launch or
interact with runtime services and may require a compatible CUDA/vLLM or external-system
environment. Use `--help` for their exact arguments; [`../scripts/README.md`](../scripts/README.md)
maps the shell wrappers to modules.

## Current publication path

The builder reads tracked raw evidence from:

```text
results/lifecycle-latency/raw/
results/request-driven-switch/raw/
results/backup-reuse-reclaim/raw/
results/exact-disk/raw/
```

It deterministically regenerates summaries, figures, result-family READMEs, and metadata.
Validators then recompute summary values and enforce experiment semantics. This path creates
no new measurements and does not require a GPU:

```bash
uv run python -m llm_switch_bench.build_all
uv run python -m llm_switch_bench.validation.validate_all
git diff --exit-code -- results
```

Git identifies tracked repository files, so package code does not generate whole-tree digest
lists. It deliberately retains cryptographic identity for external lifecycle executables and
for exact-disk runtime payload/chunk bytes.

## Development rules

- Add reusable behavior here, not in shell wrappers.
- Prefer module-relative imports within `llm_switch_bench`; never insert source directories
  into `sys.path`.
- Keep repository-bound operations explicit. `common.provenance.repository_root()` locates a
  checkout from the working directory/package ancestry and fails rather than guessing.
- Put unit/integration tests in `tests/` and keep CPU imports independent of GPU availability.
- Record runtime identity at execution: benchmark and engine/controller Git state, actually
  imported module path/version, configuration or digest, model, external executable/image,
  and environment.
- Treat physical reclaim as a resource post-condition, not an allocator counter alone.
- Never describe deterministic regeneration as a fresh benchmark run.

Run before contributing:

```bash
uv run pytest tests -q
uv run ruff check src tests
uv run ruff format --check src tests
scripts/build_all.sh
scripts/validate_all.sh
```
