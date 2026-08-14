# Installed source package

`src/vllm_switch_bench/` is the single home for reusable Python code. Install the package;
do not run source files by path or rely on an implicit `PYTHONPATH`.

## Package map

| Path | Responsibility |
|---|---|
| `adapters/` | Lifecycle contracts for vLLM, llama-swap, and SwapServeLLM |
| `common/` | HTTP, traces, schemas, sampling, process resources, and provenance |
| `experiments/<family>/run.py` | Live producer or adapter dispatcher for one experiment |
| `experiments/<family>/artifacts.py` | Family-owned summary, figure, and publication builder |
| `families.py` | Authoritative five-family registry |
| `publication.py` | Shared JSON, metadata, and result-publication mechanics |
| `artifacts.py` | Registry-driven build command |
| `validation/` | Family-specific semantic validators and registry-driven validation |
| `plotting/` | Shared deterministic figure style |
| `check_docs.py` | Documentation topology and link checker |

GPU-specific dependencies are imported inside live runner functions so builders, validators,
CLI help, and package imports remain usable in CPU environments.

## Console commands

```text
vllm-switch-bench-build-all
vllm-switch-bench-validate-all
vllm-switch-bench-check-docs
vllm-switch-bench-lifecycle
vllm-switch-bench-vllm-profiling
vllm-switch-bench-request-driven-switch
vllm-switch-bench-backup-reuse-reclaim
vllm-switch-bench-exact-disk
```

The first three operate on retained repository artifacts. The five experiment commands can
launch or contact runtime services; use their `--help` output and the corresponding
[`docs/experiments/`](../docs/experiments/) protocol.

## Design rules

- Keep measurement, aggregation, and validation owned by the relevant experiment family.
- Put only genuinely cross-family mechanics in `common/`, `publication.py`, or `plotting/`.
- Keep raw evidence immutable during deterministic builds.
- Test observable behavior and scientific invariants, not documentation wording or private
  implementation layout.
- Keep CPU import paths independent of CUDA/vLLM availability.
- Capture runtime identity at measurement time and distinguish logical accounting from
  physical resource observations.
