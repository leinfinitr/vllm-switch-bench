# Shell entry points

Shell files in this directory are thin wrappers around installed package modules. They may
locate the checkout and select `uv`, but experiment and publication logic belongs under
`src/vllm_switch_bench/`.

## Live experiments

| Wrapper | Purpose |
|---|---|
| `lifecycle-latency.sh` | Dispatch a lifecycle run to vLLM, llama-swap, or SwapServeLLM |
| `vllm-profiling.sh` | Profile cold reload and vLLM L1/L2 activation phases |
| `request-driven-switch.sh` | Replay one frozen OpenAI-compatible trace |
| `backup-reuse-reclaim.sh` | Measure same-process backup reuse or pressure reclaim |
| `exact-disk.sh` | Capture an exact-disk lifecycle command and its evidence |

These commands may require GPU/runtime services and are not executed by CPU CI. Their
protocols and exact examples are under [`../docs/experiments/`](../docs/experiments/).
Live output belongs under ignored `results/tmp/`.

`lifecycle-latency.sh` and `backup-reuse-reclaim.sh` import vLLM for their in-process modes.
Pass `--python /path/to/vllm/.venv/bin/python` after installing this package in that
environment; the Python entry point re-executes itself before importing vLLM.

## Repository checks

| Wrapper | Purpose |
|---|---|
| `build_all.sh [family]` | Rebuild all or one family's summaries, figures, README, and metadata |
| `validate_all.sh` | Validate result-tree closure and all scientific contracts |
| `promote.sh <family> ...` | Stage, validate, and optionally atomically publish reviewed live evidence |
| `docs.sh` | Check experiment-family topology and local documentation links |
| `tracked-ignore.sh` | Reject tracked files matched by ignore rules |
| `check_bash.sh` | Syntax-check every top-level shell wrapper |

The full deterministic gate is documented in [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
All wrappers except `check_bash.sh` forward arguments and exit status to one package module.

Run `scripts/promote.sh <family> --help` for the exact raw inputs required by a family. First
omit `--apply` and review the candidate; applying requires a new candidate root and preserves
the replaced family under that ignored root's `previous/` directory.
