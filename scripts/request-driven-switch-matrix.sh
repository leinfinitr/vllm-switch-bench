#!/usr/bin/env bash
set -euo pipefail
ROOT="${LLM_SWITCH_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
exec uv run python -m llm_switch_bench.experiments.request_driven_switch.run_matrix "$@"
