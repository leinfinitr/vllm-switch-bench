#!/usr/bin/env bash
set -euo pipefail
ROOT="${VLLM_SWITCH_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
exec uv run python -m vllm_switch_bench.validation.validate_all "$@"
