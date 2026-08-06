#!/usr/bin/env bash
set -euo pipefail

exec uv run python -m llm_switch_bench.experiments.backup_reuse_reclaim.pin_compare "$@"
