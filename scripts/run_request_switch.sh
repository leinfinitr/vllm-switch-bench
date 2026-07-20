#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:9000}
TRACE=${TRACE:-$ROOT/configs/traces/request-switch-alternating.jsonl}
OUTPUT=${OUTPUT:-$ROOT/results/tmp/request-switch/run.jsonl}
"$ROOT/.venv/bin/python" "$ROOT/src/bench_request_driven_switch.py" \
  --base-url "$BASE_URL" \
  --manifest "$TRACE" \
  --output "$OUTPUT"
