#!/usr/bin/env bash
set -euo pipefail

ROOT="${LLM_SWITCH_BENCH_ROOT:-/home/ljl/research-systems/llm-switch-bench}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
OUT_DIR="${OUT_DIR:-results/profiling/sleep_l1_pin_compare}"
CUDA_HOME="${CUDA_HOME:-/home/ljl/cuda-13.0}"
REPEATS="${REPEATS:-3}"
METHOD="${METHOD:-sleep_l1}"

cd "${ROOT}"
export CUDA_HOME
export PATH="$(dirname "${PYTHON}"):${CUDA_HOME}/bin:${PATH}"

cmd=(
  "${PYTHON}"
  "src/bench_vllm_pin_compare.py"
  --method "${METHOD}"
  --python "${PYTHON}"
  --cuda-home "${CUDA_HOME}"
  --out-dir "${OUT_DIR}"
  --repeats "${REPEATS}"
)

if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  models_args=(${MODELS})
  cmd+=(--models "${models_args[@]}")
fi

if [[ -n "${PIN_MODES:-}" ]]; then
  # shellcheck disable=SC2206
  pin_args=(${PIN_MODES})
  cmd+=(--pin-modes "${pin_args[@]}")
fi

if [[ -n "${PROMPTS:-}" ]]; then
  # shellcheck disable=SC2206
  prompts_args=(${PROMPTS})
  cmd+=(--prompts "${prompts_args[@]}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
fi

printf 'Running %s profiling reproduction:\n' "${METHOD}"
printf '  %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
