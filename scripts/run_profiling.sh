#!/usr/bin/env bash
set -euo pipefail

ROOT="${LLM_SWITCH_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
OUT_DIR="${OUT_DIR:-results/profiling/sleep_l1_pin_compare}"
REPEATS="${REPEATS:-3}"
METHOD="${METHOD:-sleep_l1}"

if [[ -z "${MODEL_SPECS:-}" ]]; then
  echo "MODEL_SPECS is required (NAME=PATH[,GPU_UTIL] ...)" >&2
  exit 2
fi

cd "${ROOT}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

# shellcheck disable=SC2206
model_specs=(${MODEL_SPECS})

cmd=(
  "${PYTHON}"
  "src/bench_vllm_pin_compare.py"
  --method "${METHOD}"
  --python "${PYTHON}"
  --out-dir "${OUT_DIR}"
  --repeats "${REPEATS}"
  --models "${model_specs[@]}"
)

if [[ -n "${CUDA_HOME:-}" ]]; then
  cmd+=(--cuda-home "${CUDA_HOME}")
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
