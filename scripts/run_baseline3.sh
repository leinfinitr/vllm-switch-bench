#!/usr/bin/env bash
set -euo pipefail

ROOT="${LLM_SWITCH_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/baseline3.local.yaml}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/baselines/baseline3}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing local config: ${CONFIG}" >&2
  echo "copy configs/baseline3.example.yaml and replace all placeholders" >&2
  exit 2
fi

cd "${ROOT}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

cmd=(
  "${PYTHON}"
  "src/bench_baseline3.py"
  --config "${CONFIG}"
  --out-dir "${OUT_DIR}"
)

if [[ -n "${SYSTEMS:-}" ]]; then
  # shellcheck disable=SC2206
  systems_args=(${SYSTEMS})
  cmd+=(--systems "${systems_args[@]}")
fi

if [[ -n "${PROMPTS:-}" ]]; then
  # shellcheck disable=SC2206
  prompts_args=(${PROMPTS})
  cmd+=(--prompts "${prompts_args[@]}")
fi

if [[ -n "${REPEATS:-}" ]]; then
  cmd+=(--repeats "${REPEATS}")
fi

printf 'Running baseline reproduction:\n'
printf '  %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
