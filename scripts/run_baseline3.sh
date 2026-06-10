#!/usr/bin/env bash
set -euo pipefail

ROOT="${LLM_SWITCH_BENCH_ROOT:-/home/ljl/research-systems/llm-switch-bench}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/baseline3.local.yaml}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/baselines/baseline3/qwen2p5_0p5b}"
CUDA_HOME="${CUDA_HOME:-/home/ljl/cuda-13.0}"

cd "${ROOT}"
export CUDA_HOME
export PATH="$(dirname "${PYTHON}"):${CUDA_HOME}/bin:${PATH}"

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
