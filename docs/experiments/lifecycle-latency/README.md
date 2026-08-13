# Lifecycle latency experiment

## Question

Across the retained single-node, single-GPU model matrix, how long do distinct model
**sleep** and **wake** lifecycle boundaries take for Proposed, vLLM L1, vLLM L2,
SwapServeLLM, and llama-swap?

## Metric

The primary metrics are `sleep_s` and `wake_s`, reported independently as the median and
interquartile range (Q1–Q3) across five valid cycles for each of three model sizes.

- For vLLM and SwapServeLLM, the values are explicit lifecycle operation durations.
- For llama-swap, sleep and wake are source-instrumented process-state intervals, not its complete
request latency.

## Method

The retained matrix covers `qwen2.5-0.5b`, `qwen2.5-1.5b`, and `qwen2.5-3b`, five cycles per
model/system/phase. The builder reads family raw JSON, filters llama-swap rows by model,
recomputes Q1/median/Q3, and writes 30 aggregate cells to JSON/CSV plus the figure. The
validator requires the exact 3 × 5 × 2 matrix, five samples per cell, finite positive values,
output equality where recorded, exact external executable contracts, and byte-for-byte
raw-to-summary recomputation.

External SwapServeLLM and profiled llama-swap executables are referenced by modified repository [SwapServeLLM](https://github.com/leinfinitr/SwapServeLLM) and [llama-swap](https://github.com/leinfinitr/llama-swap/tree/research/profiling)

## Retained result

The retained medians show Proposed sleep/wake at approximately 
- `0.051/0.292 s` for `0.5B`
- `0.106/0.337 s` for `1.5B`
- `0.157/0.477 s` for `3B`

1. vLLM L1 wake medians are similar in this observation
2. vLLM L2 wake increases from about `0.431` to `1.478 s` across the model sizes.
3. SwapServeLLM medians range from `0.441–0.931 s` for sleep and `0.432–1.017 s` for wake.
4. The instrumented llama-swap wake interval is much larger (about `11.273–13.276 s`).

- [Lifecycle latency figure (PNG)](../../../results/lifecycle-latency/figures/lifecycle-latency.png)
- [Lifecycle latency figure (PDF)](../../../results/lifecycle-latency/figures/lifecycle-latency.pdf)
- [Machine-readable summary](../../../results/lifecycle-latency/summary.json)
- [Result-family notes](../../../results/lifecycle-latency/README.md)

## Threats to validity

- Only one local single-GPU environment and three Qwen model sizes are represented.
- Five cycles per cell support descriptive medians/IQRs, not broad inferential claims.
- Page-cache, allocator state, warmup, CUDA graph behavior, and external process startup can
  materially affect boundaries.
- Systems expose different native mechanisms; source-instrumented process transitions are
  not semantically identical to in-process allocator sleep/wake.

## Limitations

ServerlessLLM is not in this current numeric family because its automatic scale-to-zero
contract was not established.

## Reproduce

### Deterministic CPU rebuild and validation

From the repository root:

```bash
uv sync --frozen --group dev
scripts/build_all.sh
uv run python -m llm_switch_bench.validation.lifecycle_latency.validate
scripts/validate_all.sh
git diff --exit-code -- results/lifecycle-latency
```

Repeat the build/validation/diff sequence once more to verify clean deterministic output.

### Live measurement

Run one external baseline at a time on an otherwise idle GPU. The commands below use the
local `0.5B` checkout paths as concrete examples; change model-specific names, paths, ports,
and memory budgets together for another cell. Store all generated configs, logs, profiles,
and results under `results/tmp/`.

#### Profiled llama-swap

The lifecycle adapter does not launch llama-swap. It requires a llama-swap checkout with the
opt-in lifecycle profiler and a local config whose child-process Python and `PATH` identify
the intended vLLM environment.

Build the maintained local checkout and verify its tests. Lifecycle profiling is opt-in and
has no effect unless `LLAMA_SWAP_LIFECYCLE_PROFILE_PATH` is set:

```bash
cd ~/research-systems
git clone https://github.com/leinfinitr/llama-swap.git
cd llama-swap
git checkout research/profiling
go test ./internal/process ./internal/router ./internal/server
go build -o build/llama-swap .
```

Create `results/tmp/llama-swap-lifecycle/qwen-0.5b.yaml` with this single-model config:

```yaml
healthCheckTimeout: 360
logLevel: info
logToStdout: both
startPort: 18200
sendLoadingState: false
unloadTimeout: 30
models:
  qwen-0.5b:
    cmd: >-
      /home/ljl/research-systems/vllm-upstream/.venv/bin/python
      -m vllm.entrypoints.openai.api_server
      --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct
      --served-model-name qwen-0.5b
      --host 127.0.0.1 --port ${PORT}
      --max-model-len 1024 --gpu-memory-utilization 0.45
      --dtype half --enforce-eager --trust-remote-code
    proxy: http://127.0.0.1:${PORT}
    checkEndpoint: /health
    env:
      - CUDA_HOME=/home/ljl/cuda-13.0
      - PATH=/home/ljl/cuda-13.0/bin:/home/ljl/research-systems/vllm-upstream/.venv/bin:/usr/local/bin:/usr/bin:/bin
```

Terminal 1 starts the instrumented proxy and retains its state-machine events:

```bash
cd ~/research-systems/llm-switch-bench
mkdir -p results/tmp/llama-swap-lifecycle
rm -f results/tmp/llama-swap-lifecycle/qwen-0.5b.profile.jsonl

LLAMA_SWAP_LIFECYCLE_PROFILE_PATH="$PWD/results/tmp/llama-swap-lifecycle/qwen-0.5b.profile.jsonl" \
  /home/ljl/research-systems/llama-swap/build/llama-swap \
  --config "$PWD/results/tmp/llama-swap-lifecycle/qwen-0.5b.yaml" \
  --listen 127.0.0.1:18100 \
  2>&1 | tee results/tmp/llama-swap-lifecycle/qwen-0.5b.router.log
```

Terminal 2 checks the control plane and runs five complete cycles:

```bash
cd ~/research-systems/llm-switch-bench
curl --noproxy '*' -fsS http://127.0.0.1:18100/health
curl --noproxy '*' -fsS http://127.0.0.1:18100/running

uv run python -m llm_switch_bench.adapters.llama_swap \
  --base-url http://127.0.0.1:18100 \
  --models qwen-0.5b \
  --cycles 5 \
  --unload-timeout-s 30 \
  --repo /home/ljl/research-systems/llama-swap \
  --lifecycle-profile "$PWD/results/tmp/llama-swap-lifecycle/qwen-0.5b.profile.jsonl" \
  --output "$PWD/results/tmp/llama-swap-lifecycle/qwen-0.5b.json"
```

Stop terminal 1 with `Ctrl-C`. Confirm ports `18100` and `18200`, vLLM/EngineCore children,
and GPU compute processes are gone before starting another model.

For the `1.5B` and `3B` cells, change the model ID, checkpoint path, profile/output filenames, and child port; run
each model in a fresh llama-swap process. The adapter reports source-state wake
`stopped -> starting -> ready` separately from full request latency, and requires unload,
idle-GPU, and output-validity postconditions.

#### SwapServeLLM

SwapServeLLM requires rootless Podman, NVIDIA CDI, and a local vLLM image. The local
`SwapServeLLM` checkout includes the lifecycle support needed here: explicit successful
control responses, `--max-model-len 1024`, and an optional read-only `/models` bind mount.

```bash
cd ~/research-systems
git clone https://github.com/leinfinitr/SwapServeLLM.git
SWAPSERVE_REPO=/home/ljl/research-systems/SwapServeLLM
BENCH_REPO=/home/ljl/research-systems/llm-switch-bench
SETUP="$BENCH_REPO/results/tmp/swapservellm-lifecycle"
mkdir -p "$SETUP/logs"

cd "$SWAPSERVE_REPO"
go build \
  -tags 'containers_image_openpgp exclude_graphdriver_btrfs' \
  -o "$SETUP/SwapServeLLM" .
podman image inspect docker.io/vllm/vllm-openai:latest \
  --format '{{.Id}} {{json .RepoDigests}}'
```

Before using a newly cloned checkout, confirm that
`pkg/containers/vllm_launcher.go` contains `SWAPSERVE_HOST_MODEL_ROOT`; otherwise that clone
does not yet include the required model-mount support.

Create `$SETUP/config.json`; the router is fixed at port `8000` and the backend example uses
`18081`:

```json
{
  "swap_in_logfile": "/home/ljl/research-systems/llm-switch-bench/results/tmp/swapservellm-lifecycle/logs/swap-in.log",
  "swap_out_logfile": "/home/ljl/research-systems/llm-switch-bench/results/tmp/swapservellm-lifecycle/logs/swap-out.log",
  "cold_start_logfile": "/home/ljl/research-systems/llm-switch-bench/results/tmp/swapservellm-lifecycle/logs/cold-start.log",
  "model_latency_logfile": "/home/ljl/research-systems/llm-switch-bench/results/tmp/swapservellm-lifecycle/logs/model-latency.log",
  "router_logfile": "/home/ljl/research-systems/llm-switch-bench/results/tmp/swapservellm-lifecycle/logs/router.log",
  "openai_api_key": "dummy",
  "backend_response_timeout": 300,
  "service_ready_timeout": 180,
  "max_waiting_requests": 1000,
  "hugging_face_token": "",
  "model_list": [{
    "backend_name": "vllm-01",
    "model_name": "/models/hf/Qwen2.5-0.5B-Instruct",
    "container_image": "docker.io/vllm/vllm-openai:latest",
    "initialization_timeout": "10m",
    "gpu_memory_utilization": "0.70",
    "container_port": "18081"
  }]
}
```

Terminal 1 starts the rootless runtime and router without pulling a mutable replacement
image. Both lowercase and uppercase no-proxy variables are intentional:

```bash
cd /home/ljl/research-systems/llm-switch-bench
SETUP="$PWD/results/tmp/swapservellm-lifecycle"
systemctl --user start podman.socket
cd "$SETUP"

PATH=/tmp/swapserve-extra:$PATH \
SWAPSERVE_CONFIG_PATH="$SETUP/config.json" \
SWAPSERVE_HF_CACHE=/home/ljl/models/hf \
SWAPSERVE_HOST_MODEL_ROOT=/home/ljl/models \
SWAPSERVE_SKIP_PULL=1 \
OPENAI_API_KEY=dummy \
no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost \
  "$SETUP/SwapServeLLM" 2>&1 | tee "$SETUP/router-process.log"
```

After `Listening and serving HTTP on 0.0.0.0:8000`, terminal 2 runs the lifecycle adapter:

```bash
cd /home/ljl/research-systems/llm-switch-bench
curl --noproxy '*' -fsS http://127.0.0.1:8000/v1/models

uv run python -m llm_switch_bench.adapters.swapservellm_lifecycle \
  --base-url http://127.0.0.1:8000 \
  --model /models/hf/Qwen2.5-0.5B-Instruct \
  --cycles 5 \
  --api-key dummy \
  --output "$PWD/results/tmp/swapservellm-lifecycle/qwen-0.5b.json"
```

Stop terminal 1, then remove only SwapServeLLM-labelled containers and stop the user socket:

```bash
podman ps -aq --filter label=SwapServeLLM=1 | xargs -r podman rm -f
systemctl --user stop podman.socket
rm -f "$SETUP"/gpu_pids-*.txt
```

Confirm ports `8000` and `18081`, Podman containers, and GPU compute processes are gone.

Run `1.5B` and `3B` as separate fresh router/container cells. The retained local settings
used backend ports `18082`/`18083` and GPU utilization `0.70`/`0.80`; record this resource
asymmetry rather than treating the rows as a same-budget mechanism comparison. SwapServeLLM
`sleep_s` includes vLLM L1 unload, CUDA checkpoint, and container pause; `wake_s` includes
resume, CUDA restore, readiness, and vLLM wake.

#### vLLM

```bash
cd /home/ljl/research-systems/llm-switch-bench

STAGE="$PWD/results/tmp/vllm-lifecycle"
MODEL=/home/ljl/models/hf/Qwen2.5-0.5B-Instruct
PROP=/home/ljl/research-systems/vllm
BASE=/home/ljl/research-systems/vllm-upstream

mkdir -p "$STAGE"/{proposed,vllm-l1,vllm-l2}

source "$PROP/.venv/bin/activate"
"$PROP/.venv/bin/python" -m llm_switch_bench.adapters.vllm_sleep \
  --sleep-level 1 \
  --model "$MODEL" \
  --model-name qwen-0.5b \
  --system-name Proposed \
  --cycles 5 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 1024 \
  --vllm-repo "$PROP" \
  --output "$STAGE/proposed/qwen-0.5b.json"

deactivate
source "$BASE/.venv/bin/activate"
"$BASE/.venv/bin/python" -m llm_switch_bench.adapters.vllm_sleep \
  --sleep-level 1 \
  --model "$MODEL" \
  --model-name qwen-0.5b \
  --system-name "vLLM L1" \
  --cycles 5 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 1024 \
  --vllm-repo "$BASE" \
  --output "$STAGE/vllm-l1/qwen-0.5b.json"

"$BASE/.venv/bin/python" -m llm_switch_bench.adapters.vllm_sleep \
  --sleep-level 2 \
  --model "$MODEL" \
  --model-name qwen-0.5b \
  --system-name "vLLM L2" \
  --cycles 5 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 1024 \
  --vllm-repo "$BASE" \
  --output "$STAGE/vllm-l2/qwen-0.5b.json"
```