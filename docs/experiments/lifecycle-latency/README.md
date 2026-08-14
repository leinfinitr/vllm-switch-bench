# Lifecycle latency

## Research question and metric

How long do explicit model sleep and wake operations take across the frozen model/runtime
matrix? `sleep_s` starts immediately before the native sleep or unload operation and ends
when that operation returns. `wake_s` starts immediately before native wake or load and ends
when the runtime reports ready. Both are seconds. Request queueing and inference are excluded,
and sleep and wake are never added into a synthetic switch metric.

Each of the 30 model/system/phase cells contains five successful cycles. The summary reports
the median and the second/fourth order statistics as Q1/Q3. A cycle is successful only when
both lifecycle operations return, the runtime-specific sleep postcondition holds, and
post-wake output equals the reference output. Failed attempts stay under `results/tmp/` and
must not be promoted.

The frozen matrix is three Qwen2.5 Instruct models (0.5B, 1.5B, and 3B) and five mechanisms:
Proposed, stock vLLM L1, stock vLLM L2, SwapServeLLM, and instrumented llama-swap. See the
[campaign contract](../../../results/lifecycle-latency/config/campaign.json). Native
boundaries are not identical: llama-swap observes process unload/cold start, while
SwapServeLLM includes its checkpoint/container control path.

## Retained result

The 2026-08-13 local rerun used an RTX 3080. Proposed median sleep/wake was approximately
`0.052/0.132 s`, `0.077/0.242 s`, and `0.155/0.481 s` for 0.5B, 1.5B, and 3B. Stock vLLM L2
median wake increased from `0.267 s` to `1.584 s`. Instrumented llama-swap wake was about
`11.3-13.3 s`. Every retained post-wake output matched its reference.

- [PNG figure](../../../results/lifecycle-latency/figures/lifecycle-latency.png)
- [PDF figure](../../../results/lifecycle-latency/figures/lifecycle-latency.pdf)
- [JSON summary](../../../results/lifecycle-latency/summary.json)
- [CSV summary](../../../results/lifecycle-latency/summary.csv)

## Reproduce the measurement

Run from the benchmark repository root on an idle GPU. Install this package in both vLLM
environments, and use clean, pinned runtime checkouts. The runner rejects a vLLM import that
does not come from the declared `--vllm-repo`.

```bash
uv sync --frozen --group dev

BENCH_ROOT=$PWD
RUN_ROOT="$BENCH_ROOT/results/tmp/lifecycle-latency/run-001"
PROPOSED_REPO=/path/to/vllm-switch
PROPOSED_PYTHON="$PROPOSED_REPO/.venv/bin/python"
STOCK_REPO=/path/to/vllm-upstream
STOCK_PYTHON="$STOCK_REPO/.venv/bin/python"
MODEL_ROOT=/path/to/models

"$PROPOSED_PYTHON" -m pip install -e . --no-deps
"$STOCK_PYTHON" -m pip install -e . --no-deps
```

Collect Proposed and stock vLLM L1/L2. The 3B model uses `0.80` GPU utilization; the smaller
models use `0.45`.

```bash
for spec in \
  qwen-0.5b:Qwen2.5-0.5B-Instruct:0.45 \
  qwen-1.5b:Qwen2.5-1.5B-Instruct:0.45 \
  qwen-3b:Qwen2.5-3B-Instruct:0.80
do
  name=${spec%%:*}
  remainder=${spec#*:}
  directory=${remainder%%:*}
  utilization=${spec##*:}

  scripts/lifecycle-latency.sh vllm \
    --python "$PROPOSED_PYTHON" \
    --vllm-repo "$PROPOSED_REPO" \
    --sleep-level 1 \
    --model "$MODEL_ROOT/$directory" \
    --model-name "$name" \
    --system-name Proposed \
    --cycles 5 \
    --gpu-memory-utilization "$utilization" \
    --max-model-len 1024 \
    --dtype float16 \
    --output "$RUN_ROOT/proposed/$name.json"

  for level in 1 2
  do
    scripts/lifecycle-latency.sh vllm \
      --python "$STOCK_PYTHON" \
      --vllm-repo "$STOCK_REPO" \
      --sleep-level "$level" \
      --model "$MODEL_ROOT/$directory" \
      --model-name "$name" \
      --system-name "vLLM L$level" \
      --cycles 5 \
      --gpu-memory-utilization "$utilization" \
      --max-model-len 1024 \
      --dtype float16 \
      --output "$RUN_ROOT/vllm-l$level/$name.json"
  done
done
```

The external adapters consume already running services; they do not own those processes.
Use a profiling-enabled llama-swap checkout, copy the complete template, and replace every
`/absolute/path/...` placeholder. Keep the model names, ports, timeouts, eager mode, dtype,
maximum length, and `0.80` utilization unchanged. Build the binary from the pinned checkout:

```bash
LLAMA_REPO=/path/to/llama-swap
STOCK_REPO=/path/to/vllm-upstream
LLAMA_CONFIG="$RUN_ROOT/llama-swap/config.yaml"
LLAMA_BINARY="$LLAMA_REPO/build/llama-swap"
LLAMA_PROFILE="$RUN_ROOT/llama-swap/profile.jsonl"

mkdir -p "$RUN_ROOT/llama-swap"
cp docs/experiments/lifecycle-latency/llama-swap.example.yaml "$LLAMA_CONFIG"
$EDITOR "$LLAMA_CONFIG"

git -C "$LLAMA_REPO" status --short --branch
git -C "$STOCK_REPO" status --short --branch
(
  cd "$LLAMA_REPO"
  go build -o "$LLAMA_BINARY" .
)
sha256sum "$LLAMA_CONFIG" "$LLAMA_BINARY"
```

Start llama-swap in this shell, retain its PID, and wait for the router. The selected
llama-swap revision must implement `LLAMA_SWAP_LIFECYCLE_PROFILE_PATH`; an ordinary upstream
build cannot produce the unload/load boundary consumed by this adapter.

```bash
LLAMA_SWAP_LIFECYCLE_PROFILE_PATH="$LLAMA_PROFILE" \
  "$LLAMA_BINARY" \
    --config "$LLAMA_CONFIG" \
    --listen 127.0.0.1:18100 \
    >"$RUN_ROOT/llama-swap/server.log" 2>&1 &
LLAMA_PID=$!

curl --retry 120 --retry-delay 1 --retry-all-errors -fsS \
  http://127.0.0.1:18100/v1/models

scripts/lifecycle-latency.sh llama-swap \
  --base-url http://127.0.0.1:18100 \
  --models qwen-0.5b qwen-1.5b qwen-3b \
  --cycles 5 \
  --repo "$LLAMA_REPO" \
  --config "$LLAMA_CONFIG" \
  --binary "$LLAMA_BINARY" \
  --lifecycle-profile "$LLAMA_PROFILE" \
  --output "$RUN_ROOT/llama-swap/lifecycle.json"

kill -TERM "$LLAMA_PID"
wait "$LLAMA_PID" || true
```

SwapServeLLM requires Podman, NVIDIA CDI, `cuda-checkpoint`, and a compatible vLLM image.
Its source dispatches on the literal image key `docker.io/vllm/vllm-openai:latest`; therefore
resolve that tag to a deliberately selected immutable image before the run and retain the
inspection output. Do not treat the tag itself as image identity.

Apply the retained compatibility patch to a clean pinned checkout. It fixes lifecycle HTTP
status handling, fixes maximum model length at 1024, and mounts the host model root read-only.
Build one binary and record its digest:

```bash
SWAPSERVE_REPO=/path/to/SwapServeLLM
SWAPSERVE_BINARY="$RUN_ROOT/swapserve/SwapServeLLM"
CUDA_CHECKPOINT=/absolute/path/to/cuda-checkpoint
SWAPSERVE_HOST_MODEL_ROOT=/absolute/path/to/models

mkdir -p "$RUN_ROOT/swapserve"
git -C "$SWAPSERVE_REPO" status --short --branch
git -C "$SWAPSERVE_REPO" apply --check \
  "$BENCH_ROOT/results/lifecycle-latency/config/swapserve-local-compat.patch"
git -C "$SWAPSERVE_REPO" apply \
  "$BENCH_ROOT/results/lifecycle-latency/config/swapserve-local-compat.patch"
(
  cd "$SWAPSERVE_REPO"
  go build -mod=vendor -o "$SWAPSERVE_BINARY" .
)
sha256sum "$SWAPSERVE_BINARY" "$CUDA_CHECKPOINT"
podman image inspect docker.io/vllm/vllm-openai:latest \
  >"$RUN_ROOT/swapserve/image-inspect.json"
```

Run one SwapServe process and one container set at a time because the router listens on fixed
port 8000. For each model, copy
[`swapserve.example.json`](swapserve.example.json), replace its run/log paths, backend/model
name, container port, and model directory, then execute the following block. For example,
the 0.5B adapter's `MODEL_IN_CONTAINER` is
`/models/hf/Qwen2.5-0.5B-Instruct`; the other directories are analogous.

```bash
name=qwen-0.5b
MODEL_IN_CONTAINER=/models/hf/Qwen2.5-0.5B-Instruct
SWAPSERVE_RUN="$RUN_ROOT/swapserve/$name"
SWAPSERVE_CONFIG="$SWAPSERVE_RUN/config.json"

mkdir -p "$SWAPSERVE_RUN/logs" "$SWAPSERVE_RUN/runtime" "$SWAPSERVE_RUN/hf-cache"
cp docs/experiments/lifecycle-latency/swapserve.example.json "$SWAPSERVE_CONFIG"
$EDITOR "$SWAPSERVE_CONFIG"
sha256sum "$SWAPSERVE_CONFIG"

(
  cd "$SWAPSERVE_RUN/runtime"
  exec env \
    SWAPSERVE_CONFIG_PATH="$SWAPSERVE_CONFIG" \
    SWAPSERVE_HOST_MODEL_ROOT="$SWAPSERVE_HOST_MODEL_ROOT" \
    SWAPSERVE_HF_CACHE="$SWAPSERVE_RUN/hf-cache" \
    SWAPSERVE_SKIP_PULL=1 \
    PATH="$(dirname "$CUDA_CHECKPOINT"):$PATH" \
    OPENAI_API_KEY=dummy \
    "$SWAPSERVE_BINARY"
) >"$SWAPSERVE_RUN/server.log" 2>&1 &
SWAPSERVE_PID=$!

curl --retry 360 --retry-delay 1 --retry-all-errors -fsS \
  -H 'Authorization: Bearer dummy' http://127.0.0.1:8000/v1/models

scripts/lifecycle-latency.sh swapservellm \
  --base-url http://127.0.0.1:8000 \
  --model "$MODEL_IN_CONTAINER" \
  --cycles 5 \
  --api-key dummy \
  --repo "$SWAPSERVE_REPO" \
  --config "$SWAPSERVE_CONFIG" \
  --binary "$SWAPSERVE_BINARY" \
  --cuda-checkpoint "$CUDA_CHECKPOINT" \
  --container-image docker.io/vllm/vllm-openai:latest \
  --output "$RUN_ROOT/swapserve/$name.json"

kill -TERM "$SWAPSERVE_PID"
wait "$SWAPSERVE_PID" || true
```

Repeat that block for 1.5B and 3B only after port 8000, run-owned containers, and GPU
allocations from the previous model are gone. SwapServe writes `gpu_pids-*.txt` in its
working directory, which is why every process receives a run-owned `runtime/` directory.

Stop only run-owned services. Check their ports, child processes, and GPU allocations before
continuing. Inspect every JSON file for five rows, matching output, the expected runtime
commit/import path, and the intended configuration. Keep any failed JSON separately.

## Update `results/`

First stage and validate without changing tracked results. `--candidate-root` must not
already exist.

```bash
scripts/promote.sh lifecycle-latency \
  --candidate-root "$RUN_ROOT/candidate-dry" \
  --collected-at YYYY-MM-DD \
  --proposed "$RUN_ROOT/proposed" \
  --vllm-l1 "$RUN_ROOT/vllm-l1" \
  --vllm-l2 "$RUN_ROOT/vllm-l2" \
  --swapserve "$RUN_ROOT/swapserve" \
  --llama-swap "$RUN_ROOT/llama-swap/lifecycle.json"
```

Review the candidate summary, raw provenance, and figure. Apply with a new candidate root:

```bash
scripts/promote.sh lifecycle-latency \
  --apply \
  --candidate-root "$RUN_ROOT/candidate-apply" \
  --collected-at YYYY-MM-DD \
  --proposed "$RUN_ROOT/proposed" \
  --vllm-l1 "$RUN_ROOT/vllm-l1" \
  --vllm-l2 "$RUN_ROOT/vllm-l2" \
  --swapserve "$RUN_ROOT/swapserve" \
  --llama-swap "$RUN_ROOT/llama-swap/lifecycle.json"

scripts/build_all.sh lifecycle-latency
uv run python -m llm_switch_bench.validation.lifecycle_latency.validate
git diff -- results/lifecycle-latency
```

The replaced family is retained under `$RUN_ROOT/candidate-apply/previous/`. Run the build
and validator again and require no second-pass diff before committing.

## Threats and limitations

This is one host/GPU and five observations per cell. Page cache, allocator history, CUDA
graph state, storage, and external startup affect the boundaries. The benchmark checkout was
dirty during collection, although its commit, status, and working-tree fingerprint are
retained. Cross-runtime values describe native operations and do not prove semantically
identical work, throughput, capacity, or tail behavior. SwapServe still requires a literal
`latest` configuration key, but the adapter resolves it to and retains the local image ID,
manifest digest, repository digests, and embedded vLLM build identity used for the run. The
retained local vLLM rows identify model directories by path but predate model-config digest
capture; a future rerun should close that portability gap rather than treating the path as a
registry revision.
