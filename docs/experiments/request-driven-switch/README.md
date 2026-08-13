# Request-driven switch experiment

## Question

When a frozen open-loop trace alternates model names at an OpenAI-compatible endpoint, what
request completion latency and failure behavior are observed for Proposed and llama-swap?

## Metric

The primary metric is per-request completion latency in seconds, measured from client
dispatch through complete semantic streamed output. The summary reports request count,
failed count, minimum, median, and maximum completion latency for each system.

A retained request is successful only when it has no recorded error, HTTP status 200, and a
complete SSE stream (`stream_done=true`). Semantic TTFT and TPOT are retained in request
rows, but the current family figure plots completion latency against the trace's scheduled
offset. Dispatch lag remains observable and is not silently subtracted.

## Method

Both systems replayed the same 20 immutable request identities (`w1-000` through `w1-019`),
model sequence, and absolute scheduled offsets from the alternating trace. Requests are
streaming, deterministic-temperature chat completions with bounded output length. The
builder recomputes descriptive latency summaries from retained JSON rows and plots each
request on the scheduled timeline.

The validator requires exactly 20 unique requests per system, the supported identity
sequence, identical `(request_id, model, scheduled_offset)` tuples across systems, strict
success for every retained row, finite positive completion latency, exactly the two expected
systems, and exact raw-to-summary recomputation.

## Retained result

All 20 retained requests per system satisfy the current success predicate. Proposed has a
median completion latency of about `0.859 s` (`0.278–1.081 s` observed range); llama-swap has a
median of about `12.868 s` (`0.595–25.138 s` observed range). The timeline shows the variation
under the alternating local trace.

- [Request timeline (PNG)](../../../results/request-driven-switch/figures/request-timeline.png)
- [Request timeline (PDF)](../../../results/request-driven-switch/figures/request-timeline.pdf)
- [Machine-readable summary](../../../results/request-driven-switch/summary.json)
- [Result-family notes](../../../results/request-driven-switch/README.md)

llama-swap reserves unprocessed requests in a queue while the backend isn't ready. 
When the backend is ready, it processes the queued requests in parallel. The queueing behavior can lead to a large variation in completion latency.

For example, in [Request timeline (PNG)](../../../results/request-driven-switch/figures/request-timeline.png),
w1-000, w1-002, w1-004, w1-006 and w1-008 are processed at the same time, while w1-001, w1-003, w1-005, w1-007, w1-009, w1-011, w1-013, w1-015 and w1-017 are processed at the same time. Thus, the completion latency of w1-008 and w1-017 is 
near the proposed, while the completion latency of other requests is much larger than the proposed and change in an arithmetic sequence, where the difference is request scheduled offset in [request-switch-alternating.jsonl](../../../configs/traces/request-switch-alternating.jsonl)

## Threats to validity

- The family contains one 20-request alternating trace in one local single-GPU setting.
- A request-visible metric includes routing, queueing, process/model switching, first-token
  delay, and token generation; it does not isolate lifecycle phase cost.
- Open-loop overlap means an earlier switch can affect later request latency.
- The systems may differ in process lifetime, cache state, scheduler behavior, and transport
  implementation despite sharing request identities.
- Minimum/median/maximum over 20 requests are descriptive and provide no confidence interval
  or throughput characterization.

## Limitations

The family does not cover [burst](../../../configs/traces/request-switch-burst.jsonl)/[steady](../../../configs/traces/request-switch-steady.jsonl) workloads, multiple arrival rates, concurrency
scaling, failures, ServerlessLLM, SwapServeLLM, or cluster deployments. The deterministic
rebuild validates retained rows and interpretation; it cannot supply missing runtime
provenance.

## Reproduce

### Deterministic CPU rebuild and validation

From the benchmark repository root:

```bash
uv sync --frozen --group dev
scripts/build_all.sh
uv run python -m llm_switch_bench.validation.request_driven_switch.validate
scripts/validate_all.sh
git diff --exit-code -- results/request-driven-switch
```

Repeat the build/validation/diff sequence once more. This regenerates summary/figures from
tracked evidence and does not contact a serving endpoint or generate measurements.

### Live single-trace measurement

The retained family compares the same alternating `qwen-1.5b`/`qwen-3b` trace against two
self-routing endpoints: the vLLM Switch controller (labelled Proposed) and llama-swap. Run
the systems **one at a time** on an otherwise idle GPU. The runner schedules all 20 requests
from one monotonic trace origin; it does not serialize a later request behind an earlier one.

Set machine-local paths first. These are explicit placeholders, not retained producer paths:

```bash
export BENCH_REPO=/path/to/llm-switch-bench
export VLLM_REPO=/path/to/vllm
export VLLM_UPSTREAM_REPO=/path/to/vllm-upstream
export CONTROLLER_REPO=/path/to/vllm-switch-controller
export LLAMA_SWAP_REPO=/path/to/llama-swap
export MODEL_ROOT=/path/to/huggingface-models
export RUN_ROOT="$BENCH_REPO/results/tmp/request-driven-switch"

mkdir -p "$RUN_ROOT"
```

Record the five commits, dirty states, actual imported `vllm.__file__`, model revisions, GPU,
driver, and behavior-affecting configs beside each run. The commands below use loopback only
and assume ports `8101`, `8102`, `9000`, `18100`, `18200`, and `18201` are free.

```bash
for port in 8101 8102 9000 18100 18200 18201; do
  if ss -ltn "sport = :$port" | grep -q LISTEN; then
    echo "port already in use: $port" >&2
    exit 1
  fi
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
```

Treat any existing listener or GPU compute process as an ownership collision; do not stop it
unless it belongs to this run.

This family measures request-visible E2E latency plus strict request failure, not lifecycle
phase latency or semantic-quality equivalence between different engines. The frozen manifest
fixes prompt and decoding semantics; retain its SHA-256 with each run.

#### Proposed: vLLM Switch controller

The controller owns routing and launches two single-model backends sequentially. The
compatible vLLM Switch environment must expose `/sleep`, `/wake_up`, and `/is_sleeping` and
must include its virtual environment in `PATH` so JIT helpers such as `ninja` are found.

Create the run-local launcher config:

```bash
mkdir -p "$RUN_ROOT/proposed"
cat > "$RUN_ROOT/proposed/config.yaml" <<EOF
models:
  qwen-1.5b:
    backend_url: http://127.0.0.1:8101
    served_model_name: qwen-1.5b
    sleep_level: 1
    wake_tags: null
    cwd: "$VLLM_REPO"
    env:
      VLLM_SERVER_DEV_MODE: "1"
      PYTHONPATH: "$VLLM_REPO"
      PATH: "$VLLM_REPO/.venv/bin:/usr/local/bin:/usr/bin:/bin"
    launch_command:
      - "$VLLM_REPO/.venv/bin/python"
      - -m
      - vllm.entrypoints.openai.api_server
      - --model
      - "$MODEL_ROOT/Qwen2.5-1.5B-Instruct"
      - --host
      - 127.0.0.1
      - --port
      - "8101"
      - --served-model-name
      - qwen-1.5b
      - --enable-sleep-mode
      - --enable-request-id-headers
      - --gpu-memory-utilization
      - "0.70"
      - --max-model-len
      - "1024"
      - --enforce-eager
      - --dtype
      - half
  qwen-3b:
    backend_url: http://127.0.0.1:8102
    served_model_name: qwen-3b
    sleep_level: 1
    wake_tags: null
    cwd: "$VLLM_REPO"
    env:
      VLLM_SERVER_DEV_MODE: "1"
      PYTHONPATH: "$VLLM_REPO"
      PATH: "$VLLM_REPO/.venv/bin:/usr/local/bin:/usr/bin:/bin"
    launch_command:
      - "$VLLM_REPO/.venv/bin/python"
      - -m
      - vllm.entrypoints.openai.api_server
      - --model
      - "$MODEL_ROOT/Qwen2.5-3B-Instruct"
      - --host
      - 127.0.0.1
      - --port
      - "8102"
      - --served-model-name
      - qwen-3b
      - --enable-sleep-mode
      - --enable-request-id-headers
      - --gpu-memory-utilization
      - "0.70"
      - --max-model-len
      - "1024"
      - --enforce-eager
      - --dtype
      - half
controller:
  host: 127.0.0.1
  port: 9000
  policy: always_sleep_previous
  startup_awake_model: qwen-1.5b
  request_timeout_s: 600
  switch_timeout_s: 600
  metrics_path: "$RUN_ROOT/proposed/controller-events.jsonl"
EOF

cd "$CONTROLLER_REPO"
uv sync --frozen --dev
```

This request-routing run intentionally leaves the optional CPU-backup coordinator disabled;
that mechanism is evaluated separately by `backup-reuse-reclaim`. Terminal 1 starts the
controller:

```bash
cd "$CONTROLLER_REPO"
uv run vllm-switch-controller --config "$RUN_ROOT/proposed/config.yaml" \
  2>&1 | tee "$RUN_ROOT/proposed/controller.log"
```

Terminal 2 prepares the launcher-owned pool. The command returns only after both backends
have become healthy, each has been slept and verified in order, and `qwen-1.5b` has been
woken. Keep the PID file: it is the bounded cleanup authority.

```bash
cd "$CONTROLLER_REPO"
rm -f "$RUN_ROOT/proposed/pids.json"
uv run vllm-switch-launch \
  --config "$RUN_ROOT/proposed/config.yaml" \
  --pid-file "$RUN_ROOT/proposed/pids.json"

curl --noproxy '*' -fsS http://127.0.0.1:9000/health
curl --noproxy '*' -fsS http://127.0.0.1:9000/admin/state
curl --noproxy '*' -fsS http://127.0.0.1:9000/v1/models
curl --noproxy '*' -fsS http://127.0.0.1:8101/is_sleeping
curl --noproxy '*' -fsS http://127.0.0.1:8102/is_sleeping
```

Terminal 3 replays the exact retained alternating trace:

```bash
cd "$BENCH_REPO"
uv sync --frozen --group dev
scripts/request-driven-switch.sh \
  --base-url http://127.0.0.1:9000 \
  --manifest configs/traces/request-switch-alternating.jsonl \
  --timeout-s 600 \
  --output "$RUN_ROOT/proposed/alternating.jsonl"
```

Stop only launcher-owned process groups, then stop terminal 1 with `Ctrl-C`:

```bash
cd "$CONTROLLER_REPO"
uv run vllm-switch-stop --pid-file "$RUN_ROOT/proposed/pids.json"
```

#### llama-swap

Build the maintained checkout and create a run-local two-model config. The child commands
must use the intended upstream vLLM environment, not the benchmark virtual environment:

```bash
cd "$LLAMA_SWAP_REPO"
export GO=$(command -v go)
test -n "$GO"
"$GO" test ./internal/process ./internal/router ./internal/server
"$GO" build -o build/llama-swap .

cd "$BENCH_REPO"
mkdir -p "$RUN_ROOT/llama-swap"
export LLAMA_CONFIG="$RUN_ROOT/llama-swap/config.yaml"
```

Create the config file with absolute paths.

```bash
uv run python - <<'PY'
import os
from pathlib import Path

vllm = Path(os.environ["VLLM_UPSTREAM_REPO"])
models = Path(os.environ["MODEL_ROOT"])
for path in (vllm, models):
    if any(character.isspace() for character in str(path)):
        raise SystemExit(f"llama-swap command paths must not contain whitespace: {path}")
text = f"""healthCheckTimeout: 360
logLevel: info
logToStdout: both
startPort: 18200
sendLoadingState: false
unloadTimeout: 30
models:
  qwen-1.5b:
    cmd: >-
      {vllm}/.venv/bin/python
      -m vllm.entrypoints.openai.api_server
      --model {models}/Qwen2.5-1.5B-Instruct
      --served-model-name qwen-1.5b
      --host 127.0.0.1 --port ${{PORT}}
      --max-model-len 1024 --gpu-memory-utilization 0.70
      --dtype half --enforce-eager
    proxy: http://127.0.0.1:${{PORT}}
    checkEndpoint: /health
    env:
      - PATH={vllm}/.venv/bin:/usr/local/bin:/usr/bin:/bin
  qwen-3b:
    cmd: >-
      {vllm}/.venv/bin/python
      -m vllm.entrypoints.openai.api_server
      --model {models}/Qwen2.5-3B-Instruct
      --served-model-name qwen-3b
      --host 127.0.0.1 --port ${{PORT}}
      --max-model-len 1024 --gpu-memory-utilization 0.70
      --dtype half --enforce-eager
    proxy: http://127.0.0.1:${{PORT}}
    checkEndpoint: /health
    env:
      - PATH={vllm}/.venv/bin:/usr/local/bin:/usr/bin:/bin
"""
Path(os.environ["LLAMA_CONFIG"]).write_text(text, encoding="utf-8")
PY
```

Then start one owned process group:

```bash
export LLAMA_PID_FILE="$RUN_ROOT/llama-swap/router.pid"
setsid bash -c '
  printf "%s\n" "$$" > "$LLAMA_PID_FILE"
  exec "$LLAMA_SWAP_REPO/build/llama-swap" \
    --config "$RUN_ROOT/llama-swap/config.yaml" \
    --listen 127.0.0.1:18100
' > "$RUN_ROOT/llama-swap/router.log" 2>&1 &

ready=0
for _ in $(seq 1 60); do
  test -s "$LLAMA_PID_FILE"
  LLAMA_PGID=$(cat "$LLAMA_PID_FILE")
  kill -0 "$LLAMA_PGID"
  if curl --noproxy '*' -fsS http://127.0.0.1:18100/health; then
    ready=1
    break
  fi
  sleep 1
done
test "$ready" -eq 1
test "$(ps -o pgid= -p "$LLAMA_PGID" | tr -d ' ')" = "$LLAMA_PGID"
curl --noproxy '*' -fsS http://127.0.0.1:18100/running
```

Replay the same trace:

```bash
cd "$BENCH_REPO"
scripts/request-driven-switch.sh \
  --base-url http://127.0.0.1:18100 \
  --manifest configs/traces/request-switch-alternating.jsonl \
  --timeout-s 600 \
  --output "$RUN_ROOT/llama-swap/alternating.jsonl"
```

Stop only the llama-swap process group captured above:

```bash
LLAMA_PGID=$(cat "$RUN_ROOT/llama-swap/router.pid")
kill -TERM -- "-$LLAMA_PGID"
for _ in $(seq 1 300); do
  kill -0 -- "-$LLAMA_PGID" 2>/dev/null || break
  sleep 0.1
done
kill -KILL -- "-$LLAMA_PGID" 2>/dev/null || true
```

#### Inspect success and optional repeated workloads

The single-trace command exits nonzero if any request lacks HTTP 2xx, semantic output, or a
complete SSE marker. Independently inspect each output:

```bash
cd "$BENCH_REPO"
uv run python - "$RUN_ROOT/proposed/alternating.jsonl" \
  "$RUN_ROOT/llama-swap/alternating.jsonl" <<'PY'
import json
import sys
from pathlib import Path

for value in sys.argv[1:]:
    path = Path(value)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    failed = [
        row for row in rows
        if not 200 <= int(row.get("status") or 0) < 300
        or row.get("error")
        or not row.get("stream_done")
        or row.get("semantic_ttft_ms") is None
        or not str(row.get("output_text") or "").strip()
    ]
    print(path, {"requests": len(rows), "failed": len(failed)})
    assert len(rows) == 20 and not failed
PY
```

To run the steady, alternating, and burst traces three times against **one already-running
endpoint**, use a fresh output directory. This helper is an endpoint-local matrix; it does
not launch services or make two independently managed systems safe to run concurrently on
one GPU.

```bash
export ROUTER_PORT=9000
export SYSTEM=proposed
test ! -e "$RUN_ROOT/${SYSTEM}-three-trace-r3"
scripts/request-driven-switch-matrix.sh \
  --base-url "http://127.0.0.1:$ROUTER_PORT" \
  --repeats 3 \
  --out-dir "$RUN_ROOT/${SYSTEM}-three-trace-r3"
```

#### Cleanup verification

Verify router/backend ports, owned PID/PGID records, and GPU compute processes are gone.
Do not use an unscoped `pkill` or stop unrelated processes on a shared host.
