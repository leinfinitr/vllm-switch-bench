# SwapServeLLM local reproduction notes

This document records the verified local path used for the baseline3 SwapServeLLM comparison on this machine.

## What SwapServeLLM measures here

The maintained row uses `swapout_swapin`:

1. Send an inference request through the SwapServeLLM router.
2. Call `/api/swapout` for the model.
3. SwapServeLLM unloads vLLM model state, CUDA-checkpoints GPU worker threads, and pauses the podman container.
4. Call `/api/swapin` for the same model.
5. SwapServeLLM unpauses the container, restores CUDA state, waits for the backend, reloads the model, and serves another request.

This is a system-level hotswap baseline. It is not vLLM Sleep Mode.

## Verified runtime assumptions

- Podman is installed on the host.
- User podman socket can be started with `systemctl --user start podman.socket`.
- Rootless podman + NVIDIA CDI works only with the local compatibility patch that uses numeric supplemental groups on this machine.
- `docker.io/vllm/vllm-openai:latest` is pulled into podman storage.
- `cuda-checkpoint` is available in `PATH`; a user-local binary under `/tmp/cuda-checkpoint-exp` is sufficient.
- SwapServeLLM patched branch: `/home/ljl/research-systems/SwapServeLLM`, branch `fix/local-rootless-swapserve`, pushed to GitLab research remote.

## Why the patch is required

The unmodified upstream path had these blockers on this machine:

- Config path was fixed to `/etc/SwapServeLLM/config.json`.
- vLLM model/cache mount was fixed to `/root/.cache/huggingface`.
- `systemctl start podman.socket` assumed a system service path.
- Rootless podman did not provide a usable slirp4netns container IP for the backend.
- Current `vllm/vllm-openai:latest` expects `vllm serve ...` style CLI.
- Warmup sent an invalid chat-completions payload using `prompt` instead of `messages`.
- `sudo cuda-checkpoint --get-state` and `sudo podman unpause` did not work in this user session.

## Prepare dependencies

```bash
systemctl --user start podman.socket

http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 \
podman pull docker.io/vllm/vllm-openai:latest

mkdir -p /tmp/cuda-checkpoint-exp
http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 \
curl -L --fail \
  https://github.com/NVIDIA/cuda-checkpoint/raw/refs/heads/main/bin/x86_64_Linux/cuda-checkpoint \
  -o /tmp/cuda-checkpoint-exp/cuda-checkpoint
chmod +x /tmp/cuda-checkpoint-exp/cuda-checkpoint
```

Verify CUDA checkpoint can see a vLLM PID only after the backend is running. Earlier validation showed `--get-state`, `--toggle`, and restore worked without sudo.

## Build SwapServeLLM

```bash
cd /home/ljl/research-systems/SwapServeLLM
git checkout fix/local-rootless-swapserve

/home/ljl/program/go/bin/go test \
  -tags 'containers_image_openpgp exclude_graphdriver_btrfs' ./...

/home/ljl/program/go/bin/go build \
  -tags 'containers_image_openpgp exclude_graphdriver_btrfs' \
  -o /tmp/swapserve-exp/swapservellm .
```

The tags avoid missing system development headers for gpgme/btrfs on this shared host.

## Create local config

```bash
mkdir -p /tmp/swapserve-exp/logs
cat > /tmp/swapserve-exp/config.json <<'JSON'
{
  "openai_api_key": "dummy",
  "swap_in_logfile": "/tmp/swapserve-exp/logs/swapin.log",
  "swap_out_logfile": "/tmp/swapserve-exp/logs/swapout.log",
  "cold_start_logfile": "/tmp/swapserve-exp/logs/coldstart.log",
  "model_latency_logfile": "/tmp/swapserve-exp/logs/model_latency.log",
  "service_ready_timeout": 180,
  "backend_response_timeout": 300,
  "max_waiting_requests": 16,
  "model_list": [
    {
      "backend_name": "vllm-qwen2p5-0p5b",
      "model_name": "/home/ljl/models/hf/Qwen2.5-0.5B-Instruct",
      "container_image": "docker.io/vllm/vllm-openai:latest",
      "initialization_timeout": "5m",
      "container_port": "8001",
      "gpu_memory_utilization": "0.45"
    }
  ]
}
JSON
```

## Launch router and backend

```bash
cd /tmp/swapserve-exp
PATH=/tmp/cuda-checkpoint-exp:$PATH \
SWAPSERVE_CONFIG_PATH=/tmp/swapserve-exp/config.json \
SWAPSERVE_HF_CACHE=/home/ljl/models/hf \
SWAPSERVE_SKIP_PULL=1 \
OPENAI_API_KEY=*** \
http_proxy=http://127.0.0.1:7890 \
https_proxy=http://127.0.0.1:7890 \
/tmp/swapserve-exp/swapservellm
```

Expected services:

- Router: `http://127.0.0.1:8000`
- vLLM backend: `http://127.0.0.1:8001`

Verify router:

```bash
curl http://127.0.0.1:8000/v1/models
```

## Run the adapter

```bash
cd /home/ljl/research-systems/llm-switch-bench
. .venv/bin/activate

python src/bench_swapserve_llm.py \
  --repo /home/ljl/research-systems/SwapServeLLM \
  --base-url http://127.0.0.1:8000 \
  --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct \
  --api-key dummy \
  --log-dir /tmp/swapserve-exp/logs \
  --prompts short_short long_short short_long \
  --repeats 3 \
  --out-dir results/baselines/swapserve_llm/qwen2p5_0p5b
```

## Current result source

Standalone SwapServeLLM result:

`results/baselines/swapserve_llm/qwen2p5_0p5b/20260605_145529`

Merged baseline3 result:

`results/baselines/baseline3/qwen2p5_0p5b/20260605_145529`

## Known pitfalls

- `swapout.log` and `swapin.log` contain summary rows only. Detailed stage timings are printed to router stdout.
- The OpenAI-compatible stream currently does not expose an explicit TTFT/FTTF metric. The benchmark leaves `ttft_*` and derived `tpot_*` fields blank for SwapServeLLM instead of using first client chunk latency as a semantic TTFT substitute.
- The router forwards client headers to the backend; benchmark requests need `Authorization: Bearer ***` when the backend was started with `--api-key dummy`.
- Host ports 8000 and 8001 must be free before launching.
- The current compatibility patch is machine-specific in its numeric groups (`44`, `992`). Revalidate on other hosts.
