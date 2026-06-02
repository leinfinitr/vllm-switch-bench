# Baseline 3: Engine-agnostic process checkpoint / hotswap

Baseline 3 compares systems that treat the inference engine process as a swappable unit rather than relying only on vLLM-internal memory policy.

Conceptual definition:

- The inference engine process can be checkpointed and restored.
- Swap out / swap in happens at the system layer.
- The mechanism can be engine-agnostic: vLLM, Ollama, SGLang, TensorRT-LLM, etc.
- This is useful as a related-system comparison, but it is not the same as vLLM's internal fine-grained Sleep Mode policy.

Current comparison systems:

- vLLM: imported from the clean-HBM Baseline1/2 source run.
- ServerlessLLM: real Docker Compose runtime, measured with `scale_to_zero_restore`.
- SwapServeLLM: rootless podman + CUDA checkpoint runtime, measured with `swapout_swapin`.

## Current implementation

Driver:

`src/bench_baseline3.py`

Adapters:

- `src/bench_serverless_llm.py`
- `src/bench_swapserve_llm.py`

Config:

- `configs/baseline3.local.yaml`
- `configs/baseline3.local.example.yaml`

## Environment prerequisites

Common:

```bash
cd /home/ljl/research-systems/llm-switch-bench
. .venv/bin/activate
python -m pytest tests -q
```

External repositories:

```text
/home/ljl/research-systems/ServerlessLLM
/home/ljl/research-systems/SwapServeLLM
/home/ljl/models/hf/Qwen2.5-0.5B-Instruct
```

ServerlessLLM requires Docker + NVIDIA Docker.

SwapServeLLM requires:

- podman installed
- user podman socket running
- `docker.io/vllm/vllm-openai:latest` pulled into podman storage
- `cuda-checkpoint` available in `PATH`
- SwapServeLLM compatibility patch branch/commit that supports local rootless runtime

See:

- `docs/systems/serverlessllm.md`
- `docs/systems/swapservellm.md`

## Start ServerlessLLM runtime

The verified local path uses the Docker Compose file in ServerlessLLM and a writable model store.

```bash
cd /home/ljl/research-systems/ServerlessLLM/examples/docker
export MODEL_FOLDER=/home/ljl/research-systems/llm-switch-bench/runtime/serverlessllm-models
mkdir -p "$MODEL_FOLDER"
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

docker compose up -d
curl http://127.0.0.1:8343/health
```

Important: ServerlessLLM writes its own `vllm/<model>/rank_*` state under `MODEL_FOLDER`. Keep this store separate from the raw Hugging Face checkpoint.

## Start SwapServeLLM runtime

Use the patched SwapServeLLM branch already pushed to the GitLab research remote:

```bash
cd /home/ljl/research-systems/SwapServeLLM
git checkout fix/local-rootless-swapserve
```

Prepare runtime dependencies:

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

Example local config path:

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
  "backend_configs": [
    {
      "backend_name": "vllm-qwen2p5-0p5b",
      "model_name": "/home/ljl/models/hf/Qwen2.5-0.5B-Instruct",
      "container_image": "docker.io/vllm/vllm-openai:latest",
      "initialization_timeout": "5m",
      "container_port": "8001",
      "gpu_memory_utilization": 0.45
    }
  ]
}
JSON
```

Build and launch:

```bash
cd /home/ljl/research-systems/SwapServeLLM
/home/ljl/program/go/bin/go build \
  -tags 'containers_image_openpgp exclude_graphdriver_btrfs' \
  -o /tmp/swapserve-exp/swapservellm .

cd /tmp/swapserve-exp
PATH=/tmp/cuda-checkpoint-exp:$PATH \
SWAPSERVE_CONFIG_PATH=/tmp/swapserve-exp/config.json \
SWAPSERVE_HF_CACHE=/home/ljl/models/hf \
SWAPSERVE_SKIP_PULL=1 \
OPENAI_API_KEY=dummy \
http_proxy=http://127.0.0.1:7890 \
https_proxy=http://127.0.0.1:7890 \
/tmp/swapserve-exp/swapservellm
```

Verify:

```bash
curl http://127.0.0.1:8000/v1/models
```

## Run baseline3 comparison

After both external runtimes are up:

```bash
cd /home/ljl/research-systems/llm-switch-bench
. .venv/bin/activate

python src/bench_baseline3.py \
  --config configs/baseline3.local.yaml \
  --systems vllm serverless_llm swapserve_llm \
  --prompts short_short long_short short_long \
  --repeats 3 \
  --out-dir results/baselines/baseline3/qwen2p5_0p5b
```

If only SwapServeLLM needs to be rerun:

```bash
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

Analyze a baseline3 run:

```bash
python src/analyze_baseline3.py \
  results/baselines/baseline3/qwen2p5_0p5b/<timestamp> \
  --out docs/reports/baseline3-qwen2p5-0p5b.md
```

## Curated result

Current baseline3 result:

`results/baselines/baseline3/qwen2p5_0p5b/20260602_161100`

Report:

`docs/reports/baseline3-qwen2p5-0p5b.md`

## Known interpretation caveats

- `serverless_llm/delete_register` is intentionally marked unsupported because controller delete removes metadata but does not reliably stop the router/runtime.
- ServerlessLLM `scale_to_zero_restore` measures a real external runtime path; the first conversion/register cost is distinct from raw-HF vLLM cold reload.
- SwapServeLLM `swapout_swapin` uses CUDA checkpoint/restore and podman-managed containers; it is a system-level hotswap baseline, not a vLLM Sleep Mode policy.
- SwapServeLLM detailed stage timings are emitted on router stdout; `swapout.log` and `swapin.log` contain summary rows only.
