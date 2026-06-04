# ServerlessLLM local reproduction notes

This document records the verified path used for the baseline3 ServerlessLLM comparison on this machine.

## What ServerlessLLM measures here

The maintained baseline3 rows now include both ServerlessLLM methods:

- `delete_register`: register the model, send a prompt-matched warm request to start the backend, measure ready latency with a second request, delete the model/router and wait for idle GPU memory, register again, then estimate restore latency from a restore warm request minus a second active request.
- `scale_to_zero_restore`: register the model, send a prompt-matched warm request to start the backend, measure ready latency with a second request, wait until GPU memory returns to the idle threshold, then estimate restore latency from a restore warm request minus a second active request.

`scale_to_zero_restore` is the closer serverless-serving baseline. `delete_register` is kept as a deterministic lifecycle comparison now that the local ServerlessLLM checkout has a router cleanup fix.

## Verified runtime assumptions

- Docker works.
- NVIDIA Docker works.
- ServerlessLLM is at `/home/ljl/research-systems/ServerlessLLM`.
- Raw HF checkpoint is at `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`.
- ServerlessLLM Docker Compose exposes the controller on `127.0.0.1:8343`.
- The Docker Compose worker uses `/models` as `STORAGE_PATH` and mounts `${MODEL_FOLDER}:/models`.
- The worker also mounts `${HOST_MODEL_FOLDER:-/home/ljl/models}:/host-models`, which is the path used by the benchmark payload.

## Important architecture detail

`ServerlessLLM/sllm/store_manager.py` starts the store manager, and `StoreManager.register()` calls `download_vllm_model()` for backend `vllm`.

For local testing this means:

- Passing `backend_config.pretrained_model_name_or_path=/home/ljl/models/hf/Qwen2.5-0.5B-Instruct` or `/host-models/...` selects the source checkpoint.
- It does not skip ServerlessLLM's conversion/storage path.
- ServerlessLLM writes its own `vllm/<model>/rank_*` format under the model store.

Keep the writable model store separate from the raw Hugging Face checkpoint directory.

## Start ServerlessLLM

```bash
cd /home/ljl/research-systems/ServerlessLLM/examples/docker
export MODEL_FOLDER=/home/ljl/research-systems/llm-switch-bench/runtime/serverlessllm-models
export HOST_MODEL_FOLDER=/home/ljl/models
mkdir -p "$MODEL_FOLDER"
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost

docker compose up -d
curl --noproxy '*' http://127.0.0.1:8343/health
```

Expected health response:

```json
{"status":"ok"}
```

If the Docker image cannot be rebuilt because external downloads time out, the last validated run used `docker compose create`, copied patched ServerlessLLM Python files into the created containers, and then ran `docker compose start`. This is a validation workaround, not a replacement for rebuilding the image when network access is available.

## Run the adapter directly

```bash
cd /home/ljl/research-systems/llm-switch-bench
. .venv/bin/activate

python src/bench_serverless_llm.py \
  --repo /home/ljl/research-systems/ServerlessLLM \
  --model /host-models/hf/Qwen2.5-0.5B-Instruct \
  --registered-model-name qwen2p5-0p5b \
  --base-url http://127.0.0.1:8343 \
  --prompts short_short long_short short_long \
  --repeats 1 \
  --max-model-len 2048 \
  --methods delete_register scale_to_zero_restore \
  --scale-zero-poll-interval 0.001 \
  --out-dir results/baselines/serverless_llm/qwen2p5_0p5b
```

Use `--max-model-len 2048` for the current prompt set; `long_short` exceeds 512 tokens after chat templating.

## Current result source

ServerlessLLM rows in the curated baseline3 report come from:

`results/baselines/serverless_llm/qwen2p5_0p5b/20260604_164857`

The merged baseline3 result that includes these rows is:

`results/baselines/baseline3/qwen2p5_0p5b/20260604_164857`

## Known pitfalls

- Do not use local Python startup as the default path unless all ServerlessLLM Python dependencies and store components are installed; the verified path here is Docker Compose.
- Do not point the writable ServerlessLLM model store at the raw HF checkpoint directory.
- Do not use controller `/health` alone as model readiness; the benchmark uses a prompt-matched warm request and samples ready memory after that request completes.
- ServerlessLLM startup latency is intentionally left empty because the external Docker runtime is assumed already running.
- ServerlessLLM currently does not expose external streaming TTFT for these rows, so `ttft_available=false` and `tpot_available=false` are expected.
