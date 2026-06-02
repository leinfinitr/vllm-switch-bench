# ServerlessLLM local reproduction notes

This document records the verified path used for the baseline3 ServerlessLLM comparison on this machine.

## What ServerlessLLM measures here

The maintained baseline3 row uses `scale_to_zero_restore`:

1. Register Qwen2.5-0.5B with ServerlessLLM.
2. Send an inference request.
3. Wait until GPU memory returns to the idle threshold, indicating scale-to-zero.
4. Send another request and measure the restore path.

`delete_register` is not used as a valid baseline row because, in the current ServerlessLLM checkout, controller delete removes metadata but does not reliably stop the runtime/router. The benchmark records this as an unsupported row rather than fabricating performance numbers.

## Verified runtime assumptions

- Docker works.
- NVIDIA Docker works.
- ServerlessLLM is at `/home/ljl/research-systems/ServerlessLLM`.
- Raw HF checkpoint is at `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`.
- ServerlessLLM Docker Compose exposes the controller on `127.0.0.1:8343`.
- The Docker Compose worker uses `/models` as `STORAGE_PATH` and mounts `${MODEL_FOLDER}:/models`.

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
mkdir -p "$MODEL_FOLDER"
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

docker compose up -d
curl http://127.0.0.1:8343/health
```

Expected health response:

```json
{"status":"ok"}
```

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
  --repeats 3 \
  --methods scale_to_zero_restore \
  --out-dir results/baselines/serverless_llm/qwen2p5_0p5b
```

## Current result source

ServerlessLLM rows in the curated baseline3 result come from:

`results/baselines/baseline3/qwen2p5_0p5b/20260602_161100`

## Known pitfalls

- Do not use local Python startup as the default path unless all ServerlessLLM Python dependencies and store components are installed; the verified path here is Docker Compose.
- Do not point the writable ServerlessLLM model store at the raw HF checkpoint directory.
- Do not interpret `delete_register` as reliable scale-down; it is kept as an unsupported row in reports.
