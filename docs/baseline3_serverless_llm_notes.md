# ServerlessLLM baseline notes

Current verified facts on this machine:

- Docker + docker compose work.
- NVIDIA Docker works (`docker run --rm --gpus all nvidia/cuda... nvidia-smi`).
- `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct` exists with HF safetensors.
- `llm-switch-bench/.venv` does not currently have `ray`.
- ServerlessLLM Python-local start path also needs `serverless-llm-store`, GPUtil, and speedtest-cli deps.

Important architecture detail:

- `ServerlessLLM/sllm/store_manager.py` always imports and starts `StoreManager`.
- `StoreManager.register()` always calls `download_vllm_model()` for backend `vllm`.
- `download_vllm_model()` converts the input HF checkpoint into ServerlessLLM's `vllm/<model>/rank_*` storage format using `save_serverless_llm_state`.
- Therefore, passing `backend_config.pretrained_model_name_or_path=/home/ljl/models/hf/Qwen2.5-0.5B-Instruct` does NOT skip conversion; it only changes the source checkpoint.

Implication:

- A real ServerlessLLM baseline requires a working ServerlessLLM runtime, not just the HTTP API stubs.
- The cleanest execution path is Docker Compose from `ServerlessLLM/examples/docker/docker-compose.yml` with `MODEL_FOLDER` pointing at a writable model-store directory.
- That model store should be distinct from the raw HF checkpoint source, because ServerlessLLM writes its own `vllm/` format there.

Recommended next execution path:

1. Create a writable experiment store, e.g. `/home/ljl/research-systems/llm-switch-bench/runtime/serverlessllm-models`.
2. Start SLLM via Docker Compose with `MODEL_FOLDER` set to that store.
3. Use `bench_serverless_llm.py` against `http://127.0.0.1:8343`, with:
   - source model path: `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`
   - registered model name alias: `qwen2p5-0p5b`
4. Expect first registration to include conversion cost into ServerlessLLM format.

Caveat for fairness:

- This baseline measures ServerlessLLM's real checkpoint conversion/loading path, not a pure vLLM raw-HF reload. That is acceptable for Baseline3 as long as the report states the different state transition clearly.
