# ServerlessLLM 本机运行说明

ServerlessLLM 在 Baseline3 中提供 serverless serving 参考。当前保留两个方法：

- `delete_register`：删除模型注册并等待显存回落，再重新注册和恢复。
- `scale_to_zero_restore`：等待 worker scale-to-zero 后，用恢复请求估算 restore latency。

## 启动运行时

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

`MODEL_FOLDER` 是 ServerlessLLM 自己的可写模型存储，不要指向原始 Hugging Face checkpoint 目录。

## 单独运行 adapter

```bash
cd /home/ljl/research-systems/llm-switch-bench
.venv/bin/python src/bench_serverless_llm.py   --repo /home/ljl/research-systems/ServerlessLLM   --model /host-models/hf/Qwen2.5-0.5B-Instruct   --registered-model-name qwen2p5-0p5b   --base-url http://127.0.0.1:8343   --prompts short_short long_short short_long   --repeats 1   --max-model-len 2048   --methods delete_register scale_to_zero_restore   --out-dir results/baselines/serverless_llm/qwen2p5_0p5b
```

## 注意事项

- `long_short` 经过 chat template 后超过 512 tokens，当前 prompt set 需要 `--max-model-len 2048`。
- `/health` 只能说明 controller 存活，不能说明模型 ready；benchmark 会发送 warm request。
- 这些结果没有外部 streaming TTFT，因此 `ttft_available=false` 是预期现象。
