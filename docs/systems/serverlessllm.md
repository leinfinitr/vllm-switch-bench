# ServerlessLLM 本机运行说明

ServerlessLLM 在 Baseline3 中提供 serverless serving 参考。当前 adapter 保留
两个历史方法：

- `delete_register`：删除模型注册并等待显存回落，再重新注册和恢复。
- `scale_to_zero_restore`：等待 worker scale-to-zero 后，用恢复请求估算 restore latency。

> **当前状态：功能 smoke 可运行，正式 lifecycle 被阻塞。** 现有
> `src/bench_serverless_llm.py` 固定 `load_format: auto`，没有测量
> ServerlessLLM fast loader；它还要求自动 scale-to-zero 后模型从
> `/v1/models` 消失，而正常 scale-to-zero 会保留 registration。`delete_register`
> 路径的 idle wait 已经包含在外层计时中，现有实现又重复相加。因此下方
> adapter 命令仅用于复现历史 Baseline3/API 调试，结果不能进入当前论文图。

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

## 历史 adapter（仅调试）

```bash
cd /home/ljl/research-systems/llm-switch-bench
.venv/bin/python src/bench_serverless_llm.py   --repo /home/ljl/research-systems/ServerlessLLM   --model /host-models/hf/Qwen2.5-0.5B-Instruct   --registered-model-name qwen2p5-0p5b   --base-url http://127.0.0.1:8343   --prompts short_short long_short short_long   --repeats 1   --max-model-len 2048   --methods delete_register scale_to_zero_restore   --out-dir results/baselines/serverless_llm/qwen2p5_0p5b
```

## 注意事项

- `long_short` 经过 chat template 后超过 512 tokens，当前 prompt set 需要 `--max-model-len 2048`。
- `/health` 只能说明 controller 存活，不能说明模型 ready；benchmark 会发送 warm request。
- 这些结果没有外部 streaming TTFT，因此 `ttft_available=false` 是预期现象。

## 正式测试前必须修复

1. 注册 payload 省略 `load_format: auto`，让 `/models/vllm/<name>` 的转换后
   checkpoint 走 `serverless_llm` loader。
2. 自动 sleep 的完成条件是 backend actor/process 消失、scheduler GPU
   reservation 归还、GPU 显存回到校准后的 idle 阈值；模型应保持 registered。
3. wake 边界需明确为 request-visible restore，或新增 backend-ready 观测；不能
   用“第一次请求减第二次请求”冒充直接测量值而不标注估算。
4. `delete_register` 不重复累计 idle wait，并在异常路径执行 actor/registration
   cleanup。
5. 每个模型至少保留 5 个 fresh-run 样本、恢复后正确推理和物理释放证据。

2026-07-23 按本文档实际运行的 0.5B smoke 能启动、注册和推理，但
`delete_register` 返回 200 后，artifact 保留的旧镜像 gate 中
`ray::VllmBackend` 仍持有 5810 MiB GPU memory；
这只证明旧镜像的 delete gate 未完成，不代表 current-source 自动
scale-to-zero 的结果。当前状态记录在
`results/osdi_20260723/raw/serverless/status.json`。
