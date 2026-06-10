# SwapServeLLM 本机运行说明

SwapServeLLM 在 Baseline3 中提供容器级 swapout/swapin 参考。它不是 vLLM Sleep Mode，而是通过 router、podman 和 CUDA checkpoint 管理 vLLM backend。

## 依赖

- `/home/ljl/research-systems/SwapServeLLM`，分支 `fix/local-rootless-swapserve`。
- rootless podman 和用户 podman socket。
- `docker.io/vllm/vllm-openai:latest` 已在 podman storage 中。
- `cuda-checkpoint` 在 `PATH` 中。

```bash
systemctl --user start podman.socket
```

## 构建

```bash
cd /home/ljl/research-systems/SwapServeLLM
git checkout fix/local-rootless-swapserve
/home/ljl/program/go/bin/go build   -tags 'containers_image_openpgp exclude_graphdriver_btrfs'   -o /tmp/swapserve-exp/swapservellm .
```

## 启动

使用 `/tmp/swapserve-exp/config.json` 配置模型、日志和 backend 端口后启动：

```bash
cd /tmp/swapserve-exp
PATH=/tmp/cuda-checkpoint-exp:$PATH SWAPSERVE_CONFIG_PATH=/tmp/swapserve-exp/config.json SWAPSERVE_HF_CACHE=/home/ljl/models/hf SWAPSERVE_SKIP_PULL=1 OPENAI_API_KEY=dummy http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 /tmp/swapserve-exp/swapservellm
```

验证：

```bash
curl http://127.0.0.1:8000/v1/models
```

## 单独运行 adapter

```bash
cd /home/ljl/research-systems/llm-switch-bench
.venv/bin/python src/bench_swapserve_llm.py   --repo /home/ljl/research-systems/SwapServeLLM   --base-url http://127.0.0.1:8000   --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct   --api-key dummy   --log-dir /tmp/swapserve-exp/logs   --prompts short_short long_short short_long   --repeats 3   --out-dir results/baselines/swapserve_llm/qwen2p5_0p5b
```

## 注意事项

- router 默认端口是 `8000`，backend 示例端口是 `8001`。
- `swapout.log` / `swapin.log` 只含摘要，详细阶段耗时在 router stdout。
- 当前 patch 包含本机 rootless podman 兼容处理，换机器需要重新验证。
