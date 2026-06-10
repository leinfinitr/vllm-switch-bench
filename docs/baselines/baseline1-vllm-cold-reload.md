# Baseline1：vLLM 冷启动

Baseline1 衡量最直接的按需服务方式：停止旧服务，重新启动 vLLM，加载模型，完成 warmup 后处理请求。它显存占用最低，但切换延迟最高。

## 运行方式

```bash
cd /home/ljl/research-systems/llm-switch-bench
PATH=$PWD/.venv/bin:/home/ljl/cuda-13.0/bin:$PATH CUDA_HOME=/home/ljl/cuda-13.0 .venv/bin/python src/bench_vllm_lifecycle.py   --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct   --python .venv/bin/python   --workdir /home/ljl/research-systems/llm-switch-bench   --methods cold_reload   --prompts short_short long_short short_long   --repeats 3   --ready-timeout-s 360   --gpu-memory-utilization 0.45   --max-model-len 1024   --port 0   --out-dir results/baselines/vllm/qwen2p5_0p5b
```

## 关注指标

- `startup_latency_s`：vLLM 服务启动到可访问模型的时间。
- `evict_latency_s`：停止旧服务的时间。
- `restore_latency_s`：重新启动并可服务的时间。
- `ttft_before_s` / `ttft_after_s`：切换前后的首 token 延迟。
- `latency_before_s` / `latency_after_s`：切换前后的请求端到端延迟。

## 解释

冷启动是所有方案的上界参考。Qwen2.5-0.5B 上 cold reload 约 15 秒，远慢于 vLLM sleep/wake，但它能释放几乎全部模型显存。
