# Baseline2：vLLM Sleep Mode

Baseline2 关注 vLLM 内部 sleep/wake 能力。当前仓库实现的是单模型 lifecycle 近似：启动一个 vLLM 服务，完成一次请求后调用 sleep，再 wake 并再次请求。它用于衡量 vLLM 内部显存释放和恢复成本，不包含多模型调度器。

## sleep level

- `sleep_l1`：权重备份到 CPU，KV cache 丢弃；wake 时把权重拷回 GPU。
- `sleep_l2`：权重和 KV cache 都从 GPU 释放；wake 时重新加载权重。

## 运行方式

```bash
cd /home/ljl/research-systems/llm-switch-bench
PATH=$PWD/.venv/bin:/home/ljl/cuda-13.0/bin:$PATH CUDA_HOME=/home/ljl/cuda-13.0 .venv/bin/python src/bench_vllm_lifecycle.py   --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct   --python .venv/bin/python   --workdir /home/ljl/research-systems/llm-switch-bench   --methods sleep_l1 sleep_l2   --prompts short_short long_short short_long   --repeats 3   --ready-timeout-s 360   --gpu-memory-utilization 0.45   --max-model-len 1024   --port 0   --out-dir results/baselines/vllm/qwen2p5_0p5b
```

## profiling 对比

```bash
METHOD=sleep_l1 OUT_DIR=results/profiling/sleep_l1_pin_compare scripts/run_profiling.sh
METHOD=sleep_l2 OUT_DIR=results/profiling/sleep_l2_pin_compare scripts/run_profiling.sh
```

详细结论见 `docs/reports/vllm-pin-compare.md`。

## repeated sleep / pool 验证

```bash
.venv/bin/python src/bench_vllm_repeated_sleep_l1.py \
  --out-dir results/profiling/phase1_two_model_pool \
  --iterations 5
```

最新 curated 结果位于 `results/profiling/phase1_two_model_pool/20260702_165801/`，对应报告见 `docs/reports/phase1-two-model-pool.md`。结果显示首次 `sleep_l1` 需要 pinned CPU backup allocation，后续同模型重复 sleep 会复用 CPU backup pool：Qwen2.5-0.5B 的 sleep 从 0.422s 降到约 0.050s，Qwen2.5-1.5B 从 0.991s 降到约 0.102s。

## 解释

在 0.5B 上，`sleep_l1` wake 约 0.11 秒，`sleep_l2` restore 约 0.25 秒，都明显快于 cold reload。profiling 显示 `sleep_l1` 的关键瓶颈是 pinned CPU backup 分配；`sleep_l2` 没有 CPU backup，pin/no-pin 开关基本不影响 sleep 阶段。最近的 reload profiling 进一步把 `sleep_l2` restore 拆成 `wake_weights`、`reload_weights`、`wake_kv_cache`，其中 Qwen2.5-1.5B 的 `reload_weights` 平均约 0.496s，是 L2 restore 的主要组成。
