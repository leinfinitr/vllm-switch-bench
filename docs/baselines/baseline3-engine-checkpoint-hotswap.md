# Baseline3：系统级 checkpoint / hotswap

Baseline3 用来对比 vLLM 内部 Sleep Mode 之外的系统级切换方案。这里的代表系统是 ServerlessLLM 和 SwapServeLLM；它们把推理引擎进程或容器作为可切换对象，而不是只修改 vLLM 内部 allocator 行为。

## 运行入口

配置文件：`configs/baseline3.local.yaml`

```bash
scripts/run_baseline3.sh
```

等价直接命令：

```bash
.venv/bin/python src/bench_baseline3.py   --config configs/baseline3.local.yaml   --out-dir results/baselines/baseline3/qwen2p5_0p5b
```

如果只需要生成报告：

```bash
.venv/bin/python src/tool/analyze_baseline3.py   results/baselines/baseline3/qwen2p5_0p5b/<timestamp>   --out docs/reports/baseline3-qwen2p5-0p5b.md
```

## 外部系统要求

- ServerlessLLM 需要 Docker / NVIDIA Docker，见 `docs/systems/serverlessllm.md`。
- SwapServeLLM 需要 rootless podman、`cuda-checkpoint` 和本机兼容 patch，见 `docs/systems/swapservellm.md`。

## 解释

Baseline3 的价值是提供横向参考：ServerlessLLM 和 SwapServeLLM 的切换发生在 vLLM 外部，适合作为系统方案对比；但它们不是 vLLM PR 主线里的内部优化点。当前研究主线更适合聚焦 vLLM Sleep Mode 内部路径。
