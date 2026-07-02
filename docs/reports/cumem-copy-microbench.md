# CuMem / PCIe copy microbench

本文整理最近 `src/microbench/` 新增脚本对应的 curated 结果，用于解释 vLLM `sleep_l1` 中 pinned CPU backup allocation、D2H/H2D copy 和 CuMem create/map 的成本来源。相关提交为 `4f16c65`。

## 入口与结果

脚本入口：

- `src/microbench/microbench_pcie_copy.py`
- `src/microbench/microbench_cumem_sleep_copy.py`
- `src/microbench/microbench_cumem_safetensor_sizes.py`

最新 curated 输出：

- `results/profiling/microbench/microbench_pcie_copy_20260702_184602.csv`
- `results/profiling/microbench/cumem_copy_microbench_20260702_185031_single/`
- `results/profiling/microbench/cumem_safetensor_sizes_microbench_20260702_185343/`

## PCIe copy 基线

`microbench_pcie_copy.py` 同时测 torch `copy_` 和 vLLM `CudaRTLibrary.cudaMemcpy`。下表只列 vLLM cudart 路径的 mean GB/s：

| host memory | H2D GB/s | D2H GB/s | 说明 |
|---|---:|---:|---|
| pinned | 19.41-20.01 | 19.32-19.83 | 256MiB 到 6GiB 基本稳定在约 20GB/s |
| pageable | 13.18-13.67 | 13.71-13.83 | 比 pinned 慢约三成 |

这个结果解释了 pin/no-pin profiling 中的权衡：pageable backup 能绕开 pinned allocation，但会让 D2H/H2D copy 明显退化。

## CuMem synthetic allocations

`microbench_cumem_sleep_copy.py` 在 `CuMemAllocator` memory pool 中创建 synthetic CUDA tensors，然后直接调用 `allocator.sleep()` 和 `allocator.wake_up()`。

| case | chunks | sleep | CPU backup alloc | D2H GB/s | wake | create/map | H2D GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1GiB:1` | 1 | 0.298s | 0.194s | 19.67 | 0.081s | 0.002s | 13.71 |
| `1GiB:41` | 41 | 0.349s | 0.247s | 18.40 | 0.085s | 0.028s | 19.07 |
| `3GiB:1` | 1 | 0.958s | 0.751s | 19.79 | 0.254s | 0.003s | 12.84 |
| `3GiB:76` | 76 | 1.102s | 0.884s | 18.75 | 0.241s | 0.073s | 19.12 |
| `6GiB:1` | 1 | 1.866s | 1.495s | 19.74 | 0.558s | 0.234s | 19.92 |
| `6GiB:104` | 104 | 1.582s | 1.196s | 19.11 | 0.469s | 0.116s | 18.27 |

D2H/H2D copy 本身仍接近 PCIe 带宽上限；sleep 延迟的额外部分主要来自 CPU backup allocation，wake 侧则包含 create/map。

## safetensors 粒度

`microbench_cumem_safetensor_sizes.py` 按模型 safetensors header 中每个 tensor 的 byte size 创建 synthetic allocations，更接近真实权重粒度。

| model | chunks | total | sleep | CPU backup alloc | D2H GB/s | wake | create/map | H2D GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 290 | 0.92GiB | 0.415s | 0.315s | 18.16 | 0.079s | 0.022s | 17.30 |
| Qwen2.5-1.5B | 338 | 2.88GiB | 0.915s | 0.693s | 17.92 | 0.242s | 0.067s | 17.70 |
| Qwen2.5-3B | 434 | 5.75GiB | 2.091s | 1.705s | 18.53 | 0.471s | 0.132s | 18.19 |

## 结论

1. Pinned host memory 的 copy 带宽稳定高于 pageable host memory，因此 no-pin 不是稳定优化。
2. `sleep_l1` 的大头不是单纯 D2H copy，而是首次 pinned CPU backup allocation；权重越大，这部分越明显。
3. safetensors 粒度结果与真实模型 profiling 同向：Qwen2.5-3B 的 CPU backup allocation 已到秒级。
4. repeated sleep pool 复用能直接消除后续 sleep 的 backup allocation，相关结果见 `docs/reports/phase1-two-model-pool.md`。
