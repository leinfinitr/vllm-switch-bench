# sleep_l2 pinned vs non-pinned backup comparison

All successful rows use `sleep_l2`, `short_short`, `repeats=3`. 3B uses `gpu_memory_utilization=0.85`; 0.5B/1.5B use 0.55.

## Mean comparison

| model | gpu util | pinned switch | non-pinned switch | delta | delta % | pinned evict | non-pinned evict | pinned restore | non-pinned restore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2p5_0p5b | 0.55 | 0.2914 | 0.2897 | -0.0017 | -0.6% | 0.0587 | 0.0590 | 0.2326 | 0.2307 |
| qwen2p5_1p5b | 0.55 | 0.6978 | 0.6978 | 0.0000 | 0.0% | 0.1150 | 0.1152 | 0.5828 | 0.5826 |
| qwen2p5_3b | 0.85 | 1.5278 | 1.5836 | 0.0558 | 3.7% | 0.1807 | 0.1811 | 1.3471 | 1.4025 |

## Key breakdown means

| model | pin | n | cpu backup alloc | D2H copy | H2D copy | create map | unmap release |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2p5_0p5b | true | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0231 | 0.0471 |
| qwen2p5_0p5b | false | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0232 | 0.0472 |
| qwen2p5_1p5b | true | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0705 | 0.0989 |
| qwen2p5_1p5b | false | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0707 | 0.0991 |
| qwen2p5_3b | true | 3 | 0.0000 | 0.0000 | 0.0000 | 0.1299 | 0.1643 |
| qwen2p5_3b | false | 3 | 0.0000 | 0.0000 | 0.0000 | 0.1306 | 0.1644 |
