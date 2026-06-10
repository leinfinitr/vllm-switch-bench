# sleep_l1 pinned vs non-pinned backup comparison
All successful rows use `sleep_l1`, `short_short`, `repeats=3`. 3B required `gpu_memory_utilization=0.85`; 0.5B/1.5B used default 0.55.
## Mean comparison
| model | gpu util | pinned switch | non-pinned switch | delta | delta % | pinned evict | non-pinned evict | pinned restore | non-pinned restore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2p5_0p5b | 0.55 | 0.5666 | 0.4848 | -0.0818 | -14.4% | 0.4605 | 0.3259 | 0.1061 | 0.1589 |
| qwen2p5_1p5b | 0.55 | 1.2421 | 1.3561 | 0.1141 | 9.2% | 0.9723 | 0.9343 | 0.2697 | 0.4218 |
| qwen2p5_3b | 0.85 | 2.6142 | 2.5484 | -0.0658 | -2.5% | 2.1147 | 1.7671 | 0.4995 | 0.7813 |

## Key breakdown means
| model | pin | n | cpu backup alloc | D2H copy | H2D copy | create map | unmap release |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2p5_0p5b | true | 3 | 0.3697 | 0.0559 | 0.0592 | 0.0441 | 0.0267 |
| qwen2p5_0p5b | false | 3 | 0.0003 | 0.2896 | 0.0830 | 0.0466 | 0.0269 |
| qwen2p5_1p5b | true | 3 | 0.7592 | 0.1698 | 0.1782 | 0.0885 | 0.0331 |
| qwen2p5_1p5b | false | 3 | 0.0006 | 0.8888 | 0.2525 | 0.0920 | 0.0332 |
| qwen2p5_3b | true | 3 | 1.7357 | 0.3245 | 0.3611 | 0.1352 | 0.0429 |
| qwen2p5_3b | false | 3 | 0.0009 | 1.7155 | 0.4946 | 0.1395 | 0.0382 |
