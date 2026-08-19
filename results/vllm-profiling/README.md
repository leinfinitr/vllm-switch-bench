# vLLM profiling

Question: which sleep and wake phases dominate vLLM L1/L2 and vllm-switch CPU/exact-disk mechanisms under the retained local scope?

- Configuration: [`config/campaign.json`](config/campaign.json)
- Raw evidence: [`raw/profile-samples.json`](raw/profile-samples.json)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/vllm-profiling.pdf`](figures/vllm-profiling.pdf) ([PNG](figures/vllm-profiling.png))
- Method and limitations: [`../../docs/experiments/vllm-profiling/README.md`](../../docs/experiments/vllm-profiling/README.md)

The retained comparison uses three independent process blocks with three cycles per block.
It separates first and steady L1/CPU/exact-disk behavior, and validates cold versus warm
page-cache state for vLLM L2 using residency and physical-read evidence.
