# vLLM profiling

Question: which activation phases dominate stock vLLM L1/L2 and Proposed CPU/exact-disk backup restoration under the retained local scope?

- Configuration: [`config/campaign.json`](config/campaign.json)
- Raw evidence: [`raw/profile-samples.json`](raw/profile-samples.json)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/vllm-profiling.pdf`](figures/vllm-profiling.pdf) ([PNG](figures/vllm-profiling.png))
- Method and limitations: [`../../docs/experiments/vllm-profiling/README.md`](../../docs/experiments/vllm-profiling/README.md)

The retained comparison contains five post-warm-up samples collected on 2026-08-13 for a cold-load reference, stock vLLM L1/L2, Proposed CPU backup, and Proposed exact-disk restore. Engine revisions and process-reuse conditions differ, so this is a descriptive local mechanism profile rather than a release-matched ranking.
