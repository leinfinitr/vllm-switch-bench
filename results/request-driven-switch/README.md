# Request-driven switch

Question: what completion latency was observed for the frozen 20-request alternating-model schedule?

- Configuration: [`config/workload.json`](config/workload.json)
- Raw evidence: vllm-switch and llama-swap JSONL rows plus sibling runtime manifests under [`raw/`](raw/)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/request-timeline.pdf`](figures/request-timeline.pdf) ([PNG](figures/request-timeline.png))
- Method and limitations: [`../../docs/experiments/request-driven-switch/README.md`](../../docs/experiments/request-driven-switch/README.md)

The validator binds every supplied dispatch field to the frozen trace and requires 20 unique strict-success rows per system plus raw-to-summary equality. The 2026-08-13 rerun retains runtime repository, configuration, executable, model, workload, and environment provenance in each sibling run manifest.
