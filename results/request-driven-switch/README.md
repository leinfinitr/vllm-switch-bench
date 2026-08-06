# Request-driven switch

Question: what completion latency was observed for the frozen 20-request alternating-model schedule?

- Configuration: [`config/workload.json`](config/workload.json)
- Raw evidence: Proposed and llama-swap arrays/JSONL under [`raw/`](raw/)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/request-timeline.pdf`](figures/request-timeline.pdf) ([PNG](figures/request-timeline.png))
- Method and limitations: [`../../docs/experiments/request-driven-switch/README.md`](../../docs/experiments/request-driven-switch/README.md)

The validator requires 20 unique request IDs per system, strict success, a shared frozen identity sequence, and raw-to-summary equality. The historical producer did not runtime-bind controller/engine commits, import path, or config hash, so this is a historical local observation rather than exact fresh-clone runtime reproduction. No new data was generated, and the canonical GPU rerun is incomplete.
