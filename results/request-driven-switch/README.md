# Request-driven switch

Question: what completion latency was observed for the frozen 20-request alternating-model schedule?

- Configuration: [`config/workload.json`](config/workload.json)
- Raw evidence: Proposed and llama-swap JSON arrays consumed by the builder plus retained JSONL source rows under [`raw/`](raw/)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/request-timeline.pdf`](figures/request-timeline.pdf) ([PNG](figures/request-timeline.png))
- Method and limitations: [`../../docs/experiments/request-driven-switch/README.md`](../../docs/experiments/request-driven-switch/README.md)

The validator binds every supplied dispatch field to the retained frozen trace and requires 20 unique strict-success rows per system plus raw-to-summary equality. The historical producer did not runtime-bind controller/engine commits, dirty states, executable/import paths, configuration hash, or model revision, so this is a historical local observation rather than exact fresh-checkout runtime reproduction. No new data was generated, and the canonical GPU rerun is incomplete.
