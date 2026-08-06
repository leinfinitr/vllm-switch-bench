# Lifecycle latency

Question: how long are separate sleep and wake lifecycle phases across the retained three-model/five-system matrix?

- Configuration: [`config/campaign.json`](config/campaign.json)
- Raw evidence: lifecycle JSON under [`raw/`](raw/)
- Summary: [`summary.json`](summary.json) and [`summary.csv`](summary.csv)
- Figure: [`figures/lifecycle-latency.pdf`](figures/lifecycle-latency.pdf) ([PNG](figures/lifecycle-latency.png))
- Method and limitations: [`../../docs/experiments/lifecycle-latency/README.md`](../../docs/experiments/lifecycle-latency/README.md)

The summary contains exactly 30 cells (3 models × 5 systems × 2 phases), each based on five historical v0.1 samples. Sleep and wake remain distinct. SwapServeLLM and profiled llama-swap binaries are external release assets whose immutable contracts are in metadata. No new measurement was run during this refactor, and the canonical GPU rerun is incomplete.
