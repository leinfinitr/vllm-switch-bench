# Lifecycle latency

Question: how long are separate sleep and wake lifecycle phases across the retained three-model/five-system matrix?

- Configuration: [`config/campaign.json`](config/campaign.json) and the retained
  [`SwapServeLLM compatibility patch`](config/swapserve-local-compat.patch)
- Raw evidence: lifecycle JSON under [`raw/`](raw/)
- Summary: [`summary.json`](summary.json) and [`summary.csv`](summary.csv)
- Figure: [`figures/lifecycle-latency.pdf`](figures/lifecycle-latency.pdf) ([PNG](figures/lifecycle-latency.png))
- Method and limitations: [`../../docs/experiments/lifecycle-latency/README.md`](../../docs/experiments/lifecycle-latency/README.md)

The summary contains exactly 30 cells (3 models × 5 systems × 2 phases), each based on five local samples collected on 2026-08-13. Sleep and wake remain distinct. Raw evidence binds the runtime checkout or external binary and configuration used by each producer; full commands and limitations are in the experiment protocol.
