# Backup reuse and reclaim

Question: do repeated sleeps reuse clean CPU backups without another D2H copy, and does pressure reclaim complete logically and physically?

- Configuration: [`config/claims.json`](config/claims.json)
- Raw evidence: three five-event model profiles and one pressure-release observation under [`raw/`](raw/)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/backup-reuse.pdf`](figures/backup-reuse.pdf) ([PNG](figures/backup-reuse.png))
- Method and limitations: [`../../docs/experiments/backup-reuse-reclaim/README.md`](../../docs/experiments/backup-reuse-reclaim/README.md)

The validator checks positive reused bytes/count, zero repeated D2H time, matching requested/released bytes, zero pending accounting, and material RSS/`MemAvailable` evidence. No new measurement was run during this refactor, and the canonical GPU rerun is incomplete.
