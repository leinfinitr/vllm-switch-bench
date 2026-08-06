# Result families

`results/` contains only current claim-supporting evidence. It is not a local run archive.

| Family | Retained evidence | Deterministic outputs |
|---|---|---|
| [`lifecycle-latency`](lifecycle-latency/) | 13 lifecycle JSON files covering 3 models and 5 systems | 30 aggregate cells, CSV, PDF/PNG |
| [`request-driven-switch`](request-driven-switch/) | two strict 20-request alternating traces | two aggregate rows, timeline PDF/PNG |
| [`backup-reuse-reclaim`](backup-reuse-reclaim/) | three five-cycle reuse profiles and one pressure-release observation | reuse/reclaim summary, PDF/PNG |
| [`exact-disk`](exact-disk/) | seven runtime integrity, footprint, event, and output files | claim summary, PDF/PNG |

Every family has the same lifecycle:

```text
config + raw evidence -> build_all -> summary + figures -> validate_all
```

`metadata.json` declares the exact retained file set and any external retrieval contract. Semantic validators do more than file/schema checks: they recompute aggregates and verify family-specific success, cardinality, integrity, and post-condition claims.

## Evidence boundary

These files were migrated from the v0.1 evidence at tag `v0.1.8`; no new benchmark was run during this refactor. The 30 lifecycle aggregate cells and two request-driven aggregate rows retain their original numeric values.

The v0.1 E2E producer data did not runtime-bind controller/engine commits, imported module path, or configuration hash. It is a historical local observation, not an exact fresh-clone runtime reproduction. The canonical GPU rerun is not complete.

SwapServeLLM and profiled llama-swap executables are external release assets, not tracked files. Their immutable URL, size, and SHA-256 contracts are in lifecycle metadata. Exact-disk payload and chunk digests remain because they authenticate runtime correctness; the 1 GiB payload itself is omitted.

## Rebuild and validate

```bash
scripts/build_all.sh
scripts/validate_all.sh
```

A publication-quality change must make two consecutive builds byte-identical and leave no diff. Live runs belong in ignored `results/tmp/`; promote only reviewed minimum evidence into one of the four current families.
