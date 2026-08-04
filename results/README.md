# Result artifact policy

`results/` is a curated evidence tree, not a dump of every local run. The v0.1 release has one canonical bundle; older families remain only because references, schemas, or provenance make deletion unsafe.

## Status index

| Path | Status | Policy |
|---|---|---|
| [`release-v0.1/`](release-v0.1/) | **Current release candidate** | Single canonical v0.1 bundle; existing data is exploratory until final GPU rerun |
| `request_switch/latest/` | Historical | Earlier request-driven artifact; superseded by the release bundle and retained for audit |
| `osdi_20260723/` | Superseded path | Renamed to `release-v0.1/`; no second canonical copy remains |
| `baselines/` | Historical | Legacy Baseline3 inputs and blocked rows; immutable schemas |
| `cross_system/` | Historical | Earlier cross-system matrix and blockers; not current headline data |
| `model_switch_eval/` | Historical | Earlier model-switch evaluation and secondary report export |
| `profiling/` | Historical mechanism evidence | Useful for mechanism background, not current release claims |
| `tmp/` | Local/ignored | Disposable staging and failed pilots; never cited directly |

ServerlessLLM is **blocked** for a current numeric row. Exact disk is **blocked pending final v0.1 GPU rerun**. SwapServeLLM and llama-swap are canonical baseline names; historical slugs remain unchanged inside raw data.

## Publishable bundle requirements

A cited artifact must retain:

- frozen manifests, prompts, generation fields, models, and sanitized configs;
- request- or phase-level raw evidence and structured failure diagnostics;
- run-start source commits/dirty state, executable/import path, image digest, and environment identity;
- functional correctness and physical post-conditions where claimed;
- deterministic summary/figure builders;
- a publication-subset manifest and a complete-bundle manifest;
- fully tracked manifest paths verified from a fresh checkout.

Failed, timed-out, incomplete, or semantically invalid samples must not appear as numeric baselines. Keep them as blocked/failed diagnostics with the attempted denominator and terminal state.

## Immutable raw evidence

Checksummed files under `release-v0.1/raw/` are immutable. Producer-machine absolute paths in those files are provenance, not current configuration, and must not be normalized. To correct or rerun an experiment, create a fresh staged bundle and replace the canonical bundle atomically after review.

## Derived files

Derived summaries and figures may be regenerated only from the declared tracked raw inputs. Builders must be deterministic; run them twice and require identical bytes. Generate `checksums.sha256` first and `all-files.sha256` last.

## Local output

Write live experiments under `results/tmp/<experiment>/<run-id>/` or another ignored staging path. Do not rely on ignored logs for a publication claim. Promote only the reviewed minimal evidence set, never entire caches, model files, secrets, or unfiltered repeated logs.

See [`../docs/release-artifact.md`](../docs/release-artifact.md) for the final-rerun transaction.
