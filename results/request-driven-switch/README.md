# Request-Driven Switch Result

This is the current claim-supporting result directory. It retains the minimum raw evidence consumed by the builder, a canonical summary, run metadata, and the PDF/PNG paper figure. See the matching experiment document under `../../docs/experiments/request-driven-switch/README.md`.

Migrated from tracked v0.1.8 evidence; no new data was generated during this refactor. The canonical GPU rerun is not complete.

Rebuild from the repository root with:

```bash
uv run python -m llm_switch_bench.artifacts request-driven-switch
uv run python -m llm_switch_bench.validation.request_driven_switch.validate
```
