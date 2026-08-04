# Project Context

`results/` retains the latest curated output for each relevant system and method;
it is not an archive of every local run.

## Repository conventions

- Put reusable benchmark adapters and shared schema in `src/`; put pure analysis and
  artifact builders in `src/tool/`.
- Put shell orchestration and artifact-specific measurement drivers in `scripts/` and
  document their provenance binding in `scripts/README.md`.
- Keep current reproduction semantics in `docs/baselines/` and `docs/systems/`.
  Do not retain implementation/process plans on the default release branch; issue or
  pull-request history owns process narrative. Preserve only durable design rationale.
- Each cited result must retain raw evidence, aggregate data, environment/source identity,
  correctness post-conditions, and checksums. Verify checksums from a fresh checkout.
- Physical release claims require both application accounting and OS/GPU-visible evidence.
- Do not publish failed, timed-out, or semantically invalid lifecycle samples as numeric
  baselines; retain the blocker evidence instead.
- The release artifact root is `results/release-v0.1/`. Existing raw evidence below
  it is immutable; the final GPU rerun must replace the bundle atomically, not mix runs.
