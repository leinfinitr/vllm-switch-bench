"""Deterministic publication for the vLLM mechanism profile."""

from __future__ import annotations

from pathlib import Path

from llm_switch_bench.experiments.vllm_profiling.plot import write_artifacts
from llm_switch_bench.publication import (
    default_results_root,
    prepare_family,
    write_family_metadata,
    write_result_readme,
)

RESULT_README = """# vLLM profiling

Question: which activation phases dominate stock vLLM L1/L2 and Proposed CPU/exact-disk backup restoration under the retained local scope?

- Configuration: [`config/campaign.json`](config/campaign.json)
- Raw evidence: [`raw/profile-samples.json`](raw/profile-samples.json)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/vllm-profiling.pdf`](figures/vllm-profiling.pdf) ([PNG](figures/vllm-profiling.png))
- Method and limitations: [`../../docs/experiments/vllm-profiling/README.md`](../../docs/experiments/vllm-profiling/README.md)

The retained comparison contains five post-warm-up samples collected on 2026-08-13 for a cold-load reference, stock vLLM L1/L2, Proposed CPU backup, and Proposed exact-disk restore. Engine revisions and process-reuse conditions differ, so this is a descriptive local mechanism profile rather than a controlled cross-system ranking.
"""


def build(results_root: Path | None = None) -> None:
    family = (results_root or default_results_root()) / "vllm-profiling"
    prepare_family(family)
    write_artifacts(family / "raw" / "profile-samples.json", family)
    write_result_readme(family, RESULT_README)
    write_family_metadata(
        "vllm-profiling",
        family,
        config=["config/campaign.json"],
        validation={"methods": 5, "samples_per_method": 5, "phase_accounting": True},
        extra={
            "evidence_label": (
                "descriptive local mechanism comparison; not a controlled cross-system ranking"
            )
        },
    )
