"""Deterministic publication for the vLLM mechanism profile."""

from __future__ import annotations

from pathlib import Path

from vllm_switch_bench.experiments.vllm_profiling.plot import write_artifacts
from vllm_switch_bench.publication import (
    default_results_root,
    prepare_family,
    write_family_metadata,
    write_result_readme,
)

RESULT_README = """# vLLM profiling

Question: which sleep and wake phases dominate vLLM L1/L2 and vllm-switch CPU/exact-disk mechanisms under the retained local scope?

- Configuration: [`config/campaign.json`](config/campaign.json)
- Raw evidence: [`raw/profile-samples.json`](raw/profile-samples.json)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/vllm-profiling.pdf`](figures/vllm-profiling.pdf) ([PNG](figures/vllm-profiling.png))
- Method and limitations: [`../../docs/experiments/vllm-profiling/README.md`](../../docs/experiments/vllm-profiling/README.md)

The retained comparison uses three independent process blocks with three cycles per block.
It separates first and steady L1/CPU/exact-disk behavior, and validates cold versus warm
page-cache state for vLLM L2 using residency and physical-read evidence.
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
        validation={
            "methods": 9,
            "samples_per_method": 3,
            "process_blocks_per_method": 3,
            "cycles_per_process": 3,
            "operations": ["sleep", "wake"],
            "phase_accounting": True,
            "l2_cache_state_validation": True,
        },
        extra={
            "evidence_label": (
                "controlled local mechanism comparison with process-block and cache-state controls"
            )
        },
    )
