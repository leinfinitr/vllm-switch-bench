"""Authoritative registry for the five published experiment families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentFamily:
    slug: str
    builder: str
    validator: str
    figure_stem: str


FAMILIES = (
    ExperimentFamily(
        slug="lifecycle-latency",
        builder="llm_switch_bench.experiments.lifecycle_latency.artifacts:build",
        validator="llm_switch_bench.validation.lifecycle_latency.validate:validate_family",
        figure_stem="lifecycle-latency",
    ),
    ExperimentFamily(
        slug="vllm-profiling",
        builder="llm_switch_bench.experiments.vllm_profiling.artifacts:build",
        validator="llm_switch_bench.validation.vllm_profiling.validate:validate_family",
        figure_stem="vllm-profiling",
    ),
    ExperimentFamily(
        slug="request-driven-switch",
        builder="llm_switch_bench.experiments.request_driven_switch.artifacts:build",
        validator="llm_switch_bench.validation.request_driven_switch.validate:validate_family",
        figure_stem="request-timeline",
    ),
    ExperimentFamily(
        slug="backup-reuse-reclaim",
        builder="llm_switch_bench.experiments.backup_reuse_reclaim.artifacts:build",
        validator="llm_switch_bench.validation.backup_reuse_reclaim.validate:validate_family",
        figure_stem="backup-reuse",
    ),
    ExperimentFamily(
        slug="exact-disk",
        builder="llm_switch_bench.experiments.exact_disk.artifacts:build",
        validator="llm_switch_bench.validation.exact_disk.validate:validate_family",
        figure_stem="exact-disk",
    ),
)
FAMILY_NAMES = tuple(family.slug for family in FAMILIES)
FAMILIES_BY_NAME = {family.slug: family for family in FAMILIES}
