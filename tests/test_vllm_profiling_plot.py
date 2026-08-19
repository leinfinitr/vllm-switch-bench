from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_switch_bench.experiments.vllm_profiling.plot import (
    METHOD_ORDER,
    RAW_METHOD_ORDER,
    aggregate_profiles,
    main,
)


def sample(method: str, index: int, total: float, phase: float | None = None) -> dict:
    phase_value = total if phase is None else phase
    wake_phases = {"GPU remap": phase_value}
    if phase_value != total:
        wake_phases["Control overhead"] = total - phase_value
    sleep_total = total / 2
    return {
        "method": method,
        "sample_index": index,
        "sleep_total_s": sleep_total,
        "sleep_phases_s": {
            "GPU unmap + release": sleep_total * 0.75,
            "Control overhead": sleep_total * 0.25,
        },
        "wake_total_s": total,
        "wake_phases_s": wake_phases,
        "source": f"{method}.json",
    }


def document() -> dict:
    samples = []
    for method in RAW_METHOD_ORDER:
        for index, total in enumerate([1.0, 1.1, 1.2], start=1):
            samples.append(sample(method, index, total, total * 0.75))
    return {
        "schema_version": 2,
        "metric_boundary": {"sleep": "sleep begin to done", "wake": "wake begin to ready"},
        "model": "model-a",
        "frozen_scope": {"sample_count_per_method": 3},
        "stability_rule": {},
        "phase_semantics": {},
        "sources": [],
        "samples": samples,
    }


def test_aggregate_profiles_selects_real_median_sample_and_spread():
    summary = aggregate_profiles(document())

    assert [row["method"] for row in summary["methods"]] == list(METHOD_ORDER)
    row = summary["methods"][0]
    assert row["sleep"]["median_s"] == pytest.approx(0.55)
    assert row["wake"]["median_s"] == pytest.approx(1.1)
    assert row["wake"]["min_s"] == pytest.approx(1.0)
    assert row["wake"]["max_s"] == pytest.approx(1.2)
    assert row["wake"]["representative_sample_index"] == 2
    assert sum(row["wake"]["representative_phases_s"].values()) == pytest.approx(1.1)


def test_aggregate_profiles_rejects_non_closing_breakdown():
    payload = document()
    payload["samples"][0]["sleep_phases_s"] = {"GPU unmap + release": 0.25}

    with pytest.raises(ValueError, match="phases sum"):
        aggregate_profiles(payload)


def test_aggregate_profiles_rejects_missing_samples():
    payload = document()
    payload["samples"].pop()

    with pytest.raises(ValueError, match="has 2 samples"):
        aggregate_profiles(payload)


def test_main_writes_summary_png_and_pdf(tmp_path: Path):
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(document()), encoding="utf-8")
    output = tmp_path / "output"

    assert main(["--input", str(input_path), "--output-dir", str(output)]) == 0
    assert (output / "summary.json").stat().st_size > 0
    assert (output / "figures" / "vllm-profiling.png").stat().st_size > 0
    assert (output / "figures" / "vllm-profiling.pdf").stat().st_size > 0
