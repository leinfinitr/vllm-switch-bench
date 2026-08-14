"""Deterministic summaries and figures for lifecycle latency."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from vllm_switch_bench.plotting.style import apply_paper_style, save_figure
from vllm_switch_bench.publication import (
    default_results_root,
    prepare_family,
    read_json,
    write_family_metadata,
    write_json,
    write_result_readme,
)

MODELS = ("qwen-0.5b", "qwen-1.5b", "qwen-3b")
SYSTEMS = ("vllm-switch", "vLLM L1", "vLLM L2", "SwapServeLLM", "llama-swap")
PHASES = ("sleep", "wake")
RAW_MAP = {
    "vllm-switch": "vllm-switch",
    "vLLM L1": "vllm-l1",
    "vLLM L2": "vllm-l2",
    "SwapServeLLM": "swapserve",
}
EXTERNAL_CONTRACTS = {
    "SwapServeLLM": {
        "size_bytes": 54_774_096,
        "sha256": "7d463c42e3d0c965cba078d77a2abb053ba02f2a27a2303d32e5dccecffae091",
    },
    "llama-swap-profiled": {
        "size_bytes": 20_973_543,
        "sha256": "196148236fad99b32cb86c04d9297cfe0eaca68d204c920e3aeff290d04a024b",
    },
}


def external_contracts(family: Path) -> dict[str, Any]:
    provenance = family / "provenance.json"
    if not provenance.is_file() or read_json(provenance).get("status") != "local-rerun":
        return EXTERNAL_CONTRACTS
    llama = read_json(family / "raw" / "llama-swap" / "lifecycle.json")["binary"]
    swapserve = [
        read_json(family / "raw" / "swapserve" / f"{model}.json")["environment"]["binary"]
        for model in MODELS
    ]
    if any(item != swapserve[0] for item in swapserve[1:]):
        raise ValueError("SwapServeLLM lifecycle inputs used different binaries")
    images = [
        read_json(family / "raw" / "swapserve" / f"{model}.json")["environment"]["container_image"]
        for model in MODELS
    ]
    if any(item != images[0] for item in images[1:]):
        raise ValueError("SwapServeLLM lifecycle inputs used different container images")
    checkpoint_tools = [
        read_json(family / "raw" / "swapserve" / f"{model}.json")["environment"]["cuda_checkpoint"]
        for model in MODELS
    ]
    if any(item != checkpoint_tools[0] for item in checkpoint_tools[1:]):
        raise ValueError("SwapServeLLM lifecycle inputs used different cuda-checkpoint tools")
    return {
        "SwapServeLLM": swapserve[0],
        "SwapServeLLM-container-image": images[0],
        "SwapServeLLM-cuda-checkpoint": checkpoint_tools[0],
        "llama-swap-profiled": llama,
    }


RESULT_README = """# Lifecycle latency

Question: how long are separate sleep and wake lifecycle phases across the retained three-model/five-system matrix?

- Configuration: [`config/campaign.json`](config/campaign.json) and the retained
  [`SwapServeLLM compatibility patch`](config/swapserve-local-compat.patch)
- Raw evidence: lifecycle JSON under [`raw/`](raw/)
- Summary: [`summary.json`](summary.json) and [`summary.csv`](summary.csv)
- Figure: [`figures/lifecycle-latency.pdf`](figures/lifecycle-latency.pdf) ([PNG](figures/lifecycle-latency.png))
- Method and limitations: [`../../docs/experiments/lifecycle-latency/README.md`](../../docs/experiments/lifecycle-latency/README.md)

The summary contains exactly 30 cells (3 models \u00d7 5 systems \u00d7 2 phases), each based on five local samples collected on 2026-08-13. Sleep and wake remain distinct. Raw evidence binds the runtime checkout or external binary and configuration used by each producer; full commands and limitations are in the experiment protocol.
"""


def lifecycle_raw_path(system: str, model: str, family_dir: Path) -> Path:
    if system == "llama-swap":
        return family_dir / "raw" / "llama-swap" / "lifecycle.json"
    return family_dir / "raw" / RAW_MAP[system] / f"{model}.json"


def lifecycle_rows(system: str, model: str, family_dir: Path) -> list[dict[str, Any]]:
    rows = read_json(lifecycle_raw_path(system, model, family_dir))["rows"]
    if system == "llama-swap":
        rows = [row for row in rows if row["model"] == model]
    return rows


def quantiles(values: list[float]) -> tuple[float, float, float]:
    """Return the retained five-sample quartiles."""

    if len(values) != 5:
        raise ValueError(f"expected five lifecycle samples, got {len(values)}")
    ordered = sorted(values)
    return ordered[1], statistics.median(ordered), ordered[3]


def summary_rows(family_dir: Path | None = None) -> list[dict[str, Any]]:
    family = family_dir or default_results_root() / "lifecycle-latency"
    summary: list[dict[str, Any]] = []
    for model in MODELS:
        for system in SYSTEMS:
            samples = lifecycle_rows(system, model, family)
            for phase in PHASES:
                key = phase if system == "llama-swap" else f"{phase}_s"
                values = [
                    float(sample[key]["state_machine_latency_s"])
                    if system == "llama-swap"
                    else float(sample[key])
                    for sample in samples
                ]
                q1, median, q3 = quantiles(values)
                summary.append(
                    {
                        "model": model,
                        "system": system,
                        "phase": phase,
                        "n": len(values),
                        "median_s": median,
                        "q1_s": q1,
                        "q3_s": q3,
                    }
                )
    return summary


def write_figure(summary: list[dict[str, Any]], family_dir: Path) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), sharey=True)
    x = range(len(SYSTEMS))
    for index, (axis, model) in enumerate(zip(axes, MODELS, strict=True)):
        model_rows = [row for row in summary if row["model"] == model]
        for phase, offset, marker in (("sleep", -0.12, "o"), ("wake", 0.12, "^")):
            rows = [
                next(row for row in model_rows if row["system"] == system and row["phase"] == phase)
                for system in SYSTEMS
            ]
            medians = [float(row["median_s"]) for row in rows]
            errors = [
                [float(row["median_s"]) - float(row["q1_s"]) for row in rows],
                [float(row["q3_s"]) - float(row["median_s"]) for row in rows],
            ]
            axis.errorbar(
                [item + offset for item in x],
                medians,
                yerr=errors,
                label=phase.capitalize(),
                marker=marker,
                linestyle="none",
                color="#222222",
                capsize=2,
                markersize=3.5,
            )
        axis.set_title(model)
        axis.set_xticks(list(x), SYSTEMS, rotation=38, ha="right")
        axis.set_yscale("log")
        if index == 0:
            axis.set_ylabel("Latency (s, log scale)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, family_dir / "figures" / "lifecycle-latency")


def build(results_root: Path | None = None) -> None:
    family = (results_root or default_results_root()) / "lifecycle-latency"
    prepare_family(family)
    summary = summary_rows(family)
    write_json(family / "summary.json", {"lifecycle": summary})
    fields = ("model", "system", "phase", "n", "median_s", "q1_s", "q3_s")
    with (family / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary:
            handle.write(",".join(str(row[field]) for field in fields) + "\n")
    write_figure(summary, family)
    write_result_readme(family, RESULT_README)
    write_family_metadata(
        "lifecycle-latency",
        family,
        config=["config/campaign.json", "config/swapserve-local-compat.patch"],
        validation={"aggregate_cells": 30, "samples_per_cell": 5},
        extra={"external_artifacts": external_contracts(family)},
    )
