"""Aggregate and plot the retained vLLM sleep- and wake-latency profiles."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from vllm_switch_bench.plotting.style import apply_paper_style, save_figure

RAW_METHOD_ORDER = (
    "Cold load",
    "vLLM L1",
    "vLLM L2",
    "CPU backup",
    "Exact disk",
)
METHOD_LABELS = {
    "Cold load": "Cold load",
    "vLLM L1": "vLLM L1",
    "vLLM L2": "vLLM L2",
    "CPU backup": "vllm-switch CPU backup",
    "Exact disk": "vllm-switch exact disk",
}
METHOD_ORDER = tuple(METHOD_LABELS[method] for method in RAW_METHOD_ORDER)
PROFILE_OPERATIONS = ("sleep", "wake")
PHASE_ORDER = (
    "Process shutdown",
    "CPU backup allocation",
    "GPU→CPU copy",
    "GPU unmap + release",
    "Process + engine startup",
    "CPU→GPU copy",
    "GPU remap",
    "Checkpoint load",
    "KV-cache remap",
    "Disk read + hash + H2D pipeline",
    "Control overhead",
)
PHASE_COLORS = {
    "Process shutdown": "#374151",
    "CPU backup allocation": "#F0E442",
    "GPU→CPU copy": "#D55E00",
    "GPU unmap + release": "#8A9A2A",
    "Process + engine startup": "#6B7280",
    "CPU→GPU copy": "#E69F00",
    "GPU remap": "#009E73",
    "Checkpoint load": "#CC79A7",
    "KV-cache remap": "#0072B2",
    "Disk read + hash + H2D pipeline": "#56B4E9",
    "Control overhead": "#D9D9D9",
}


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def aggregate_profiles(document: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate sleep and wake independently using real median-nearest profiles."""

    samples = document.get("samples")
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    grouped: dict[str, list[dict[str, Any]]] = {method: [] for method in RAW_METHOD_ORDER}
    for index, raw in enumerate(samples):
        if not isinstance(raw, dict):
            raise ValueError(f"samples[{index}] must be an object")
        method = raw.get("method")
        if method not in grouped:
            raise ValueError(f"unknown method: {method!r}")
        sample_index = raw.get("sample_index")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError(f"samples[{index}].sample_index must be an integer")
        sample: dict[str, Any] = {
            "sample_index": sample_index,
            "source": raw.get("source"),
        }
        for operation in PROFILE_OPERATIONS:
            total_field = f"{operation}_total_s"
            phases_field = f"{operation}_phases_s"
            total = _number(raw.get(total_field), name=f"samples[{index}].{total_field}")
            phases = raw.get(phases_field)
            if not isinstance(phases, dict) or not phases:
                raise ValueError(f"samples[{index}].{phases_field} must be a non-empty object")
            normalized_phases: dict[str, float] = {}
            for phase, value in phases.items():
                if phase not in PHASE_ORDER:
                    raise ValueError(f"unknown phase: {phase!r}")
                normalized_phases[phase] = _number(
                    value, name=f"samples[{index}].{phases_field}[{phase!r}]"
                )
            phase_sum = sum(normalized_phases.values())
            if not math.isclose(phase_sum, total, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f"sample {method!r}/{sample_index!r} {operation} phases sum to "
                    f"{phase_sum}, not total {total}"
                )
            sample[operation] = {"total_s": total, "phases_s": normalized_phases}
        grouped[method].append(sample)

    rows: list[dict[str, Any]] = []
    expected_count = int(document.get("frozen_scope", {}).get("sample_count_per_method", 5))
    for method in RAW_METHOD_ORDER:
        method_samples = grouped[method]
        if len(method_samples) != expected_count:
            raise ValueError(
                f"method {method!r} has {len(method_samples)} samples; expected {expected_count}"
            )
        row: dict[str, Any] = {
            "method": METHOD_LABELS[method],
            "n": len(method_samples),
            "source": method_samples[0]["source"],
        }
        for operation in PROFILE_OPERATIONS:
            totals = [sample[operation]["total_s"] for sample in method_samples]
            median = statistics.median(totals)
            representative = min(
                method_samples,
                key=lambda sample: (
                    abs(sample[operation]["total_s"] - median),
                    sample["sample_index"],
                ),
            )
            row[operation] = {
                "median_s": median,
                "min_s": min(totals),
                "max_s": max(totals),
                "representative_sample_index": representative["sample_index"],
                "representative_total_s": representative[operation]["total_s"],
                "representative_phases_s": representative[operation]["phases_s"],
            }
        rows.append(row)

    return {
        "schema_version": 2,
        "title": document.get("title"),
        "evidence_label": document.get("evidence_label"),
        "comparability": document.get("comparability"),
        "metric_boundary": document.get("metric_boundary"),
        "model": document.get("model"),
        "frozen_scope": document.get("frozen_scope"),
        "stability_rule": document.get("stability_rule"),
        "phase_semantics": document.get("phase_semantics"),
        "sources": document.get("sources"),
        "methods": rows,
    }


def _draw_panel(
    ax,
    rows: list[Mapping[str, Any]],
    *,
    operation: str,
    seconds: bool,
    title: str,
) -> None:
    x = np.arange(len(rows), dtype=float)
    scale = 1.0 if seconds else 1000.0
    bottoms = np.zeros(len(rows), dtype=float)
    for phase in PHASE_ORDER:
        values = np.asarray(
            [
                float(row[operation]["representative_phases_s"].get(phase, 0.0)) * scale
                for row in rows
            ]
        )
        if not np.any(values > 0):
            continue
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.68,
            color=PHASE_COLORS[phase],
            edgecolor="black",
            linewidth=0.35,
            label=phase,
            zorder=2,
        )
        bottoms += values

    medians = np.asarray([float(row[operation]["median_s"]) * scale for row in rows])
    lower = medians - np.asarray([float(row[operation]["min_s"]) * scale for row in rows])
    upper = np.asarray([float(row[operation]["max_s"]) * scale for row in rows]) - medians
    ax.errorbar(
        x,
        medians,
        yerr=np.vstack([lower, upper]),
        fmt="D",
        markersize=3.2,
        markerfacecolor="white",
        markeredgecolor="black",
        color="black",
        capsize=2.5,
        linewidth=0.8,
        label="Median [min, max]",
        zorder=4,
    )
    for position, row, top in zip(x, rows, bottoms, strict=True):
        value = float(row[operation]["median_s"])
        label = f"{value:.3f} s" if seconds else f"{value * 1000:.0f} ms"
        ax.annotate(
            label,
            (position, top),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.set_title(title, y=1.07)
    ax.set_xticks(x, [str(row["method"]) for row in rows], rotation=15, ha="right")
    unit = "s" if seconds else "ms"
    ax.set_ylabel(f"{operation.title()} latency ({unit})")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(bottom=0)


def plot_profiles(summary: Mapping[str, Any], output_base: Path) -> list[Path]:
    """Plot sleep above wake while preserving a readable cold-process wake scale."""

    apply_paper_style()
    rows = list(summary["methods"])
    cold = [row for row in rows if row["method"] == "Cold load"]
    warm = [row for row in rows if row["method"] != "Cold load"]
    fig = plt.figure(figsize=(7.6, 6.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 3.3])
    sleep_ax = fig.add_subplot(grid[0, :])
    cold_ax = fig.add_subplot(grid[1, 0])
    warm_ax = fig.add_subplot(grid[1, 1])
    _draw_panel(
        sleep_ax,
        rows,
        operation="sleep",
        seconds=False,
        title="(a) Sleep Latency Profiling",
    )
    _draw_panel(
        cold_ax,
        cold,
        operation="wake",
        seconds=True,
        title="(b) Cold process",
    )
    _draw_panel(
        warm_ax,
        warm,
        operation="wake",
        seconds=False,
        title="(c) In-process",
    )

    handles, labels = warm_ax.get_legend_handles_labels()
    cold_handles, cold_labels = cold_ax.get_legend_handles_labels()
    sleep_handles, sleep_labels = sleep_ax.get_legend_handles_labels()
    by_label = dict(
        zip(
            sleep_labels + cold_labels + labels,
            sleep_handles + cold_handles + handles,
            strict=False,
        )
    )
    ordered = [phase for phase in PHASE_ORDER if phase in by_label]
    if "Median [min, max]" in by_label:
        ordered.append("Median [min, max]")
    fig.legend(
        [by_label[label] for label in ordered],
        ordered,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(f"Local sleep and wake profiles — {summary['model']}", y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.79), h_pad=3.0)
    wake_top = max(cold_ax.get_position().y1, warm_ax.get_position().y1)
    fig.text(
        0.5,
        wake_top + 0.055,
        "Wake Latency Profiling",
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    return save_figure(fig, output_base)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_artifacts(input_path: Path, output_dir: Path) -> list[Path]:
    """Build the deterministic summary and figures from one retained input."""

    document = json.loads(input_path.read_text(encoding="utf-8"))
    summary = aggregate_profiles(document)
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    outputs = plot_profiles(summary, output_dir / "figures" / "vllm-profiling")
    return [summary_path, *outputs]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    for path in write_artifacts(args.input, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
