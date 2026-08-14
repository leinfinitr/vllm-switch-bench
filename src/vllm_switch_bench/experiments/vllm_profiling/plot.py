"""Aggregate and plot the retained vLLM activation-latency profiles."""

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
PHASE_ORDER = (
    "Process + engine startup",
    "CPU→GPU copy",
    "GPU remap",
    "Checkpoint load",
    "KV-cache remap",
    "Disk read + hash + H2D pipeline",
    "Control overhead",
)
PHASE_COLORS = {
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
    """Select one real median-nearest profile and preserve total-latency spread."""

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
        total = _number(raw.get("total_s"), name=f"samples[{index}].total_s")
        phases = raw.get("phases_s")
        if not isinstance(phases, dict) or not phases:
            raise ValueError(f"samples[{index}].phases_s must be a non-empty object")
        normalized_phases: dict[str, float] = {}
        for phase, value in phases.items():
            if phase not in PHASE_ORDER:
                raise ValueError(f"unknown phase: {phase!r}")
            normalized_phases[phase] = _number(value, name=f"samples[{index}].phases_s[{phase!r}]")
        phase_sum = sum(normalized_phases.values())
        if not math.isclose(phase_sum, total, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"sample {method!r}/{raw.get('sample_index')!r} phases sum to "
                f"{phase_sum}, not total {total}"
            )
        sample_index = raw.get("sample_index")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError(f"samples[{index}].sample_index must be an integer")
        grouped[method].append(
            {
                "sample_index": sample_index,
                "total_s": total,
                "phases_s": normalized_phases,
                "source": raw.get("source"),
            }
        )

    rows: list[dict[str, Any]] = []
    expected_count = int(document.get("frozen_scope", {}).get("sample_count_per_method", 5))
    for method in RAW_METHOD_ORDER:
        method_samples = grouped[method]
        if len(method_samples) != expected_count:
            raise ValueError(
                f"method {method!r} has {len(method_samples)} samples; expected {expected_count}"
            )
        totals = [sample["total_s"] for sample in method_samples]
        median = statistics.median(totals)
        representative = min(
            method_samples,
            key=lambda sample: (abs(sample["total_s"] - median), sample["sample_index"]),
        )
        rows.append(
            {
                "method": METHOD_LABELS[method],
                "n": len(method_samples),
                "median_s": median,
                "min_s": min(totals),
                "max_s": max(totals),
                "representative_sample_index": representative["sample_index"],
                "representative_total_s": representative["total_s"],
                "representative_phases_s": representative["phases_s"],
                "source": representative["source"],
            }
        )

    return {
        "schema_version": 1,
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
    seconds: bool,
    title: str,
) -> None:
    x = np.arange(len(rows), dtype=float)
    scale = 1.0 if seconds else 1000.0
    bottoms = np.zeros(len(rows), dtype=float)
    for phase in PHASE_ORDER:
        values = np.asarray(
            [float(row["representative_phases_s"].get(phase, 0.0)) * scale for row in rows]
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

    medians = np.asarray([float(row["median_s"]) * scale for row in rows])
    lower = medians - np.asarray([float(row["min_s"]) * scale for row in rows])
    upper = np.asarray([float(row["max_s"]) * scale for row in rows]) - medians
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
        value = float(row["median_s"])
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
    ax.set_ylabel("Activation latency (s)" if seconds else "Activation latency (ms)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(bottom=0)


def _draw_share_panel(ax, rows: list[Mapping[str, Any]]) -> None:
    """Show normalized shares without sacrificing the absolute-latency panel."""

    x = np.arange(len(rows), dtype=float)
    bottoms = np.zeros(len(rows), dtype=float)
    for phase in PHASE_ORDER:
        values = np.asarray(
            [
                float(row["representative_phases_s"].get(phase, 0.0))
                / float(row["representative_total_s"])
                * 100.0
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
            zorder=2,
        )
        for position, bottom, value in zip(x, bottoms, values, strict=True):
            if value >= 8.0:
                ax.text(
                    position,
                    bottom + value / 2.0,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black",
                )
        bottoms += values
    ax.set_title("(c) Phase share")
    ax.set_xticks(x, [str(row["method"]) for row in rows], rotation=15, ha="right")
    ax.set_ylabel("Share of activation latency (%)")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)


def plot_profiles(summary: Mapping[str, Any], output_base: Path) -> list[Path]:
    """Plot cold load separately so sub-second phase proportions remain visible."""

    apply_paper_style()
    rows = list(summary["methods"])
    cold = [row for row in rows if row["method"] == "Cold load"]
    warm = [row for row in rows if row["method"] != "Cold load"]
    fig = plt.figure(figsize=(7.2, 5.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], width_ratios=[1.0, 3.3])
    cold_ax = fig.add_subplot(grid[0, 0])
    warm_ax = fig.add_subplot(grid[0, 1])
    share_ax = fig.add_subplot(grid[1, :])
    _draw_panel(cold_ax, cold, seconds=True, title="(a) Cold process")
    _draw_panel(warm_ax, warm, seconds=False, title="(b) In-process activation")
    _draw_share_panel(share_ax, rows)

    handles, labels = warm_ax.get_legend_handles_labels()
    cold_handles, cold_labels = cold_ax.get_legend_handles_labels()
    by_label = dict(zip(cold_labels + labels, cold_handles + handles, strict=False))
    ordered = [phase for phase in PHASE_ORDER if phase in by_label]
    if "Median [min, max]" in by_label:
        ordered.append("Median [min, max]")
    fig.legend(
        [by_label[label] for label in ordered],
        ordered,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(f"Local activation profiles — {summary['model']}", y=0.9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.82))
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
