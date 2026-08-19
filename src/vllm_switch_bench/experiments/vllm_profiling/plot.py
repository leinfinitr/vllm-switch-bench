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
    "vLLM L1 First",
    "vLLM L1 Steady",
    "vLLM L2 Cold",
    "vLLM L2 Warm",
    "CPU backup First",
    "CPU backup Steady",
    "Exact disk First",
    "Exact disk Steady",
)
METHOD_LABELS = {
    "Cold load": "Cold load",
    "vLLM L1 First": "vLLM L1 first",
    "vLLM L1 Steady": "vLLM L1 steady",
    "vLLM L2 Cold": "vLLM L2 cold",
    "vLLM L2 Warm": "vLLM L2 warm",
    "CPU backup First": "vllm-switch CPU first",
    "CPU backup Steady": "vllm-switch CPU steady",
    "Exact disk First": "vllm-switch disk first",
    "Exact disk Steady": "vllm-switch disk steady",
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
SLEEP_PHASE_ORDER = (
    "Process shutdown",
    "CPU backup allocation",
    "GPU→CPU copy",
    "GPU unmap + release",
    "Control overhead",
)
WAKE_PHASE_ORDER = (
    "GPU remap",
    "CPU-GPU copy",
    "Checkpoint load",
    "Control overhead",
)
DISPLAY_PHASE_ORDER = {
    "sleep": SLEEP_PHASE_ORDER,
    "wake": WAKE_PHASE_ORDER,
}
WAKE_PHASE_LABELS = {
    "CPU→GPU copy": "CPU-GPU copy",
    "KV-cache remap": "GPU remap",
    "Disk read + hash + H2D pipeline": "Checkpoint load",
}
LEGEND_PHASE_ORDER = (
    "Process shutdown",
    "CPU backup allocation",
    "GPU→CPU copy",
    "GPU unmap + release",
    "GPU remap",
    "CPU-GPU copy",
    "Checkpoint load",
    "Control overhead",
)
PHASE_COLORS = {
    "Process shutdown": "#374151",
    "CPU backup allocation": "#F0E442",
    "GPU→CPU copy": "#D55E00",
    "GPU unmap + release": "#8A9A2A",
    "Process + engine startup": "#6B7280",
    "CPU→GPU copy": "#E69F00",
    "CPU-GPU copy": "#E69F00",
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
    expected_count = int(document.get("frozen_scope", {}).get("sample_count_per_method", 3))
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
        "schema_version": 3,
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


def _display_phases(phases: Mapping[str, Any], operation: str) -> dict[str, float]:
    labels = WAKE_PHASE_LABELS if operation == "wake" else {}
    displayed: dict[str, float] = {}
    for phase, value in phases.items():
        label = labels.get(phase, phase)
        displayed[label] = displayed.get(label, 0.0) + float(value)
    return displayed


def _plotted_rows(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in summary["methods"] if row["method"] != "Cold load"]


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
    displayed_phases = [
        _display_phases(row[operation]["representative_phases_s"], operation) for row in rows
    ]
    for phase in DISPLAY_PHASE_ORDER[operation]:
        values = np.asarray([float(phases.get(phase, 0.0)) * scale for phases in displayed_phases])
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
        spread = float(row[operation]["max_s"]) - float(row[operation]["median_s"])
        lower_spread = float(row[operation]["median_s"]) - float(row[operation]["min_s"])
        large_spread = max(spread, lower_spread) > max(0.04, value * 0.15)
        ax.annotate(
            label,
            (position, top),
            xytext=(-8 if large_spread else 0, 4),
            textcoords="offset points",
            ha="right" if large_spread else "center",
            va="bottom",
            fontsize=6.5,
            bbox=(
                {"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4}
                if large_spread
                else None
            ),
        )
    ax.set_title(title, y=1.07)
    ax.set_xticks(x, [str(row["method"]) for row in rows], rotation=15, ha="right")
    unit = "s" if seconds else "ms"
    ax.set_ylabel(f"{operation.title()} latency ({unit})")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(bottom=0)


def plot_profiles(summary: Mapping[str, Any], output_base: Path) -> list[Path]:
    """Plot the in-process sleep and wake phase breakdowns."""

    apply_paper_style()
    rows = _plotted_rows(summary)
    fig, (sleep_ax, wake_ax) = plt.subplots(2, 1, figsize=(10.2, 7.0))
    _draw_panel(
        sleep_ax,
        rows,
        operation="sleep",
        seconds=False,
        title="(a) Sleep Latency Profiling",
    )
    _draw_panel(
        wake_ax,
        rows,
        operation="wake",
        seconds=False,
        title="(b) Wake Latency Profiling",
    )

    wake_handles, wake_labels = wake_ax.get_legend_handles_labels()
    sleep_handles, sleep_labels = sleep_ax.get_legend_handles_labels()
    by_label = dict(
        zip(
            sleep_labels + wake_labels,
            sleep_handles + wake_handles,
            strict=False,
        )
    )
    ordered = [phase for phase in LEGEND_PHASE_ORDER if phase in by_label]
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
    fig.tight_layout(rect=(0, 0.035, 1, 0.87), h_pad=2.5)
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
