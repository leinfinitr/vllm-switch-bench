from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

SLEEP_COMPONENTS = [
    ("sleep_allocator_cpu_backup_alloc_s", "CPU backup alloc", "#4E79A7"),
    ("sleep_allocator_copy_d2h_s", "D2H copy", "#F28E2B"),
    ("sleep_allocator_unmap_release_s", "unmap & release", "#59A14F"),
    ("sleep_other_s", "sleep other", "#B07AA1"),
]
LINESTYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v")
LINE_COLORS = ("#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#8C564B")


def parse_float(value: Any) -> float:
    if value in (None, "", "None"):
        return 0.0
    return float(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        row["step_index"] = int(row["step_index"])
        row["sleep_latency_s"] = parse_float(row.get("sleep_latency_s"))
        row["infer_latency_s"] = parse_float(row.get("infer_latency_s"))
        for key, _, _ in SLEEP_COMPONENTS:
            if key != "sleep_other_s":
                row[key] = parse_float(row.get(key))
        accounted = sum(
            float(row[key]) for key, _, _ in SLEEP_COMPONENTS if key != "sleep_other_s"
        )
        row["sleep_other_s"] = max(0.0, float(row["sleep_latency_s"]) - accounted)
    return sorted(rows, key=lambda row: row["step_index"])


def configure_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render(rows: list[dict[str, Any]], out_path: Path, title: str) -> Path:
    configure_paper_style()
    model_names = sorted({str(row["model_name"]) for row in rows})
    model_index = {name: index for index, name in enumerate(model_names)}
    step_indices = [row["step_index"] for row in rows]
    fig, ax_sleep = plt.subplots(figsize=(7.2, 2.85))
    ax_infer = ax_sleep.twinx()

    bottoms = [0.0] * len(rows)
    width = 0.68
    for key, label, color in SLEEP_COMPONENTS:
        values = [float(row[key]) for row in rows]
        for idx, (row, value) in enumerate(zip(rows, values)):
            ax_sleep.bar(
                row["step_index"],
                value,
                width=width,
                bottom=bottoms[idx],
                color=color,
                edgecolor="black"
                if model_index[str(row["model_name"])] % 2
                else "white",
                linewidth=0.45,
                hatch="///" if model_index[str(row["model_name"])] % 2 else None,
                zorder=2,
            )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model_name"])].append(row)
    for model, model_rows in sorted(by_model.items()):
        index = model_index[model]
        ax_infer.plot(
            [row["step_index"] for row in model_rows],
            [row["infer_latency_s"] for row in model_rows],
            color=LINE_COLORS[index % len(LINE_COLORS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            marker=MARKERS[index % len(MARKERS)],
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=f"{model} infer",
            zorder=4,
        )

    ax_sleep.set_xlabel("step index")
    ax_sleep.set_ylabel("sleep latency (s)")
    ax_infer.set_ylabel("infer latency (s)")
    ax_sleep.set_xticks(step_indices)
    ax_sleep.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_sleep.grid(axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.7, zorder=0)
    ax_sleep.set_axisbelow(True)
    ax_sleep.set_xlim(min(step_indices) - 0.7, max(step_indices) + 0.7)
    ax_sleep.set_ylim(bottom=0)
    ax_infer.set_ylim(bottom=0)
    ax_sleep.set_title(title)

    for ax in (ax_sleep, ax_infer):
        ax.spines["top"].set_visible(False)
    ax_sleep.spines["right"].set_visible(False)
    ax_infer.spines["left"].set_visible(False)

    sleep_handles = [
        Patch(facecolor=color, edgecolor="white", label=label)
        for _, label, color in SLEEP_COMPONENTS
    ]
    line_handles, line_labels = ax_infer.get_legend_handles_labels()
    handles = sleep_handles + line_handles
    labels: list[str] = [str(handle.get_label()) for handle in sleep_handles]
    labels.extend(str(label) for label in line_labels)
    ax_sleep.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.34),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot repeated sleep-pool breakdown and inference latency."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/reports/figures/phase1-two-model-pool-breakdown.pdf"),
    )
    parser.add_argument(
        "--title", default="Repeated sleep backup pool: breakdown and inference latency"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.csv)
    out_path = render(rows, args.out, args.title)
    print(out_path)
    print(out_path.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
