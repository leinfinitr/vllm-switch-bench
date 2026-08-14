from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "vllm-switch": "#0072B2",
    "vLLM L1": "#E69F00",
    "vLLM L2": "#009E73",
    "SwapServeLLM": "#D55E00",
    "llama-swap": "#CC79A7",
    "exact-disk": "#56B4E9",
}
MARKERS = {
    "vllm-switch": "o",
    "vLLM L1": "s",
    "vLLM L2": "^",
    "SwapServeLLM": "D",
    "llama-swap": "v",
    "exact-disk": "P",
}
HATCHES = {
    "vllm-switch": "",
    "vLLM L1": "//",
    "vLLM L2": "\\\\",
    "SwapServeLLM": "xx",
    "llama-swap": "..",
    "exact-disk": "--",
}


def apply_paper_style() -> None:
    """Apply the repository's deterministic single-column paper style."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.figsize": (3.4, 2.15),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
        }
    )


def save_figure(fig: mpl.figure.Figure, output_base: Path) -> list[Path]:
    """Write vector PDF and 300-DPI PNG with deterministic PDF metadata."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf = output_base.with_suffix(".pdf")
    png = output_base.with_suffix(".png")
    metadata = {
        "Creator": "vllm-switch-bench",
        "Producer": "vllm-switch-bench",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, metadata=metadata)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return [pdf, png]


def system_color(system: str) -> str:
    return PALETTE.get(system, "#000000")


def system_marker(system: str) -> str:
    return MARKERS.get(system, "o")


def system_hatch(system: str) -> str:
    return HATCHES.get(system, "")


def ordered_legend(ax: mpl.axes.Axes, order: Iterable[str]) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ordered_labels = [label for label in order if label in by_label]
    if ordered_labels:
        ax.legend([by_label[label] for label in ordered_labels], ordered_labels, frameon=False)
