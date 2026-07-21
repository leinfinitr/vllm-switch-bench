#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "results/model_switch_eval/latest"


def main() -> None:
    summary = json.loads((LATEST / "summary.json").read_text())
    systems = ["proposed", "vllm-stock-l1", "swapserve", "llama-swap", "serverlessllm"]
    labels = ["Proposed", "vLLM L1", "SwapServeLLM", "llama-swap*", "ServerlessLLM"]
    models = ["qwen-1.5b", "qwen-3b"]
    colors = ["#2878B5", "#9AC9DB"]
    fig, ax = plt.subplots(figsize=(9.3, 4.8))
    x = np.arange(len(systems))
    width = 0.36
    for i, model in enumerate(models):
        vals = []
        for system in systems:
            row = summary["lifecycle"].get(system, {}).get(model)
            vals.append(np.nan if row is None else row["switch_s"]["median"])
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=model, color=colors[i])
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width()/2, val),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Switch time = sleep + wake (s, log scale)")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Single-GPU model lifecycle comparison (median, steady-state)")
    ax.text(
        0.01,
        0.02,
        "llama-swap*: request-visible transition (not phase-decomposed); "
        "ServerlessLLM 3B: failed runnable gate; n=4–5 cycles/cell",
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(LATEST / "lifecycle-switch-time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    workloads = ["alternating", "burst"]
    x = np.arange(len(workloads))
    width = 0.32
    for i, system in enumerate(["proposed", "llama-swap"]):
        vals = [summary["e2e"][w][system]["elapsed_s"]["median"] for w in workloads]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label="Proposed" if system == "proposed" else "llama-swap")
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.1f}s", (bar.get_x()+bar.get_width()/2,val),
                        xytext=(0,4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("Trace makespan (s)")
    ax.set_xticks(x, ["Alternating", "Burst/locality"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("20-request self-routing workload")
    ax.set_ylim(0, 60)
    ax.text(
        0.01,
        0.98,
        "Proposed: n=2 alternating, n=1 burst; llama-swap: n=3 each",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(LATEST / "e2e-makespan.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
