from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lifecycle = summary["lifecycle"]
    model_keys = sorted(
        lifecycle,
        key=lambda key: min(method["gpu_ready_mib_median"] for method in lifecycle[key].values()),
    )
    if len(model_keys) != 2:
        raise ValueError("expected exactly two lifecycle model groups")
    labels = ["Cold process", "vLLM L1", "vLLM L2"]
    methods = ["cold_reload", "sleep_l1", "sleep_l2"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    markers = ["o", "s"]
    for index, key in enumerate(model_keys):
        values = [lifecycle[key][method]["activation_ms"]["median"] for method in methods]
        ci = [lifecycle[key][method]["activation_ms"]["median_ci95"] for method in methods]
        errors = np.array(
            [
                [value - bounds[0] for value, bounds in zip(values, ci, strict=True)],
                [bounds[1] - value for value, bounds in zip(values, ci, strict=True)],
            ]
        )
        offset = (index - 0.5) * 0.12
        ax.errorbar(
            x + offset,
            values,
            yerr=errors,
            fmt=markers[index],
            markersize=8,
            capsize=4,
            linewidth=1.6,
            label="1.5B" if index == 0 else "3B",
        )
        for xpos, value in zip(x + offset, values, strict=True):
            annotation_offset = 16 if index == 0 else 8
            ax.annotate(
                f"{value:.0f}",
                (xpos, value),
                xytext=(0, annotation_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.set_yscale("log")
    ax.set_ylim(600, 22000)
    ax.set_ylabel("Activation latency (ms, log scale)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "lifecycle-latency.png", dpi=200)
    plt.close(fig)

    workloads = ["alternating", "burst-local"]
    proposed = summary["traces"]["proposed/request-traces-final"]
    llama = summary["traces"]["llama-swap/request-traces-final"]
    systems = [
        (
            "Proposed L1 reuse",
            [
                proposed['["proposed","request-switch-alternating.jsonl"]']["run_median_ttft_ms"][
                    "median"
                ],
                proposed['["proposed","request-switch-burst.jsonl"]']["run_median_ttft_ms"][
                    "median"
                ],
            ],
        ),
        (
            "llama-swap cold process",
            [
                llama['["llama-swap","request-switch-alternating.jsonl"]']["run_median_ttft_ms"][
                    "median"
                ],
                llama['["llama-swap","request-switch-burst.jsonl"]']["run_median_ttft_ms"][
                    "median"
                ],
            ],
        ),
    ]
    x = np.arange(len(workloads))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for index, (label, values) in enumerate(systems):
        offset = (index - 0.5) * 0.12
        ax.plot(
            x + offset,
            values,
            markers[index],
            markersize=8,
            linestyle="none",
            label=label,
        )
        for xpos, value in zip(x + offset, values, strict=True):
            ax.annotate(
                f"{value:.0f}",
                (xpos, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.set_yscale("log")
    ax.set_ylim(10, 40000)
    ax.set_ylabel("Run-median semantic TTFT (ms, log scale)")
    ax.set_xticks(x, workloads)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "trace-ttft.png", dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
