#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "osdi_20260723"
RAW = RESULT / "raw"
FIG = RESULT / "figures"
MODEL_ORDER = ["qwen-0.5b", "qwen-1.5b", "qwen-3b"]
MODEL_TITLES = {
    "qwen-0.5b": "Qwen2.5-0.5B",
    "qwen-1.5b": "Qwen2.5-1.5B",
    "qwen-3b": "Qwen2.5-3B",
}
SYSTEM_ORDER = ["Proposed", "vLLM L1", "vLLM L2", "SwapServeLLM", "llama-swap"]
SYSTEM_STYLE = {
    "Proposed": {"color": "#0072B2", "hatch": "", "marker": "o"},
    "vLLM L1": {"color": "#E69F00", "hatch": "//", "marker": "s"},
    "vLLM L2": {"color": "#CC79A7", "hatch": "..", "marker": "D"},
    "SwapServeLLM": {"color": "#56B4E9", "hatch": "\\\\", "marker": "P"},
    "llama-swap": {"color": "#009E73", "hatch": "xx", "marker": "^"},
}


def json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def lifecycle_samples() -> tuple[dict[tuple[str, str, str], list[float]], dict[str, Any]]:
    values: dict[tuple[str, str, str], list[float]] = {}
    provenance: dict[str, Any] = {}
    for system, directory in (("Proposed", "proposed"), ("vLLM L1", "vllm-stock")):
        for model in MODEL_ORDER:
            data = json_file(RAW / directory / f"{model}.json")
            provenance[f"{system}:{model}"] = data["environment"]["vllm"]
            values[(system, model, "sleep")] = [float(row["sleep_s"]) for row in data["rows"]]
            values[(system, model, "wake")] = [float(row["wake_s"]) for row in data["rows"]]
    llama = json_file(RAW / "llama-swap" / "lifecycle.json")
    provenance["llama-swap"] = llama["llama_swap_repo"]
    for model in MODEL_ORDER:
        rows = [row for row in llama["rows"] if row["model"] == model and row["ok"]]
        values[("llama-swap", model, "sleep")] = [
            float(
                row["sleep"].get(
                    "state_machine_latency_s", row["sleep"]["latency_s"]
                )
            )
            for row in rows
        ]
        values[("llama-swap", model, "wake")] = [
            float(
                row["wake"].get(
                    "state_machine_latency_s", row["wake"]["latency_s"]
                )
            )
            for row in rows
        ]
        l2_path = sorted((RAW / "vllm-l2-runs" / model).glob("*/summary.json"))[-1]
        l2_rows = [row for row in json_file(l2_path) if row["ok"]]
        values[("vLLM L2", model, "sleep")] = [
            float(row["evict"]["latency_s"]) for row in l2_rows
        ]
        values[("vLLM L2", model, "wake")] = [
            float(row["restore"]["latency_s"]) for row in l2_rows
        ]
        provenance[f"vLLM L2:{model}"] = json_file(
            l2_path.with_name("metadata.json")
        )["engine_git"]
        swapserve = json_file(RAW / "swapserve" / f"{model}.json")
        values[("SwapServeLLM", model, "sleep")] = [
            float(row["sleep_s"]) for row in swapserve["rows"]
        ]
        values[("SwapServeLLM", model, "wake")] = [
            float(row["wake_s"]) for row in swapserve["rows"]
        ]
    provenance["SwapServeLLM"] = {
        "path": "/home/ljl/research-systems/SwapServeLLM",
        "commit": "69f8aec0b11e49124f70754dc5149c36fd8327a5",
        "benchmark_patch": "raw/swapserve/benchmark.patch",
    }
    provenance["ServerlessLLM"] = json_file(RAW / "serverless" / "status.json")
    return values, provenance


def median_iqr(samples: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(samples)
    med = float(np.median(arr))
    return med, med - float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75)) - med


def plot_lifecycle(values: dict[tuple[str, str, str], list[float]]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for model in MODEL_ORDER:
        fig, ax = plt.subplots(figsize=(3.25, 2.15), constrained_layout=True)
        x = np.arange(2)
        offsets = dict(zip(SYSTEM_ORDER, np.linspace(-0.28, 0.28, len(SYSTEM_ORDER))))
        for system in SYSTEM_ORDER:
            meds, lows, highs = [], [], []
            for phase in ("sleep", "wake"):
                med, low, high = median_iqr(values[(system, model, phase)])
                meds.append(med)
                lows.append(low)
                highs.append(high)
            style = SYSTEM_STYLE[system]
            ax.errorbar(
                x + offsets[system],
                meds,
                yerr=np.asarray([lows, highs]),
                color=style["color"],
                marker=style["marker"],
                linestyle="none",
                markeredgecolor="black",
                markeredgewidth=0.45,
                capsize=2.3,
                elinewidth=0.8,
                label=system,
            )
        ax.set_yscale("log")
        ax.set_xticks(x, ["Sleep", "Wake"])
        ax.set_ylabel("Latency (s, log scale)")
        ax.text(
            0.02,
            0.96,
            MODEL_TITLES[model],
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
        )
        ax.grid(axis="y", which="major", color="#b0b0b0", linewidth=0.45, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.7, handlelength=1.1)
        for suffix in ("pdf", "png"):
            fig.savefig(FIG / f"lifecycle-{model}.{suffix}", bbox_inches="tight")
        plt.close(fig)


def plot_e2e() -> dict[str, Any]:
    paths = {
        "Proposed": RAW / "proposed" / "e2e-alternating.json",
        "llama-swap": RAW / "llama-swap" / "e2e-alternating.json",
    }
    rows = {system: json_file(path) for system, path in paths.items()}
    fig, ax = plt.subplots(figsize=(3.35, 2.05), constrained_layout=True)
    for system in ("Proposed", "llama-swap"):
        y = [float(row["completion_latency_ms"]) / 1000 for row in rows[system]]
        x = np.arange(1, len(y) + 1)
        style = SYSTEM_STYLE[system]
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            markevery=1,
            linestyle="-" if system == "Proposed" else "--",
            markersize=3.7,
            label=system,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Request sequence number")
    ax.set_ylabel("Request latency (s, log scale)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.6, 20.4)
    ax.grid(axis="both", color="#b0b0b0", linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for suffix in ("pdf", "png"):
        fig.savefig(FIG / f"e2e-alternating-request-latency.{suffix}", bbox_inches="tight")
    plt.close(fig)
    return {
        system: {
            "requests": len(system_rows),
            "failed": sum(bool(row.get("error")) or int(row.get("status", 0)) != 200 for row in system_rows),
            "median_s": statistics.median(float(row["completion_latency_ms"]) / 1000 for row in system_rows),
            "min_s": min(float(row["completion_latency_ms"]) / 1000 for row in system_rows),
            "max_s": max(float(row["completion_latency_ms"]) / 1000 for row in system_rows),
        }
        for system, system_rows in rows.items()
    }


def write_summary(values: dict[tuple[str, str, str], list[float]], provenance: dict[str, Any], e2e: dict[str, Any]) -> None:
    table_rows = []
    for model in MODEL_ORDER:
        for system in SYSTEM_ORDER:
            for phase in ("sleep", "wake"):
                samples = values[(system, model, phase)]
                med, low, high = median_iqr(samples)
                table_rows.append(
                    {
                        "model": model,
                        "system": system,
                        "phase": phase,
                        "n": len(samples),
                        "median_s": med,
                        "q1_s": med - low,
                        "q3_s": med + high,
                    }
                )
    RESULT.mkdir(parents=True, exist_ok=True)
    with (RESULT / "lifecycle-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(table_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(table_rows)
    summary = {
        "lifecycle": table_rows,
        "e2e": e2e,
        "provenance": provenance,
        "plot_policy": {
            "lifecycle": "median points with IQR; log y-axis; five state-machine samples per system/model",
            "e2e": "one fresh 20-request open-loop alternating trace per system; completion latency; 1.5 s scheduled spacing",
            "style": "single-column OSDI-like dimensions; serif fonts; color-blind-safe palette; vector PDF and 300-DPI PNG",
        },
    }
    (RESULT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    configure()
    values, provenance = lifecycle_samples()
    plot_lifecycle(values)
    e2e = plot_e2e()
    write_summary(values, provenance, e2e)
    print(RESULT / "summary.json")


if __name__ == "__main__":
    main()
