from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_summary(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    proposed = summary
    labels = ["W0 steady", "W1 alternating", "W2 burst"]
    ttft = [proposed[key]["semantic_ttft_ms"]["median"] for key in ("w0", "w1", "w2")]
    ttft_p95 = [proposed[key]["semantic_ttft_ms"]["p95"] for key in ("w0", "w1", "w2")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(labels, ttft, color="#4C78A8", label="median")
    ax.errorbar(
        labels,
        ttft,
        yerr=[[0] * len(ttft), [p - m for p, m in zip(ttft_p95, ttft)]],
        fmt="none",
        color="black",
        capsize=4,
        label="median to p95",
    )
    ax.bar_label(bars, fmt="%.1f", label_type="edge", padding=2, color="#1f1f1f", fontsize=8)
    for x, (median_value, p95_value) in enumerate(zip(ttft, ttft_p95)):
        horizontal = 20 if p95_value - median_value < 20 else 5
        ax.annotate(
            f"p95 {p95_value:.1f}",
            (x, p95_value),
            xytext=(horizontal, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_ylabel("Semantic TTFT (ms)")
    ax.set_title("Request-visible latency by frozen workload")
    ax.legend()
    outputs.append(_save(fig, output_dir / "request-workloads.png"))

    controller = summary["controller"]
    fields = [("sleep", "sleep_latency_ms"), ("wake", "wake_latency_ms"), ("drain", "request_drain_ms")]
    values = [controller[field]["median"] for _, field in fields]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar([name for name, _ in fields], values, color=["#F58518", "#54A24B", "#E45756"])
    ax.bar_label(bars, fmt="%.2f")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Controller switch-path decomposition")
    outputs.append(_save(fig, output_dir / "switch-breakdown.png"))

    ablation = summary["profile_ablation"]
    names: list[str] = []
    values = []
    for model, data in ablation.items():
        names.extend([f"{model}\nfirst miss", f"{model}\nclean reuse", f"{model}\nwake"])
        values.extend([data["first_miss"]["latency_s"] * 1000, data["clean_reuse_latency_s_median"] * 1000, data["wake_latency_s_median"] * 1000])
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bars = ax.bar(names, values, color=["#E45756", "#72B7B2", "#54A24B"] * len(ablation))
    ax.bar_label(bars, fmt="%.0f", fontsize=8)
    ax.set_ylabel("Allocator phase latency (ms)")
    ax.set_title("Pinned-backup first miss, clean reuse, and wake")
    outputs.append(_save(fig, output_dir / "backup-ablation.png"))

    pressure = summary["pressure"]["p1"]
    rss = min(pressure["client_rss_delta_bytes"].values())
    mem = pressure["memavailable_delta_bytes"]
    values_gib = [-rss / 2**30, mem / 2**30]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(["Worker RSS decrease", "MemAvailable increase"], values_gib, color=["#B279A2", "#59A14F"])
    ax.bar_label(bars, fmt="%.2f GiB")
    ax.set_ylabel("Physical memory change (GiB)")
    ax.set_title("Controlled CPU backup release")
    outputs.append(_save(fig, output_dir / "physical-reclaim.png"))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot request-driven switch summary")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    outputs = plot_summary(summary, Path(args.output_dir))
    print(json.dumps([str(path) for path in outputs]))


if __name__ == "__main__":
    main()
