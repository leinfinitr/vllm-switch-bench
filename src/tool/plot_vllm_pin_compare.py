from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLEEP_COMPONENTS = [
    ("cpu_backup_alloc_s_mean", "CPU backup alloc", "#4E79A7"),
    ("copy_d2h_s_mean", "D2H copy", "#F28E2B"),
    ("unmap_release_s_mean", "unmap/release", "#59A14F"),
    ("empty_cache_s_mean", "empty_cache", "#E15759"),
    ("sleep_other_s", "sleep other", "#B07AA1"),
]
WAKE_COMPONENTS = [
    ("copy_h2d_s_mean", "H2D copy", "#F28E2B"),
    ("create_map_s_mean", "create/map", "#76B7B2"),
    ("wake_other_s", "wake other", "#B07AA1"),
]


def parse_float(value: Any) -> float:
    if value in (None, "", "None"):
        return 0.0
    return float(value)


def metric(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def load_summary(path: Path, method: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            item = dict(row)
            item["method"] = item.get("method") or method
            for key, value in list(item.items()):
                if key.endswith("_mean") or key in {"gpu_memory_utilization", "n"}:
                    item[key] = parse_float(value)
            item["sleep_other_s"] = max(
                0.0,
                metric(item, "allocator_sleep_s_mean")
                - metric(item, "cpu_backup_alloc_s_mean")
                - metric(item, "copy_d2h_s_mean")
                - metric(item, "unmap_release_s_mean")
                - metric(item, "empty_cache_s_mean"),
            )
            item["wake_other_s"] = max(
                0.0,
                metric(item, "allocator_wake_s_mean")
                - metric(item, "copy_h2d_s_mean")
                - metric(item, "create_map_s_mean"),
            )
            rows.append(item)
        return rows


def load_all(sleep_l1_dir: Path, sleep_l2_dir: Path) -> list[dict[str, Any]]:
    return load_summary(sleep_l1_dir / "analysis_summary" / "summary.csv", "sleep_l1") + load_summary(
        sleep_l2_dir / "analysis_summary" / "summary.csv", "sleep_l2"
    )


def stack_bars(ax: Any, labels: list[str], rows: list[dict[str, Any]], components: list[tuple[str, str, str]]) -> None:
    bottoms = [0.0] * len(rows)
    x = list(range(len(rows)))
    for key, label, color in components:
        vals = [float(row.get(key, 0.0)) for row in rows]
        if max(vals, default=0.0) < 0.003:
            continue
        ax.bar(x, vals, bottom=bottoms, label=label, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("seconds")
    ax.grid(axis="y", alpha=0.25)


def render_model(model: str, rows: list[dict[str, Any]], out_path: Path) -> Path:
    ordered = []
    labels = []
    for method in ["sleep_l1", "sleep_l2"]:
        for pin in ["true", "false"]:
            match = next(row for row in rows if row["model"] == model and row["method"] == method and row["pin_memory"] == pin)
            ordered.append(match)
            labels.append(f"{method}\n{'pin' if pin == 'true' else 'no-pin'}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    fig.suptitle(f"vLLM Sleep/Wake breakdown: {model}")
    stack_bars(axes[0], labels, ordered, SLEEP_COMPONENTS)
    axes[0].set_title("Sleep / evict breakdown")
    stack_bars(axes[1], labels, ordered, WAKE_COMPONENTS)
    axes[1].set_title("Wake / restore breakdown")
    handles, labels_legend = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels_legend.extend(l)
    dedup = dict(zip(labels_legend, handles))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.30, wspace=0.22)
    fig.legend(dedup.values(), dedup.keys(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_all(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)
    outputs = []
    for model in sorted(by_model):
        outputs.append(render_model(model, by_model[model], out_dir / f"vllm-pin-compare-{model}.png"))
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot vLLM sleep_l1/sleep_l2 pin/no-pin breakdown figures.")
    parser.add_argument("--sleep-l1-dir", type=Path, default=Path("results/profiling/sleep_l1_pin_compare"))
    parser.add_argument("--sleep-l2-dir", type=Path, default=Path("results/profiling/sleep_l2_pin_compare"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/reports/figures"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_all(args.sleep_l1_dir, args.sleep_l2_dir)
    outputs = render_all(rows, args.out_dir)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
