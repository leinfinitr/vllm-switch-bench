from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLEEP_COMPONENTS = [
    ("cpu_backup_alloc_s_mean", "CPU backup alloc", "#4E79A7"),
    ("copy_d2h_s_mean", "D2H copy", "#F28E2B"),
    ("unmap_release_s_mean", "unmap & release", "#59A14F"),
    ("empty_cache_s_mean", "empty cache", "#E15759"),
    ("sleep_other_s", "sleep other", "#B07AA1"),
]
WAKE_COMPONENTS = [
    ("create_map_s_mean", "create map", "#76B7B2"),
    ("copy_h2d_s_mean", "H2D copy", "#F28E2B"),
    ("reload_weights_s_mean", "reload weights", "#EDC948"),
    ("wake_other_s", "restore other", "#B07AA1"),
]


def parse_float(value: Any) -> float:
    if value in (None, "", "None"):
        return 0.0
    return float(value)


def metric(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_restore_step_means(result_dir: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    step_means: dict[tuple[str, str, str], dict[str, float]] = {}
    for summary_path in result_dir.glob("qwen2p5_*/pin_*/*/summary.json"):
        model = summary_path.parts[-4]
        pin_memory = summary_path.parts[-3].replace("pin_", "")
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        if not rows:
            continue
        method = rows[0].get("method", "")
        reload_weights = []
        restore = []
        copy_h2d = []
        create_map = []
        profile_path = summary_path.with_name("sleep_profile_summary.csv")
        allocator_by_repeat: dict[int, dict[str, float]] = defaultdict(
            lambda: {"copy_h2d_s": 0.0, "create_map_s": 0.0}
        )
        if profile_path.exists():
            with profile_path.open("r", encoding="utf-8", newline="") as handle:
                for profile_row in csv.DictReader(handle):
                    if profile_row.get("phase") != "allocator_wake_up":
                        continue
                    operation = profile_row.get("operation") or ""
                    if not operation.startswith("wake"):
                        continue
                    repeat_index = int(profile_row.get("repeat_index") or 0)
                    allocator_by_repeat[repeat_index]["copy_h2d_s"] += parse_float(
                        profile_row.get("copy_h2d_s")
                    )
                    allocator_by_repeat[repeat_index]["create_map_s"] += parse_float(
                        profile_row.get("create_map_s")
                    )
        for row in rows:
            repeat_index = int(row.get("repeat_index") or 0)
            restore.append(parse_float((row.get("restore") or {}).get("latency_s")))
            steps = (row.get("restore") or {}).get("steps") or {}
            reload_weights.append(parse_float((steps.get("reload_weights") or {}).get("latency_s")))
            allocator = allocator_by_repeat.get(repeat_index, {})
            copy_h2d.append(float(allocator.get("copy_h2d_s", 0.0)))
            create_map.append(float(allocator.get("create_map_s", 0.0)))
        step_means[(model, method, pin_memory)] = {
            "restore_total_s_mean": mean(restore),
            "reload_weights_s_mean": mean(reload_weights),
            "copy_h2d_s_mean": mean(copy_h2d),
            "create_map_s_mean": mean(create_map),
        }
    return step_means


def load_summary(path: Path, method: str, step_means: dict[tuple[str, str, str], dict[str, float]] | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            item = dict(row)
            item["method"] = item.get("method") or method
            for key, value in list(item.items()):
                if key.endswith("_mean") or key in {"gpu_memory_utilization", "n"}:
                    item[key] = parse_float(value)
            extra = (step_means or {}).get((str(item["model"]), str(item["method"]), str(item["pin_memory"])), {})
            item["reload_weights_s_mean"] = extra.get("reload_weights_s_mean", 0.0)
            if extra:
                item["copy_h2d_s_mean"] = extra.get("copy_h2d_s_mean", metric(item, "copy_h2d_s_mean"))
                item["create_map_s_mean"] = extra.get("create_map_s_mean", metric(item, "create_map_s_mean"))
            restore_total_s = extra.get("restore_total_s_mean", metric(item, "restore_latency_s_mean"))
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
                restore_total_s
                - metric(item, "copy_h2d_s_mean")
                - metric(item, "create_map_s_mean")
                - metric(item, "reload_weights_s_mean"),
            )
            rows.append(item)
        return rows


def load_all(sleep_l1_dir: Path, sleep_l2_dir: Path) -> list[dict[str, Any]]:
    l1_steps = load_restore_step_means(sleep_l1_dir)
    l2_steps = load_restore_step_means(sleep_l2_dir)
    return load_summary(sleep_l1_dir / "analysis_summary" / "summary.csv", "sleep_l1", l1_steps) + load_summary(
        sleep_l2_dir / "analysis_summary" / "summary.csv", "sleep_l2", l2_steps
    )


def stack_bars(ax: Any, labels: list[str], rows: list[dict[str, Any]], components: list[tuple[str, str, str]]) -> None:
    bottoms = [0.0] * len(rows)
    x = list(range(len(rows)))
    for key, label, color in components:
        vals = [float(row.get(key, 0.0)) for row in rows]
        if max(vals, default=0.0) < 0.003:
            continue
        for index, val in enumerate(vals):
            ax.bar(
                x[index],
                val,
                bottom=bottoms[index],
                label=label if index == 0 else None,
                color=color,
                edgecolor="black" if rows[index]["pin_memory"] == "false" else "white",
                linewidth=0.6 if rows[index]["pin_memory"] == "false" else 0.3,
                hatch="//" if rows[index]["pin_memory"] == "false" else None,
            )
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("seconds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.25))


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
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.22, wspace=0.22)
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
    parser.add_argument("--out-dir", type=Path, default=Path("results/tmp/figures/vllm-pin-compare"))
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
