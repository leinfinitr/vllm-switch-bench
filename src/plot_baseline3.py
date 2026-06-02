from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METHOD_SPECS: list[tuple[str, str, str]] = [
    ("vllm", "sleep_l1", "vLLM Sleep L1"),
    ("vllm", "sleep_l2", "vLLM Sleep L2"),
    ("swapserve_llm", "swapout_swapin", "SwapServeLLM"),
    ("serverless_llm", "scale_to_zero_restore", "ServerlessLLM"),
]
DEFAULT_LABELS: dict[tuple[str, str], str] = {
    (system, method): label for system, method, label in DEFAULT_METHOD_SPECS
}


NUMERIC_FIELDS = [
    "evict_latency_s",
    "restore_latency_s",
    "latency_before_s",
    "latency_after_s",
    "tokens_per_s_before",
    "tokens_per_s_after",
]


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _parse_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def load_summary_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed = dict(row)
            parsed["ok"] = _parse_bool(row.get("ok"))
            for field in NUMERIC_FIELDS:
                parsed[field] = _parse_float(row.get(field))
            rows.append(parsed)
    return rows


def parse_method_specs(method_specs: list[str] | None) -> list[tuple[str, str, str]]:
    if not method_specs:
        return DEFAULT_METHOD_SPECS
    parsed: list[tuple[str, str, str]] = []
    for spec in method_specs:
        parts = spec.split(":", 2)
        if len(parts) < 2:
            raise ValueError(f"invalid method spec: {spec!r}; expected system:method[:label]")
        system, method = parts[0], parts[1]
        label = parts[2] if len(parts) == 3 else f"{system}:{method}"
        parsed.append((system, method, label))
    return parsed


def aggregate_method_metrics(
    rows: list[dict[str, Any]],
    methods: list[tuple[str, str]] | list[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    if methods is None:
        specs = DEFAULT_METHOD_SPECS
    else:
        specs = [
            (item[0], item[1], item[2] if len(item) >= 3 else DEFAULT_LABELS.get((item[0], item[1]), f"{item[0]}:{item[1]}"))  # type: ignore[index]
            for item in methods
        ]

    aggregated: list[dict[str, Any]] = []
    for system, method, label in specs:
        selected = [row for row in rows if row.get("system") == system and row.get("method") == method and row.get("ok")]
        if not selected:
            continue
        entry: dict[str, Any] = {
            "system": system,
            "method": method,
            "label": label,
            "count": len(selected),
        }
        for field in NUMERIC_FIELDS:
            values = [row[field] for row in selected if row.get(field) is not None]
            entry[field] = mean(values) if values else None
        aggregated.append(entry)
    return aggregated


def _ratio(after: float | None, before: float | None) -> float | None:
    if after is None or before in (None, 0):
        return None
    return after / before


def render_comparison_figure(aggregated: list[dict[str, Any]], out_path: str | Path, title: str = "Baseline3 switch comparison") -> Path:
    if not aggregated:
        raise ValueError("no aggregated rows to plot")

    labels = [row["label"] for row in aggregated]
    x = list(range(len(labels)))
    width = 0.36

    evict = [row.get("evict_latency_s") for row in aggregated]
    restore = [row.get("restore_latency_s") for row in aggregated]
    latency_before = [row.get("latency_before_s") for row in aggregated]
    latency_after = [row.get("latency_after_s") for row in aggregated]
    tps_before = [row.get("tokens_per_s_before") for row in aggregated]
    tps_after = [row.get("tokens_per_s_after") for row in aggregated]
    latency_ratio = [_ratio(a, b) for a, b in zip(latency_after, latency_before)]
    throughput_ratio = [_ratio(a, b) for a, b in zip(tps_after, tps_before)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), constrained_layout=False)
    fig.suptitle(title)

    ax = axes[0][0]
    ax.bar([i - width / 2 for i in x], evict, width=width, label="evict", color="#4E79A7")
    ax.bar([i + width / 2 for i in x], restore, width=width, label="restore", color="#F28E2B")
    ax.set_title("Switch overhead")
    ax.set_ylabel("seconds")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.legend()

    ax = axes[0][1]
    ax.bar([i - width / 2 for i in x], latency_before, width=width, label="before switch", color="#59A14F")
    ax.bar([i + width / 2 for i in x], latency_after, width=width, label="after switch", color="#E15759")
    ax.set_title("Inference client latency")
    ax.set_ylabel("seconds")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.legend()

    ax = axes[1][0]
    ax.bar([i - width / 2 for i in x], tps_before, width=width, label="before switch", color="#76B7B2")
    ax.bar([i + width / 2 for i in x], tps_after, width=width, label="after switch", color="#EDC948")
    ax.set_title("Inference throughput")
    ax.set_ylabel("tokens / second")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.legend()

    ax = axes[1][1]
    ax.bar([i - width / 2 for i in x], latency_ratio, width=width, label="latency after / before", color="#B07AA1")
    ax.bar([i + width / 2 for i in x], throughput_ratio, width=width, label="throughput after / before", color="#FF9DA7")
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
    ax.set_title("Relative post-switch impact")
    ax.set_ylabel("ratio")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.legend()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Baseline3 run directory containing summary.csv")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Baseline3 switch comparison")
    parser.add_argument(
        "--method",
        dest="method_specs",
        action="append",
        help="Method selection in system:method[:label] form. Repeatable. Defaults to vLLM sleep_l1/sleep_l2 + SwapServeLLM + ServerlessLLM.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    rows = load_summary_rows(run_dir / "summary.csv")
    specs = parse_method_specs(args.method_specs)
    aggregated = aggregate_method_metrics(rows, methods=specs)
    out = render_comparison_figure(aggregated, args.out, title=args.title)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
