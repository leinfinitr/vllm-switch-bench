from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("system", "unknown"), row.get("method", "unknown"), row.get("prompt_name", "unknown"))].append(row)
    return grouped


def _fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _extract_latency(row: dict[str, Any], key: str) -> float | None:
    value = (row.get(key) or {}).get("latency_s")
    if value is None:
        return None
    return float(value)


def _extract_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def build_report(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    grouped = group_rows(rows)
    lines = [
        "# Baseline3 report",
        "",
        f"Model: {metadata.get('model', '-')}",
        "",
        "## Aggregated rows",
        "",
        "| system | method | prompt | n | ok_runs | mean_startup_s | mean_restore_s | mean_evict_s | estimated_restore_runs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (system, method, prompt), items in sorted(grouped.items()):
        ok_items = [r for r in items if r.get("ok")]
        startup_vals = [lat for lat in (_extract_float(r, "startup_latency_s") for r in ok_items) if lat is not None]
        restore_vals = [lat for lat in (_extract_latency(r, "restore") for r in ok_items) if lat is not None]
        evict_vals = [lat for lat in (_extract_latency(r, "evict") for r in ok_items) if lat is not None]
        estimated = sum(1 for r in ok_items if r.get("restore_latency_estimated"))
        lines.append(
            f"| {system} | {method} | {prompt} | {len(items)} | {len(ok_items)} | "
            f"{_fmt_float(mean(startup_vals) if startup_vals else None)} | "
            f"{_fmt_float(mean(restore_vals) if restore_vals else None)} | "
            f"{_fmt_float(mean(evict_vals) if evict_vals else None)} | {estimated} |"
        )

    if any(r.get("restore_latency_estimated") for r in rows):
        lines.extend([
            "",
            "Note: rows with restore_latency_estimated=True estimate restore latency as first post-evict request latency minus second active request latency.",
        ])

    unsupported = [r for r in rows if r.get("unsupported")]
    lines.extend(["", "## Unsupported / blocked rows", ""])
    if unsupported:
        for row in unsupported:
            lines.append(f"- {row.get('system')} / {row.get('method')} / {row.get('prompt_name')}: {row.get('error')}")
    else:
        lines.append("- None")

    lines.extend(["", "## Stage breakdown excerpts", ""])
    for row in rows:
        stage = row.get("stage_breakdown") or {}
        if stage:
            lines.append(f"- {row.get('system')} / {row.get('method')} / {row.get('prompt_name')}: {json.dumps(stage, ensure_ascii=False, sort_keys=True)}")
    if not any(row.get("stage_breakdown") for row in rows):
        lines.append("- None")

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    rows = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    report = build_report(rows, metadata)
    Path(args.out).write_text(report, encoding="utf-8")
    print(args.out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
