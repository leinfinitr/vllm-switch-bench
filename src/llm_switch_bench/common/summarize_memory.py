#!/usr/bin/env python3
"""Extract phase-specific memory snapshots from lifecycle event logs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PHASES = [
    "run_start",
    "api_ready",
    "infer_before_end",
    "evict_end",
    "restore_end",
    "infer_after_end",
    "run_end",
]


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.4f}"


def resolve_path(repo_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def event_by_name(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for ev in events:
        if ev.get("event") == name:
            return ev
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dir", type=Path)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    args = p.parse_args()

    rows = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    snapshots: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        events = load_events(resolve_path(args.repo_root, row["event_log"]))
        for phase in PHASES:
            ev = event_by_name(events, phase)
            if not ev:
                continue
            rec = {
                "run_id": row["run_id"],
                "method": row["method"],
                "prompt_name": row["prompt_name"],
                "phase": phase,
                "gpu_used_mib": ev.get("gpu_used_mib"),
                "cpu_used_mib": ev.get("cpu_used_mib"),
                "proc_rss_mib": ev.get("proc_rss_mib"),
                "proc_uss_mib": ev.get("proc_uss_mib"),
            }
            snapshots.append(rec)
            key = (row["method"], phase)
            for field in ["gpu_used_mib", "cpu_used_mib", "proc_rss_mib", "proc_uss_mib"]:
                if rec[field] is not None:
                    grouped[key][field].append(float(rec[field]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run_id",
                "method",
                "prompt_name",
                "phase",
                "gpu_used_mib",
                "cpu_used_mib",
                "proc_rss_mib",
                "proc_uss_mib",
            ],
        )
        writer.writeheader()
        writer.writerows(snapshots)

    lines = [
        "# Phase memory summary",
        "",
        f"Result directory: `{args.result_dir}`",
        "",
        "| method | phase | gpu used avg MiB | cpu used avg MiB | proc RSS avg MiB | proc USS avg MiB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in sorted({r["method"] for r in rows}):
        for phase in PHASES:
            vals = grouped.get((method, phase), {})
            lines.append(
                f"| {method} | {phase} | {fmt(mean(vals.get('gpu_used_mib', [])))} | {fmt(mean(vals.get('cpu_used_mib', [])))} | {fmt(mean(vals.get('proc_rss_mib', [])))} | {fmt(mean(vals.get('proc_uss_mib', [])))} |"
            )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_csv)
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
