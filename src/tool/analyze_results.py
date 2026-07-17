#!/usr/bin/env python3
"""Summarize vLLM lifecycle benchmark output directories."""
from __future__ import annotations

import argparse
import csv

import statistics
from collections import defaultdict
from pathlib import Path



def fnum(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def stats(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"avg": None, "min": None, "max": None}
    return {"avg": statistics.mean(vals), "min": min(vals), "max": max(vals)}


def fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def load_rows(result_dir: Path) -> list[dict[str, str]]:
    with (result_dir / "summary.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))




def build_report(result_dir: Path, repo_root: Path) -> str:
    rows = load_rows(result_dir)

    by = defaultdict(list)
    for r in rows:
        by[(r["method"], r["prompt_name"], r["ok"])].append(r)

    lines = [
        "# Qwen2.5-0.5B vLLM lifecycle benchmark results",
        "",
        f"Result directory: `{result_dir}`",
        "",
        "## Summary by method / prompt / success",
        "",
        "| method | prompt | ok | n | startup avg s | evict avg s | restore avg s | TTFT before avg s | TTFT after avg s | latency before avg s | latency after avg s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metric_fields = [
        "startup_latency_s",
        "evict_latency_s",
        "restore_latency_s",
        "ttft_before_s",
        "ttft_after_s",
        "latency_before_s",
        "latency_after_s",
    ]
    for key in sorted(by):
        vs = by[key]
        agg = {field: stats([v for r in vs if (v := fnum(r.get(field))) is not None]) for field in metric_fields}
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(vs)} | "
            f"{fmt(agg['startup_latency_s']['avg'])} | {fmt(agg['evict_latency_s']['avg'])} | {fmt(agg['restore_latency_s']['avg'])} | "
            f"{fmt(agg['ttft_before_s']['avg'])} | {fmt(agg['ttft_after_s']['avg'])} | "
            f"{fmt(agg['latency_before_s']['avg'])} | {fmt(agg['latency_after_s']['avg'])} |"
        )

    lines += ["", "## Ready / evicted memory", "", "| method | prompt | ready gpu avg MiB | evicted gpu avg MiB | ready cpu avg MiB | evicted cpu avg MiB |", "|---|---|---:|---:|---:|---:|"]
    for key in sorted(by):
        vs = by[key]
        ready_gpu = stats([v for r in vs if (v := fnum(r.get("memory_gpu_used_ready_mib"))) is not None])
        evict_gpu = stats([v for r in vs if (v := fnum(r.get("memory_gpu_used_evict_mib"))) is not None])
        ready_cpu = stats([v for r in vs if (v := fnum(r.get("memory_cpu_used_ready_mib"))) is not None])
        evict_cpu = stats([v for r in vs if (v := fnum(r.get("memory_cpu_used_evict_mib"))) is not None])
        lines.append(
            f"| {key[0]} | {key[1]} | {fmt(ready_gpu['avg'])} | {fmt(evict_gpu['avg'])} | "
            f"{fmt(ready_cpu['avg'])} | {fmt(evict_cpu['avg'])} |"
        )

    failed = [r for r in rows if r.get("ok") != "True"]
    lines += ["", "## Failures", ""]
    if not failed:
        lines.append("No failed rows.")
    else:
        lines.append(f"Failed rows: {len(failed)}")
        lines.append("")
        causes = sorted({(r.get("error") or "see summary.json")[:300] for r in failed})
        lines.append("Observed failure snippets:")
        for cause in causes[:5]:
            lines.append(f"- `{cause}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("result_dir", type=Path)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    report = build_report(args.result_dir, args.repo_root)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
