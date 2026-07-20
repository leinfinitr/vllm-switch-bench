from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(root: Path) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("w[012]-r*.jsonl")):
        workload = path.name.split("-", 1)[0]
        grouped[workload].extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    summary: dict[str, Any] = {}
    for workload, rows in grouped.items():
        ttft = [float(row["semantic_ttft_ms"]) for row in rows if row.get("semantic_ttft_ms")]
        latency = [
            float(row["completion_latency_ms"])
            for row in rows
            if row.get("completion_latency_ms")
        ]
        successful = [
            row
            for row in rows
            if row.get("status")
            and int(row["status"]) < 400
            and not row.get("error")
            and row.get("stream_done", True)
        ]
        summary[workload] = {
            "requests": len(rows),
            "success": len(successful),
            "failed": len(rows) - len(successful),
            "semantic_ttft_ms": {
                "median": median(ttft) if ttft else None,
                "p95": percentile(ttft, 0.95),
            },
            "completion_latency_ms": {
                "median": median(latency) if latency else None,
                "p95": percentile(latency, 0.95),
            },
        }
    return summary


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return {
        "median": median(values) if values else None,
        "p95": percentile(values, 0.95),
    }


def summarize_controller_switches(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requests = [row for row in rows if str(row.get("path", "")).startswith("/v1/")]
    switches = [row for row in requests if row.get("switch_needed") is True]
    return {
        "requests": len(requests),
        "switches": len(switches),
        "steady_hits": sum(row.get("switch_needed") is False for row in requests),
        "switch_latency_ms": _distribution(switches, "switch_latency_ms"),
        "sleep_latency_ms": _distribution(switches, "sleep_latency_ms"),
        "wake_latency_ms": _distribution(switches, "wake_latency_ms"),
        "request_drain_ms": _distribution(switches, "request_drain_ms"),
        "queue_wait_ms": _distribution(requests, "queue_wait_ms"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize request switch matrix JSONL")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--controller-events")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = summarize(Path(args.input_dir))
    if args.controller_events:
        data["controller"] = summarize_controller_switches(Path(args.controller_events))
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
