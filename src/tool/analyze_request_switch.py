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
        successful = [row for row in rows if row.get("status") and int(row["status"]) < 400]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize request switch matrix JSONL")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = summarize(Path(args.input_dir))
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
