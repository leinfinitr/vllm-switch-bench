from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(
    values: list[float], statistic: Callable[[list[float]], float], seed: int, samples: int
) -> list[float] | None:
    if not values:
        return None
    generator = random.Random(seed)
    boot = [
        statistic([generator.choice(values) for _ in range(len(values))])
        for _ in range(samples)
    ]
    low = percentile(boot, 0.025)
    high = percentile(boot, 0.975)
    assert low is not None and high is not None
    return [low, high]


def success(row: dict[str, Any]) -> bool:
    status = row.get("status")
    return bool(
        status is not None
        and 200 <= int(status) < 300
        and not row.get("error")
        and row.get("stream_done") is True
    )


def summarize_lifecycle(path: Path, seed: int, samples: int) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["method"])].append(row)
    result: dict[str, Any] = {}
    for method, method_rows in sorted(groups.items()):
        good = [row for row in method_rows if row.get("ok") is True]
        restore_ms = [float(row["restore"]["latency_s"]) * 1000 for row in good]
        evict_ms = [float(row["evict"]["latency_s"]) * 1000 for row in good]
        activation_ms = [a + b for a, b in zip(evict_ms, restore_ms, strict=True)]
        result[method] = {
            "runs": len(method_rows),
            "success": len(good),
            "activation_ms": metric_summary(activation_ms, seed, samples),
            "evict_ms": metric_summary(evict_ms, seed + 1, samples),
            "restore_ms": metric_summary(restore_ms, seed + 2, samples),
            "gpu_ready_mib_median": statistics.median(
                float(row["memory_gpu_used_ready_mib"]) for row in good
            )
            if good
            else None,
            "gpu_evicted_mib_median": statistics.median(
                float(row["memory_gpu_used_evict_mib"]) for row in good
            )
            if good
            else None,
        }
    return result


def metric_summary(values: list[float], seed: int, samples: int) -> dict[str, Any]:
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "median_ci95": bootstrap_ci(values, statistics.median, seed, samples),
    }


def summarize_trace_dir(path: Path, seed: int, samples: int) -> dict[str, Any]:
    matrix = json.loads((path / "matrix.json").read_text(encoding="utf-8"))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in matrix:
        output = path / run["output"]
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        if digest != run["output_sha256"]:
            raise ValueError(f"checksum mismatch: {output}")
        rows = [json.loads(line) for line in output.read_text().splitlines() if line]
        groups[(run["system"], run["manifest"])].append(
            {"run": run, "rows": rows}
        )
    result: dict[str, Any] = {}
    for (system, manifest), runs in sorted(groups.items()):
        run_ttft: list[float] = []
        run_e2e: list[float] = []
        total = 0
        succeeded = 0
        pooled_ttft: list[float] = []
        pooled_e2e: list[float] = []
        for item in runs:
            rows = item["rows"]
            total += len(rows)
            good = [row for row in rows if success(row)]
            succeeded += len(good)
            ttft = [float(row["semantic_ttft_ms"]) for row in good]
            e2e = [float(row["completion_latency_ms"]) for row in good]
            pooled_ttft.extend(ttft)
            pooled_e2e.extend(e2e)
            if ttft:
                run_ttft.append(statistics.median(ttft))
            if e2e:
                run_e2e.append(statistics.median(e2e))
        result[f"{system}:{manifest}"] = {
            "runs": len(runs),
            "requests": total,
            "success": succeeded,
            "failure_rate": 1 - succeeded / total if total else None,
            "run_median_ttft_ms": metric_summary(run_ttft, seed, samples),
            "run_median_e2e_ms": metric_summary(run_e2e, seed + 1, samples),
            "pooled_ttft_ms": metric_summary(pooled_ttft, seed + 2, samples),
            "pooled_e2e_ms": metric_summary(pooled_e2e, seed + 3, samples),
        }
    return result


def trace_group_name(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifecycle", action="append", default=[])
    parser.add_argument("--trace-dir", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--external-systems")
    args = parser.parse_args()
    output = {
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "lifecycle": {
            Path(value).parent.name: summarize_lifecycle(
                Path(value), args.seed + index * 10, args.bootstrap_samples
            )
            for index, value in enumerate(args.lifecycle)
        },
        "traces": {
            trace_group_name(Path(value)): summarize_trace_dir(
                Path(value), args.seed + 1000 + index * 10, args.bootstrap_samples
            )
            for index, value in enumerate(args.trace_dir)
        },
    }
    if args.external_systems:
        output["external_systems"] = json.loads(
            Path(args.external_systems).read_text(encoding="utf-8")
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
