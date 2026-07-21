from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_request_driven_switch import failed_record
from benchlib.request_trace import REQUIRED_FIELDS, load_manifest


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
    return not failed_record(row)


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


def request_identity_matches(
    expected: dict[str, Any], observed: dict[str, Any]
) -> bool:
    full_fields = tuple(sorted(REQUIRED_FIELDS))
    if all(field in observed for field in full_fields):
        return tuple(expected.get(field) for field in full_fields) == tuple(
            observed.get(field) for field in full_fields
        )
    return (
        expected.get("request_id"),
        expected.get("model"),
        float(expected.get("scheduled_offset_s", math.nan)),
    ) == (
        observed.get("request_id"),
        observed.get("model"),
        float(observed.get("scheduled_offset_s", math.nan)),
    )


def validate_trace_matrix(path: Path) -> list[dict[str, Any]]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    matrix = json.loads((path / "matrix.json").read_text(encoding="utf-8"))
    if int(metadata["repeats"]) <= 0:
        raise ValueError("metadata repeats must be positive")

    manifests: dict[str, tuple[Path, str, list[dict[str, Any]]]] = {}
    for item in metadata["manifests"]:
        repo_root = Path(__file__).resolve().parents[2]
        relative_path = item.get("repo_relative_path")
        manifest_path = repo_root / relative_path if relative_path else Path(item["path"])
        if not manifest_path.exists() and Path(item["path"]).exists():
            manifest_path = Path(item["path"])
        if not manifest_path.is_absolute():
            manifest_path = path / manifest_path
        rows = load_manifest(manifest_path)
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"manifest checksum mismatch: {manifest_path}")
        name = manifest_path.name
        if name in manifests:
            raise ValueError(f"duplicate manifest name: {name}")
        manifests[name] = (manifest_path, digest, rows)

    system_names = [str(system["name"]) for system in metadata["systems"]]
    if len(system_names) != len(set(system_names)):
        raise ValueError("duplicate system name in metadata")
    expected = {
        (system, manifest, repeat)
        for system in system_names
        for manifest in manifests
        for repeat in range(int(metadata["repeats"]))
    }
    observed: set[tuple[str, str, int]] = set()
    seen_outputs: set[Path] = set()
    validated: list[dict[str, Any]] = []
    for run in matrix:
        key = (str(run["system"]), str(run["manifest"]), int(run["repeat"]))
        if key in observed:
            raise ValueError(f"duplicate matrix run: {key}")
        observed.add(key)
        if key not in expected:
            raise ValueError(f"unexpected matrix run: {key}")
        if int(run.get("return_code", 1)) != 0:
            raise ValueError(f"nonzero harness return code: {key}")
        manifest_path, manifest_sha, expected_rows = manifests[key[1]]
        if run.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"run manifest checksum mismatch: {key}")
        output = (path / run["output"]).resolve()
        if output in seen_outputs:
            raise ValueError(f"duplicate output path: {output}")
        seen_outputs.add(output)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        if digest != run.get("output_sha256"):
            raise ValueError(f"checksum mismatch: {output}")
        rows = [json.loads(line) for line in output.read_text().splitlines() if line]
        if len(rows) != len(expected_rows) or int(run.get("rows", -1)) != len(rows):
            raise ValueError(f"row count mismatch: {key}")
        for index, (expected_row, row) in enumerate(
            zip(expected_rows, rows, strict=True)
        ):
            if not request_identity_matches(expected_row, row):
                raise ValueError(f"request identity mismatch: {key} row {index}")
        validated.append({"run": run, "rows": rows, "manifest_path": manifest_path})
    if observed != expected:
        missing = sorted(expected - observed)
        raise ValueError(f"matrix incomplete; missing={missing}")
    return validated


def summarize_trace_dir(path: Path, seed: int, samples: int) -> dict[str, Any]:
    validated = validate_trace_matrix(path)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in validated:
        run = item["run"]
        groups[(run["system"], run["manifest"])].append(item)
    result: dict[str, Any] = {}
    for (system, manifest), runs in sorted(groups.items()):
        cell_key = json.dumps([system, manifest], separators=(",", ":"))
        if cell_key in result:
            raise ValueError(f"duplicate trace cell: {(system, manifest)}")
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
        result[cell_key] = {
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


def lifecycle_group_name(path: Path) -> str:
    rows = json.loads(path.read_text(encoding="utf-8"))
    models = {str(row["model"]) for row in rows}
    if len(models) != 1:
        raise ValueError(f"lifecycle summary must contain exactly one model: {path}")
    return Path(next(iter(models))).name


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
            lifecycle_group_name(Path(value)): summarize_lifecycle(
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
