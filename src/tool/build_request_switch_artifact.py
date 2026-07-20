from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from analyze_request_switch import summarize, summarize_controller_switches


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def profile_summary(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    sleeps = [row for row in rows if row.get("phase") == "allocator_sleep"]
    wakes = [row for row in rows if row.get("phase") == "allocator_wake_up"]
    misses = [row for row in sleeps if float(row.get("copy_d2h_s", 0)) > 0]
    reuse = [
        row
        for row in sleeps
        if float(row.get("copy_d2h_s", 0)) == 0
        and int(row.get("cpu_backup_reused_bytes", 0)) > 0
    ]
    return {
        "first_miss": misses[0],
        "clean_reuse_count": len(reuse),
        "clean_reuse_latency_s_median": median(row["latency_s"] for row in reuse),
        "clean_reuse_copy_d2h_s_values": sorted({row["copy_d2h_s"] for row in reuse}),
        "clean_reuse_bytes_median": median(row["cpu_backup_reused_bytes"] for row in reuse),
        "wake_latency_s_median": median(row["latency_s"] for row in wakes),
        "wake_copy_h2d_s_median": median(row["copy_h2d_s"] for row in wakes),
        "total_sleep_events": len(sleeps),
        "total_wake_events": len(wakes),
        "reclaimed_miss_count": sum(
            int(row.get("cpu_backup_release_bytes", 0)) > 0
            and float(row.get("copy_d2h_s", 0)) > 0
            for row in sleeps
        ),
    }


def build_summary(input_dir: Path, provenance: Path) -> dict[str, Any]:
    summary = summarize(input_dir)
    summary["controller"] = summarize_controller_switches(input_dir / "controller-events.jsonl")
    metadata = load_json(input_dir / "metadata.json")
    summary["metadata"] = metadata
    manifests: dict[str, Any] = {}
    for run in metadata["runs"]:
        workload = run["workload"]
        if workload in manifests:
            continue
        manifest = input_dir / Path(run["manifest"]).name
        rows = load_jsonl(manifest)
        duration = float(rows[-1]["scheduled_offset_s"]) - float(rows[0]["scheduled_offset_s"])
        manifests[workload] = {
            "file": manifest.name,
            "sha256": run["manifest_sha256"],
            "requests": len(rows),
            "scheduled_duration_s": duration,
            "offered_rate_rps": (len(rows) - 1) / duration,
        }
    summary["manifests"] = manifests
    for workload, manifest in manifests.items():
        rates = []
        for run in metadata["runs"]:
            if run["workload"] != workload:
                continue
            rows = load_jsonl(input_dir / run["output"])
            start = min(row["client_dispatch_offset_s"] for row in rows)
            finish = max(
                row["client_dispatch_offset_s"] + row["completion_latency_ms"] / 1000
                for row in rows
            )
            rates.append((len(rows) - 1) / (finish - start))
        summary[workload].update(
            {
                "repeats": metadata["repeats"],
                "scheduled_duration_s": manifest["scheduled_duration_s"],
                "offered_rate_rps": manifest["offered_rate_rps"],
                "achieved_rate_rps_median": median(rates),
            }
        )
    summary["profile_ablation"] = {
        "1.5b": profile_summary(input_dir / "1.5b-sleep-profile.jsonl"),
        "3b": profile_summary(input_dir / "3b-sleep-profile.jsonl"),
    }
    summary["pressure"] = load_json(input_dir / "pressure-evidence.json")
    summary["provenance"] = load_json(provenance)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complete curated request-switch summary")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = build_summary(Path(args.input_dir), Path(args.provenance))
    Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
