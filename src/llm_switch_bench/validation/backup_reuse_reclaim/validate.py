from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_switch_bench.artifacts import MODELS, backup_summary
from llm_switch_bench.validation.common import RESULTS, require, validate_metadata


def validate_family(path: Path | None = None) -> None:
    family = path or RESULTS / "backup-reuse-reclaim"
    validate_metadata(family, "backup-reuse-reclaim")
    for model in MODELS:
        data = json.loads(
            (family / "raw" / "proposed" / f"{model}.json").read_text(encoding="utf-8")
        )
        events = data.get("sleep_events", [])
        require(len(events) == 5, f"backup: {model} expected five sleep profiles")
        for event in events:
            require(
                int(event.get("cpu_backup_reuse_count", 0)) > 0,
                f"backup: {model} missing reusable count",
            )
            require(
                int(event.get("cpu_backup_reused_bytes", 0)) > 0,
                f"backup: {model} missing reusable bytes",
            )
            require(
                float(event.get("copy_d2h_s", 0.0)) == 0.0,
                f"backup: {model} repeated D2H copy observed",
            )
    reclaim = json.loads(
        (family / "raw" / "proposed" / "controller-pressure-release.json").read_text(
            encoding="utf-8"
        )
    )
    requested = int(reclaim["release_response"]["queued_bytes"])
    released = int(
        reclaim["before"]["pool_stats"]["total_bytes"]
        - reclaim["after"]["pool_stats"]["total_bytes"]
    )
    require(requested == released, "backup: requested and released bytes differ")
    require(
        int(reclaim["after"]["pool_stats"]["pending_release_bytes"]) == 0,
        "backup: pending release bytes remain",
    )
    require(
        int(reclaim["after"]["pool_stats"]["pending_release_request_count"]) == 0,
        "backup: pending release requests remain",
    )
    require(int(reclaim["memavailable_delta_bytes"]) > 0, "backup: MemAvailable did not increase")
    require(
        any(int(delta) < 0 for delta in reclaim["client_rss_delta_bytes"].values()),
        "backup: no client RSS decrease evidence",
    )
    summary = json.loads((family / "summary.json").read_text(encoding="utf-8"))
    require(summary == backup_summary(family), "backup: raw recomputation does not equal summary")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate backup reuse/reclaim result semantics")
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    validate_family(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
