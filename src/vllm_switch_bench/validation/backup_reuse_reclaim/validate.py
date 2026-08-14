from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_switch_bench.experiments.backup_reuse_reclaim.artifacts import (
    reclaim_metrics,
    reclaim_path,
    reuse_events,
    reuse_path,
    reuse_rows,
    summary as backup_summary,
)
from vllm_switch_bench.experiments.lifecycle_latency.artifacts import MODELS
from vllm_switch_bench.validation.common import (
    default_results_root,
    require,
    validate_metadata,
)

MATERIAL_RELEASE_RATIO = 0.5


def require_coordinator_provenance(reclaim: dict) -> None:
    environment = reclaim.get("environment", {})
    coordinator = environment.get("coordinator")
    require(isinstance(coordinator, dict), "backup: coordinator provenance is missing")
    repo = coordinator.get("repo", {})
    config = coordinator.get("config", {})
    require(repo.get("git_commit"), "backup: coordinator commit is missing")
    require(repo.get("repo_path"), "backup: coordinator checkout path is missing")
    digest = config.get("sha256")
    require(
        isinstance(digest, str) and len(digest) == 64,
        "backup: coordinator config digest is missing",
    )


def validate_family(path: Path | None = None) -> None:
    family = path or default_results_root() / "backup-reuse-reclaim"
    metadata = validate_metadata(family, "backup-reuse-reclaim")
    for model in MODELS:
        data = json.loads(reuse_path(family, model).read_text(encoding="utf-8"))
        events = reuse_events(data)
        rows = reuse_rows(data)
        require(len(events) == 5, f"backup: {model} expected five reuse events")
        require(
            len({int(row.get("cycle", row.get("iteration", -1))) for row in rows}) == 5,
            f"backup: {model} lifecycle cycles are not unique",
        )
        require(
            all(
                row.get("output_match", row.get("output_matches_reference")) is True for row in rows
            ),
            f"backup: {model} output mismatch",
        )
        if metadata["status"] == "local-rerun":
            require(data.get("ok") is True, f"backup: {model} runner reported failure")
            require(not data.get("assertion_failures"), f"backup: {model} assertions failed")
        for event in events:
            require(event.get("phase") == "allocator_sleep", f"backup: {model} wrong event phase")
            require(
                int(event.get("cpu_backup_reuse_count", 0)) > 0,
                f"backup: {model} missing reuse count",
            )
            require(
                int(event.get("cpu_backup_reused_bytes", 0))
                == int(event.get("backup_bytes", -1))
                > 0,
                f"backup: {model} reused-byte accounting mismatch",
            )
            require(
                float(event.get("copy_d2h_s", -1.0)) == 0.0,
                f"backup: {model} repeated D2H copy observed",
            )

    reclaim = json.loads(reclaim_path(family).read_text(encoding="utf-8"))
    if metadata["status"] == "local-rerun":
        require(reclaim.get("ok") is True, "backup: reclaim runner reported failure")
        require(not reclaim.get("assertion_failures"), "backup: reclaim assertions failed")
        require_coordinator_provenance(reclaim)
        metrics = reclaim_metrics(reclaim)
        queued = int(metrics["requested_bytes"])
        require(queued > 0, "backup: release request was not accepted")
        require(
            int(metrics["released_delta_bytes"]) == queued,
            "backup: logical release bytes differ from request",
        )
        require(int(metrics["pending_release_bytes"]) == 0, "backup: pending bytes remain")
        require(
            int(metrics["pending_release_request_count"]) == 0,
            "backup: pending requests remain",
        )
        material = max(1, int(queued * MATERIAL_RELEASE_RATIO))
        rss_drop = -sum(min(int(value), 0) for value in metrics["client_rss_delta_bytes"].values())
        require(rss_drop >= material, "backup: process RSS recovery is not material")
        require(
            int(metrics["memavailable_delta_bytes"]) >= material,
            "backup: MemAvailable recovery is not material",
        )
        require(metrics["flush_success"] is True, "backup: host-cache flush failed")
        summary = json.loads((family / "summary.json").read_text(encoding="utf-8"))
        require(summary == backup_summary(family), "backup: raw recomputation differs")
        return

    before = reclaim["before"]
    after = reclaim["after"]
    before_pool = before["pool_stats"]
    after_pool = after["pool_stats"]
    response = reclaim["release_response"]
    queued = int(response["queued_bytes"])
    reclaimable = int(before_pool["ram_reclaimable_without_disk_bytes"])
    require(response.get("ok") is True and queued > 0, "backup: release request was not accepted")
    require(queued == reclaimable, "backup: queued bytes do not match reclaimable footprint")
    released = int(before_pool["total_bytes"] - after_pool["total_bytes"])
    require(released == queued, "backup: logical release bytes differ from request")
    require(int(after_pool["pending_release_bytes"]) == 0, "backup: pending bytes remain")
    require(
        int(after_pool["pending_release_request_count"]) == 0,
        "backup: pending requests remain",
    )

    before_clients = before["clients"]
    after_clients = after["clients"]
    require(set(before_clients) == set(after_clients), "backup: client incarnations changed")
    require(
        all(before_clients[name]["pid"] == after_clients[name]["pid"] for name in before_clients),
        "backup: client PIDs changed",
    )
    declared_rss = reclaim["client_rss_delta_bytes"]
    require(set(declared_rss) == set(before_clients), "backup: RSS delta clients mismatch")
    for name in before_clients:
        observed = int(after_clients[name]["process_tree_rss_bytes"]) - int(
            before_clients[name]["process_tree_rss_bytes"]
        )
        require(observed == int(declared_rss[name]), "backup: RSS delta snapshot mismatch")
    process_rss_drop = -sum(min(int(value), 0) for value in declared_rss.values())
    memavailable_delta = int(reclaim["memavailable_delta_bytes"])
    material = max(1, int(queued * MATERIAL_RELEASE_RATIO))
    require(process_rss_drop >= material, "backup: process RSS recovery is not material")
    require(memavailable_delta >= material, "backup: MemAvailable recovery is not material")
    require(
        memavailable_delta == int(after["memavailable_bytes"]) - int(before["memavailable_bytes"]),
        "backup: MemAvailable snapshots are inconsistent",
    )
    released_counter_delta = sum(
        int(after_clients[name]["released_bytes_total"])
        - int(before_clients[name]["released_bytes_total"])
        for name in before_clients
    )
    requested_counter_delta = sum(
        int(after_clients[name]["requested_release_bytes_total"])
        - int(before_clients[name]["requested_release_bytes_total"])
        for name in before_clients
    )
    require(
        released_counter_delta == requested_counter_delta == queued,
        "backup: client release counters do not acknowledge the request",
    )
    summary = json.loads((family / "summary.json").read_text(encoding="utf-8"))
    require(summary == backup_summary(family), "backup: raw recomputation differs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate backup reuse/reclaim semantics")
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args(argv)
    validate_family(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
