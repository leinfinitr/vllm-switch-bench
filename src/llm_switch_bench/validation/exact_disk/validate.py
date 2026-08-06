from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_switch_bench.artifacts import exact_disk_summary
from llm_switch_bench.validation.common import RESULTS, require, validate_metadata

REQUIRED_FILES = {
    "raw/exact-disk/bundle-COMMIT",
    "raw/exact-disk/bundle-manifest.json",
    "raw/exact-disk/disk-footprint.json",
    "raw/exact-disk/exact_disk_profile.jsonl",
    "raw/exact-disk/output_observation.json",
    "raw/exact-disk/payload-hash.json",
    "raw/exact-disk/run-metadata.json",
}


def validate_family(path: Path | None = None) -> None:
    family = path or RESULTS / "exact-disk"
    meta = validate_metadata(family, "exact-disk")
    require(
        REQUIRED_FILES <= set(meta["tracked_files"]),
        "exact-disk: missing seven retained evidence files",
    )
    raw = family / "raw" / "exact-disk"
    events = [
        json.loads(line)
        for line in (raw / "exact_disk_profile.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    phases = {event.get("phase") for event in events}
    require(
        {"exact_disk_spill", "exact_disk_demotion", "allocator_sleep"} <= phases,
        "exact-disk: missing required event phases",
    )
    manifest = json.loads((raw / "bundle-manifest.json").read_text(encoding="utf-8"))
    payload = json.loads((raw / "payload-hash.json").read_text(encoding="utf-8"))
    footprint = json.loads((raw / "disk-footprint.json").read_text(encoding="utf-8"))
    require(
        manifest["payload_size_bytes"] == payload["payload_size_bytes"] == 1048576000,
        "exact-disk: payload byte mismatch",
    )
    require(len(payload["payload_sha256"]) == 64, "exact-disk: payload hash missing")
    segment_bytes = sum(int(segment["size_bytes"]) for segment in manifest["segments"])
    require(
        segment_bytes == manifest["payload_size_bytes"],
        "exact-disk: manifest segment sizes do not match payload",
    )
    for segment in manifest["segments"]:
        require(segment.get("chunk_sha256"), "exact-disk: segment missing chunk checksums")
        require(
            all(len(value) == 64 for value in segment["chunk_sha256"]),
            "exact-disk: invalid chunk checksum",
        )
    observed = footprint["filesystem_observation"]
    require(
        int(observed["logical_size_bytes"]) == 1048576000,
        "exact-disk: footprint logical size mismatch",
    )
    require(
        int(observed["allocated_bytes"]) >= 1048576000,
        "exact-disk: footprint lacks material allocation",
    )
    output = json.loads((raw / "output_observation.json").read_text(encoding="utf-8"))
    require(output["before"] == output["after"], "exact-disk: restored output differs")
    demotions = output["demotion"]["results"]
    require(
        demotions
        and all(item["released"] and item["pending_release_bytes"] == 0 for item in demotions),
        "exact-disk: demotion did not complete",
    )
    summary = json.loads((family / "summary.json").read_text(encoding="utf-8"))
    require(
        summary == exact_disk_summary(family),
        "exact-disk: raw recomputation does not equal summary",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate exact-disk result semantics")
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    validate_family(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
