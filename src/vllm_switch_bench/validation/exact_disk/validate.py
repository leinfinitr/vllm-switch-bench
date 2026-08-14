from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vllm_switch_bench.experiments.exact_disk.artifacts import summary as exact_disk_summary
from vllm_switch_bench.validation.common import (
    default_results_root,
    require,
    validate_metadata,
)

REQUIRED_RAW_FILES = {
    "bundle-COMMIT",
    "bundle-manifest.json",
    "disk-footprint.json",
    "exact_disk_profile.jsonl",
    "output_observation.json",
    "payload-hash.json",
    "run-metadata.json",
}
REQUIRED_PHASES = {
    "exact_disk_spill",
    "exact_disk_demotion",
    "allocator_sleep",
    "exact_disk_restore",
    "allocator_wake_up",
}


def validate_family(path: Path | None = None) -> None:
    family = path or default_results_root() / "exact-disk"
    family_metadata = validate_metadata(family, "exact-disk")
    raw = family / "raw" / "exact-disk"
    require(
        {item.name for item in raw.iterdir() if item.is_file()} == REQUIRED_RAW_FILES,
        "exact-disk: retained raw set is not the required seven-file closure",
    )
    events = [
        json.loads(line)
        for line in (raw / "exact_disk_profile.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    phases = {event.get("phase") for event in events}
    require(REQUIRED_PHASES <= phases, "exact-disk: missing lifecycle phases")
    require(not any(event.get("fallback") is True for event in events), "exact-disk: fallback used")
    require(
        sum(int(event.get("disk_spill_bytes", 0)) for event in events) > 0,
        "exact-disk: physical spill missing",
    )
    require(
        sum(int(event.get("disk_read_bytes", 0)) for event in events) > 0,
        "exact-disk: physical restore read missing",
    )

    manifest_bytes = (raw / "bundle-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    commit = (raw / "bundle-COMMIT").read_text(encoding="utf-8").strip()
    require(
        commit == hashlib.sha256(manifest_bytes).hexdigest(),
        "exact-disk: runtime commit marker does not bind manifest bytes",
    )
    require(manifest.get("magic") == "vllm-exact-runtime-backup", "exact-disk: bad magic")
    require(int(manifest.get("schema_version", 0)) == 1, "exact-disk: bad manifest schema")
    payload_bytes = int(manifest.get("payload_size_bytes", 0))
    segments = manifest.get("segments", [])
    require(payload_bytes > 0 and segments, "exact-disk: empty payload/segments")
    require(
        sum(int(segment.get("size_bytes", 0)) for segment in segments) == payload_bytes,
        "exact-disk: segment byte accounting differs from payload",
    )
    expected_offset = 0
    for segment in segments:
        require(
            int(segment.get("offset_bytes", -1)) == expected_offset,
            "exact-disk: non-contiguous segment offsets",
        )
        size = int(segment["size_bytes"])
        expected_offset += size
        chunk_hashes = segment.get("chunk_sha256", [])
        expected_chunks = (size + int(manifest["chunk_bytes"]) - 1) // int(manifest["chunk_bytes"])
        require(len(chunk_hashes) == expected_chunks, "exact-disk: chunk count mismatch")
        require(
            all(
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
                for digest in chunk_hashes
            ),
            "exact-disk: invalid runtime chunk checksum",
        )

    payload = json.loads((raw / "payload-hash.json").read_text(encoding="utf-8"))
    require(
        int(payload["payload_size_bytes"]) == payload_bytes, "exact-disk: payload size mismatch"
    )
    digest = payload.get("payload_sha256")
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        "exact-disk: omitted payload digest invalid",
    )
    footprint = json.loads((raw / "disk-footprint.json").read_text(encoding="utf-8"))[
        "filesystem_observation"
    ]
    require(
        int(footprint["logical_size_bytes"]) == payload_bytes
        and int(footprint["allocated_bytes"]) >= payload_bytes,
        "exact-disk: physical footprint does not cover payload",
    )

    allocator_sleeps = [event for event in events if event.get("phase") == "allocator_sleep"]
    if family_metadata["status"] == "local-rerun":
        require(bool(allocator_sleeps), "exact-disk: expected allocator sleep evidence")
        sleep = next(
            (
                event
                for event in allocator_sleeps
                if int(event.get("cpu_backup_release_bytes", 0)) == payload_bytes
            ),
            None,
        )
        require(sleep is not None, "exact-disk: host-backup release evidence is missing")
        assert sleep is not None
    else:
        require(len(allocator_sleeps) == 1, "exact-disk: expected one allocator sleep")
        sleep = allocator_sleeps[0]
    require(
        int(sleep.get("backup_bytes", 0))
        == int(sleep.get("cpu_backup_release_bytes", -1))
        == payload_bytes,
        "exact-disk: host-backup release does not match payload",
    )
    require(
        int(sleep.get("cpu_backup_host_cache_flush_errors", -1)) == 0
        and int(sleep.get("cpu_backup_host_cache_flush_count", 0)) > 0
        and int(sleep.get("cpu_backup_release_count", 0)) > 0,
        "exact-disk: host-cache release did not complete",
    )

    metadata = json.loads((raw / "run-metadata.json").read_text(encoding="utf-8"))
    if family_metadata["status"] == "local-rerun":
        require(
            metadata.get("environment", {}).get("vllm_repo", {}).get("commit"),
            "exact-disk: engine commit missing",
        )
        require(
            metadata.get("model") and metadata.get("command"),
            "exact-disk: run config missing",
        )
    else:
        require(
            metadata.get("engine", {}).get("collection_commit"),
            "exact-disk: engine commit missing",
        )
        require(
            metadata.get("model") and metadata.get("launch_parameters"),
            "exact-disk: run config missing",
        )
    output = json.loads((raw / "output_observation.json").read_text(encoding="utf-8"))
    require(output.get("before") == output.get("after"), "exact-disk: output differs after restore")
    demotions = output.get("demotion", {}).get("results", [])
    require(
        bool(demotions)
        and all(
            item.get("released") is True
            and int(item.get("pending_release_bytes", -1)) == 0
            and int(item.get("released_bytes_total", 0))
            == int(item.get("requested_bytes", -1))
            == payload_bytes
            for item in demotions
        ),
        "exact-disk: demotion release did not complete",
    )
    summary = json.loads((family / "summary.json").read_text(encoding="utf-8"))
    require(summary == exact_disk_summary(family), "exact-disk: raw recomputation differs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate exact-disk result semantics")
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args(argv)
    validate_family(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
