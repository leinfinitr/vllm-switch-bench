#!/usr/bin/env python3
"""Verify the release-v0.1 artifact manifests and publication closure."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "release-v0.1"
PUBLICATION = ARTIFACT / "checksums.sha256"
COMPLETE = ARTIFACT / "all-files.sha256"


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid SHA-256 row") from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{path}:{line_number}: invalid digest")
        rows.append((digest, relative))
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    return rows


def tracked_paths() -> set[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return {item.decode() for item in output.split(b"\0") if item}


def verify_manifest(path: Path, tracked: set[str]) -> list[str]:
    failures = []
    seen = set()
    for expected, relative in parse_manifest(path):
        if relative in seen:
            failures.append(f"duplicate path in {path.name}: {relative}")
            continue
        seen.add(relative)
        candidate = ROOT / relative
        if not candidate.is_file():
            failures.append(f"missing artifact file: {relative}")
            continue
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"checksum mismatch: {relative}")
        if relative not in tracked:
            failures.append(f"manifest path is not tracked: {relative}")
    return failures


def main() -> int:
    tracked = tracked_paths()
    failures = []
    for manifest in (PUBLICATION, COMPLETE):
        if not manifest.is_file():
            failures.append(f"missing manifest: {manifest.relative_to(ROOT)}")
        else:
            failures.extend(verify_manifest(manifest, tracked))

    complete_paths = {relative for _, relative in parse_manifest(COMPLETE)} if COMPLETE.is_file() else set()
    declared = {
        path.relative_to(ROOT).as_posix()
        for path in ARTIFACT.rglob("*")
        if path.is_file() and path.name != COMPLETE.name
    }
    if complete_paths != declared:
        missing = sorted(declared - complete_paths)
        extra = sorted(complete_paths - declared)
        if missing:
            failures.append(f"complete manifest omits: {missing}")
        if extra:
            failures.append(f"complete manifest has unexpected paths: {extra}")

    # Nested producer manifests must be self-contained release-relative indexes,
    # never pointers back to a collector's local ignored tree.
    for manifest in sorted((ARTIFACT / "provenance").glob("*.sha256")):
        for _digest, relative in parse_manifest(manifest):
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                failures.append(f"non-portable nested provenance path: {relative}")
            elif not (ARTIFACT / relative).is_file():
                failures.append(f"nested provenance path is missing: {relative}")

    summary = ARTIFACT / "summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        patch_path = payload.get("provenance", {}).get("SwapServeLLM", {}).get(
            "benchmark_patch"
        )
        if patch_path and not (ARTIFACT / patch_path).is_file():
            failures.append(f"summary references missing path: {patch_path}")

    exact_profile = ARTIFACT / "raw/exact-disk/exact_disk_profile.jsonl"
    exact_metadata = ARTIFACT / "raw/exact-disk/run-metadata.json"
    pressure_path = ARTIFACT / "raw/proposed/controller-pressure-release.json"
    if exact_profile.is_file():
        events = [
            json.loads(line)
            for line in exact_profile.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        phases = {event.get("phase") for event in events}
        required_phases = {"exact_disk_spill", "exact_disk_demotion", "exact_disk_restore"}
        if not required_phases <= phases:
            failures.append("exact-disk profile omits required lifecycle phases")
        if any(event.get("fallback") is True for event in events):
            failures.append("exact-disk profile contains a fallback event")
        if sum(int(event.get("disk_spill_bytes", 0)) for event in events) <= 0:
            failures.append("exact-disk profile has no physical spill bytes")
        if sum(int(event.get("disk_read_bytes", 0)) for event in events) <= 0:
            failures.append("exact-disk profile has no physical restore bytes")
    else:
        failures.append("exact-disk profile is missing")

    if exact_metadata.is_file():
        metadata = json.loads(exact_metadata.read_text(encoding="utf-8"))
        if not metadata.get("engine", {}).get("collection_commit"):
            failures.append("exact-disk run metadata lacks engine commit")
        if not metadata.get("model") or not metadata.get("launch_parameters"):
            failures.append("exact-disk run metadata lacks model or launch parameters")
    else:
        failures.append("exact-disk run metadata is missing")

    if pressure_path.is_file():
        pressure = json.loads(pressure_path.read_text(encoding="utf-8"))
        if int(pressure.get("memavailable_delta_bytes", 0)) <= 0:
            failures.append("controller pressure evidence lacks positive MemAvailable delta")
        if not any(
            int(delta) < 0 for delta in pressure.get("client_rss_delta_bytes", {}).values()
        ):
            failures.append("controller pressure evidence lacks a decreasing process RSS")
    else:
        failures.append("controller physical-memory pressure evidence is missing")

    if failures:
        print("release artifact verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(
        "release artifact verification: ok "
        f"({len(parse_manifest(PUBLICATION))} publication files, "
        f"{len(parse_manifest(COMPLETE))} complete files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
