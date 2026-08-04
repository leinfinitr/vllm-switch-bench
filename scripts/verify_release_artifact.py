#!/usr/bin/env python3
"""Verify the release-v0.1 artifact manifests and publication closure."""

from __future__ import annotations

import hashlib
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
