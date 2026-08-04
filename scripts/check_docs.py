#!/usr/bin/env python3
"""Check current documentation links, language, and portable path policy."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "CODE_OF_CONDUCT.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / "docs" / "baselines").glob("*.md")),
    *sorted((ROOT / "docs" / "systems").glob("*.md")),
    ROOT / "results" / "README.md",
    ROOT / "results" / "release-v0.1" / "README.md",
    ROOT / "scripts" / "README.md",
    ROOT / "src" / "README.md",
]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HAN_RE = re.compile(r"[\u3400-\u9fff]")
PRIVATE_PATH_RE = re.compile(r"/home/[A-Za-z0-9._-]+")


def tracked_text_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    paths = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        path = ROOT / raw.decode()
        if (
            path.is_file()
            and path.parts[len(ROOT.parts)] != "results"
            and "archive" not in path.parts
        ):
            try:
                path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            paths.append(path)
    return paths


def main() -> int:
    failures: list[str] = []
    for doc in CURRENT_DOCS:
        if not doc.is_file():
            failures.append(f"missing current document: {doc.relative_to(ROOT)}")
            continue
        text = doc.read_text(encoding="utf-8")
        if HAN_RE.search(text):
            failures.append(f"non-English text in current document: {doc.relative_to(ROOT)}")
        if PRIVATE_PATH_RE.search(text):
            failures.append(f"private absolute path in current document: {doc.relative_to(ROOT)}")
        for target in LINK_RE.findall(text):
            target = target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.exists():
                failures.append(
                    f"broken link in {doc.relative_to(ROOT)}: {target}"
                )

    for path in tracked_text_files():
        if "archive" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if PRIVATE_PATH_RE.search(text):
            failures.append(f"private absolute path in tracked current file: {path.relative_to(ROOT)}")

    if failures:
        print("documentation/portability check failed:", file=sys.stderr)
        for failure in sorted(set(failures)):
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"documentation/portability check: ok ({len(CURRENT_DOCS)} current docs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
