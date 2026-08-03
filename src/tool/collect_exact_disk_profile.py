#!/usr/bin/env python3
"""Verify local raw exact-disk evidence and build curated assertions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchlib.exact_disk import (
    ExactDiskRequirements,
    build_curated_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--curated-dir", type=Path, required=True)
    parser.add_argument("--expected-source-medium", default="disk")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument(
        "--no-require-command-success",
        dest="require_command_success",
        action="store_false",
    )
    parser.add_argument(
        "--no-require-spill", dest="require_spill", action="store_false"
    )
    parser.add_argument(
        "--no-require-read", dest="require_read", action="store_false"
    )
    parser.add_argument(
        "--no-require-worker-rss", dest="require_worker_rss", action="store_false"
    )
    parser.add_argument(
        "--no-require-mem-available",
        dest="require_mem_available",
        action="store_false",
    )
    parser.add_argument(
        "--no-require-disk-footprint-growth",
        dest="require_disk_footprint_growth",
        action="store_false",
    )
    parser.add_argument(
        "--no-require-output-equality",
        dest="require_output_equality",
        action="store_false",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requirements = ExactDiskRequirements(
        require_command_success=args.require_command_success,
        require_spill=args.require_spill,
        require_read=args.require_read,
        expected_source_medium=args.expected_source_medium or None,
        allow_fallback=args.allow_fallback,
        require_worker_rss=args.require_worker_rss,
        require_mem_available=args.require_mem_available,
        require_disk_footprint_growth=args.require_disk_footprint_growth,
        require_output_equality=args.require_output_equality,
    )
    try:
        result = build_curated_artifacts(
            args.raw_dir, args.curated_dir, requirements=requirements
        )
    except ValueError as exc:
        print(f"cannot curate exact-disk evidence: {exc}", file=sys.stderr)
        return 2
    payload = {
        "ok": result["assertions"]["ok"],
        "summary": str(args.curated_dir.resolve() / "summary.json"),
        "assertions": str(args.curated_dir.resolve() / "assertions.json"),
        "failures": result["assertions"]["failures"],
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
