from __future__ import annotations

import argparse
from pathlib import Path

from llm_switch_bench.artifacts import build_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build all current result families")
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args(argv)
    build_all(args.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
