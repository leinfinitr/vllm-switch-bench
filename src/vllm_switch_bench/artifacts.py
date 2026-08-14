"""Build dispatch for published result families."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Callable

from vllm_switch_bench.families import FAMILIES, FAMILIES_BY_NAME, FAMILY_NAMES
from vllm_switch_bench.publication import default_results_root

Builder = Callable[[Path | None], None]


def _load_builder(target: str) -> Builder:
    module_name, attribute = target.split(":", maxsplit=1)
    return getattr(import_module(module_name), attribute)


def build_family(name: str, results_root: Path | None = None) -> None:
    _load_builder(FAMILIES_BY_NAME[name].builder)(results_root)


def build_all(results_root: Path | None = None) -> None:
    root = results_root or default_results_root()
    for family in FAMILIES:
        _load_builder(family.builder)(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic result artifacts")
    parser.add_argument("family", nargs="?", default="all", choices=("all", *FAMILY_NAMES))
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args(argv)
    if args.family == "all":
        build_all(args.results_root)
    else:
        build_family(args.family, args.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
