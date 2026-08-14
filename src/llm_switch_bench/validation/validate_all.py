from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Callable

from llm_switch_bench.families import FAMILIES
from llm_switch_bench.validation.common import default_results_root, validate_top_level_results

Validator = Callable[[Path | None], None]


def _load_validator(target: str) -> Validator:
    module_name, attribute = target.split(":", maxsplit=1)
    return getattr(import_module(module_name), attribute)


def validate_all(results_root: Path | None = None) -> None:
    root = results_root or default_results_root()
    validate_top_level_results(root)
    for family in FAMILIES:
        _load_validator(family.validator)(root / family.slug)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate all current result families")
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args(argv)
    validate_all(args.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
