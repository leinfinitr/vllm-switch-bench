from __future__ import annotations

import argparse
from pathlib import Path

from llm_switch_bench.validation.backup_reuse_reclaim.validate import (
    validate_family as validate_backup,
)
from llm_switch_bench.validation.common import default_results_root, validate_top_level_results
from llm_switch_bench.validation.exact_disk.validate import validate_family as validate_exact_disk
from llm_switch_bench.validation.lifecycle_latency.validate import (
    validate_family as validate_lifecycle,
)
from llm_switch_bench.validation.request_driven_switch.validate import (
    validate_family as validate_request,
)
from llm_switch_bench.validation.vllm_profiling.validate import (
    validate_family as validate_vllm_profiling,
)


def validate_all(results_root: Path | None = None) -> None:
    root = results_root or default_results_root()
    validate_top_level_results(root)
    validate_lifecycle(root / "lifecycle-latency")
    validate_vllm_profiling(root / "vllm-profiling")
    validate_request(root / "request-driven-switch")
    validate_backup(root / "backup-reuse-reclaim")
    validate_exact_disk(root / "exact-disk")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate all current result families")
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args(argv)
    validate_all(args.results_root)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
