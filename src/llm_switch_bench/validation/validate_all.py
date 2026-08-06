from __future__ import annotations

from llm_switch_bench.validation.backup_reuse_reclaim.validate import (
    validate_family as validate_backup,
)
from llm_switch_bench.validation.common import validate_top_level_results
from llm_switch_bench.validation.exact_disk.validate import validate_family as validate_exact_disk
from llm_switch_bench.validation.lifecycle_latency.validate import (
    validate_family as validate_lifecycle,
)
from llm_switch_bench.validation.request_driven_switch.validate import (
    validate_family as validate_request,
)


def main() -> int:
    validate_top_level_results()
    validate_lifecycle()
    validate_request()
    validate_backup()
    validate_exact_disk()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
