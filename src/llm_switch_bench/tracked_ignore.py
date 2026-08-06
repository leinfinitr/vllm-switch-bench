from __future__ import annotations

import subprocess

from llm_switch_bench.common.provenance import repository_root


def main() -> int:
    root = repository_root()
    output = subprocess.check_output(
        ["git", "ls-files", "-ci", "--exclude-standard"], cwd=root, text=True
    )
    if output.strip():
        raise SystemExit(f"tracked ignored files present:\n{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
