from __future__ import annotations

from llm_switch_bench.artifacts import build_all


def main() -> int:
    build_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
