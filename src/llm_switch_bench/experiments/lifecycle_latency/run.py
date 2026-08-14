"""Dispatch lifecycle-latency collection to a system-specific adapter."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from llm_switch_bench.adapters import (
    llama_swap_lifecycle,
    swapservellm_lifecycle,
    vllm_lifecycle,
)

SYSTEMS = {
    "vllm": vllm_lifecycle.main,
    "llama-swap": llama_swap_lifecycle.main,
    "swapservellm": swapservellm_lifecycle.main,
}


def main(argv: Sequence[str] | None = None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Collect one lifecycle-latency system using its retained-result adapter."
    )
    parser.add_argument("system", choices=tuple(SYSTEMS), help="Adapter to invoke")
    if not values or values[0] in {"-h", "--help"}:
        parser.parse_args(values)
    system = values.pop(0)
    if system not in SYSTEMS:
        parser.error(f"argument system: invalid choice: {system!r}")
    return SYSTEMS[system](values)


if __name__ == "__main__":
    raise SystemExit(main())
