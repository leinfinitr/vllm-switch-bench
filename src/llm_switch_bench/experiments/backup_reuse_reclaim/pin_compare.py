#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llm_switch_bench.common.provenance import repository_root

ROOT = repository_root()
DEFAULT_PYTHON = ROOT / ".venv/bin/python"
PIN_MODES = ("true", "false")
METHODS = ("sleep_l1", "sleep_l2")


@dataclass(frozen=True)
class ModelCase:
    name: str
    path: str
    gpu_memory_utilization: float


def parse_model_case(value: str) -> ModelCase:
    """Parse NAME=PATH[,GPU_MEMORY_UTILIZATION]."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "model must be NAME=PATH or NAME=PATH,GPU_MEMORY_UTILIZATION"
        )
    name, remainder = value.split("=", 1)
    path, separator, utilization = remainder.rpartition(",")
    if not separator:
        path = remainder
        utilization_value = 0.55
    else:
        try:
            utilization_value = float(utilization)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid GPU memory utilization in model spec: {value!r}"
            ) from exc
    if not name or not path:
        raise argparse.ArgumentTypeError(f"invalid model spec: {value!r}")
    if not 0 < utilization_value <= 1:
        raise argparse.ArgumentTypeError("GPU memory utilization must be in (0, 1]")
    return ModelCase(name, path, utilization_value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model-agnostic sleep pinned/non-pinned profiling."
    )
    parser.add_argument("--method", choices=METHODS, default="sleep_l1")
    parser.add_argument("--out-dir", default="results/profiling/sleep_l1_pin_compare")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--cuda-home", default=os.environ.get("CUDA_HOME"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompts", nargs="+", default=["short_short"])
    parser.add_argument(
        "--models",
        nargs="+",
        type=parse_model_case,
        required=True,
        metavar="NAME=PATH[,GPU_UTIL]",
    )
    parser.add_argument("--pin-modes", nargs="+", choices=PIN_MODES, default=list(PIN_MODES))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def bench_command(
    args: argparse.Namespace, case: ModelCase, pin_mode: str, out_dir: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "llm_switch_bench.experiments.lifecycle_latency.run",
        "--model",
        case.path,
        "--python",
        str(args.python),
        "--workdir",
        str(ROOT),
        "--methods",
        args.method,
        "--prompts",
        *args.prompts,
        "--repeats",
        str(args.repeats),
        "--port",
        "0",
        "--idle-s",
        "0.2",
        "--sample-interval-s",
        "1",
        "--ready-timeout-s",
        "360",
        "--gpu-memory-utilization",
        str(case.gpu_memory_utilization),
        "--sleep-cpu-backup-pin-memory",
        pin_mode,
        "--out-dir",
        str(out_dir),
    ]


def run_one(args: argparse.Namespace, case: ModelCase, pin_mode: str) -> dict:
    out_dir = ROOT / args.out_dir / case.name / f"pin_{pin_mode}"
    cmd = bench_command(args, case, pin_mode, out_dir)
    env = os.environ.copy()
    python_bin = Path(args.python).absolute().parent
    path_parts = [str(python_bin)]
    if args.cuda_home:
        env["CUDA_HOME"] = args.cuda_home
        path_parts.append(str(Path(args.cuda_home) / "bin"))
    env["PATH"] = ":".join([*path_parts, env.get("PATH", "")])
    started = time.time()
    if args.dry_run:
        return {
            "model_name": case.name,
            "model_path": case.path,
            "method": args.method,
            "pin_mode": pin_mode,
            "gpu_memory_utilization": case.gpu_memory_utilization,
            "returncode": 0,
            "duration_s": 0.0,
            "command": cmd,
            "output": "dry-run",
            "result_dir": None,
        }
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )
    output = proc.stdout.strip()
    return {
        "model_name": case.name,
        "model_path": case.path,
        "method": args.method,
        "pin_mode": pin_mode,
        "gpu_memory_utilization": case.gpu_memory_utilization,
        "returncode": proc.returncode,
        "duration_s": time.time() - started,
        "command": cmd,
        "output": output,
        "result_dir": output.splitlines()[-1] if output else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_base = ROOT / args.out_dir
    out_base.mkdir(parents=True, exist_ok=True)
    manifest_path = out_base / f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    results = []
    with manifest_path.open("w", encoding="utf-8") as handle:
        for case in args.models:
            for pin_mode in args.pin_modes:
                print(f"RUN {args.method} {case.name} pin={pin_mode}", flush=True)
                try:
                    result = run_one(args, case, pin_mode)
                except Exception as exc:
                    result = {
                        "model_name": case.name,
                        "model_path": case.path,
                        "method": args.method,
                        "pin_mode": pin_mode,
                        "gpu_memory_utilization": case.gpu_memory_utilization,
                        "returncode": -1,
                        "duration_s": None,
                        "output": repr(exc),
                        "result_dir": None,
                    }
                result["created_at"] = datetime.now(timezone.utc).isoformat()
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps(result, ensure_ascii=False), flush=True)
    print(str(manifest_path))
    return 0 if all(result["returncode"] == 0 for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
