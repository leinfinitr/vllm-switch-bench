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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = ROOT / ".venv/bin/python"
BENCH = ROOT / "src/bench_vllm_lifecycle.py"


@dataclass(frozen=True)
class ModelCase:
    name: str
    path: str
    gpu_memory_utilization: float


DEFAULT_MODELS = [
    ModelCase("qwen2p5_0p5b", "/home/ljl/models/hf/Qwen2.5-0.5B-Instruct", 0.55),
    ModelCase("qwen2p5_1p5b", "/home/ljl/models/hf/Qwen2.5-1.5B-Instruct", 0.55),
    ModelCase("qwen2p5_3b", "/home/ljl/models/hf/Qwen2.5-3B-Instruct", 0.85),
]
PIN_MODES = ("true", "false")
METHODS = ("sleep_l1", "sleep_l2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sleep pinned/non-pinned profiling comparison.")
    parser.add_argument("--method", choices=METHODS, default="sleep_l1")
    parser.add_argument("--out-dir", default="results/profiling/sleep_l1_pin_compare")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--cuda-home", default="/home/ljl/cuda-13.0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompts", nargs="+", default=["short_short"])
    parser.add_argument("--models", nargs="+", choices=[case.name for case in DEFAULT_MODELS])
    parser.add_argument("--pin-modes", nargs="+", choices=PIN_MODES, default=list(PIN_MODES))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def selected_models(names: list[str] | None) -> list[ModelCase]:
    if not names:
        return list(DEFAULT_MODELS)
    requested = set(names)
    return [case for case in DEFAULT_MODELS if case.name in requested]


def bench_command(args: argparse.Namespace, case: ModelCase, pin_mode: str, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(BENCH),
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
    env["CUDA_HOME"] = args.cuda_home
    env["PATH"] = f"{Path(args.python).resolve().parent}:{Path(args.cuda_home) / 'bin'}:" + env.get("PATH", "")
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
        for case in selected_models(args.models):
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
