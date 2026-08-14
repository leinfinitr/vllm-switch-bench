#!/usr/bin/env python3
"""Measure in-process vLLM sleep/wake for a curated lifecycle artifact.

Unlike the service-based lifecycle runner, this driver uses the offline ``LLM``
API and emits compact lifecycle JSON for a curated result bundle. Level 1 and
level 2 intentionally have different wake transactions: level 2 must recreate
weight mappings, reload checkpoint weights, and then recreate the KV cache.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_switch_bench.common.environment import reexec_with_python
from llm_switch_bench.common.provenance import file_metadata, git_metadata, repository_root


def gpu() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()


def main(argv: Sequence[str] | None = None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Measure in-process vLLM sleep and wake phases.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the vLLM environment used for this measurement.",
    )
    parser.add_argument("--sleep-level", type=int, choices=[1, 2], default=1)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--system-name", required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--vllm-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(values)
    reexec_with_python(
        args.python,
        "llm_switch_bench.experiments.lifecycle_latency.run",
        ["vllm", *values],
        workdir=args.vllm_repo,
        import_root=args.vllm_repo,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    profile = args.output.with_suffix(".sleep_profile.jsonl").resolve()
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.unlink(missing_ok=True)
    os.environ["VLLM_SLEEP_PROFILE_PATH"] = str(profile)

    import vllm
    from vllm import LLM, SamplingParams

    imported_vllm = Path(vllm.__file__).resolve()
    expected_vllm = args.vllm_repo.resolve(strict=True)
    if not imported_vllm.is_relative_to(expected_vllm):
        raise RuntimeError(
            f"imported vLLM from {imported_vllm}, outside --vllm-repo {expected_vllm}"
        )
    model_path = Path(args.model).expanduser().resolve(strict=True)
    model_config = model_path / "config.json"

    params = SamplingParams(temperature=0, seed=1, max_tokens=8)
    prompt = "Reply with exactly OK."
    llm = LLM(
        model=args.model,
        enable_sleep_mode=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )
    reference = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
    reference_value = ([int(x) for x in reference.token_ids], reference.text)
    rows: list[dict[str, Any]] = []
    try:
        for cycle in range(args.cycles):
            started = time.perf_counter()
            llm.sleep(level=args.sleep_level)
            sleep_s = time.perf_counter() - started
            started = time.perf_counter()
            if args.sleep_level == 1:
                llm.wake_up()
            else:
                # Level 2 discards weight contents. Recreating the mappings is
                # not sufficient: restore checkpoint weights before allocating
                # KV cache so the model is inference-ready at this boundary.
                llm.wake_up(tags=["weights"])
                llm.collective_rpc("reload_weights")
                llm.wake_up(tags=["kv_cache"])
            wake_s = time.perf_counter() - started
            output = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
            observed = ([int(x) for x in output.token_ids], output.text)
            row = {
                "cycle": cycle,
                "sleep_s": sleep_s,
                "wake_s": wake_s,
                "output_match": observed == reference_value,
                "output": output.text,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        del llm
        gc.collect()

    events = (
        [
            json.loads(line)
            for line in profile.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if profile.exists()
        else []
    )
    sleep_events = [row for row in events if row.get("phase") == "allocator_sleep"]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": args.system_name,
        "model": args.model_name,
        "model_path": str(model_path),
        "model_identity": {
            "path": str(model_path),
            "config": file_metadata(model_config) if model_config.is_file() else None,
        },
        "cycles": args.cycles,
        "sleep_level": args.sleep_level,
        "rows": rows,
        "sleep_events": sleep_events,
        "medians": {
            "sleep_s": statistics.median(row["sleep_s"] for row in rows),
            "wake_s": statistics.median(row["wake_s"] for row in rows),
        },
        "mechanism_check": {
            "all_outputs_match": all(row["output_match"] for row in rows),
            "all_sleep_copy_d2h_zero": (
                all(float(event.get("copy_d2h_s", -1)) == 0 for event in sleep_events)
                if sleep_events
                else None
            ),
            "all_sleep_reused_bytes_positive": (
                all(int(event.get("cpu_backup_reused_bytes", 0)) > 0 for event in sleep_events)
                if sleep_events
                else None
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": str(Path(sys.executable).resolve()),
            "gpu": gpu(),
            "benchmark": git_metadata(repository_root()),
            "vllm": {
                **git_metadata(args.vllm_repo),
                "module_path": str(imported_vllm),
            },
        },
        "profile_path": str(profile),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output)
    return 0 if summary["mechanism_check"]["all_outputs_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
