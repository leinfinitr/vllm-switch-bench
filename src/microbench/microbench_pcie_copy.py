#!/usr/bin/env python3
"""Microbenchmark CPU<->GPU copy bandwidth on the local CUDA device.

This script compares:
- torch Tensor.copy_ with pinned/pageable host tensors
- cudaMemcpy through vLLM's CudaRTLibrary, matching the vLLM sleep path

Outputs CSV rows to stdout or --csv.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class BenchResult:
    method: str
    direction: str
    host_memory: str
    size_bytes: int
    repeats: int
    warmups: int
    mean_s: float
    median_s: float
    min_s: float
    max_s: float
    stdev_s: float
    mean_gbps: float
    median_gbps: float
    min_time_gbps: float


def _try_import_vllm_cudart():
    try:
        from vllm.distributed.device_communicators.cuda_wrapper import (  # type: ignore
            CudaRTLibrary,
        )

        return CudaRTLibrary()
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"warning: failed to import vLLM CudaRTLibrary: {exc!r}", file=sys.stderr)
        return None


def _time_sync(fn: Callable[[], None], warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    return times


def _time_cuda_events(fn: Callable[[], None], warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        # elapsed_time is milliseconds.
        times.append(start.elapsed_time(end) / 1000.0)
    return times


def _summarize(
    method: str,
    direction: str,
    host_memory: str,
    size_bytes: int,
    repeats: int,
    warmups: int,
    times: list[float],
) -> BenchResult:
    return BenchResult(
        method=method,
        direction=direction,
        host_memory=host_memory,
        size_bytes=size_bytes,
        repeats=repeats,
        warmups=warmups,
        mean_s=statistics.mean(times),
        median_s=statistics.median(times),
        min_s=min(times),
        max_s=max(times),
        stdev_s=statistics.stdev(times) if len(times) > 1 else 0.0,
        mean_gbps=size_bytes / statistics.mean(times) / 1e9,
        median_gbps=size_bytes / statistics.median(times) / 1e9,
        min_time_gbps=size_bytes / min(times) / 1e9,
    )


def parse_sizes(text: str) -> list[int]:
    sizes: list[int] = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        mult = 1
        if part.endswith("gib"):
            mult = 1024**3
            part = part[:-3]
        elif part.endswith("gb"):
            mult = 1000**3
            part = part[:-2]
        elif part.endswith("mib"):
            mult = 1024**2
            part = part[:-3]
        elif part.endswith("mb"):
            mult = 1000**2
            part = part[:-2]
        sizes.append(int(float(part) * mult))
    return sizes


def run(args: argparse.Namespace) -> list[BenchResult]:
    torch.cuda.set_device(args.device)
    print(f"device={torch.cuda.get_device_name(args.device)}", file=sys.stderr)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}", file=sys.stderr)

    libcudart = _try_import_vllm_cudart() if args.include_vllm_cudart else None
    results: list[BenchResult] = []

    for size_bytes in parse_sizes(args.sizes):
        # Allocate as uint8 so numel == bytes.
        gpu = torch.empty(size_bytes, dtype=torch.uint8, device=f"cuda:{args.device}")
        gpu.fill_(123)
        torch.cuda.synchronize()

        for pin in (True, False):
            host_memory = "pinned" if pin else "pageable"
            if pin:
                try:
                    host = torch.empty(size_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
                except Exception as exc:
                    print(
                        f"warning: failed pinned allocation size={size_bytes}: {exc!r}",
                        file=sys.stderr,
                    )
                    continue
            else:
                host = torch.empty(size_bytes, dtype=torch.uint8, device="cpu")
            host.fill_(17)
            torch.cuda.synchronize()

            # torch copy_ path. non_blocking only matters for pinned memory.
            for direction, fn in (
                ("H2D", lambda host=host, gpu=gpu: gpu.copy_(host, non_blocking=args.non_blocking)),
                ("D2H", lambda host=host, gpu=gpu: host.copy_(gpu, non_blocking=args.non_blocking)),
            ):
                times = _time_cuda_events(fn, args.warmups, args.repeats)
                results.append(
                    _summarize(
                        method=f"torch.copy_.event.nb={args.non_blocking}",
                        direction=direction,
                        host_memory=host_memory,
                        size_bytes=size_bytes,
                        repeats=args.repeats,
                        warmups=args.warmups,
                        times=times,
                    )
                )

            # vLLM's sleep path uses CudaRTLibrary.cudaMemcpy and wall time.
            if libcudart is not None:
                cpu_ptr = host.data_ptr()
                gpu_ptr = gpu.data_ptr()
                for direction, fn in (
                    ("H2D", lambda cpu_ptr=cpu_ptr, gpu_ptr=gpu_ptr: libcudart.cudaMemcpy(gpu_ptr, cpu_ptr, size_bytes)),
                    ("D2H", lambda cpu_ptr=cpu_ptr, gpu_ptr=gpu_ptr: libcudart.cudaMemcpy(cpu_ptr, gpu_ptr, size_bytes)),
                ):
                    times = _time_sync(fn, args.warmups, args.repeats)
                    results.append(
                        _summarize(
                            method="vllm.CudaRTLibrary.cudaMemcpy.wall",
                            direction=direction,
                            host_memory=host_memory,
                            size_bytes=size_bytes,
                            repeats=args.repeats,
                            warmups=args.warmups,
                            times=times,
                        )
                    )

        del gpu
        torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sizes", default="256MiB,512MiB,1024MiB,3072MiB,6144MiB")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--non-blocking", action="store_true", default=True)
    parser.add_argument("--include-vllm-cudart", action="store_true")
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    results = run(args)
    fieldnames = list(BenchResult.__dataclass_fields__.keys())
    if args.csv:
        out = open(args.csv, "w", newline="", encoding="utf-8")
    else:
        out = sys.stdout
    with out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(item.__dict__)


if __name__ == "__main__":
    main()
