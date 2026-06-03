from __future__ import annotations

import json
import subprocess
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional at runtime
    psutil = None  # type: ignore[assignment]


def parse_gpu_memory_used_mib(output: str) -> int | None:
    text = output.strip()
    if not text:
        return None
    first = text.splitlines()[0].strip()
    if not first:
        return None
    token = first.replace("MiB", "").strip().split(",")[0].strip()
    try:
        return int(float(token))
    except ValueError:
        return None


def query_gpu_memory_used_mib() -> int | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return parse_gpu_memory_used_mib(proc.stdout)


def process_tree_rss_mib(pid: int | None) -> float | None:
    if pid is None or psutil is None:
        return None
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        rss_bytes = 0
        for proc in processes:
            try:
                rss_bytes += int(proc.memory_info().rss)
            except Exception:
                continue
        return rss_bytes / 2**20
    except Exception:
        return None


def _container_rss_mib(runtime: str, names: list[str]) -> float | None:
    if not names:
        return None
    total: float = 0.0
    found = False
    for name in names:
        try:
            proc = subprocess.run(
                [runtime, "inspect", name, "--format", "json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception:
            continue
        if proc.returncode != 0 or not proc.stdout.strip():
            continue
        try:
            data: Any = json.loads(proc.stdout)
            item = data[0] if isinstance(data, list) else data
            pid = item.get("State", {}).get("Pid") or item.get("State", {}).get("pid")
            rss = process_tree_rss_mib(int(pid)) if pid else None
        except Exception:
            rss = None
        if rss is not None:
            total += rss
            found = True
    return total if found else None


def docker_container_rss_mib(container_names: list[str]) -> float | None:
    return _container_rss_mib("docker", container_names)


def podman_container_rss_mib(container_ids_or_names: list[str]) -> float | None:
    return _container_rss_mib("podman", container_ids_or_names)
