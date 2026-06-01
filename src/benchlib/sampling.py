from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, Callable

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

from .schema import Event, JsonlLogger


def run_cmd(cmd: list[str], timeout: int = 30, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=check)


def query_gpu() -> dict[str, int | None]:
    try:
        cp = run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
        )
        parts = [p.strip() for p in cp.stdout.strip().splitlines()[0].split(",")]
        return {"gpu_used_mib": int(parts[0]), "gpu_free_mib": int(parts[1]), "gpu_util_pct": int(parts[2])}
    except Exception:
        return {"gpu_used_mib": None, "gpu_free_mib": None, "gpu_util_pct": None}


def query_cpu(pid: int | None) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "cpu_used_mib": None,
        "cpu_available_mib": None,
        "proc_rss_mib": None,
        "proc_uss_mib": None,
    }
    if psutil is None:
        return result
    try:
        vm = psutil.virtual_memory()
        result["cpu_used_mib"] = int(vm.used / 2**20)
        result["cpu_available_mib"] = int(vm.available / 2**20)
    except Exception:
        pass
    if pid:
        try:
            process = psutil.Process(pid)
            info = process.memory_full_info()
            result["proc_rss_mib"] = int(info.rss / 2**20)
            result["proc_uss_mib"] = int(getattr(info, "uss", 0) / 2**20)
        except Exception:
            pass
    return result


def make_event(
    ctx: dict[str, Any],
    event: str,
    start_ts: float,
    pid: int | None = None,
    note: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Event:
    now = time.time()
    metrics: dict[str, Any] = {}
    metrics.update(query_gpu())
    metrics.update(query_cpu(pid))
    return Event(
        system=ctx["system"],
        run_id=ctx["run_id"],
        method=ctx["method"],
        model=ctx["model"],
        prompt_name=ctx["prompt_name"],
        repeat_index=ctx["repeat_index"],
        event=event,
        ts=now,
        elapsed_s=now - start_ts,
        proc_pid=pid,
        note=note,
        extra=extra,
        **metrics,
    )


class Sampler:
    def __init__(
        self,
        logger: JsonlLogger,
        ctx: dict[str, Any],
        start_ts: float,
        get_pid: Callable[[], int | None],
        interval_s: float = 0.5,
    ):
        self.logger = logger
        self.ctx = ctx
        self.start_ts = start_ts
        self.get_pid = get_pid
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self):
        while not self._stop.is_set():
            self.logger.write(make_event(self.ctx, "sample", self.start_ts, self.get_pid()))
            time.sleep(self.interval_s)
