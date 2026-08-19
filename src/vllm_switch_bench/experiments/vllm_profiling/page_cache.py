"""Linux per-file page-cache treatments for vLLM L2 profiling."""

from __future__ import annotations

import ctypes
import errno
import math
import mmap
import os
from pathlib import Path
from typing import Any, Iterable

import psutil

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_longlong,
]
_LIBC.mmap.restype = ctypes.c_void_p
_LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
_LIBC.mincore.restype = ctypes.c_int
_LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_LIBC.munmap.restype = ctypes.c_int
_MAP_FAILED = ctypes.c_void_p(-1).value


def checkpoint_files(model: str | Path) -> list[Path]:
    """Return the local safetensors payload files used by a checkpoint."""

    root = Path(model).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"page-cache treatment requires a local model directory: {root}")
    files = sorted(path for path in root.glob("*.safetensors") if path.is_file())
    if not files:
        raise ValueError(f"no safetensors files found under local model directory: {root}")
    return files


def _resident_pages(path: Path) -> tuple[int, int, int]:
    size = path.stat().st_size
    if size == 0:
        return 0, 0, 0
    page_count = math.ceil(size / _PAGE_SIZE)
    fd = os.open(path, os.O_RDONLY)
    address: int | None = None
    try:
        address = _LIBC.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if address == _MAP_FAILED:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), path)
        vector = (ctypes.c_ubyte * page_count)()
        if _LIBC.mincore(ctypes.c_void_p(address), size, vector) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), path)
        resident = sum(1 for value in vector if value & 1)
        return size, page_count, resident
    finally:
        if address not in (None, _MAP_FAILED):
            _LIBC.munmap(ctypes.c_void_p(address), size)
        os.close(fd)


def measure_page_cache(files: Iterable[Path]) -> dict[str, Any]:
    """Measure file-page residency without faulting payload bytes into memory."""

    observations = []
    total_bytes = 0
    total_pages = 0
    resident_pages = 0
    for path in files:
        size, pages, resident = _resident_pages(path)
        total_bytes += size
        total_pages += pages
        resident_pages += resident
        observations.append(
            {
                "path": str(path),
                "size_bytes": size,
                "page_count": pages,
                "resident_pages": resident,
                "resident_ratio": resident / pages if pages else 0.0,
            }
        )
    return {
        "total_bytes": total_bytes,
        "page_size_bytes": _PAGE_SIZE,
        "page_count": total_pages,
        "resident_pages": resident_pages,
        "resident_ratio": resident_pages / total_pages if total_pages else 0.0,
        "files": observations,
    }


def evict_page_cache(files: Iterable[Path]) -> dict[str, Any]:
    """Request eviction of clean checkpoint pages and report observed residency."""

    paths = list(files)
    before = measure_page_cache(paths)
    errors: list[dict[str, Any]] = []
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as exc:
            errors.append(
                {
                    "path": str(path),
                    "errno": exc.errno,
                    "error": str(exc),
                }
            )
        finally:
            os.close(fd)
    after = measure_page_cache(paths)
    return {"before": before, "after": after, "errors": errors, "ok": not errors}


def _proc_faults(pid: int) -> tuple[int, int]:
    text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = text[text.rfind(")") + 2 :].split()
    if len(fields) < 10:
        raise ValueError(f"unexpected /proc/{pid}/stat format")
    return int(fields[7]), int(fields[9])


def process_tree_io_snapshot(pid: int) -> dict[str, Any]:
    """Capture storage-I/O and fault counters for a live process tree."""

    root = psutil.Process(pid)
    processes = [root, *root.children(recursive=True)]
    pids: list[int] = []
    read_bytes = 0
    read_chars = 0
    minor_faults = 0
    major_faults = 0
    for process in processes:
        try:
            current_pid = int(process.pid)
            io = process.io_counters()
            minor, major = _proc_faults(current_pid)
        except (FileNotFoundError, ProcessLookupError, psutil.Error, PermissionError):
            continue
        pids.append(current_pid)
        read_bytes += int(io.read_bytes)
        read_chars += int(io.read_chars)
        minor_faults += minor
        major_faults += major
    if not pids:
        raise ProcessLookupError(errno.ESRCH, f"no live processes found under PID {pid}")
    return {
        "pids": sorted(pids),
        "read_bytes": read_bytes,
        "read_chars": read_chars,
        "minor_faults": minor_faults,
        "major_faults": major_faults,
    }


def process_tree_io_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        key: max(0, int(after[key]) - int(before[key]))
        for key in ("read_bytes", "read_chars", "minor_faults", "major_faults")
    }


def l2_cache_schedule(block_index: int, cycles_per_process: int) -> list[str]:
    """Rotate one cold treatment across cycle positions in successive blocks."""

    if cycles_per_process < 2:
        raise ValueError("L2 cache profiling requires at least two cycles per process")
    cold_index = block_index % cycles_per_process
    return ["cold" if index == cold_index else "warm" for index in range(cycles_per_process)]


def validate_cache_observation(
    condition: str,
    *,
    before_wake: dict[str, Any],
    io_delta: dict[str, int],
    checkpoint_bytes: int,
    cold_max_resident_ratio: float,
    cold_min_read_ratio: float,
    warm_min_resident_ratio: float,
    warm_max_read_ratio: float,
) -> tuple[bool, list[str]]:
    """Validate that a labeled L2 sample observed the intended cache state."""

    failures: list[str] = []
    resident_ratio = float(before_wake["resident_ratio"])
    read_ratio = int(io_delta["read_bytes"]) / checkpoint_bytes if checkpoint_bytes else 0.0
    if condition == "cold":
        if resident_ratio > cold_max_resident_ratio:
            failures.append(
                f"cold checkpoint residency {resident_ratio:.4f} exceeds "
                f"{cold_max_resident_ratio:.4f}"
            )
        if read_ratio < cold_min_read_ratio:
            failures.append(
                f"cold storage-read ratio {read_ratio:.4f} is below {cold_min_read_ratio:.4f}"
            )
    elif condition == "warm":
        if resident_ratio < warm_min_resident_ratio:
            failures.append(
                f"warm checkpoint residency {resident_ratio:.4f} is below "
                f"{warm_min_resident_ratio:.4f}"
            )
        if read_ratio > warm_max_read_ratio:
            failures.append(
                f"warm storage-read ratio {read_ratio:.4f} exceeds {warm_max_read_ratio:.4f}"
            )
    else:
        raise ValueError(f"unknown cache condition: {condition}")
    return not failures, failures
