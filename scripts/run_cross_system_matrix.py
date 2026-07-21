from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_request_driven_switch import failed_record, run_trace, write_jsonl
from benchlib.request_trace import REQUIRED_FIELDS, load_manifest


@dataclass(frozen=True)
class SystemSpec:
    name: str
    base_url: str
    launch: list[str] | None = None
    cwd: str | None = None
    ready_url: str | None = None
    env: dict[str, str] | None = None


def parse_system(value: str) -> SystemSpec:
    """Parse NAME=BASE_URL or NAME=BASE_URL::LAUNCH_JSON."""

    name, separator, remainder = value.partition("=")
    if not separator or not name or not remainder:
        raise argparse.ArgumentTypeError("system must be NAME=BASE_URL[::LAUNCH_JSON]")
    base_url, launch_separator, launch_json = remainder.partition("::")
    launch = None
    if launch_separator:
        parsed = json.loads(launch_json)
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) and item for item in parsed
        ):
            raise argparse.ArgumentTypeError("LAUNCH_JSON must be a non-empty string array")
        launch = parsed
    return SystemSpec(name=name, base_url=base_url.rstrip("/"), launch=launch)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(path: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    return {
        "path": str(path.resolve()),
        "commit": command("rev-parse", "HEAD"),
        "tree": command("rev-parse", "HEAD^{tree}"),
        "tracked_dirty": bool(command("status", "--short", "--untracked-files=no")),
    }


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


async def wait_ready(url: str, process: subprocess.Popen[str] | None, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    async with httpx.AsyncClient(timeout=2) as client:
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"system exited before readiness: {process.returncode}")
            try:
                response = await client.get(url)
                if 200 <= response.status_code < 300:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as exc:
                last_error = repr(exc)
            await asyncio.sleep(0.2)
    raise TimeoutError(f"readiness timeout for {url}: {last_error}")


def stop_process(process: subprocess.Popen[str] | None, timeout_s: float = 30) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, __import__("signal").SIGTERM)
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, __import__("signal").SIGKILL)
        process.wait(timeout=10)


def validate_manifest_identity(
    expected: list[dict[str, Any]], output_path: Path
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line]
    if len(rows) != len(expected):
        raise ValueError(f"row count mismatch: expected={len(expected)} actual={len(rows)}")
    frozen_fields = tuple(sorted(REQUIRED_FIELDS))
    for index, (request, row) in enumerate(zip(expected, rows, strict=True)):
        frozen = tuple(request.get(field) for field in frozen_fields)
        observed = tuple(row.get(field) for field in frozen_fields)
        if frozen != observed:
            raise ValueError(
                f"request identity mismatch at row {index}: {observed} != {frozen}"
            )
    return rows


async def run_one(
    system: SystemSpec,
    manifest_path: Path,
    repeat: int,
    output_path: Path,
    timeout_s: float,
    ready_timeout_s: float,
    log_path: Path,
) -> dict[str, Any]:
    process: subprocess.Popen[str] | None = None
    manifest = load_manifest(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    launch = system.launch
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        if launch:
            process = subprocess.Popen(
                launch,
                cwd=system.cwd,
                env={**os.environ, **(system.env or {})},
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            await wait_ready(
                system.ready_url or f"{system.base_url}/health",
                process,
                ready_timeout_s,
            )
        async with httpx.AsyncClient(timeout=timeout_s, trust_env=False) as client:
            records = await run_trace(client, system.base_url, manifest, timeout_s)
        write_jsonl(temporary_output, records)
        rows = validate_manifest_identity(manifest, temporary_output)
        temporary_output.replace(output_path)
        result: dict[str, Any] = {
            "requests": len(rows),
            "failed": sum(failed_record(row) for row in rows),
        }
        result.update(
            {
                "system": system.name,
                "repeat": repeat,
                "manifest": manifest_path.name,
                "manifest_sha256": sha256_file(manifest_path),
                "output": output_path.name,
                "output_sha256": sha256_file(output_path),
                "rows": len(rows),
                "return_code": 0,
            }
        )
        return result
    except Exception as exc:
        return {
            "system": system.name,
            "repeat": repeat,
            "manifest": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "output": output_path.name,
            "output_sha256": None,
            "return_code": 1,
            "error": repr(exc),
        }
    finally:
        temporary_output.unlink(missing_ok=True)
        stop_process(process)
        log_handle.close()
        metadata = {
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": time.monotonic() - started_monotonic,
            "launch": launch,
            "launch_shell": shlex.join(launch) if launch else None,
            "log": log_path.name,
        }
        metadata_path = output_path.with_suffix(".run.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


async def async_main(args: argparse.Namespace) -> int:
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    manifests = [Path(path).resolve() for path in args.manifests]
    for path in manifests:
        load_manifest(path)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    order = [
        (system, manifest, repeat)
        for repeat in range(args.repeats)
        for manifest in manifests
        for system in args.systems
    ]
    random.Random(args.order_seed).shuffle(order)
    matrix: list[dict[str, Any]] = []
    seen_outputs: set[Path] = set()
    for position, (system, manifest, repeat) in enumerate(order):
        stem = f"{system.name}-{manifest.stem}-r{repeat}"
        output_path = out_dir / f"{stem}.jsonl"
        if output_path.resolve() in seen_outputs:
            raise ValueError(f"duplicate output path: {output_path}")
        seen_outputs.add(output_path.resolve())
        row = await run_one(
            system,
            manifest,
            repeat,
            output_path,
            args.request_timeout_s,
            args.ready_timeout_s,
            out_dir / f"{stem}.log",
        )
        row["order_position"] = position
        matrix.append(row)
        (out_dir / "matrix.json").write_text(
            json.dumps(matrix, indent=2), encoding="utf-8"
        )

    benchmark_root = Path(__file__).resolve().parents[1]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "systems": [asdict(system) for system in args.systems],
        "manifests": [
            {
                "path": str(path),
                "repo_relative_path": str(path.relative_to(benchmark_root))
                if path.is_relative_to(benchmark_root)
                else None,
                "sha256": sha256_file(path),
            }
            for path in manifests
        ],
        "repeats": args.repeats,
        "order_seed": args.order_seed,
        "order": [
            {
                "system": system.name,
                "manifest": manifest.name,
                "repeat": repeat,
            }
            for system, manifest, repeat in order
        ],
        "machine": {
            "platform": platform.platform(),
            "python": sys.version,
        },
        "benchmark_git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return 0 if all(row["return_code"] == 0 for row in matrix) else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay immutable OpenAI traces against a randomized cross-system matrix."
    )
    parser.add_argument("--systems", type=parse_system, nargs="+", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--order-seed", type=int, default=20260721)
    parser.add_argument("--request-timeout-s", type=float, default=600)
    parser.add_argument("--ready-timeout-s", type=float, default=240)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.request_timeout_s <= 0 or args.ready_timeout_s <= 0:
        parser.error("timeouts must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
