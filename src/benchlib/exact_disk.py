"""Strict parsing and assertions for exact runtime disk-backup evidence.

The engine-owned profile is treated as immutable local raw evidence. This module
verifies that evidence before deriving compact local curated summaries; it does
not move or rewrite the producer's records.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

PROFILE_FILE = "exact_disk_profile.jsonl"
RESOURCE_FILE = "resources.jsonl"
OUTPUT_FILE = "output_observation.json"
RUN_FILE = "run.json"
MANIFEST_FILE = "evidence_manifest.json"


@dataclass(frozen=True)
class ExactDiskRequirements:
    """Fail-closed defaults for a valid exact-disk lifecycle observation."""

    require_command_success: bool = True
    require_spill: bool = True
    require_read: bool = True
    expected_source_medium: str | None = "disk"
    allow_fallback: bool = False
    require_worker_rss: bool = True
    require_mem_available: bool = True
    require_disk_footprint_growth: bool = True
    require_output_equality: bool = True


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} line {line_number} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_number} must be a JSON object")
            rows.append(row)
    return rows


def _number(row: dict[str, Any], field: str, *, context: str) -> float:
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} field {field} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{context} field {field} must be finite and non-negative")
    return numeric


def _paired_metric(
    row: dict[str, Any], byte_field: str, time_field: str, *, context: str
) -> tuple[int, float] | None:
    has_bytes = byte_field in row
    has_time = time_field in row
    if has_bytes != has_time:
        raise ValueError(
            f"{context} fields {byte_field} and {time_field} must appear together"
        )
    if not has_bytes:
        return None
    byte_value = _number(row, byte_field, context=context)
    if not byte_value.is_integer():
        raise ValueError(f"{context} field {byte_field} must be an integer")
    return int(byte_value), _number(row, time_field, context=context)


def parse_exact_disk_profile(path: Path) -> dict[str, Any]:
    """Parse the engine JSONL profile and aggregate exact-disk events strictly."""

    rows = _load_jsonl(path)
    spill_bytes = 0
    spill_s = 0.0
    read_bytes = 0
    read_s = 0.0
    source_media: set[str] = set()
    fallback_count = 0
    fallback_reasons: set[str] = set()
    exact_event_count = 0

    for index, row in enumerate(rows, start=1):
        context = f"{path} line {index}"
        spill = _paired_metric(row, "disk_spill_bytes", "disk_spill_s", context=context)
        read = _paired_metric(row, "disk_read_bytes", "disk_read_s", context=context)
        is_exact_disk = spill is not None or read is not None
        if not is_exact_disk:
            continue
        exact_event_count += 1
        if spill is not None:
            spill_bytes += spill[0]
            spill_s += spill[1]
        if read is not None:
            read_bytes += read[0]
            read_s += read[1]
            medium = row.get("source_medium")
            if not isinstance(medium, str) or not medium.strip():
                raise ValueError(
                    f"{context} exact-disk read requires a non-empty source_medium"
                )
            source_media.add(medium)
            fallback = row.get("fallback", False)
            if not isinstance(fallback, bool):
                raise ValueError(f"{context} field fallback must be boolean")
            if fallback:
                fallback_count += 1
                reason = row.get("fallback_reason")
                if isinstance(reason, str) and reason.strip():
                    fallback_reasons.add(reason)

    return {
        "profile_event_count": len(rows),
        "exact_disk_event_count": exact_event_count,
        "disk_spill_bytes": spill_bytes,
        "disk_spill_s": spill_s,
        "disk_read_bytes": read_bytes,
        "disk_read_s": read_s,
        "source_media": sorted(source_media),
        "fallback_count": fallback_count,
        "fallback_reasons": sorted(fallback_reasons),
    }


def _optional_non_negative_integer(
    row: dict[str, Any], field: str, *, context: str
) -> int | None:
    if field not in row or row[field] is None:
        return None
    value = _number(row, field, context=context)
    if not value.is_integer():
        raise ValueError(f"{context} field {field} must be an integer")
    return int(value)


def summarize_resources(path: Path) -> dict[str, Any]:
    """Summarize OS/process/disk samples without discarding the raw timeline."""

    rows = _load_jsonl(path)
    rss: list[int] = []
    mem_available: list[int] = []
    footprint: list[int] = []
    for index, row in enumerate(rows, start=1):
        context = f"{path} line {index}"
        worker_rss = _optional_non_negative_integer(
            row, "worker_rss_bytes", context=context
        )
        available = _optional_non_negative_integer(
            row, "mem_available_bytes", context=context
        )
        disk_bytes = _optional_non_negative_integer(
            row, "disk_footprint_bytes", context=context
        )
        if worker_rss is not None:
            rss.append(worker_rss)
        if available is not None:
            mem_available.append(available)
        if disk_bytes is not None:
            footprint.append(disk_bytes)

    footprint_baseline = footprint[0] if footprint else None
    footprint_peak = max(footprint) if footprint else None
    return {
        "sample_count": len(rows),
        "worker_rss_sample_count": len(rss),
        "worker_rss_first_bytes": rss[0] if rss else None,
        "worker_rss_last_bytes": rss[-1] if rss else None,
        "worker_rss_peak_bytes": max(rss) if rss else None,
        "mem_available_sample_count": len(mem_available),
        "mem_available_first_bytes": mem_available[0] if mem_available else None,
        "mem_available_last_bytes": mem_available[-1] if mem_available else None,
        "mem_available_min_bytes": min(mem_available) if mem_available else None,
        "disk_footprint_sample_count": len(footprint),
        "disk_footprint_baseline_bytes": footprint_baseline,
        "disk_footprint_final_bytes": footprint[-1] if footprint else None,
        "disk_footprint_peak_bytes": footprint_peak,
        "disk_footprint_peak_delta_bytes": (
            footprint_peak - footprint_baseline
            if footprint_peak is not None and footprint_baseline is not None
            else None
        ),
    }


def summarize_output_equality(path: Path) -> dict[str, Any]:
    """Compare deterministic before/after outputs while keeping values raw-only."""

    observation = _load_json(path)
    before = observation.get("before")
    after = observation.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError(f"{path} requires before and after JSON objects")

    def digest(value: dict[str, Any]) -> str:
        canonical = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(canonical).hexdigest()

    return {
        "available": True,
        "output_equal": before == after,
        "before_sha256": digest(before),
        "after_sha256": digest(after),
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_evidence_manifest(raw_dir: Path, file_names: Iterable[str]) -> Path:
    """Checksum immutable local raw files after the producer has stopped."""

    files: dict[str, dict[str, Any]] = {}
    for name in sorted(set(file_names)):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"raw evidence name must stay inside raw directory: {name}"
            )
        path = raw_dir / relative
        if not path.is_file():
            raise ValueError(f"raw evidence file is missing: {path}")
        files[relative.as_posix()] = {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    manifest_path = raw_dir / MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_tier": "local_raw",
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def verify_evidence_manifest(raw_dir: Path) -> dict[str, Any]:
    manifest_path = raw_dir / MANIFEST_FILE
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("raw evidence manifest schema_version must be 1")
    if manifest.get("evidence_tier") != "local_raw":
        raise ValueError("raw evidence manifest must declare evidence_tier=local_raw")
    if set(manifest) != {"schema_version", "evidence_tier", "files"}:
        raise ValueError("raw evidence manifest contains unsupported top-level fields")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("raw evidence manifest files must be an object")
    required = {PROFILE_FILE, RESOURCE_FILE, OUTPUT_FILE, RUN_FILE}
    missing = sorted(required - set(files))
    if missing:
        raise ValueError(f"raw evidence manifest is missing required files: {missing}")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"raw evidence path escapes raw directory: {name}")
        if not isinstance(expected, dict):
            raise ValueError(f"raw evidence manifest entry must be an object: {name}")
        path = raw_dir / relative
        if not path.is_file():
            raise ValueError(f"raw evidence file is missing: {path}")
        actual_digest = _file_sha256(path)
        if actual_digest != expected.get("sha256"):
            raise ValueError(f"raw evidence checksum mismatch: {name}")
        if path.stat().st_size != expected.get("size_bytes"):
            raise ValueError(f"raw evidence size mismatch: {name}")
    return manifest


def validate_exact_disk_summary(
    summary: dict[str, Any], requirements: ExactDiskRequirements
) -> list[str]:
    """Return all assertion failures rather than hiding later failed gates."""

    failures: list[str] = []
    resources = summary.get("resources")
    if not isinstance(resources, dict):
        resources = {}
    output = summary.get("output_equality")
    if not isinstance(output, dict):
        output = {}
    run = summary.get("run")
    if not isinstance(run, dict):
        run = {}
    profile = summary.get("profile")
    if not isinstance(profile, dict):
        profile = {}

    if requirements.require_command_success and run.get("command_return_code") != 0:
        failures.append("benchmark command did not exit successfully")
    if int(profile.get("exact_disk_event_count", 0) or 0) <= 0:
        failures.append("profile contains no exact-disk events")
    if requirements.require_spill and int(profile.get("disk_spill_bytes", 0) or 0) <= 0:
        failures.append("profile contains no positive disk spill bytes")
    if requirements.require_read and int(profile.get("disk_read_bytes", 0) or 0) <= 0:
        failures.append("profile contains no positive disk read bytes")
    if requirements.expected_source_medium is not None:
        source_media = profile.get("source_media", [])
        if source_media != [requirements.expected_source_medium]:
            failures.append(
                "restore source medium mismatch: "
                f"expected only {requirements.expected_source_medium!r}, "
                f"observed {source_media!r}"
            )
    fallback_count = int(profile.get("fallback_count", 0) or 0)
    if not requirements.allow_fallback and fallback_count:
        failures.append(f"observed {fallback_count} exact-disk fallback event(s)")
    if requirements.require_worker_rss:
        rss_count = int(resources.get("worker_rss_sample_count", 0) or 0)
        rss_peak = resources.get("worker_rss_peak_bytes")
        rss_last = resources.get("worker_rss_last_bytes")
        if rss_count <= 0 or rss_peak is None or rss_last is None:
            failures.append("required worker RSS evidence is missing")
        elif int(rss_last) >= int(rss_peak):
            failures.append("worker RSS did not decrease after disk demotion")
    available_sample = resources.get("mem_available_last_bytes")
    if requirements.require_mem_available and (
        int(resources.get("mem_available_sample_count", 0) or 0) <= 0
        or available_sample is None
    ):
        failures.append("required host MemAvailable evidence is missing")
    if requirements.require_disk_footprint_growth:
        disk_samples = int(resources.get("disk_footprint_sample_count", 0) or 0)
        peak_delta = resources.get("disk_footprint_peak_delta_bytes")
        # Older hand-collected summaries may only retain a peak. They still
        # prove a footprint exists, but the runner records a baseline/delta.
        if peak_delta is None:
            peak_delta = resources.get("disk_footprint_peak_bytes", 0)
        if disk_samples <= 0 or int(peak_delta or 0) <= 0:
            failures.append(
                "required positive disk footprint growth evidence is missing"
            )
    if requirements.require_output_equality:
        if output.get("available") is not True:
            failures.append("required output equality observation is missing")
        elif output.get("output_equal") is not True:
            failures.append("deterministic output changed after exact-disk restore")
    return failures


def build_curated_artifacts(
    raw_dir: Path,
    curated_dir: Path,
    requirements: ExactDiskRequirements | None = None,
) -> dict[str, Any]:
    """Verify local raw evidence and derive summary/assertion JSON files."""

    raw_dir = raw_dir.resolve()
    curated_dir = curated_dir.resolve()
    if raw_dir.name != "raw" or curated_dir.name != "curated":
        raise ValueError("raw and curated directories must be named raw and curated")
    if raw_dir.parent != curated_dir.parent:
        raise ValueError("raw and curated directories must be siblings")
    if curated_dir.exists() and any(curated_dir.iterdir()):
        raise ValueError(f"curated output directory is not empty: {curated_dir}")

    manifest = verify_evidence_manifest(raw_dir)
    run = _load_json(raw_dir / RUN_FILE)
    if run.get("evidence_tier") != "local_raw":
        raise ValueError("run metadata must declare evidence_tier=local_raw")
    summary = {
        "schema_version": 1,
        "evidence_tier": "local_curated",
        "raw_evidence": {
            "evidence_tier": "local_raw",
            "relative_dir": "../raw",
            "manifest": MANIFEST_FILE,
            "manifest_sha256": _file_sha256(raw_dir / MANIFEST_FILE),
            "file_count": len(manifest["files"]),
        },
        "run": run,
        "profile": parse_exact_disk_profile(raw_dir / PROFILE_FILE),
        "resources": summarize_resources(raw_dir / RESOURCE_FILE),
        "output_equality": summarize_output_equality(raw_dir / OUTPUT_FILE),
    }
    active_requirements = requirements or ExactDiskRequirements()
    failures = validate_exact_disk_summary(summary, active_requirements)
    assertions = {
        "schema_version": 1,
        "evidence_tier": "local_curated",
        "ok": not failures,
        "requirements": asdict(active_requirements),
        "failures": failures,
    }

    curated_dir.mkdir(parents=True, exist_ok=True)
    (curated_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (curated_dir / "assertions.json").write_text(
        json.dumps(assertions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"summary": summary, "assertions": assertions}
