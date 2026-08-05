#!/usr/bin/env python3
"""Verify the release-v0.1 artifact manifests and publication closure."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "release-v0.1"
PUBLICATION = ARTIFACT / "checksums.sha256"
COMPLETE = ARTIFACT / "all-files.sha256"
MATERIAL_RELEASE_RATIO = 0.5


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid SHA-256 row") from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{path}:{line_number}: invalid digest")
        rows.append((digest, relative))
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    return rows


def tracked_paths() -> set[str]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Release archives intentionally omit .git. The immutable complete
        # manifest becomes the packaged tracked-file inventory in that case.
        return {relative for _digest, relative in parse_manifest(COMPLETE)} | {
            PUBLICATION.relative_to(ROOT).as_posix(),
            COMPLETE.relative_to(ROOT).as_posix(),
        }
    return {item.decode() for item in output.split(b"\0") if item}


def verify_manifest(path: Path, tracked: set[str]) -> list[str]:
    failures = []
    seen = set()
    for expected, relative in parse_manifest(path):
        if relative in seen:
            failures.append(f"duplicate path in {path.name}: {relative}")
            continue
        seen.add(relative)
        candidate = ROOT / relative
        if not candidate.is_file():
            failures.append(f"missing artifact file: {relative}")
            continue
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"checksum mismatch: {relative}")
        if relative not in tracked:
            failures.append(f"manifest path is not tracked: {relative}")
    return failures


def main() -> int:
    tracked = tracked_paths()
    failures = []
    for manifest in (PUBLICATION, COMPLETE):
        if not manifest.is_file():
            failures.append(f"missing manifest: {manifest.relative_to(ROOT)}")
        else:
            failures.extend(verify_manifest(manifest, tracked))

    complete_paths = (
        {relative for _, relative in parse_manifest(COMPLETE)} if COMPLETE.is_file() else set()
    )
    declared = {
        path.relative_to(ROOT).as_posix()
        for path in ARTIFACT.rglob("*")
        if path.is_file() and path.name != COMPLETE.name
    }
    if complete_paths != declared:
        missing = sorted(declared - complete_paths)
        extra = sorted(complete_paths - declared)
        if missing:
            failures.append(f"complete manifest omits: {missing}")
        if extra:
            failures.append(f"complete manifest has unexpected paths: {extra}")

    # Nested producer manifests must be self-contained release-relative indexes,
    # never pointers back to a collector's local ignored tree.
    for manifest in sorted((ARTIFACT / "provenance").glob("*.sha256")):
        for expected_digest, relative in parse_manifest(manifest):
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                failures.append(f"non-portable nested provenance path: {relative}")
                continue
            candidate = ARTIFACT / relative
            if candidate.is_symlink():
                failures.append(f"nested provenance path is a symlink: {relative}")
            elif not candidate.is_file():
                failures.append(f"nested provenance path is missing: {relative}")
            elif hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_digest:
                failures.append(f"nested provenance checksum mismatch: {relative}")

    trace_path = ARTIFACT / (
        "raw/provenance-inputs/proposed-e2e-raw/request-switch-alternating.jsonl"
    )
    e2e_output_path = ARTIFACT / (
        "raw/provenance-inputs/proposed-e2e-raw/proposed-request-switch-alternating-r0.jsonl"
    )
    matrix_path = ARTIFACT / "raw/provenance-inputs/proposed-e2e-raw/matrix.json"
    if trace_path.is_file() and e2e_output_path.is_file() and matrix_path.is_file():
        try:
            trace_rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
            output_rows = [
                json.loads(line) for line in e2e_output_path.read_text().splitlines() if line
            ]
            matrix_rows = json.loads(matrix_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(f"cannot parse E2E trace/output identity inputs: {exc}")
        else:
            trace_by_id = {row.get("request_id"): row for row in trace_rows}
            output_by_id = {row.get("request_id"): row for row in output_rows}
            if None in trace_by_id or len(trace_by_id) != len(trace_rows):
                failures.append("E2E trace has missing or duplicate request IDs")
            if None in output_by_id or len(output_by_id) != len(output_rows):
                failures.append("E2E output has missing or duplicate request IDs")
            if set(trace_by_id) != set(output_by_id):
                failures.append("E2E trace/output request IDs differ")
            identity_fields = (
                "endpoint",
                "model",
                "prompt_name",
                "max_tokens",
                "temperature",
                "seed",
                "stream",
                "scheduled_offset_s",
            )
            for request_id in set(trace_by_id) & set(output_by_id):
                trace = trace_by_id[request_id]
                output = output_by_id[request_id]
                if any(output.get(field) != trace.get(field) for field in identity_fields):
                    failures.append(f"E2E trace/output identity mismatch: {request_id}")
                    break
                timings = (
                    output.get("completion_latency_ms"),
                    output.get("semantic_ttft_ms"),
                    output.get("dispatch_lag_ms"),
                )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                    for value in timings
                ):
                    failures.append(f"E2E output has invalid timings: {request_id}")
                    break
                if (
                    output.get("status") != 200
                    or output.get("error") is not None
                    or output.get("stream_done") is not True
                    or not str(output.get("output_text", "")).strip()
                ):
                    failures.append(f"E2E output fails strict success: {request_id}")
                    break
            expected_trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            expected_output_sha = hashlib.sha256(e2e_output_path.read_bytes()).hexdigest()
            if len(matrix_rows) != 1 or matrix_rows[0].get("manifest_sha256") != expected_trace_sha:
                failures.append("E2E matrix is not bound to the frozen trace")
            if len(matrix_rows) != 1 or matrix_rows[0].get("output_sha256") != expected_output_sha:
                failures.append("E2E matrix is not bound to the retained output")
            if len(matrix_rows) == 1 and (
                matrix_rows[0].get("return_code") != 0
                or matrix_rows[0].get("failed") != 0
                or matrix_rows[0].get("rows") != len(trace_rows)
                or matrix_rows[0].get("requests") != len(trace_rows)
            ):
                failures.append("E2E matrix summary is incomplete or unsuccessful")
            published_e2e_path = ARTIFACT / "raw/proposed/e2e-alternating.json"
            if published_e2e_path.is_file():
                published_rows = json.loads(published_e2e_path.read_text())
                if published_rows != output_rows:
                    failures.append(
                        "published Proposed E2E rows differ from the digest-bound output"
                    )
            else:
                failures.append("published Proposed E2E rows are missing")

            llama_e2e_path = ARTIFACT / "raw/llama-swap/e2e-alternating.json"
            llama_e2e_jsonl_path = ARTIFACT / "raw/llama-swap/e2e-alternating.jsonl"
            if llama_e2e_path.is_file() and llama_e2e_jsonl_path.is_file():
                try:
                    llama_rows = json.loads(llama_e2e_path.read_text())
                    llama_jsonl_rows = [
                        json.loads(line)
                        for line in llama_e2e_jsonl_path.read_text().splitlines()
                        if line
                    ]
                except (json.JSONDecodeError, OSError) as exc:
                    failures.append(f"cannot parse llama-swap E2E evidence: {exc}")
                else:
                    if llama_rows != llama_jsonl_rows:
                        failures.append("published llama-swap E2E rows differ from retained JSONL")
                    llama_by_id = {row.get("request_id"): row for row in llama_rows}
                    if None in llama_by_id or len(llama_by_id) != len(llama_rows):
                        failures.append(
                            "llama-swap E2E output has missing or duplicate request IDs"
                        )
                    if set(trace_by_id) != set(llama_by_id):
                        failures.append("llama-swap trace/output request IDs differ")
                    for request_id in set(trace_by_id) & set(llama_by_id):
                        trace = trace_by_id[request_id]
                        output = llama_by_id[request_id]
                        if any(output.get(field) != trace.get(field) for field in identity_fields):
                            failures.append(
                                f"llama-swap trace/output identity mismatch: {request_id}"
                            )
                            break
                        timings = (
                            output.get("completion_latency_ms"),
                            output.get("semantic_ttft_ms"),
                            output.get("dispatch_lag_ms"),
                        )
                        if any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(value)
                            or value < 0
                            for value in timings
                        ):
                            failures.append(
                                f"llama-swap E2E output has invalid timings: {request_id}"
                            )
                            break
                        if (
                            output.get("status") != 200
                            or output.get("error") is not None
                            or output.get("stream_done") is not True
                            or not str(output.get("output_text", "")).strip()
                        ):
                            failures.append(
                                f"llama-swap E2E output fails strict success: {request_id}"
                            )
                            break
            else:
                failures.append("llama-swap E2E evidence is incomplete")
    else:
        failures.append("E2E trace/output identity inputs are incomplete")

    summary = ARTIFACT / "summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        patch_path = payload.get("provenance", {}).get("SwapServeLLM", {}).get("benchmark_patch")
        if patch_path and not (ARTIFACT / patch_path).is_file():
            failures.append(f"summary references missing path: {patch_path}")

    exact_profile = ARTIFACT / "raw/exact-disk/exact_disk_profile.jsonl"
    exact_metadata = ARTIFACT / "raw/exact-disk/run-metadata.json"
    exact_manifest = ARTIFACT / "raw/exact-disk/bundle-manifest.json"
    exact_commit = ARTIFACT / "raw/exact-disk/bundle-COMMIT"
    exact_payload = ARTIFACT / "raw/exact-disk/payload-hash.json"
    exact_output = ARTIFACT / "raw/exact-disk/output_observation.json"
    exact_footprint = ARTIFACT / "raw/exact-disk/disk-footprint.json"
    pressure_path = ARTIFACT / "raw/proposed/controller-pressure-release.json"
    manifest_bytes_total = 0
    allocator_sleeps: list[dict[str, Any]] = []
    if exact_profile.is_file():
        events = [
            json.loads(line)
            for line in exact_profile.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        phases = {event.get("phase") for event in events}
        required_phases = {"exact_disk_spill", "exact_disk_demotion", "exact_disk_restore"}
        if not required_phases <= phases:
            failures.append("exact-disk profile omits required lifecycle phases")
        if any(event.get("fallback") is True for event in events):
            failures.append("exact-disk profile contains a fallback event")
        if sum(int(event.get("disk_spill_bytes", 0)) for event in events) <= 0:
            failures.append("exact-disk profile has no physical spill bytes")
        if sum(int(event.get("disk_read_bytes", 0)) for event in events) <= 0:
            failures.append("exact-disk profile has no physical restore bytes")
        allocator_sleeps = [event for event in events if event.get("phase") == "allocator_sleep"]
        if not allocator_sleeps or any(
            int(event.get("cpu_backup_host_cache_flush_errors", -1)) != 0
            or int(event.get("cpu_backup_host_cache_flush_count", 0)) <= 0
            or int(event.get("cpu_backup_release_count", 0)) <= 0
            or int(event.get("cpu_backup_release_bytes", 0)) < int(event.get("backup_bytes", 0))
            or int(event.get("backup_bytes", 0)) <= 0
            for event in allocator_sleeps
        ):
            failures.append("exact-disk profile lacks a material full-backup host-cache release")
    else:
        failures.append("exact-disk profile is missing")

    if exact_metadata.is_file():
        metadata = json.loads(exact_metadata.read_text(encoding="utf-8"))
        if not metadata.get("engine", {}).get("collection_commit"):
            failures.append("exact-disk run metadata lacks engine commit")
        if not metadata.get("model") or not metadata.get("launch_parameters"):
            failures.append("exact-disk run metadata lacks model or launch parameters")
    else:
        failures.append("exact-disk run metadata is missing")

    if all(
        path.is_file()
        for path in (
            exact_manifest,
            exact_commit,
            exact_payload,
            exact_output,
            exact_footprint,
        )
    ):
        manifest_bytes = exact_manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        payload_hash = json.loads(exact_payload.read_text(encoding="utf-8"))
        output = json.loads(exact_output.read_text(encoding="utf-8"))
        footprint = json.loads(exact_footprint.read_text(encoding="utf-8"))[
            "filesystem_observation"
        ]
        if (
            exact_commit.read_text(encoding="utf-8").strip()
            != hashlib.sha256(manifest_bytes).hexdigest()
        ):
            failures.append("exact-disk commit marker does not bind the manifest")
        manifest_bytes_total = int(manifest.get("payload_size_bytes", 0))
        segment_bytes_total = sum(
            int(segment.get("nbytes", segment.get("size_bytes", 0)))
            for segment in manifest.get("segments", [])
        )
        if manifest_bytes_total <= 0 or segment_bytes_total != manifest_bytes_total:
            failures.append("exact-disk manifest payload accounting is inconsistent")
        if int(payload_hash.get("payload_size_bytes", 0)) != manifest_bytes_total:
            failures.append("exact-disk payload size does not match the manifest")
        if (
            int(footprint.get("logical_size_bytes", 0)) != manifest_bytes_total
            or int(footprint.get("allocated_bytes", 0)) < manifest_bytes_total
        ):
            failures.append("exact-disk physical footprint does not cover the payload")
        payload_digest = payload_hash.get("payload_sha256")
        if not isinstance(payload_digest, str) or len(payload_digest) != 64:
            failures.append("exact-disk payload SHA-256 is invalid")
        if output.get("before") != output.get("after"):
            failures.append("exact-disk output differs after restore")
        if allocator_sleeps and any(
            int(event.get("backup_bytes", 0)) != manifest_bytes_total
            or int(event.get("cpu_backup_release_bytes", 0)) != manifest_bytes_total
            for event in allocator_sleeps
        ):
            failures.append("exact-disk allocator release does not match the authenticated payload")
        demotions = output.get("demotion", {}).get("results", [])
        if not demotions or any(
            not row.get("released")
            or int(row.get("pending_release_bytes", -1)) != 0
            or int(row.get("released_bytes_total", 0)) != int(row.get("requested_bytes", -1))
            or int(row.get("requested_bytes", 0)) != manifest_bytes_total
            for row in demotions
        ):
            failures.append("exact-disk demotion did not complete its requested release")
    else:
        failures.append(
            "exact-disk manifest, commit, payload hash, output, or footprint evidence is missing"
        )

    if pressure_path.is_file():
        pressure = json.loads(pressure_path.read_text(encoding="utf-8"))
        before = pressure.get("before", {})
        after = pressure.get("after", {})
        queued = int(pressure.get("release_response", {}).get("queued_bytes", 0))
        if not pressure.get("release_response", {}).get("ok") or queued <= 0:
            failures.append("controller pressure release was not accepted")
        before_pool = before.get("pool_stats", {})
        authenticated_reclaimable = int(before_pool.get("ram_reclaimable_without_disk_bytes", 0))
        if authenticated_reclaimable <= 0 or queued != authenticated_reclaimable:
            failures.append(
                "controller pressure queue does not match the pre-release reclaimable footprint"
            )
        material_release_bytes = max(1, int(authenticated_reclaimable * MATERIAL_RELEASE_RATIO))
        memavailable_delta = int(pressure.get("memavailable_delta_bytes", 0))
        process_rss_drop = -sum(
            min(int(delta), 0) for delta in pressure.get("client_rss_delta_bytes", {}).values()
        )
        if memavailable_delta < material_release_bytes:
            failures.append(
                "controller pressure MemAvailable recovery is not material relative to queued bytes"
            )
        if process_rss_drop < material_release_bytes:
            failures.append(
                "controller pressure process RSS recovery is not material relative to queued bytes"
            )
        if int(pressure.get("memavailable_delta_bytes", 0)) != int(
            after.get("memavailable_bytes", 0)
        ) - int(before.get("memavailable_bytes", 0)):
            failures.append("controller MemAvailable delta is inconsistent with snapshots")
        after_pool = after.get("pool_stats", {})
        if (
            int(after_pool.get("pending_release_bytes", -1)) != 0
            or int(after_pool.get("pending_release_request_count", -1)) != 0
        ):
            failures.append("controller pressure evidence retains pending releases")
        before_clients = before.get("clients", {})
        after_clients = after.get("clients", {})
        if set(before_clients) != set(after_clients):
            failures.append("controller pressure evidence changed process incarnations")
        elif any(
            int(before_clients[name].get("pid", -1)) != int(after_clients[name].get("pid", -2))
            for name in before_clients
        ):
            failures.append("controller pressure evidence changed client PIDs")
        observed_rss_deltas = pressure.get("client_rss_delta_bytes", {})
        if set(observed_rss_deltas) != set(before_clients) or any(
            int(observed_rss_deltas[name])
            != int(after_clients[name].get("process_tree_rss_bytes", 0))
            - int(before_clients[name].get("process_tree_rss_bytes", 0))
            for name in set(before_clients) & set(after_clients)
        ):
            failures.append("controller RSS deltas are inconsistent with snapshots")
        released_delta = sum(
            int(after_clients[name].get("released_bytes_total", 0))
            - int(before_clients[name].get("released_bytes_total", 0))
            for name in set(before_clients) & set(after_clients)
        )
        requested_delta = sum(
            int(after_clients[name].get("requested_release_bytes_total", 0))
            - int(before_clients[name].get("requested_release_bytes_total", 0))
            for name in set(before_clients) & set(after_clients)
        )
        if released_delta != queued or requested_delta != queued:
            failures.append("controller pressure release accounting does not match queued bytes")
    else:
        failures.append("controller physical-memory pressure evidence is missing")

    if failures:
        print("release artifact verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(
        "release artifact verification: ok "
        f"({len(parse_manifest(PUBLICATION))} publication files, "
        f"{len(parse_manifest(COMPLETE))} complete files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
