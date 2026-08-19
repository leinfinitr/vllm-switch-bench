"""Stage, validate, and optionally publish one complete live experiment family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm_switch_bench.artifacts import build_family
from vllm_switch_bench.common.provenance import git_metadata, repository_root
from vllm_switch_bench.experiments.exact_disk.evidence import verify_evidence_manifest
from vllm_switch_bench.experiments.vllm_profiling.compile import compile_profiles
from vllm_switch_bench.families import FAMILY_NAMES, FAMILIES_BY_NAME
from vllm_switch_bench.publication import default_results_root
from vllm_switch_bench.validation.validate_all import _load_validator

MODELS = ("qwen-0.5b", "qwen-1.5b", "qwen-3b")


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ValueError(f"measurement input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_process_local_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_process_local_fields(item)
            for key, item in value.items()
            if key not in {"data_ptr", "base_ptr", "allocator_ptr"}
        }
    if isinstance(value, list):
        return [_remove_process_local_fields(item) for item in value]
    return value


def _stage_lifecycle(args: argparse.Namespace, raw: Path) -> None:
    for model in MODELS:
        for source_dir, destination in (
            (args.vllm_switch, "vllm-switch"),
            (args.vllm_l1, "vllm-l1"),
            (args.vllm_l2, "vllm-l2"),
            (args.swapserve, "swapserve"),
        ):
            _copy(source_dir / f"{model}.json", raw / destination / f"{model}.json")
    _copy(args.llama_swap, raw / "llama-swap" / "lifecycle.json")


def _stage_profiling(args: argparse.Namespace, raw: Path) -> None:
    document = compile_profiles(
        args.cold_summary,
        args.vllm_blocks,
        args.switch_blocks,
    )
    _write_json(raw / "profile-samples.json", document)


def _stage_request(args: argparse.Namespace, raw: Path) -> None:
    for name, source in (("vllm-switch", args.vllm_switch), ("llama-swap", args.llama_swap)):
        _copy(source, raw / name / "e2e-alternating.jsonl")
        _copy(source.with_suffix(".run.json"), raw / name / "e2e-alternating.run.json")


def _model_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in MODELS or name in result:
            raise ValueError(f"invalid or duplicate --reuse value: {value}")
        result[name] = Path(path)
    if set(result) != set(MODELS):
        raise ValueError(f"--reuse must cover exactly: {', '.join(MODELS)}")
    return result


def _stage_backup(args: argparse.Namespace, raw: Path) -> None:
    for model, source in _model_mapping(args.reuse).items():
        _copy(source, raw / "vllm-switch" / "reuse" / f"{model}.json")
    _copy(args.reclaim, raw / "vllm-switch" / "reclaim.json")


def _runtime_bundle(run: Path) -> tuple[Path, Path, Path]:
    manifests = list((run / "runtime-bundle").glob("*/*.ready/manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"exact disk: expected one runtime manifest, found {len(manifests)}")
    manifest = manifests[0]
    return manifest, manifest.with_name("COMMIT"), manifest.with_name("data.bin")


def _stage_exact_disk(args: argparse.Namespace, raw: Path) -> None:
    run = args.run
    source_raw = run / "raw"
    verify_evidence_manifest(source_raw)
    assertions = json.loads((run / "curated" / "assertions.json").read_text(encoding="utf-8"))
    if assertions.get("ok") is not True:
        raise ValueError(f"exact disk source assertions failed: {assertions.get('failures')}")
    manifest_path, commit_path, payload_path = _runtime_bundle(run)
    runtime_manifest_bytes = manifest_path.read_bytes()
    if (
        commit_path.read_text(encoding="utf-8").strip()
        != hashlib.sha256(runtime_manifest_bytes).hexdigest()
    ):
        raise ValueError("exact disk runtime COMMIT does not bind manifest bytes")
    manifest = _remove_process_local_fields(json.loads(runtime_manifest_bytes))
    manifest_output = raw / "exact-disk" / "bundle-manifest.json"
    _write_json(manifest_output, manifest)
    (raw / "exact-disk" / "bundle-COMMIT").write_text(
        hashlib.sha256(manifest_output.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    _copy(source_raw / "exact_disk_profile.jsonl", raw / "exact-disk/exact_disk_profile.jsonl")
    _copy(source_raw / "output_observation.json", raw / "exact-disk/output_observation.json")
    _copy(source_raw / "run.json", raw / "exact-disk/run-metadata.json")
    stat = payload_path.stat()
    _write_json(
        raw / "exact-disk/payload-hash.json",
        {
            "payload_size_bytes": stat.st_size,
            "payload_sha256": _sha256(payload_path),
            "retention": "payload omitted from Git; manifest and hash retained",
        },
    )
    _write_json(
        raw / "exact-disk/disk-footprint.json",
        {
            "schema_version": 1,
            "filesystem_observation": {
                "path": "data.bin",
                "logical_size_bytes": stat.st_size,
                "allocated_blocks": stat.st_blocks,
                "block_size_bytes": 512,
                "allocated_bytes": stat.st_blocks * 512,
            },
            "interpretation": "Payload occupied physical filesystem blocks before omission.",
        },
    )


STAGERS = {
    "lifecycle-latency": _stage_lifecycle,
    "vllm-profiling": _stage_profiling,
    "request-driven-switch": _stage_request,
    "backup-reuse-reclaim": _stage_backup,
    "exact-disk": _stage_exact_disk,
}


def _write_provenance(family: Path, collected_at: str) -> None:
    benchmark = git_metadata(repository_root())
    _write_json(
        family / "provenance.json",
        {
            "schema_version": 1,
            "status": "local-rerun",
            "collected_at": collected_at,
            "note": (
                "Local GPU rerun promoted from validator-approved results/tmp evidence. "
                "Per-run raw metadata is authoritative for runtime identity."
            ),
            "promotion_benchmark": benchmark,
        },
    )


def _apply(candidate: Path, destination: Path, candidate_root: Path) -> Path:
    backup = candidate_root / "previous" / destination.name
    if backup.exists():
        raise ValueError(f"promotion backup already exists: {backup}")
    prepared = destination.parent / f".{destination.name}.promotion-{os.getpid()}"
    if prepared.exists():
        raise ValueError(f"stale promotion directory exists: {prepared}")
    shutil.copytree(candidate, prepared)
    backup.parent.mkdir(parents=True, exist_ok=True)
    destination.rename(backup)
    try:
        prepared.rename(destination)
    except Exception:
        backup.rename(destination)
        raise
    return backup


def promote(args: argparse.Namespace) -> tuple[Path, Path | None]:
    candidate_root = args.candidate_root.resolve()
    candidate = candidate_root / args.family
    if candidate_root.exists():
        raise ValueError(f"candidate root already exists: {candidate_root}")
    destination = default_results_root() / args.family
    candidate_root.mkdir(parents=True)
    shutil.copytree(destination, candidate)
    shutil.rmtree(candidate / "raw")
    raw = candidate / "raw"
    raw.mkdir()
    _write_provenance(candidate, args.collected_at)
    STAGERS[args.family](args, raw)
    build_family(args.family, candidate_root)
    _load_validator(FAMILIES_BY_NAME[args.family].validator)(candidate)
    backup = _apply(candidate, destination, candidate_root) if args.apply else None
    return candidate, backup


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    values = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILY_NAMES)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--collected-at", default=datetime.now(timezone.utc).date().isoformat())
    if not values or values[0] in {"-h", "--help"}:
        return parser.parse_args(values)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("family", choices=FAMILY_NAMES)
    known, _ = bootstrap.parse_known_args(values)
    if known.family == "lifecycle-latency":
        for option in ("vllm-switch", "vllm-l1", "vllm-l2", "swapserve"):
            parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument("--llama-swap", type=Path, required=True)
    elif known.family == "vllm-profiling":
        parser.add_argument("--cold-summary", type=Path, required=True)
        parser.add_argument("--vllm-blocks", type=Path, required=True)
        parser.add_argument("--switch-blocks", type=Path, required=True)
    elif known.family == "request-driven-switch":
        parser.add_argument("--vllm-switch", type=Path, required=True)
        parser.add_argument("--llama-swap", type=Path, required=True)
    elif known.family == "backup-reuse-reclaim":
        parser.add_argument("--reuse", action="append", default=[], required=True)
        parser.add_argument("--reclaim", type=Path, required=True)
    else:
        parser.add_argument("--run", type=Path, required=True)
    return parser.parse_args(values)


def main(argv: list[str] | None = None) -> int:
    candidate, backup = promote(parse_args(argv))
    print(candidate)
    if backup is not None:
        print(f"previous result retained at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
