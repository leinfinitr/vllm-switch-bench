from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vllm_switch_bench.experiments.vllm_profiling.plot import (
    METHOD_LABELS,
    METHOD_ORDER,
    PROFILE_OPERATIONS,
    RAW_METHOD_ORDER,
    aggregate_profiles,
)
from vllm_switch_bench.validation.common import default_results_root, require, validate_metadata


def validate_family(path: Path | None = None) -> None:
    family = path or default_results_root() / "vllm-profiling"
    validate_metadata(family, "vllm-profiling")
    raw_dir = family / "raw"
    require(
        {item.name for item in raw_dir.iterdir()} == {"profile-samples.json"},
        "vllm-profiling: unexpected raw evidence set",
    )
    document = json.loads((raw_dir / "profile-samples.json").read_text(encoding="utf-8"))
    config = json.loads((family / "config" / "campaign.json").read_text(encoding="utf-8"))
    require(document.get("schema_version") == 3, "vllm-profiling: raw schema mismatch")
    require(config.get("schema_version") == 3, "vllm-profiling: config schema mismatch")
    require(document.get("model") == config.get("model"), "vllm-profiling: model mismatch")
    scope = document.get("frozen_scope", {})
    for field in ("dtype", "max_model_len", "gpu_memory_utilization", "engine_mode"):
        require(scope.get(field) == config.get(field), f"vllm-profiling: {field} mismatch")

    methods = list(config["methods"])
    require(tuple(methods) == RAW_METHOD_ORDER, "vllm-profiling: raw method order mismatch")
    require(
        config.get("display_labels")
        == {method: label for method, label in METHOD_LABELS.items() if method != label},
        "vllm-profiling: display labels mismatch",
    )
    sample_count = int(config["samples_per_method"])
    require(sample_count == 3, "vllm-profiling: retained sample count must be three")
    require(
        config.get("metric_boundary") == document.get("metric_boundary"),
        "vllm-profiling: metric boundary mismatch",
    )
    samples = document.get("samples")
    require(isinstance(samples, list), "vllm-profiling: samples must be a list")
    require(len(samples) == len(methods) * sample_count, "vllm-profiling: sample count mismatch")
    require(
        sorted({str(item.get("method")) for item in samples}) == sorted(methods),
        "vllm-profiling: method set mismatch",
    )
    sources = document.get("sources")
    require(isinstance(sources, list) and sources, "vllm-profiling: sources missing")
    source_provenance = document.get("source_provenance")
    require(
        isinstance(source_provenance, dict) and set(source_provenance) == set(sources),
        "vllm-profiling: source provenance does not close",
    )
    for source, provenance in source_provenance.items():
        require(isinstance(provenance, dict), f"vllm-profiling: {source} provenance invalid")
        for repository in ("benchmark_repo", "engine_repo"):
            identity = provenance.get(repository, {})
            require(
                identity.get("commit") and isinstance(identity.get("dirty"), bool),
                f"vllm-profiling: {source} {repository} identity is incomplete",
            )
        require(provenance.get("model_identity"), f"vllm-profiling: {source} model missing")
        engine = provenance["engine_repo"]
        runtime = provenance.get("engine_runtime", provenance.get("runtime", {}))
        import_path = engine.get("module_path") or runtime.get("vllm_import_path")
        require(import_path, f"vllm-profiling: {source} imported vLLM path is missing")
    for method in methods:
        rows = [item for item in samples if item.get("method") == method]
        require(len(rows) == sample_count, f"vllm-profiling: {method} sample count mismatch")
        require(
            sorted(int(item["sample_index"]) for item in rows) == list(range(1, sample_count + 1)),
            f"vllm-profiling: {method} sample indexes mismatch",
        )
        for row in rows:
            for operation in PROFILE_OPERATIONS:
                total = float(row[f"{operation}_total_s"])
                phases = row.get(f"{operation}_phases_s", {})
                require(
                    math.isfinite(total) and total > 0,
                    f"vllm-profiling: invalid {operation} total",
                )
                require(
                    isinstance(phases, dict) and phases,
                    f"vllm-profiling: {operation} phases missing",
                )
                require(
                    all(
                        math.isfinite(float(value)) and float(value) >= 0
                        for value in phases.values()
                    ),
                    f"vllm-profiling: invalid {operation} phase value",
                )
                require(
                    math.isclose(
                        sum(float(value) for value in phases.values()),
                        total,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    ),
                    f"vllm-profiling: {operation} phase accounting does not close",
                )
            source = row.get("source")
            require(
                isinstance(source, str) and source in sources and not Path(source).is_absolute(),
                "vllm-profiling: invalid sample source",
            )
            if method in {"vLLM L2 Cold", "vLLM L2 Warm"}:
                cache = row.get("cache_evidence", {})
                require(cache, f"vllm-profiling: {method} cache evidence missing")
                resident = float(cache.get("resident_ratio_before_wake", -1))
                read_ratio = float(cache.get("storage_read_ratio", -1))
                if method == "vLLM L2 Cold":
                    require(
                        cache.get("treatment") == "posix_fadvise_dontneed"
                        and resident <= 0.05
                        and read_ratio >= 0.90,
                        "vllm-profiling: cold L2 cache evidence is invalid",
                    )
                else:
                    require(
                        cache.get("treatment") == "none"
                        and resident >= 0.90
                        and read_ratio <= 0.10,
                        "vllm-profiling: warm L2 cache evidence is invalid",
                    )

    comparability = document.get("comparability", {})
    require(
        comparability.get("shared_conditions") and comparability.get("cache_conditions"),
        "vllm-profiling: comparison controls missing",
    )
    require(
        comparability.get("prohibited_claim"),
        "vllm-profiling: prohibited claim missing",
    )
    expected = aggregate_profiles(document)
    summary = json.loads((family / "summary.json").read_text(encoding="utf-8"))
    require(summary == expected, "vllm-profiling: raw recomputation differs")
    require(
        [row["method"] for row in summary["methods"]] == list(METHOD_ORDER),
        "vllm-profiling: summary method order mismatch",
    )
    for suffix in ("png", "pdf"):
        require(
            (family / "figures" / f"vllm-profiling.{suffix}").is_file(),
            f"vllm-profiling: {suffix.upper()} figure missing",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate vLLM profiling result semantics")
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args(argv)
    validate_family(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
