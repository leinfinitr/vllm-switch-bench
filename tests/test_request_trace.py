import json
import math

import pytest


from llm_switch_bench.common.traces import load_manifest, validate_manifest


def test_validate_manifest_rejects_duplicate_ids_and_non_monotonic_arrivals():
    def row(request_id, scheduled):
        return {
            "request_id": request_id,
            "scheduled_offset_s": scheduled,
            "model": "a",
            "endpoint": "/v1/chat/completions",
            "prompt_name": "short_short",
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "seed": 1,
        }

    with pytest.raises(ValueError, match="duplicate request_id"):
        validate_manifest([row("r1", 0.0), row("r1", 1.0)])
    with pytest.raises(ValueError, match="non-monotonic"):
        validate_manifest([row("r1", 1.0), row("r2", 0.0)])


def test_load_manifest_preserves_frozen_rows(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [
        {
            "request_id": "r1",
            "scheduled_offset_s": 0.0,
            "model": "a",
            "endpoint": "/v1/chat/completions",
            "prompt_name": "short_short",
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "seed": 1,
        }
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert load_manifest(path) == rows


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("request_id", "", "request_id"),
        ("scheduled_offset_s", math.nan, "finite"),
        ("endpoint", "/unknown", "endpoint"),
        ("stream", False, "stream"),
        ("max_tokens", 0, "max_tokens"),
        ("prompt_name", "missing", "prompt_name"),
    ],
)
def test_validate_manifest_rejects_invalid_semantics(field, value, match):
    row = {
        "request_id": "r1",
        "scheduled_offset_s": 0.0,
        "model": "a",
        "endpoint": "/v1/chat/completions",
        "prompt_name": "short_short",
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "seed": 1,
    }
    row[field] = value
    with pytest.raises(ValueError, match=match):
        validate_manifest([row])
