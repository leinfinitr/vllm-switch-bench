import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchlib.request_trace import load_manifest, validate_manifest


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