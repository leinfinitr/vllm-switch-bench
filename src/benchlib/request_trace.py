from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from benchlib.schema import PROMPTS

REQUIRED_FIELDS = {
    "request_id",
    "scheduled_offset_s",
    "model",
    "endpoint",
    "prompt_name",
    "max_tokens",
    "temperature",
    "stream",
    "seed",
}


def validate_manifest(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("manifest must contain at least one request")
    request_ids: set[str] = set()
    previous = float("-inf")
    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"manifest row {index} missing fields: {sorted(missing)}")
        request_id = str(row["request_id"])
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if request_id in request_ids:
            raise ValueError(f"duplicate request_id: {request_id}")
        request_ids.add(request_id)
        scheduled = float(row["scheduled_offset_s"])
        if not math.isfinite(scheduled):
            raise ValueError("scheduled_offset_s must be finite")
        if scheduled < previous:
            raise ValueError("manifest has non-monotonic scheduled_offset_s")
        if scheduled < 0:
            raise ValueError("scheduled_offset_s must be non-negative")
        if row["endpoint"] not in {"/v1/chat/completions", "/v1/completions"}:
            raise ValueError("unsupported endpoint")
        if row["stream"] is not True:
            raise ValueError("stream must be true")
        if int(row["max_tokens"]) <= 0:
            raise ValueError("max_tokens must be positive")
        if str(row["prompt_name"]) not in PROMPTS:
            raise ValueError("unknown prompt_name")
        previous = scheduled


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_manifest(rows)
    return rows


def write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    validate_manifest(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
