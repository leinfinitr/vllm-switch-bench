from __future__ import annotations

import json
from unittest import mock

import requests

from llm_switch_bench.adapters import llama_swap_lifecycle
from llm_switch_bench.adapters import swapservellm_lifecycle


def test_running_models_retries_transient_router_error() -> None:
    transient = mock.Mock(status_code=502)
    transient.raise_for_status.side_effect = requests.HTTPError("bad gateway")
    success = mock.Mock()
    success.raise_for_status.return_value = None
    success.json.return_value = {"running": []}

    with (
        mock.patch.object(llama_swap_lifecycle, "local_session") as session,
        mock.patch.object(llama_swap_lifecycle.time, "sleep") as sleep,
    ):
        session.return_value.get.side_effect = [transient, success]
        assert llama_swap_lifecycle.running_models("http://127.0.0.1:18100") == []

    assert session.return_value.get.call_count == 2
    sleep.assert_called_once_with(0.05)


def test_running_models_raises_persistent_router_error() -> None:
    failed = mock.Mock(status_code=502)
    error = requests.HTTPError("bad gateway")
    failed.raise_for_status.side_effect = error

    with (
        mock.patch.object(llama_swap_lifecycle, "local_session") as session,
        mock.patch.object(llama_swap_lifecycle.time, "sleep") as sleep,
    ):
        session.return_value.get.return_value = failed
        try:
            llama_swap_lifecycle.running_models("http://127.0.0.1:18100")
        except requests.HTTPError as exc:
            assert exc is error
        else:
            raise AssertionError("expected persistent router failure")

    assert session.return_value.get.call_count == 3
    assert sleep.call_count == 2


def test_local_session_bypasses_proxy_environment() -> None:
    assert llama_swap_lifecycle.local_session().trust_env is False
    assert swapservellm_lifecycle.local_session().trust_env is False


def test_swapserve_container_image_metadata_is_stable(monkeypatch) -> None:
    monkeypatch.setattr(
        swapservellm_lifecycle.subprocess,
        "check_output",
        lambda *args, **kwargs: json.dumps(
            [
                {
                    "Id": "image-id",
                    "Digest": "sha256:digest",
                    "RepoDigests": ["repo@sha256:b", "repo@sha256:a"],
                    "Created": "2026-01-01T00:00:00Z",
                    "Architecture": "amd64",
                    "Os": "linux",
                    "Size": 123,
                    "Labels": {
                        "ai.vllm.build.commit": "commit",
                        "ai.vllm.image.tag": "vllm:v1",
                    },
                }
            ]
        ),
    )

    assert swapservellm_lifecycle.container_image_metadata("image:tag") == {
        "reference": "image:tag",
        "id": "image-id",
        "digest": "sha256:digest",
        "repo_digests": ["repo@sha256:a", "repo@sha256:b"],
        "created": "2026-01-01T00:00:00Z",
        "architecture": "amd64",
        "os": "linux",
        "size_bytes": 123,
        "vllm_build_commit": "commit",
        "vllm_image_tag": "vllm:v1",
    }
