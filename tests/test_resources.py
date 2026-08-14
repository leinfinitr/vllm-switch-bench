from __future__ import annotations


import pytest


from llm_switch_bench.common.resources import (
    docker_container_rss_mib,
    parse_gpu_memory_used_mib,
    podman_container_rss_mib,
    process_tree_rss_bytes,
    process_tree_rss_mib,
)


def test_parse_nvidia_smi_memory_used():
    assert parse_gpu_memory_used_mib("123\n") == 123
    assert parse_gpu_memory_used_mib("123 MiB\n") == 123
    assert parse_gpu_memory_used_mib("") is None
    assert parse_gpu_memory_used_mib("not-a-number") is None


def test_process_tree_rss_mib_aggregates_parent_and_children(monkeypatch):
    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def memory_info(self):
            return type("Mem", (), {"rss": self.pid * 1024 * 1024})()

        def children(self, recursive=True):
            assert recursive is True
            return [FakeProc(2), FakeProc(3)]

    monkeypatch.setattr(
        "llm_switch_bench.common.resources.psutil.Process", lambda pid: FakeProc(pid)
    )
    assert process_tree_rss_bytes(1) == 6 * 1024 * 1024
    assert process_tree_rss_mib(1) == pytest.approx(6.0)


def test_process_tree_rss_mib_returns_none_without_pid():
    assert process_tree_rss_mib(None) is None


def test_container_rss_helpers_return_none_for_empty_names():
    assert docker_container_rss_mib([]) is None
    assert podman_container_rss_mib([]) is None
