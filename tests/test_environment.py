from __future__ import annotations

import sys
import os
from pathlib import Path

from llm_switch_bench.common import environment


def test_reexec_skips_current_python(monkeypatch) -> None:
    called = False

    def fail(*_args: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(environment.os, "execv", fail)

    environment.reexec_with_python(sys.executable, "package.module", ["--flag"])

    assert called is False


def test_reexec_uses_selected_python(monkeypatch, tmp_path: Path) -> None:
    python = tmp_path / "venv" / "bin" / "python"
    captured: list[object] = []
    monkeypatch.setattr(environment.os, "execv", lambda *args: captured.extend(args))

    environment.reexec_with_python(python, "package.module", ["--flag", "value"])

    target = str(python.absolute())
    assert captured == [target, [target, "-m", "package.module", "--flag", "value"]]
    assert os.environ["PATH"].split(os.pathsep)[0] == str(python.parent.absolute())


def test_reexec_current_python_still_selects_its_tool_directory(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    environment.reexec_with_python(sys.executable, "package.module", [])

    assert os.environ["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).absolute().parent)


def test_reexec_binds_workdir_and_import_root(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/existing")

    environment.reexec_with_python(
        sys.executable,
        "package.module",
        [],
        workdir=checkout,
        import_root=checkout,
    )

    assert Path.cwd() == checkout
    assert os.environ["PYTHONPATH"].split(os.pathsep) == [str(checkout), "/existing"]
    assert sys.path[0] == str(checkout)
