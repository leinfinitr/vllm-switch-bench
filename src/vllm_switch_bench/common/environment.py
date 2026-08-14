"""Runtime-environment helpers shared by live experiment entry points."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path


def reexec_with_python(
    python: str | Path,
    module: str,
    argv: Sequence[str],
    *,
    workdir: str | Path | None = None,
    import_root: str | Path | None = None,
) -> None:
    """Bind a live run to its Python environment and source checkout."""

    target = Path(python).expanduser().absolute()
    bin_dir = str(target.parent)
    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    os.environ["PATH"] = os.pathsep.join(
        [bin_dir, *(entry for entry in path_entries if entry != bin_dir)]
    )
    if import_root is not None:
        root = str(Path(import_root).expanduser().resolve(strict=True))
        pythonpath = [
            entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry
        ]
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [root, *(entry for entry in pythonpath if entry != root)]
        )
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    if workdir is not None:
        os.chdir(Path(workdir).expanduser().resolve(strict=True))
    current = Path(sys.executable).absolute()
    if target == current:
        return
    os.execv(str(target), [str(target), "-m", module, *argv])
