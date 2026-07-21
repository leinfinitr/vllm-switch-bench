import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/tool/build_cross_system_checksums.py"
spec = importlib.util.spec_from_file_location("build_cross_system_checksums", MODULE)
assert spec and spec.loader
checksums = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checksums
spec.loader.exec_module(checksums)


def test_committed_checksum_manifest_matches_files():
    manifest_path = ROOT / "results/cross_system/latest/checksums.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]
    paths = [entry["path"] for entry in manifest["files"]]
    assert len(paths) == len(set(paths))
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert checksums.sha256(path) == entry["sha256"]
