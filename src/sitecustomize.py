import importlib.util
import pathlib

_path = pathlib.Path(__file__).with_name("vllm_compat_sitecustomize.py")
_spec = importlib.util.spec_from_file_location("vllm_compat_sitecustomize", _path)
if _spec is not None and _spec.loader is not None:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
