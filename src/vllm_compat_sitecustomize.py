"""Compatibility shim for mismatched prerelease transformers/torch in this repo env.

The existing prism-research .venv currently has vLLM 0.6.3.post1, torch 2.4.0,
and a very new transformers build. That transformers build imports
`torch.distributed.tensor.device_mesh`, while torch 2.4 exposes the same class as
`torch.distributed.device_mesh`. It also expects experimental integer dtypes from
a newer torchao/torch pair. This shim is only used by the benchmark subprocess
through PYTHONPATH; it does not modify installed packages.
"""
from __future__ import annotations

import sys
import types

try:
    import torch
    import torch.distributed.device_mesh as _device_mesh

    if not hasattr(torch, "int1"):
        _int8 = getattr(torch, "int8")
        for _name in ["int1", "int2", "int3", "int4", "int5", "int6", "int7"]:
            setattr(torch, _name, _int8)

    _tensor_pkg = types.ModuleType("torch.distributed.tensor")
    _device_mesh_mod = types.ModuleType("torch.distributed.tensor.device_mesh")
    setattr(_device_mesh_mod, "DeviceMesh", _device_mesh.DeviceMesh)
    setattr(_tensor_pkg, "DeviceMesh", _device_mesh.DeviceMesh)
    setattr(_tensor_pkg, "device_mesh", _device_mesh_mod)
    sys.modules.setdefault("torch.distributed.tensor", _tensor_pkg)
    sys.modules.setdefault("torch.distributed.tensor.device_mesh", _device_mesh_mod)

    try:
        import transformers.utils.import_utils as _tf_import_utils

        def _torchao_unavailable(*args, **kwargs):
            return False

        _tf_import_utils.is_torchao_available.cache_clear()
        _tf_import_utils.is_torchao_available = _torchao_unavailable
        try:
            import transformers.utils as _tf_utils
            setattr(_tf_utils, "is_torchao_available", _torchao_unavailable)
        except Exception:
            pass
    except Exception:
        pass

    try:
        from transformers import PreTrainedTokenizerBase

        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)
    except Exception:
        pass
except Exception:
    pass
