"""Safe tensor-only loading for trusted legacy MSINet checkpoints.

The fallback parser accepts only the small set of tensor reconstruction globals
used by PyTorch zip checkpoints. It never imports checkpoint-provided classes or
executes arbitrary pickle globals.
"""

import inspect
import io
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import torch


class _StorageType:
    def __init__(self, dtype):
        self.dtype = dtype


class _StorageRef:
    def __init__(self, dtype, key, location, size):
        self.dtype = dtype
        self.key = str(key)
        self.location = str(location)
        self.size = int(size)


class _TensorRef:
    def __init__(self, storage, offset, shape, stride):
        self.storage = storage
        self.offset = int(offset)
        self.shape = tuple(shape)
        self.stride = tuple(stride)

    def __setstate__(self, state):
        # Tensor metadata hooks do not contain tensor data and are not needed
        # when restoring an inference state_dict.
        self.state = state


def _rebuild_tensor(storage, offset, shape, stride, *unused):
    return _TensorRef(storage, offset, shape, stride)


def _rebuild_from_type(rebuild, new_type, args, state):
    tensor = rebuild(*args)
    tensor.__setstate__(state)
    return tensor


_STORAGE_TYPES = {
    "ByteStorage": torch.uint8,
    "CharStorage": torch.int8,
    "ShortStorage": torch.int16,
    "IntStorage": torch.int32,
    "LongStorage": torch.int64,
    "HalfStorage": torch.float16,
    "FloatStorage": torch.float32,
    "DoubleStorage": torch.float64,
    "BoolStorage": torch.bool,
    "BFloat16Storage": torch.bfloat16,
}


class _RestrictedTensorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if (module, name) == ("torch._tensor", "_rebuild_from_type_v2"):
            return _rebuild_from_type
        if (module, name) == ("torch", "Tensor"):
            return _TensorRef
        if module == "torch" and name in _STORAGE_TYPES:
            return _StorageType(_STORAGE_TYPES[name])
        raise pickle.UnpicklingError(
            "Blocked checkpoint global: {}.{}".format(module, name))

    def persistent_load(self, persistent_id):
        if not isinstance(persistent_id, tuple) or persistent_id[0] != "storage":
            raise pickle.UnpicklingError("Unsupported persistent checkpoint object")
        _, storage_type, key, location, size = persistent_id
        if not isinstance(storage_type, _StorageType):
            raise pickle.UnpicklingError("Unsupported storage type")
        return _StorageRef(storage_type.dtype, key, location, size)


def _materialize(value, archive, data_prefix, storage_cache):
    if isinstance(value, _TensorRef):
        storage = value.storage
        if storage.key not in storage_cache:
            member = "{}/data/{}".format(data_prefix, storage.key)
            raw = bytearray(archive.read(member))
            base = torch.frombuffer(raw, dtype=storage.dtype)
            if base.numel() != storage.size:
                raise RuntimeError(
                    "Storage size mismatch for {}: {} != {}".format(
                        member, base.numel(), storage.size))
            storage_cache[storage.key] = base
        base = storage_cache[storage.key]
        return torch.as_strided(
            base,
            size=value.shape,
            stride=value.stride,
            storage_offset=value.offset,
        )
    if isinstance(value, OrderedDict):
        return OrderedDict(
            (key, _materialize(item, archive, data_prefix, storage_cache))
            for key, item in value.items())
    if isinstance(value, dict):
        return {
            key: _materialize(item, archive, data_prefix, storage_cache)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_materialize(item, archive, data_prefix, storage_cache) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize(item, archive, data_prefix, storage_cache) for item in value)
    return value


def _load_restricted_zip(path):
    with zipfile.ZipFile(path) as archive:
        data_member = next(
            (name for name in archive.namelist() if name.endswith("/data.pkl")),
            None,
        )
        if data_member is None:
            raise RuntimeError("No data.pkl was found in the checkpoint archive")
        data_prefix = data_member.rsplit("/", 1)[0]
        metadata = _RestrictedTensorUnpickler(
            io.BytesIO(archive.read(data_member))).load()
        return _materialize(metadata, archive, data_prefix, {})


def safe_torch_load(path, map_location="cpu"):
    """Load tensors without allowing arbitrary checkpoint Python globals."""
    path = str(Path(path).expanduser())
    signature = inspect.signature(torch.load)
    if "weights_only" in signature.parameters:
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception as weights_error:
            if not zipfile.is_zipfile(path):
                raise RuntimeError(
                    "Safe weights-only loading failed for a non-zip checkpoint: {}"
                    .format(weights_error)) from weights_error
    if not zipfile.is_zipfile(path):
        raise RuntimeError(
            "This PyTorch version cannot safely load the legacy non-zip checkpoint")
    checkpoint = _load_restricted_zip(path)
    if map_location not in (None, "cpu", torch.device("cpu")):
        raise ValueError("Restricted legacy loading currently supports CPU map_location only")
    return checkpoint
