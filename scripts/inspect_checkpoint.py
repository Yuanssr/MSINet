#!/usr/bin/env python3
"""Inspect trusted tensor checkpoints without enabling arbitrary pickle code.

This utility intentionally has no unsafe fallback. Legacy checkpoints that are
not supported by the installed PyTorch safe loader are reported as unsupported.
"""

import argparse
import hashlib
import inspect
from pathlib import Path
from typing import Any, Dict

import torch


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_load(path: Path) -> Any:
    signature = inspect.signature(torch.load)
    if "weights_only" not in signature.parameters:
        raise RuntimeError(
            "This PyTorch version has no weights_only loader. Upgrade PyTorch "
            "or inspect the checkpoint in an isolated conversion environment."
        )
    return torch.load(str(path), map_location="cpu", weights_only=True)


def summarize(value: Any, prefix: str = "") -> None:
    if torch.is_tensor(value):
        print(f"{prefix}: tensor shape={tuple(value.shape)} dtype={value.dtype}")
    elif isinstance(value, Dict):
        print(f"{prefix or '<root>'}: mapping with {len(value)} keys")
        for key, item in value.items():
            summarize(item, f"{prefix}.{key}" if prefix else str(key))
    else:
        print(f"{prefix}: {type(value).__name__} = {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--no-hash", action="store_true", help="Skip SHA256 calculation.")
    args = parser.parse_args()

    path = args.checkpoint.expanduser().resolve(strict=True)
    stat = path.stat()
    print(f"path: {path}")
    print(f"size: {stat.st_size}")
    print(f"mtime_ns: {stat.st_mtime_ns}")
    if not args.no_hash:
        print(f"sha256: {sha256(path)}")

    try:
        checkpoint = safe_load(path)
    except Exception as error:
        raise SystemExit(f"safe load failed; no unsafe fallback used: {error}") from error
    summarize(checkpoint)


if __name__ == "__main__":
    main()
