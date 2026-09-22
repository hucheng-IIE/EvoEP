"""Every persistent or temporary project write is confined to ROOT."""
import json
import os
import tempfile
from pathlib import Path
from . import ROOT

def resolve_write_path(path, root=ROOT):
    root = Path(root).resolve()
    if root != ROOT:
        raise ValueError("project_root must be the installed EvoEP directory")
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"Write outside EvoEP is forbidden: {path}") from None
    return resolved

def initialize_runtime(cfg=None):
    for name in ["artifacts", "runs", "cache", "cache/huggingface", "cache/torch",
                 "cache/xdg", "cache/pip", "cache/matplotlib", "cache/wandb", "tmp"]:
        resolve_write_path(ROOT / name).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(ROOT / "tmp")
    return ROOT

def atomic_write_json(path, obj):
    path = resolve_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name+".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def atomic_save_npz(path, **arrays):
    import numpy as np
    path = resolve_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def object_hash(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":")).encode()).hexdigest()
