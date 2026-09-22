import os
import tempfile
import torch
from ..paths import resolve_write_path,object_hash
from ..reproducibility import restore_rng_state

def save_checkpoint(path,state):
    path=resolve_write_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix=".pt");os.close(fd)
    try:
        torch.save(state,tmp)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def load_checkpoint(path,expected_manifest=None):
    # Checkpoints are generated locally by this project, not arbitrary downloaded pickle.
    path=resolve_write_path(path)
    state=torch.load(path,map_location="cpu",weights_only=False)
    if expected_manifest is not None and state["manifest_hash"]!=object_hash(expected_manifest):
        raise ValueError("Checkpoint/artifact mismatch")
    return state

def append_train_log(path,stats):
    import json
    path=resolve_write_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f:f.write(json.dumps(stats,allow_nan=False)+"\n")
