import json
from pathlib import Path
import numpy as np
from ..paths import file_hash,object_hash,atomic_write_json
from .split import validate_split

def audit_artifacts(path,verify_hashes=True):
    path=Path(path); m=json.loads((path/"manifest.json").read_text()); s=json.loads((path/"split.json").read_text())
    validate_split(s)
    if object_hash({k:v for k,v in s.items() if k!="split_hash"})!=s["split_hash"]: raise ValueError("Split hash invalid")
    if s["split_hash"]!=m["split_hash"]: raise ValueError("Manifest split hash mismatch")
    if verify_hashes:
        for name,digest in m["output_hashes"].items():
            if file_hash(path/name)!=digest: raise ValueError(f"Artifact checksum changed: {name}")
    events=np.load(path/"events.npz")["events"]
    if events.ndim!=2 or events.shape[1]!=4 or (np.diff(events[:,3])<0).any(): raise ValueError("Invalid event order")
    for phase,file,group in [("train","train",[]),("validation","val",s["val_unseen_ids"]),("test","test",s["test_unseen_ids"])]:
        data=np.load(path/f"labels_{file}.npz")
        ids=data["candidate_ids"];times=data["cutoffs"];y=data["labels"]
        if ids.tolist()!=sorted(s["seen_ids"]+group): raise ValueError("Candidate registry mismatch")
        if y.shape!=(len(times),len(ids)) or not np.isin(y,[0,1]).all(): raise ValueError("Invalid labels")
        a,b=m["bounds"][phase]; H=m["data_config"]["horizon_days"]
        if ((times+1<a)|(times+H>b)).any(): raise ValueError("Window crosses temporal split")
    report={"passed":True,"hashes_verified":verify_hashes,"events":len(events),"split_hash":s["split_hash"]}
    atomic_write_json(path/"audit.json",report)
    return report

def assert_history_contract(batch,episode,cfg,seen_ids):
    allowed=set(seen_ids)-set(episode.pseudo_ids)
    for sample in batch.samples:
        for snap in sample.snapshots:
            if not sample.cutoff_day-cfg.data.history_days<snap.day<=sample.cutoff_day:
                raise ValueError("Future/out-of-window history")
            if not set(snap.edge_type).issubset(allowed): raise ValueError("Forbidden relation in graph")
            for dates in snap.article_days:
                if any(day>snap.day for day in dates): raise ValueError("Future article")
