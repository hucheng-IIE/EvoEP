import json
from pathlib import Path
from itertools import chain
import numpy as np
from ..config import artifact_path
from ..paths import atomic_write_json,atomic_save_npz,file_hash,resolve_write_path
from .readers import (inspect_source,read_id_mapping,iter_quadruples,iter_event_text_links,
                      align_event_links,deduplicate_records,load_calendar)
from .ontology import load_ontology,build_descriptions
from .split import count_train_positive_days,make_type_split
from .articles import build_article_store
from .windows import enumerate_windows,build_daily_presence,build_window_targets

def prepare_country(cfg):
    src=Path(cfg.data_root)/cfg.dataset; out=resolve_write_path(artifact_path(cfg))
    sources=inspect_source(cfg.data_root,cfg.dataset)
    source_hashes={name:file_hash(src/name) for name in sources}
    ontology_path=Path(cfg.data_root)/"CAMEO"/"dict_id2ont.json"
    source_hashes["ontology"]=file_hash(ontology_path)
    if cfg.data.article_availability_path:
        source_hashes["availability"]=file_hash(cfg.data.article_availability_path)
    if (out/"manifest.json").exists():
        manifest=json.loads((out/"manifest.json").read_text())
        if manifest["source_hashes"]!=source_hashes: raise ValueError("Source changed: use a new artifact ID")
        from .audit import audit_artifacts
        audit_artifacts(out)
        return manifest
    if out.exists() and any(out.iterdir()): raise ValueError(f"Incomplete artifact directory: {out}; inspect before retry")
    out.mkdir(parents=True,exist_ok=True)
    entities=read_id_mapping(src/"entity2id.txt"); relations=read_id_mapping(src/"relation2id.txt")
    stat=list(map(int,(src/"stat.txt").read_text().split()))
    if stat!=[len(entities),len(relations)]: raise ValueError("stat/mapping mismatch")
    definitions=build_descriptions(relations,load_ontology(ontology_path))
    raw=[];bounds={};raw_counts={}
    for phase in ("train","valid","test"):
        rows=list(align_event_links(iter_quadruples(src/(phase+".txt")),iter_event_text_links(src/(phase+"_w_md5s.txt"))))
        if not rows: raise ValueError("Empty temporal split")
        bounds["validation" if phase=="valid" else phase]=[rows[0].day,rows[-1].day]
        raw_counts[phase]=len(rows)
        for e in rows:
            if e.head_id not in entities or e.tail_id not in entities or e.relation_id not in relations:
                raise ValueError("Out-of-range event ID")
        raw.extend(rows)
    if bounds["train"][1]>=bounds["validation"][0] or bounds["validation"][1]>=bounds["test"][0]:
        raise ValueError("Overlapping time splits")
    calendar=load_calendar(src/(cfg.dataset+".csv"))
    if calendar["min_day"]!=0 or calendar["max_day"]!=bounds["test"][1]: raise ValueError("Calendar/event range mismatch")
    records=list(deduplicate_records(raw)); del raw
    split=make_type_split(count_train_positive_days(records,bounds["train"][1]),definitions,cfg)
    events=np.asarray([(e.head_id,e.relation_id,e.tail_id,e.day) for e in records],dtype=np.int64)
    atomic_save_npz(out/"events.npz",events=events)
    atomic_write_json(out/"split.json",split)
    atomic_write_json(out/"descriptions.json",definitions)
    atomic_write_json(out/"entities.json",entities)
    article_audit=build_article_store(records,src,out/"articles.sqlite",split,cfg.data.article_availability_path)
    for phase,key in [("train","train"),("validation","val"),("test","test")]:
        ids=sorted(split["seen_ids"]+(split["val_unseen_ids"] if phase=="validation" else split["test_unseen_ids"] if phase=="test" else []))
        windows=enumerate_windows(bounds,phase,cfg.data.history_days,cfg.data.horizon_days,cfg.data.require_full_history)
        labels=build_window_targets(build_daily_presence(events,calendar["max_day"]+1,ids),windows)
        atomic_save_npz(out/f"labels_{key}.npz",candidate_ids=np.asarray(ids,dtype=np.int64),
                       cutoffs=np.asarray([w.cutoff_day for w in windows],dtype=np.int64),labels=labels)
    atomic_write_json(out/"isolation.json",{"policy":cfg.protocol,"text_policy":cfg.data.text_policy,
                      "forbidden_type_ids":split["val_unseen_ids"]+split["test_unseen_ids"]+split["excluded_ids"],
                      **article_audit})
    outputs=["events.npz","split.json","descriptions.json","entities.json","articles.sqlite",
             "labels_train.npz","labels_val.npz","labels_test.npz","isolation.json"]
    manifest={"schema_version":1,"dataset":cfg.dataset,"artifact_dir":str(out),
              "source_hashes":source_hashes,"output_hashes":{n:file_hash(out/n) for n in outputs},
              "split_hash":split["split_hash"],"bounds":bounds,"num_days":calendar["max_day"]+1,
              "num_entities":len(entities),"num_relations":len(relations),
              "raw_rows":raw_counts,"canonical_rows":len(records),"data_config":dict(cfg.data),
              "text_time_proxy":cfg.data.text_policy=="linked_event_proxy",**article_audit}
    atomic_write_json(out/"manifest.json",manifest)
    from .audit import audit_artifacts
    audit_artifacts(out)
    return manifest
