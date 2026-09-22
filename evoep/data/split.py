import numpy as np
from ..paths import object_hash

def count_train_positive_days(records,train_end):
    counts={}
    for e in records:
        if e.day<=train_end: counts.setdefault(e.relation_id,set()).add(e.day)
    return {r:len(days) for r,days in counts.items()}

def make_type_split(counts,definitions,cfg):
    ids=sorted((int(r) for r in definitions),key=lambda r:definitions[str(r)]["canonical_code"])
    eligible=[r for r in ids if counts.get(r,0)>=cfg.data.min_train_positive_days]
    rng=np.random.default_rng(cfg.split_seed); rng.shuffle(eligible)
    n=len(eligible); nv=int(n*cfg.data.val_unseen_fraction); nt=int(n*cfg.data.test_unseen_fraction)
    spec={"seen_ids":sorted(eligible[nv+nt:]),"val_unseen_ids":sorted(eligible[:nv]),
          "test_unseen_ids":sorted(eligible[nv:nv+nt]),
          "excluded_ids":sorted(set(ids)-set(eligible)),"split_seed":cfg.split_seed,
          "train_positive_days":{str(k):v for k,v in counts.items()},
          "canonical_codes":{r:definitions[r]["canonical_code"] for r in definitions}}
    validate_split(spec)
    spec["split_hash"]=object_hash(spec)
    return spec

def validate_split(spec):
    groups=[set(spec[k]) for k in ["seen_ids","val_unseen_ids","test_unseen_ids","excluded_ids"]]
    if len(groups[0])<2 or not groups[1] or not groups[2]: raise ValueError("Insufficient eligible types")
    if sum(map(len,groups))!=len(set.union(*groups)): raise ValueError("Type sets overlap")
    if set.union(*groups)!=set(map(int,spec["canonical_codes"])): raise ValueError("Split coverage")
