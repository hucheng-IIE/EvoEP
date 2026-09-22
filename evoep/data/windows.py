import numpy as np
from .schemas import WindowSpec

def enumerate_windows(bounds,phase,W,H,require_full_history=True,min_day=0):
    start,end=bounds[phase]
    first=max(start-1,min_day+W-1 if require_full_history else min_day)
    return [WindowSpec(t,max(min_day,t-W+1),t+1,t+H,phase)
            for t in range(first,end-H+1)]

def build_daily_presence(events,num_days,type_ids):
    mapping={int(r):i for i,r in enumerate(type_ids)}
    out=np.zeros((num_days,len(type_ids)),dtype=bool)
    for _,r,_,day in events:
        if int(r) in mapping: out[int(day),mapping[int(r)]]=True
    return out

def build_window_targets(presence,windows):
    prefix=np.concatenate([np.zeros((1,presence.shape[1]),dtype=np.int64),
                           np.cumsum(presence,axis=0,dtype=np.int64)])
    if not windows: return np.empty((0,presence.shape[1]),dtype=np.float32)
    if any(w.target_end>=len(presence) for w in windows): raise ValueError("Incomplete future window")
    return np.stack([(prefix[w.target_end+1]-prefix[w.target_start])>0
                     for w in windows]).astype(np.float32)
