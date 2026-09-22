import hashlib
import numpy as np

def sample_day_events(event_ids,cap,seed,day):
    if len(event_ids)<=cap: return np.asarray(event_ids,dtype=np.int64)
    def priority(eid):
        return hashlib.sha256(f"{seed}:{day}:{int(eid)}".encode()).digest()
    return np.asarray(sorted(sorted(event_ids,key=priority)[:cap]),dtype=np.int64)
