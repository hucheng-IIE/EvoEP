import json
import numpy as np
from ..paths import atomic_write_json
from .metrics import sigmoid,column_f1,harmonic_mean

def apply_calibration(logits,seen_mask,gamma):
    return np.asarray(logits,dtype=np.float64)-gamma*np.asarray(seen_mask)[None,:]

def fit_calibration(val_logits,val_labels,seen_mask,cfg):
    seen=np.asarray(seen_mask,dtype=bool)
    if not seen.any() or seen.all():raise ValueError("Calibration requires both seen and unseen")
    c=cfg.calibration
    thresholds=np.arange(c.threshold_min,c.threshold_max+c.threshold_step/2,c.threshold_step)
    candidates=[];raw_candidates=[]
    for gamma in c.gamma_grid:
        p=sigmoid(apply_calibration(val_logits,seen,gamma))
        for threshold in thresholds:
            f1=column_f1(val_labels,p>threshold)
            score=harmonic_mean(float(f1[seen].mean()),float(f1[~seen].mean()))
            candidate={"gamma":float(gamma),"threshold":float(round(threshold,10)),"score":score}
            candidates.append(candidate)
            if gamma==0:raw_candidates.append(candidate)
    def rank(x):return (-x["score"],abs(x["gamma"]),abs(x["threshold"]-.5),x["gamma"],x["threshold"])
    best=min(candidates,key=rank);raw=min(raw_candidates,key=rank)
    return {**best,"raw_threshold":raw["threshold"],"raw_score":raw["score"],
            "objective":c.objective,"gamma_grid":list(c.gamma_grid),"thresholds":thresholds.tolist()}

def save_calibration(state,path):atomic_write_json(path,state)

def load_calibration(path,checkpoint_hash):
    with open(path) as f:state=json.load(f)
    if state["checkpoint_hash"]!=checkpoint_hash:raise ValueError("Calibration/checkpoint mismatch")
    return state
