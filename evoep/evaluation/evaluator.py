import json
import numpy as np
import torch
from ..data.dataset import EpisodeCollator
from ..data.audit import assert_history_contract
from ..training.episodes import evaluation_episode
from .metrics import metric_report,sigmoid
from .calibration import apply_calibration

@torch.inference_mode()
def collect_predictions(model,dataset,phase,candidate_features=None):
    if phase!=dataset.phase or phase not in ("validation","test"):raise ValueError("Invalid evaluation phase")
    model.eval()
    candidate_features=dataset.candidate_features() if candidate_features is None else candidate_features
    seen_features=dataset.seen_features();episode=evaluation_episode(dataset.split["seen_ids"])
    collator=EpisodeCollator(dataset);outputs=[];targets=[];cutoffs=[];stats={}
    limit=dataset.cfg.train.max_eval_windows or len(dataset)
    limit=min(limit,len(dataset))
    for start in range(0,limit,dataset.cfg.train.batch_size):
        windows=[dataset[i] for i in range(start,min(start+dataset.cfg.train.batch_size,limit))]
        history,target=collator(windows,episode)
        assert_history_contract(history,episode,dataset.cfg,dataset.split["seen_ids"])
        output=model(history,seen_features,candidate_features)
        if not torch.isfinite(output.logits).all():raise ValueError("Nonfinite evaluation logits")
        outputs.append(output.logits.cpu().numpy());targets.append(target.labels.numpy())
        cutoffs.extend(w.cutoff_day for w in windows)
        for k,v in dataset.last_stats.items():stats[k]=stats.get(k,0)+v
    if not outputs:raise ValueError("No evaluation windows")
    return {"cutoffs":np.asarray(cutoffs),"candidate_ids":dataset.candidate_ids,
            "raw_logits":np.concatenate(outputs),"labels":np.concatenate(targets),"history_stats":stats}

def evaluate_predictions(table,seen_ids,calibration):
    z=table["raw_logits"];seen=np.isin(table["candidate_ids"],seen_ids)
    adjusted=apply_calibration(z,seen,calibration["gamma"])
    return {"raw":metric_report(table["labels"],z,sigmoid(z)>calibration["raw_threshold"],table["candidate_ids"],seen_ids),
            "calibrated":metric_report(table["labels"],adjusted,sigmoid(adjusted)>calibration["threshold"],table["candidate_ids"],seen_ids),
            "history_stats":table["history_stats"]}
