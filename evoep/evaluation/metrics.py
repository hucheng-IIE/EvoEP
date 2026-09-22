import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

def sigmoid(logits):
    x=np.asarray(logits,dtype=np.float64)
    return np.exp(-np.logaddexp(0,-x))

def harmonic_mean(a,b):
    if a is None or b is None:return None
    return 0. if a+b==0 else float(2*a*b/(a+b))

def column_f1(labels,predicted):
    y=np.asarray(labels,dtype=bool);p=np.asarray(predicted,dtype=bool)
    tp=(y&p).sum(0);denom=y.sum(0)+p.sum(0)
    return np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom!=0)

def compute_per_type_metrics(labels,logits,predicted,candidate_ids):
    y=np.asarray(labels,dtype=bool);z=np.asarray(logits);p=np.asarray(predicted,dtype=bool)
    if y.shape!=z.shape or p.shape!=y.shape or len(candidate_ids)!=y.shape[1]:raise ValueError("Metric shapes")
    rows=[]
    for i,r in enumerate(candidate_ids):
        pos=int(y[:,i].sum());neg=len(y)-pos;tp=int((y[:,i]&p[:,i]).sum());npred=int(p[:,i].sum())
        precision=tp/npred if npred else 0.;recall=tp/pos if pos else 0.
        rows.append({"relation_id":int(r),"positive_days":pos,"negative_days":neg,
          "prevalence":pos/len(y),"ap":float(average_precision_score(y[:,i],z[:,i])) if pos else None,
          "roc_auc":float(roc_auc_score(y[:,i],z[:,i])) if pos and neg else None,
          "precision":precision,"recall":recall,"f1":float(column_f1(y[:,i:i+1],p[:,i:i+1])[0])})
    return rows

def aggregate_group_metrics(rows,labels,logits,predicted,group_mask):
    mask=np.asarray(group_mask,dtype=bool);selected=[r for r,m in zip(rows,mask) if m]
    def mean(key):
        values=[r[key] for r in selected if r[key] is not None]
        return float(np.mean(values)) if values else None
    y=np.asarray(labels)[:,mask];z=np.asarray(logits)[:,mask];p=np.asarray(predicted)[:,mask]
    if not selected:raise ValueError("Empty metric group")
    return {"types":len(selected),"ap_valid_types":sum(r["ap"] is not None for r in selected),
            "auc_valid_types":sum(r["roc_auc"] is not None for r in selected),
            "macro_ap":mean("ap"),"macro_roc_auc":mean("roc_auc"),"macro_f1":mean("f1"),
            "macro_precision":mean("precision"),"macro_recall":mean("recall"),
            "micro_ap":float(average_precision_score(y.ravel(),z.ravel())) if y.sum() else None,
            "micro_f1":float(column_f1(y.reshape(-1,1),p.reshape(-1,1))[0]),
            "brier":float(np.mean((sigmoid(z)-y)**2))}

def metric_report(labels,logits,predicted,candidate_ids,seen_ids):
    rows=compute_per_type_metrics(labels,logits,predicted,candidate_ids)
    seen=np.isin(candidate_ids,seen_ids)
    groups={name:aggregate_group_metrics(rows,labels,logits,predicted,mask)
            for name,mask in [("seen",seen),("unseen",~seen),("overall",np.ones(len(seen),dtype=bool))]}
    groups["harmonic_macro_ap_seen_unseen"]=harmonic_mean(groups["seen"]["macro_ap"],groups["unseen"]["macro_ap"])
    groups["harmonic_macro_f1_seen_unseen"]=harmonic_mean(groups["seen"]["macro_f1"],groups["unseen"]["macro_f1"])
    return {"groups":groups,"per_type":rows}

def summarize_runs(reports):
    summary={}
    for group in ["seen","unseen","overall"]:
        summary[group]={}
        for metric in ["macro_ap","macro_f1","micro_ap","micro_f1"]:
            values=[r["calibrated"]["groups"][group][metric] for r in reports]
            valid=[v for v in values if v is not None]
            summary[group][metric]={"mean":float(np.mean(valid)) if valid else None,
                "std":float(np.std(valid,ddof=1)) if len(valid)>1 else 0. if valid else None,
                "valid_runs":len(valid)}
    return summary
