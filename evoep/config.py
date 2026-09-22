"""Validated nested configuration; unknown options fail rather than disappear."""
from copy import deepcopy
from pathlib import Path
import yaml
from . import ROOT
from .paths import object_hash

DEFAULTS = {
 "project_root": str(ROOT), "data_root": str(ROOT / "data"), "dataset": "EG",
 "protocol": "strict_zero_shot", "split_seed": 42, "train_seed": 42,
 "data": {"history_days":7,"horizon_days":1,"require_full_history":True,
  "min_train_positive_days":5,"val_unseen_fraction":.15,"test_unseen_fraction":.15,
  "deduplicate_events":True,"max_events_per_day":128,"sampling_seed":42,
  "text_policy":"verified_only","article_availability_path":None,"max_articles_per_event":4},
 "text":{"model_path":None,"revision":None,"pooling":"mean","l2_normalize":True,
  "frozen":True,"random_init":False,"random_seed":0,
  "max_type_tokens":128,"max_entity_tokens":32,"max_article_tokens":256,
  "encode_batch_size":32},
 "model":{"name":"evoep","hidden_dim":128,"graph_layers":1,"temporal_layers":2,
  "attention_heads":4,"temporal_ffn_dim":512,"experts":4,"expert_bottleneck":32,"retrieval_top_k":0,"transition_prior":False,"transition_temperature":0.2,"ablation_no_evolution":False,"dropout":.1},
 "train":{"batch_size":4,"pseudo_fraction":.2,"lambda_pseudo":1.0,"lambda_balance":.01,"lambda_rank":0.0,
  "epochs":50,"patience":8,"lr":.0003,"weight_decay":.0001,"grad_clip":1.0,
  "optimizer":"adamw","precision":"fp32","num_workers":0,"device":"cpu","cpu_threads":4,
  "selection_metric":"harmonic_macro_ap_seen_unseen","refit_after_validation":False,
  "max_train_windows":None,"max_eval_windows":None}
}

class Config(dict):
    def __setattr__(self, key, value):
        self[key] = Config.wrap(value)
    def __getattr__(self, key):
        try: return self[key]
        except KeyError as e: raise AttributeError(key) from e
    @classmethod
    def wrap(cls, value):
        return cls({k:cls.wrap(v) for k,v in value.items()}) if isinstance(value,dict) else value

def _merge(dst, src, prefix=""):
    for key, value in src.items():
        if key not in dst: raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(dst[key],dict):
            if not isinstance(value,dict): raise ValueError(f"Expected mapping: {prefix}{key}")
            _merge(dst[key],value,prefix+key+".")
        else: dst[key]=value

def load_config(path=None, overrides=()):
    cfg = deepcopy(DEFAULTS)
    if path:
        with open(path,encoding="utf-8") as f: _merge(cfg,yaml.safe_load(f) or {})
    for override in overrides:
        key,value=override.split("=",1)
        parts=key.split("."); node=cfg
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part],dict): raise ValueError(key)
            node=node[part]
        _merge(node,{parts[-1]:yaml.safe_load(value)},key.rsplit(".",1)[0]+".")
    cfg=Config.wrap(cfg); validate_config(cfg)
    return cfg

def validate_config(cfg):
    if Path(cfg.project_root).resolve()!=ROOT: raise ValueError("External project_root forbidden")
    if cfg.protocol!="strict_zero_shot": raise ValueError("Only strict_zero_shot implemented")
    if not cfg.dataset or Path(cfg.dataset).name!=cfg.dataset: raise ValueError("Invalid dataset")
    d,m,t=cfg.data,cfg.model,cfg.train
    for x in [d.history_days,d.horizon_days,d.max_events_per_day,d.max_articles_per_event,
              d.min_train_positive_days,m.hidden_dim,m.attention_heads,m.experts,
              m.expert_bottleneck,m.temporal_ffn_dim,t.batch_size,t.epochs,t.patience,t.cpu_threads]:
        if not isinstance(x,int) or isinstance(x,bool) or x<1: raise ValueError("Expected positive integer")
    if m.hidden_dim%m.attention_heads or m.hidden_dim%2: raise ValueError("Invalid hidden dimension")
    if not isinstance(m.retrieval_top_k,int) or isinstance(m.retrieval_top_k,bool) or m.retrieval_top_k<0: raise ValueError("Invalid retrieval_top_k")
    if not isinstance(m.transition_prior,bool): raise ValueError("Invalid transition_prior")
    if not isinstance(m.ablation_no_evolution,bool): raise ValueError("Invalid ablation_no_evolution")
    if not isinstance(m.transition_temperature,(int,float)) or isinstance(m.transition_temperature,bool) or m.transition_temperature<=0: raise ValueError("Invalid transition_temperature")
    if not (0<d.val_unseen_fraction<1 and 0<d.test_unseen_fraction<1 and
            d.val_unseen_fraction+d.test_unseen_fraction<1): raise ValueError("Invalid type fractions")
    if not 0<=t.pseudo_fraction<1: raise ValueError("Invalid pseudo fraction")
    if d.text_policy not in ("verified_only","linked_event_proxy","disabled"): raise ValueError("text_policy")
    if cfg.text.pooling not in ("mean","cls"): raise ValueError("pooling")
    if not cfg.text.frozen: raise ValueError("This implementation requires a frozen language encoder")
    if not isinstance(cfg.text.random_init,bool): raise ValueError("Invalid random_init")
    if (not isinstance(cfg.text.random_seed,int) or isinstance(cfg.text.random_seed,bool) or
            cfg.text.random_seed<0): raise ValueError("Invalid random_seed")
    if t.optimizer!="adamw" or t.precision!="fp32" or t.num_workers!=0: raise ValueError("Unsupported training mode")
    if t.refit_after_validation: raise ValueError("Refit would invalidate type isolation")
    if not d.deduplicate_events: raise ValueError("Canonical preparation requires deduplication")
    if m.graph_layers<0 or m.temporal_layers<0 or not 0<=m.dropout<1: raise ValueError("Invalid model configuration")
    for key in ("max_train_windows","max_eval_windows"):
        if t[key] is not None and (not isinstance(t[key],int) or t[key]<1): raise ValueError(key)
    if t.selection_metric!="harmonic_macro_ap_seen_unseen": raise ValueError("selection_metric")

def artifact_id(cfg):
    return object_hash({"schema":1,"dataset":cfg.dataset,"data_root":str(Path(cfg.data_root).resolve()),
                        "data":cfg.data,"split_seed":cfg.split_seed})[:16]

def artifact_path(cfg):
    return ROOT/"artifacts"/cfg.dataset/artifact_id(cfg)
