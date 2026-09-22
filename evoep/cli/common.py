import argparse
import json
from pathlib import Path
import torch
from .. import ROOT
from ..config import load_config,artifact_path
from ..paths import initialize_runtime,resolve_write_path,file_hash
from ..reproducibility import choose_device
from ..data.audit import audit_artifacts
from ..data.dataset import EventWindowDataset
from ..models.text import TextFeatureStore
from ..models.registry import build_model
from ..training.checkpoint import load_checkpoint

def config_parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument("--config",required=True)
    p.add_argument("--set",action="append",default=[],metavar="KEY=VALUE")
    return p

def get_config(args):
    cfg=load_config(args.config,args.set);initialize_runtime(cfg);return cfg

def run_path(value):
    path=Path(value)
    return resolve_write_path(path if path.is_absolute() else ROOT/path)

def load_run(run_dir,phase,checkpoint="best.pt"):
    run_dir=run_path(run_dir)
    cfg=load_config(run_dir/"config.resolved.yaml");initialize_runtime(cfg)
    art=artifact_path(cfg);audit_artifacts(art)
    manifest=json.loads((art/"manifest.json").read_text())
    ckpath=resolve_write_path(run_dir/checkpoint)
    state=load_checkpoint(ckpath,manifest)
    device=choose_device(cfg);store=TextFeatureStore(cfg,art,device="cpu")
    if store.provenance!=state["encoder_provenance"]:raise ValueError("Encoder provenance mismatch")
    data=EventWindowDataset(art,phase,cfg,store)
    model=build_model(cfg,state["text_feature_dim"]).to(device)
    model.load_state_dict(state["model"],strict=True);model.eval()
    return cfg,run_dir,state,model,data,store,file_hash(ckpath)
