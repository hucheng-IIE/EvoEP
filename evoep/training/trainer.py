"""Reusable training engine: model factories do not own data, metrics, or checkpointing."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from ..config import artifact_path
from ..paths import resolve_write_path,atomic_write_json,object_hash
from ..reproducibility import seed_everything,get_rng_state,restore_rng_state,capture_environment,choose_device
from ..data.dataset import EventWindowDataset,EpisodeCollator
from ..data.audit import audit_artifacts,assert_history_contract
from ..models.text import TextFeatureStore
from ..models.registry import build_model
from ..evaluation.evaluator import collect_predictions
from ..evaluation.metrics import metric_report,sigmoid
from .episodes import sample_episode
from .losses import prediction_losses,total_loss,ranking_loss
from .checkpoint import save_checkpoint,load_checkpoint,append_train_log

def training_signature(cfg):
    config=json.loads(json.dumps(cfg))
    # Epoch extension and hardware changes are allowed; semantics are not.
    for key in ["epochs","device","cpu_threads"]:config["train"].pop(key,None)
    return object_hash(config)

class Trainer:
    def __init__(self,cfg,run_dir,resume=False):
        self.cfg=cfg;self.run_dir=resolve_write_path(run_dir)
        exists=self.run_dir.exists() and any(self.run_dir.iterdir())
        if exists and not resume:raise FileExistsError(f"Run exists: {self.run_dir}; pass --resume")
        if resume and not (self.run_dir/"last.pt").exists():raise FileNotFoundError("No last.pt to resume")
        self.run_dir.mkdir(parents=True,exist_ok=True)
        self.artifacts=artifact_path(cfg);audit_artifacts(self.artifacts)
        self.device=choose_device(cfg);seed_everything(cfg.train_seed)
        self.store=TextFeatureStore(cfg,self.artifacts,device=self.device)
        self.train_data=EventWindowDataset(self.artifacts,"train",cfg,self.store)
        self.val_data=EventWindowDataset(self.artifacts,"validation",cfg,self.store)
        self.model=build_model(cfg,self.store.dim).to(self.device)
        self.optimizer=torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                        lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
        self.manifest=self.train_data.manifest;self.start_epoch=0;self.best=-float("inf");self.bad_epochs=0
        self.rng=np.random.default_rng(cfg.train_seed)
        self.candidate_features=self.train_data.candidate_features()
        self.seen_features=self.train_data.seen_features()
        self.store.release_encoder()
        if resume:
            state=load_checkpoint(self.run_dir/"last.pt",self.manifest)
            if state["training_signature"]!=training_signature(cfg):raise ValueError("Resume changes training semantics")
            if state["encoder_provenance"]!=self.store.provenance:raise ValueError("Encoder changed")
            self.model.load_state_dict(state["model"],strict=True);self.optimizer.load_state_dict(state["optimizer"])
            self.start_epoch=state["epoch"]+1;self.best=state["best"];self.bad_epochs=state["bad_epochs"]
            self.rng.bit_generator.state=state["episode_rng"];restore_rng_state(state["rng"])
        atomic_write_json(self.run_dir/"config.resolved.json",cfg)
        import yaml
        resolve_write_path(self.run_dir/"config.resolved.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(cfg)),sort_keys=False))
        atomic_write_json(self.run_dir/"environment.json",capture_environment())
        self.partial=cfg.train.max_train_windows is not None or cfg.train.max_eval_windows is not None

    def train_epoch(self,epoch):
        start_time=time.monotonic();self.model.train();sums={};n=0
        limit=min(self.cfg.train.max_train_windows or len(self.train_data),len(self.train_data))
        order=self.rng.permutation(limit)
        collator=EpisodeCollator(self.train_data);routing=[];stats={}
        for start in range(0,limit,self.cfg.train.batch_size):
            windows=[self.train_data[int(i)] for i in order[start:start+self.cfg.train.batch_size]]
            ep=sample_episode(self.train_data.split["seen_ids"],self.cfg.train.pseudo_fraction,self.rng)
            history,targets=collator(windows,ep)
            assert_history_contract(history,ep,self.cfg,self.train_data.split["seen_ids"])
            self.optimizer.zero_grad(set_to_none=True)
            output=self.model(history,self.seen_features,self.candidate_features)
            parts=prediction_losses(output.logits,targets.labels,ep.mask(targets.candidate_ids))
            balance=self.model.routing_balance_loss(output) if hasattr(self.model,"routing_balance_loss") else None
            rank=ranking_loss(output.logits,targets.labels,
                ep.mask(targets.candidate_ids),self.cfg.train.lambda_pseudo)
            loss,logs=total_loss(parts,output.router_probs,self.cfg,balance,rank)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.train.grad_clip,error_if_nonfinite=True)
            self.optimizer.step()
            for k,v in logs.items():sums[k]=sums.get(k,0)+v*len(windows)
            routing.append(output.router_probs.detach().mean((0,1)).cpu().numpy())
            for k,v in self.train_data.last_stats.items():stats[k]=stats.get(k,0)+v
            n+=len(windows)
        return {"epoch":epoch,**{k:v/n for k,v in sums.items()},"windows":n,
                "seconds":time.monotonic()-start_time,"router_mean":np.mean(routing,axis=0).tolist(),**stats}

    def validate(self):
        table=collect_predictions(self.model,self.val_data,"validation")
        report=metric_report(table["labels"],table["raw_logits"],sigmoid(table["raw_logits"])>.5,
                             table["candidate_ids"],self.train_data.split["seen_ids"])
        score=report["groups"]["harmonic_macro_ap_seen_unseen"]
        if score is None:raise ValueError("Validation groups lack positive labels; protocol cannot select a model")
        return score,report

    def state(self,epoch):
        return {"model":self.model.state_dict(),"optimizer":self.optimizer.state_dict(),
                "epoch":epoch,"best":self.best,"bad_epochs":self.bad_epochs,
                "rng":get_rng_state(),"episode_rng":self.rng.bit_generator.state,"config":dict(self.cfg),
                "training_signature":training_signature(self.cfg),"manifest_hash":object_hash(self.manifest),
                "encoder_provenance":self.store.provenance,"text_feature_dim":self.store.dim,
                "seen_ids":self.train_data.split["seen_ids"],"split_hash":self.train_data.split["split_hash"],
                "partial_run":self.partial}

    def fit(self):
        if self.start_epoch>=self.cfg.train.epochs:raise ValueError("No epochs remaining")
        for epoch in range(self.start_epoch,self.cfg.train.epochs):
            stats=self.train_epoch(epoch);score,report=self.validate()
            improved=score>self.best
            if improved:self.best=score;self.bad_epochs=0
            else:self.bad_epochs+=1
            stats.update(validation_score=score,best=self.best,partial_run=self.partial)
            append_train_log(self.run_dir/"train.jsonl",stats)
            atomic_write_json(self.run_dir/"validation.latest.json",report)
            if improved:save_checkpoint(self.run_dir/"best.pt",self.state(epoch))
            save_checkpoint(self.run_dir/"last.pt",self.state(epoch))
            print(json.dumps(stats),flush=True)
            if self.bad_epochs>=self.cfg.train.patience:break
        best=load_checkpoint(self.run_dir/"best.pt",self.manifest)
        self.model.load_state_dict(best["model"],strict=True)
        self.store.release_encoder()
        return self.run_dir/"best.pt"

    def close(self):
        self.train_data.close();self.val_data.close();self.store.close()
