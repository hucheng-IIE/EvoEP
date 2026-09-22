"""Frozen offline language encoder and content-addressed raw-feature cache."""
import json
import sqlite3
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..paths import file_hash,object_hash,resolve_write_path,atomic_write_json
from .. import ROOT

def encoder_provenance(cfg):
    path=Path(cfg.text.model_path or "")
    if not cfg.text.model_path or not path.is_dir() or not cfg.text.revision:
        raise ValueError("Set text.model_path to an existing local encoder and text.revision explicitly")
    names=["config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.txt"]
    weights=list(path.glob("*.safetensors"))
    if not weights: weights=list(path.glob("pytorch_model*.bin"))
    if not weights: raise ValueError("No local encoder weights")
    hashes={p.name:file_hash(p) for p in [path/n for n in names]+sorted(weights) if p.is_file()}
    payload={"model_path":str(path.resolve()),"revision":cfg.text.revision,"files":hashes,
             "pooling":cfg.text.pooling,"normalize":cfg.text.l2_normalize,
             "random_init":cfg.text.random_init,"random_seed":cfg.text.random_seed,
             "limits":{k:cfg.text[k] for k in ("max_type_tokens","max_entity_tokens","max_article_tokens")}}
    payload["hash"]=object_hash(payload)
    return payload

class FrozenTextEncoder(nn.Module):
    def __init__(self,cfg,device="cpu"):
        super().__init__()
        from transformers import AutoConfig,AutoModel,AutoTokenizer
        self.cfg=cfg
        self.tokenizer=AutoTokenizer.from_pretrained(cfg.text.model_path,local_files_only=True)
        if cfg.text.random_init:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.text.random_seed)
                model_cfg=AutoConfig.from_pretrained(cfg.text.model_path,local_files_only=True)
                self.encoder=AutoModel.from_config(model_cfg)
        else:
            self.encoder=AutoModel.from_pretrained(cfg.text.model_path,local_files_only=True)
        self.encoder.requires_grad_(False); self.encoder.eval(); self.encoder.to(device)
        self.dim=int(self.encoder.config.hidden_size)

    def train(self,mode=True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.inference_mode()
    def encode(self,texts,kind):
        if not texts: return torch.empty((0,self.dim))
        limit=self.cfg.text[{"type":"max_type_tokens","entity":"max_entity_tokens","article":"max_article_tokens"}[kind]]
        results=[]
        for start in range(0,len(texts),self.cfg.text.encode_batch_size):
            tokens=self.tokenizer(texts[start:start+self.cfg.text.encode_batch_size],padding=True,
                                  truncation=True,max_length=limit,return_tensors="pt")
            tokens={k:v.to(next(self.encoder.parameters()).device) for k,v in tokens.items()}
            h=self.encoder(**tokens).last_hidden_state
            if self.cfg.text.pooling=="cls": features=h[:,0]
            else:
                mask=tokens["attention_mask"].unsqueeze(-1)
                features=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
            if self.cfg.text.l2_normalize: features=torch.nn.functional.normalize(features,dim=-1)
            if not torch.isfinite(features).all(): raise ValueError("Nonfinite language features")
            results.append(features.float().cpu())
        return torch.cat(results)

class TextFeatureStore:
    def __init__(self,cfg,artifact_dir,device="cpu"):
        self.cfg=cfg; self.provenance=encoder_provenance(cfg); self.device=device
        # Frozen features are functions of (encoder, kind, text), not of the held-out
        # type split. Share them within a dataset while labels/article permissions remain
        # isolated in each artifact directory.
        path=resolve_write_path(ROOT/"cache"/"text"/cfg.dataset/self.provenance["hash"])
        path.mkdir(parents=True,exist_ok=True)
        db=resolve_write_path(path/"features.sqlite")
        self.conn=sqlite3.connect(db,timeout=120)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS features(key TEXT PRIMARY KEY,dim INTEGER,vector BLOB)")
        legacy=Path(artifact_dir)/"text"/self.provenance["hash"]/"features.sqlite"
        if legacy.exists() and legacy.resolve()!=db.resolve():
            self.conn.execute("ATTACH DATABASE ? AS legacy",(str(legacy.resolve()),))
            self.conn.execute("INSERT OR IGNORE INTO features SELECT key,dim,vector FROM legacy.features")
            self.conn.commit();self.conn.execute("DETACH DATABASE legacy")
        atomic_write_json(path/"provenance.json",self.provenance)
        self.encoder=None
        with open(Path(cfg.text.model_path)/"config.json") as f:self.dim=int(json.load(f)["hidden_size"])
        self.memory={}

    def get(self,texts,kind):
        keys=[object_hash([self.provenance["hash"],kind,text]) for text in texts]
        missing=[]
        for key,text in zip(keys,texts):
            if key in self.memory: continue
            row=self.conn.execute("SELECT dim,vector FROM features WHERE key=?",(key,)).fetchone()
            if row:
                if row[0]!=self.dim: raise ValueError("Cache dimension mismatch")
                self.memory[key]=np.frombuffer(row[1],dtype=np.float32).copy()
            else: missing.append((key,text))
        unique=dict(missing)
        if unique:
            if self.encoder is None:
                with torch.random.fork_rng(devices=[]):
                    self.encoder=FrozenTextEncoder(self.cfg,self.device)
            vectors=self.encoder.encode(list(unique.values()),kind).numpy()
            for key,vector in zip(unique,vectors):
                self.memory[key]=vector.copy()
                self.conn.execute("INSERT OR REPLACE INTO features VALUES(?,?,?)",(key,self.dim,vector.tobytes()))
            self.conn.commit()
        return torch.from_numpy(np.stack([self.memory[k] for k in keys])) if keys else torch.empty((0,self.dim))

    def release_encoder(self):
        self.encoder=None
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def close(self): self.conn.close()

class TypeProjector(nn.Module):
    def __init__(self,D,d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(D,d),nn.LayerNorm(d,eps=1e-5))
    def forward(self,x):return self.net(x)

class EntityProjector(TypeProjector):
    pass
