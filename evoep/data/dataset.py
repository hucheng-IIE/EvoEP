import json
from pathlib import Path
import numpy as np
import torch
from .schemas import Snapshot,HistorySample,HistoryBatch,TargetBatch,WindowSpec
from .articles import ArticleStore
from .sampling import sample_day_events

class EventWindowDataset:
    """Labels are phase-local. Complete records are reduced to seen-only history immediately."""
    def __init__(self,artifacts,phase,cfg,text_store=None):
        self.path=Path(artifacts); self.phase=phase;self.cfg=cfg;self.text_store=text_store
        self.manifest=json.loads((self.path/"manifest.json").read_text())
        self.split=json.loads((self.path/"split.json").read_text())
        self.definitions=json.loads((self.path/"descriptions.json").read_text())
        self.entities=json.loads((self.path/"entities.json").read_text())
        key={"train":"train","validation":"val","test":"test"}[phase]
        with np.load(self.path/f"labels_{key}.npz") as data:
            self.cutoffs=data["cutoffs"];self.candidate_ids=data["candidate_ids"];self.labels=data["labels"]
        events=np.load(self.path/"events.npz")["events"]
        eligible=np.isin(events[:,1],self.split["seen_ids"])
        self.event_ids=np.flatnonzero(eligible);self.events=events[eligible];del events
        self.by_day={int(day):np.flatnonzero(self.events[:,3]==day) for day in np.unique(self.events[:,3])}
        self.article_store=ArticleStore(self.path/"articles.sqlite",cfg.data.text_policy,cfg.data.max_articles_per_event)
        self.row_for_cutoff={int(t):i for i,t in enumerate(self.cutoffs)}
        self.last_stats={}

    def __len__(self):return len(self.cutoffs)

    def __getitem__(self,index):
        t=int(self.cutoffs[index])
        return WindowSpec(t,max(0,t-self.cfg.data.history_days+1),t+1,t+self.cfg.data.horizon_days,self.phase)

    def candidate_features(self):
        descriptions=[self.definitions[str(r)]["description"] for r in self.candidate_ids]
        if self.cfg.model.name=="lamp":
            path=self.path/"lamp_causes.json"
            if not path.exists(): raise FileNotFoundError("Run prepare_lamp_causes before LAMP training")
            causes=json.loads(path.read_text())["types"]
            descriptions=[text+" Potential earlier causes: "+"; ".join(causes[str(int(r))]["causes"])
                          for r,text in zip(self.candidate_ids,descriptions)]
        return self.text_store.get(descriptions,"type")

    def seen_features(self):
        ids=np.asarray(self.split["seen_ids"],dtype=np.int64)
        return ids,self.text_store.get([self.definitions[str(r)]["description"] for r in ids],"type")

    def entity_features(self,ids):
        return self.text_store.get([self.entities[str(int(e))] for e in ids],"entity")

    def build_history(self,windows,episode):
        samples=[];available=0;selected=0;text_count=0
        for w in windows:
            snaps=[];window_ids=[]
            for day in range(w.history_start,w.cutoff_day+1):
                positions=self.by_day.get(day,np.empty(0,dtype=np.int64))
                positions=positions[~np.isin(self.events[positions,1],episode.pseudo_ids)]
                available+=len(positions)
                all_ids=self.event_ids[positions]
                ids=sample_day_events(all_ids,self.cfg.data.max_events_per_day,self.cfg.data.sampling_seed,day)
                if not len(ids):continue
                loc=np.searchsorted(self.event_ids,ids);rows=self.events[loc]
                selected+=len(rows);window_ids.extend(ids.tolist())
                nodes=np.unique(rows[:,[0,2]])
                edge_index=np.stack([np.searchsorted(nodes,rows[:,0]),np.searchsorted(nodes,rows[:,2])])
                docs=self.article_store.select_allowed_articles(ids,[day]*len(ids),episode.pseudo_ids)
                feature_list=[];article_days=[]
                for eid in ids:
                    articles=docs[int(eid)]; article_days.append([x[2] for x in articles])
                    if articles:
                        if self.text_store is None: raise RuntimeError("Need a text store for available articles")
                        features=self.text_store.get([x[1] for x in articles],"article")
                        feature_list.append(features.mean(0));text_count+=1
                    else:feature_list.append(None)
                snap=Snapshot(day,nodes,edge_index,rows[:,1],ids,feature_list,article_days)
                snap.entity_features=self.entity_features(nodes) if self.text_store is not None else None
                snaps.append(snap)
            samples.append(HistorySample(w.cutoff_day,snaps,window_ids))
        self.last_stats={"history_available":available,"history_selected":selected,"events_with_text":text_count}
        return HistoryBatch(samples)

    def targets(self,windows):
        indices=[self.row_for_cutoff[w.cutoff_day] for w in windows]
        return TargetBatch(self.candidate_ids,torch.from_numpy(self.labels[indices]),self.phase)

    def close(self):self.article_store.close()

class EpisodeCollator:
    def __init__(self,dataset):self.dataset=dataset
    def __call__(self,windows,episode):
        return self.dataset.build_history(windows,episode),self.dataset.targets(windows)
