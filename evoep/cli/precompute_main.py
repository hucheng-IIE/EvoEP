"""Batch frozen features once; no trainable fitting and no target labels."""
import json
import torch
from .common import config_parser,get_config
from ..config import artifact_path
from ..models.text import TextFeatureStore
from ..data.dataset import EventWindowDataset
from ..training.episodes import evaluation_episode

def main():
    p=config_parser("Precompute permitted history text for train/validation")
    p.add_argument("--encoder-device",default="cpu")
    args=p.parse_args();cfg=get_config(args)
    torch.set_num_threads(cfg.train.cpu_threads)
    store=TextFeatureStore(cfg,artifact_path(cfg),device=args.encoder_device)
    try:
        for phase in ["train","validation"]:
            data=EventWindowDataset(artifact_path(cfg),phase,cfg,store)
            try:
                data.candidate_features();data.seen_features()
                import numpy as np
                rows=data.events[data.events[:,3]<=int(data.cutoffs[-1])]
                store.get([data.entities[str(int(i))] for i in np.unique(rows[:,[0,2]])],"entity")
                # All permitted seen-history edges, not just one sampled episode.
                # A pseudo-episode may choose different edges after removal.
                pending={}; count=0
                for day in sorted(data.by_day):
                    if day>int(data.cutoffs[-1]):break
                    positions=data.by_day[day]
                    ids=data.event_ids[positions]
                    docs=data.article_store.select_allowed_articles(ids,[day]*len(ids))
                    for articles in docs.values():
                        for md5,text,_ in articles:pending[md5]=text
                    if len(pending)>=256:
                        store.get(list(pending.values()),"article")
                        count+=len(pending);pending={}
                        if count%2048<256:print(json.dumps({"phase":phase,"article_visits":count,"day":day}),flush=True)
                if pending:store.get(list(pending.values()),"article");count+=len(pending)
                print(json.dumps({"phase":phase,"article_visits":count,"complete":True}),flush=True)
            finally:data.close()
    finally:store.close()

if __name__=="__main__":main()
