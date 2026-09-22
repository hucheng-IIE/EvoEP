from dataclasses import dataclass
import numpy as np

@dataclass
class EpisodeSpec:
    pseudo_ids:tuple
    retained_ids:tuple
    def mask(self,candidate_ids):
        return np.isin(candidate_ids,self.pseudo_ids)

def sample_episode(seen_ids,pseudo_fraction,rng):
    seen=tuple(map(int,seen_ids))
    if pseudo_fraction==0:return EpisodeSpec((),seen)
    if len(seen)<2:raise ValueError("Need at least two seen types")
    n=min(len(seen)-1,max(1,round(pseudo_fraction*len(seen))))
    p=tuple(sorted(map(int,rng.choice(seen,n,replace=False))))
    return EpisodeSpec(p,tuple(r for r in seen if r not in p))

def evaluation_episode(seen_ids):
    return EpisodeSpec((),tuple(map(int,seen_ids)))
