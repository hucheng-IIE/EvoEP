from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch

@dataclass(frozen=True)
class EventRecord:
    head_id:int
    relation_id:int
    tail_id:int
    day:int
    md5s:tuple=()
    source_split:str=""
    event_id:int=-1

@dataclass(frozen=True)
class WindowSpec:
    cutoff_day:int
    history_start:int
    target_start:int
    target_end:int
    phase:str

@dataclass
class Snapshot:
    day:int
    node_ids:np.ndarray
    edge_index:np.ndarray
    edge_type:np.ndarray
    event_ids:np.ndarray
    # Raw frozen mean article features; None means unavailable.
    article_features:list=field(default_factory=list)
    article_days:list=field(default_factory=list)
    entity_features:Any=None

@dataclass
class HistorySample:
    cutoff_day:int
    snapshots:list
    selected_event_ids:list

@dataclass
class HistoryBatch:
    samples:list

@dataclass
class TargetBatch:
    candidate_ids:np.ndarray
    labels:torch.Tensor
    phase:str

@dataclass
class EvoEPOutput:
    logits:torch.Tensor
    router_probs:torch.Tensor
    attention:Any=None
    routing_details:Any=None

@dataclass
class HistoryMemory:
    values:torch.Tensor
    valid:torch.Tensor
    days:torch.Tensor
    event_ids:list
