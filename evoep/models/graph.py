import torch
from torch import nn
from .text import EntityProjector

def mlp(a,b,c,dropout=0.):
    return nn.Sequential(nn.Linear(a,b),nn.GELU(),nn.Dropout(dropout),nn.Linear(b,c))

class SemanticGraphLayer(nn.Module):
    def __init__(self,d,dropout):
        super().__init__()
        self.incoming=mlp(2*d,2*d,d,dropout);self.outgoing=mlp(2*d,2*d,d,dropout)
        self.self_linear=nn.Linear(d,d);self.norm=nn.LayerNorm(d)
        self.dropout=nn.Dropout(dropout)

    def forward(self,node_h,edge_index,edge_z):
        source,target=edge_index
        incoming=torch.zeros_like(node_h);outgoing=torch.zeros_like(node_h)
        if source.numel():
            incoming.index_add_(0,target,self.incoming(torch.cat([node_h[source],edge_z],-1)))
            outgoing.index_add_(0,source,self.outgoing(torch.cat([node_h[target],edge_z],-1)))
            ni=torch.bincount(target,minlength=len(node_h)).clamp_min(1).unsqueeze(-1)
            no=torch.bincount(source,minlength=len(node_h)).clamp_min(1).unsqueeze(-1)
            incoming=incoming/ni;outgoing=outgoing/no
        delta=self.self_linear(node_h)+incoming+outgoing
        return self.norm(node_h+self.dropout(torch.nn.functional.gelu(delta)))

class SnapshotGraphEncoder(nn.Module):
    def __init__(self,D,d,layers,dropout):
        super().__init__()
        self.entities=EntityProjector(D,d)
        self.layers=nn.ModuleList([SemanticGraphLayer(d,dropout) for _ in range(layers)])
    def forward(self,entity_features,edge_index,edge_z):
        h=self.entities(entity_features)
        for layer in self.layers:h=layer(h,edge_index,edge_z)
        return h
