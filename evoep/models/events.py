import math
import torch
from torch import nn
from .graph import mlp

class RelativeTimeEncoder(nn.Module):
    def __init__(self,d,history_days):
        super().__init__()
        if d%2:raise ValueError("Time dimension must be even")
        self.history_days=history_days
        self.register_buffer("omega",10000**(-torch.arange(0,d,2,dtype=torch.float32)/d))
    def forward(self,elapsed_days):
        if ((elapsed_days<0)|(elapsed_days>=self.history_days)).any():raise ValueError("Invalid elapsed days")
        x=elapsed_days.float().unsqueeze(-1)/max(self.history_days-1,1)
        phase=x*self.omega
        return torch.stack([phase.sin(),phase.cos()],dim=-1).flatten(-2)

class HistoricalEventEncoder(nn.Module):
    def __init__(self,D,d,history_days,dropout,use_time=True):
        super().__init__()
        self.use_time=bool(use_time)
        self.article=nn.Sequential(nn.Linear(D,d),nn.LayerNorm(d))
        self.missing_text=nn.Parameter(torch.randn(d)/math.sqrt(d))
        self.presence=nn.Linear(1,d)
        self.time=RelativeTimeEncoder(d,history_days)
        self.fusion=mlp(6*d,2*d,d,dropout);self.norm=nn.LayerNorm(d)
    def forward(self,node_h,edge_index,edge_z,articles,elapsed):
        source,target=edge_index
        a=[];present=[]
        for item in articles:
            if item is None:a.append(self.missing_text);present.append(0.)
            else:a.append(self.article(item.to(node_h.device)));present.append(1.)
        if not len(a):return node_h.new_empty((0,node_h.shape[-1]))
        a=torch.stack(a)
        present=self.presence(node_h.new_tensor(present).unsqueeze(-1))
        time=self.time(elapsed) if self.use_time else node_h.new_zeros((len(elapsed),node_h.shape[-1]))
        return self.norm(self.fusion(torch.cat([node_h[source],edge_z,node_h[target],a,
                                               time,present],dim=-1)))
