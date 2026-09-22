import math
import torch
from torch import nn
from ..data.schemas import HistoryMemory

def build_temporal_masks(days,valid,heads):
    B,S=days.shape
    allowed=(days[:,None,:]<=days[:,:,None]) & valid[:,None,:] & valid[:,:,None]
    allowed[:,:,0]=True
    allowed[:,0,:]=False;allowed[:,0,0]=True
    # Padded queries may read only null, never become keys.
    allowed &= valid[:,None,:]
    return (~allowed).unsqueeze(1).expand(B,heads,S,S).reshape(B*heads,S,S)

class TemporalBlock(nn.Module):
    def __init__(self,d,heads,ffn_dim,dropout):
        super().__init__()
        self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d)
        self.attention=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(d,ffn_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(ffn_dim,d))
        self.drop=nn.Dropout(dropout)
    def forward(self,x,mask):
        norm=self.n1(x)
        x=x+self.drop(self.attention(norm,norm,norm,attn_mask=mask,need_weights=False)[0])
        return x+self.drop(self.ffn(self.n2(x)))

class TemporalEvolutionEncoder(nn.Module):
    def __init__(self,d,heads,layers,ffn_dim,dropout):
        super().__init__();self.heads=heads
        self.null=nn.Parameter(torch.randn(d)/math.sqrt(d))
        self.layers=nn.ModuleList([TemporalBlock(d,heads,ffn_dim,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(d)

    def pack(self,vectors,days,event_ids):
        length=max([len(x) for x in vectors],default=0)+1
        B=len(vectors);d=self.null.numel()
        x=self.null.new_zeros((B,length,d));valid=torch.zeros((B,length),dtype=torch.bool,device=x.device)
        dates=torch.full((B,length),-1,dtype=torch.long,device=x.device)
        for b,(v,ts) in enumerate(zip(vectors,days)):
            x[b,0]=self.null;valid[b,0]=True
            if len(v):
                x[b,1:len(v)+1]=v;valid[b,1:len(v)+1]=True
                dates[b,1:len(v)+1]=torch.as_tensor(ts,device=x.device)
        return x,dates,valid

    def forward(self,vectors,days,event_ids):
        x,dates,valid=self.pack(vectors,days,event_ids)
        mask=build_temporal_masks(dates,valid,self.heads)
        for layer in self.layers:x=layer(x,mask)
        x=self.norm(x).masked_fill(~valid.unsqueeze(-1),0)
        return HistoryMemory(x,valid,dates,event_ids)

class DailyGRUEncoder(TemporalEvolutionEncoder):
    """Baseline: average same-day event tokens, then causal GRU across days."""
    def __init__(self,d,heads,layers,ffn_dim,dropout):
        super().__init__(d,heads,0,ffn_dim,dropout)
        self.gru=nn.GRU(d,d,batch_first=True)
    def forward(self,vectors,days,event_ids):
        sequences=[];daylists=[]
        for x,ts in zip(vectors,days):
            unique=sorted(set(ts)); daylists.append(unique)
            if unique:
                td=torch.tensor(ts,device=x.device)
                means=torch.stack([x[td==day].mean(0) for day in unique]).unsqueeze(0)
                y,_=self.gru(means);sequences.append(y[0])
            else:sequences.append(x)
        x,dates,valid=self.pack(sequences,daylists,event_ids)
        return HistoryMemory(self.norm(x).masked_fill(~valid.unsqueeze(-1),0),valid,dates,event_ids)
