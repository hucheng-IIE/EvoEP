import torch
import torch.nn.functional as F

def prediction_losses(logits,targets,pseudo_mask):
    if logits.shape!=targets.shape or not torch.isfinite(logits).all():raise ValueError("Invalid logits")
    mask=torch.as_tensor(pseudo_mask,dtype=torch.bool,device=logits.device)
    if mask.all():raise ValueError("Empty retained set")
    losses=F.binary_cross_entropy_with_logits(logits,targets.to(logits.device),reduction="none")
    return {"seen":losses[:,~mask].mean(),
            "pseudo":losses[:,mask].mean() if mask.any() else logits.sum()*0}


def _ranking_group(logits,targets,mask):
    losses=[]
    for row in range(logits.shape[0]):
        positive=logits[row][mask & (targets[row]>0)]
        negative=logits[row][mask & (targets[row]<=0)]
        if positive.numel() and negative.numel():
            negative=negative.topk(min(16,negative.numel())).values
            losses.append(torch.nn.functional.softplus(
                negative.unsqueeze(0)-positive.unsqueeze(1)).mean())
    return torch.stack(losses).mean() if losses else logits.sum()*0

def ranking_loss(logits,targets,pseudo_mask,lambda_pseudo=1.0):
    pseudo=torch.as_tensor(pseudo_mask,dtype=torch.bool,device=logits.device)
    targets=targets.to(logits.device)
    return (_ranking_group(logits,targets,~pseudo)+
            lambda_pseudo*_ranking_group(logits,targets,pseudo))

def load_balance_loss(alpha):
    mean=alpha.mean(dim=(0,1))
    return ((mean-1/alpha.shape[-1])**2).sum()

def total_loss(parts,alpha,cfg,balance_override=None,rank=None):
    balance=load_balance_loss(alpha) if balance_override is None else balance_override
    rank=parts["seen"].new_zeros(()) if rank is None else rank
    loss=parts["seen"]+cfg.train.lambda_pseudo*parts["pseudo"]+cfg.train.lambda_balance*balance+cfg.train.lambda_rank*rank
    return loss,{"loss":float(loss.detach()),"seen":float(parts["seen"].detach()),
                 "pseudo":float(parts["pseudo"].detach()),"balance":float(balance.detach()),"rank":float(rank.detach())}
