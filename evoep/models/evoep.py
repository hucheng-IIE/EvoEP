"""EvoEP: description-conditioned evolution over shared seen-event history."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .text import TypeProjector
from .graph import SnapshotGraphEncoder, mlp
from .events import HistoricalEventEncoder
from .temporal import TemporalBlock, build_temporal_masks
from ..data.schemas import EvoEPOutput
from ..training.losses import load_balance_loss


@dataclass
class EvoEPMemory:
    values: torch.Tensor
    keys: torch.Tensor
    valid: torch.Tensor
    days: torch.Tensor
    event_ids: list


class BottleneckMoE(nn.Module):
    """A dense mixture of bottleneck experts with learned routing."""

    def __init__(self, dim, experts, bottleneck, dropout):
        super().__init__()
        self.router = nn.Linear(dim, experts)
        self.experts = nn.ModuleList([
            mlp(dim, bottleneck, dim) for _ in range(experts)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        weights = self.router(value).softmax(-1)
        expert_values = torch.stack(
            [expert(value) for expert in self.experts], dim=-2)
        output = (expert_values * weights.unsqueeze(-1)).sum(-2)
        return self.dropout(output), weights


class SharedEvolutionLayer(nn.Module):
    """Evolve shared history, then update candidate semantic and graph states."""
    def __init__(self, d, heads, ffn, experts, bottleneck, dropout):
        super().__init__()
        self.heads = heads
        self.temporal = TemporalBlock(d, heads, ffn, dropout)
        self.history_norm = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout,
                                           batch_first=True)
        self.semantic_fusion = mlp(3*d, 2*d, d, dropout)
        self.semantic_moe = BottleneckMoE(d, experts, bottleneck, dropout)
        self.graph_fusion = mlp(3*d, 2*d, d, dropout)
        self.graph_moe = BottleneckMoE(d, experts, bottleneck, dropout)
        self.semantic_update = mlp(d, 2*d, d, dropout)
        self.graph_update = mlp(d, 2*d, d, dropout)
        self.semantic_norm = nn.LayerNorm(d)
        self.graph_norm = nn.LayerNorm(d)

    def forward(self, semantic, graph, history, valid, days,
                retrieval_indices, retrieval_valid, return_attention=False):
        mask = build_temporal_masks(days, valid, self.heads)
        history = self.temporal(history, mask)
        history = self.history_norm(history).masked_fill(~valid.unsqueeze(-1), 0)
        batch, candidates, selected_count = retrieval_indices.shape
        dim = history.shape[-1]
        gather_index = retrieval_indices.unsqueeze(-1).expand(
            batch, candidates, selected_count, dim)
        selected = torch.gather(
            history.unsqueeze(1).expand(-1, candidates, -1, -1),
            2, gather_index)
        context, attention = self.cross(
            semantic.reshape(batch*candidates, 1, dim),
            selected.reshape(batch*candidates, selected_count, dim),
            selected.reshape(batch*candidates, selected_count, dim),
            key_padding_mask=~retrieval_valid.reshape(batch*candidates,
                                                       selected_count),
            need_weights=return_attention, average_attn_weights=False)
        context = context.reshape(batch, candidates, dim)
        if attention is not None:
            attention = attention.reshape(batch, candidates, self.heads,
                                          selected_count)

        semantic_input = self.semantic_fusion(torch.cat(
            [semantic, context, semantic * context], dim=-1))
        semantic_delta, semantic_alpha = self.semantic_moe(semantic_input)
        semantic = self.semantic_norm(
            semantic + self.semantic_update(semantic_delta))

        graph_input = self.graph_fusion(torch.cat(
            [graph, context, graph * context], dim=-1))
        graph_delta, graph_alpha = self.graph_moe(graph_input)
        graph = self.graph_norm(graph + self.graph_update(graph_delta))
        return history, semantic, graph, semantic_alpha, graph_alpha, attention


class SemanticTransitionPrior(nn.Module):
    """Score target types from recent source-type descriptions in history."""
    def __init__(self, dim, temperature):
        super().__init__()
        self.temperature = float(temperature)
        self.source = nn.Linear(dim, dim, bias=False)
        self.target = nn.Linear(dim, dim, bias=False)
        self.log_decay = nn.Parameter(torch.tensor(-1.5))
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, candidates, keys, valid, days):
        # candidates: [B,C,D], keys: [B,S,D]. The bilinear similarity learns
        # which observed seen type tends to precede each semantic target.
        query = F.normalize(self.target(candidates), dim=-1)
        source = F.normalize(self.source(keys), dim=-1)
        compatibility = torch.einsum("bcd,bsd->bcs", query, source)
        compatibility = compatibility / self.temperature
        latest = days.masked_fill(~valid, torch.iinfo(days.dtype).min).max(1).values
        age = (latest[:, None] - days).clamp_min(0).to(compatibility.dtype)
        compatibility = compatibility - F.softplus(self.log_decay) * age.unsqueeze(1)
        compatibility = compatibility.masked_fill(~valid.unsqueeze(1), -torch.inf)
        count = valid.sum(1).clamp_min(1).to(compatibility.dtype)
        pooled = torch.logsumexp(compatibility, dim=-1) - count.log().unsqueeze(1)
        return self.strength * pooled


class EvoEP(nn.Module):
    """All candidates use descriptions to retrieve candidate-specific history."""
    def __init__(self, cfg, text_feature_dim):
        super().__init__()
        m = cfg.model
        d = m.hidden_dim
        self.hidden_dim = d
        self.retrieval_top_k = m.retrieval_top_k
        self.no_evolution = m.ablation_no_evolution
        self.type_projector = TypeProjector(text_feature_dim, d)
        self.graph = SnapshotGraphEncoder(text_feature_dim, d, m.graph_layers,
                                          m.dropout)
        self.events = HistoricalEventEncoder(text_feature_dim, d,
                                              cfg.data.history_days, m.dropout,
                                              use_time=not self.no_evolution)
        self.empty_history = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.transition_prior = (SemanticTransitionPrior(d, m.transition_temperature)
                                 if m.transition_prior else None)
        self.graph_seed = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.layers = nn.ModuleList([] if self.no_evolution else [
            SharedEvolutionLayer(d, m.attention_heads, m.temporal_ffn_dim,
                                 m.experts, m.expert_bottleneck, m.dropout)
            for _ in range(m.temporal_layers)
        ])
        if self.no_evolution:
            self.set_cross = nn.MultiheadAttention(
                d, m.attention_heads, dropout=m.dropout, batch_first=True)
            self.set_fusion = mlp(3*d, 2*d, d, m.dropout)
            self.set_norm = nn.LayerNorm(d)
        self.final_fusion = mlp(3*d, 2*d, d, m.dropout)
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def encode_history(self, history_batch, seen_type_features):
        device = next(self.parameters()).device
        ids, raw = seen_type_features
        projected = self.type_projector(raw.to(device))
        id_to_row = {int(value): row for row, value in enumerate(ids)}
        vectors, keys, dates, event_ids = [], [], [], []
        for sample in history_batch.samples:
            sample_vectors, sample_keys, sample_days, sample_ids = [], [], [], []
            for snapshot in sorted(sample.snapshots, key=lambda item: item.day):
                edge_index = torch.as_tensor(snapshot.edge_index, dtype=torch.long,
                                             device=device)
                rows = torch.tensor([id_to_row[int(value)]
                                     for value in snapshot.edge_type],
                                    dtype=torch.long, device=device)
                relation = projected[rows]
                nodes = self.graph(snapshot.entity_features.to(device),
                                   edge_index, relation)
                elapsed = torch.full((len(snapshot.event_ids),),
                    sample.cutoff_day - snapshot.day, dtype=torch.long,
                    device=device)
                event = self.events(nodes, edge_index, relation,
                                    snapshot.article_features, elapsed)
                sample_vectors.append(event)
                sample_keys.append(relation)
                sample_days.extend([snapshot.day] * len(snapshot.event_ids))
                sample_ids.extend(map(int, snapshot.event_ids))
            vectors.append(torch.cat(sample_vectors) if sample_vectors else
                           projected.new_empty((0, self.hidden_dim)))
            keys.append(torch.cat(sample_keys) if sample_keys else
                        projected.new_empty((0, self.hidden_dim)))
            dates.append(sample_days)
            event_ids.append(sample_ids)

        length = max(max((len(value) for value in vectors), default=0), 1)
        batch = len(vectors)
        values = projected.new_zeros((batch, length, self.hidden_dim))
        key_tensor = projected.new_zeros((batch, length, self.hidden_dim))
        valid = torch.zeros((batch, length), dtype=torch.bool, device=device)
        day_tensor = torch.full((batch, length), -1, dtype=torch.long,
                                device=device)
        for row, (value, key, day) in enumerate(zip(vectors, keys, dates)):
            if len(value):
                values[row, :len(value)] = value
                key_tensor[row, :len(value)] = key
                valid[row, :len(value)] = True
                day_tensor[row, :len(value)] = torch.as_tensor(day, device=device)
            else:
                values[row, 0] = self.empty_history
                key_tensor[row, 0] = self.empty_history
                valid[row, 0] = True
        return EvoEPMemory(values, key_tensor, valid, day_tensor, event_ids)

    def retrieval_plan(self, semantic, memory):
        """Select candidate-specific history using relation-description similarity."""
        batch, candidates, _ = semantic.shape
        steps = memory.values.shape[1]
        count = steps if self.retrieval_top_k == 0 else min(
            self.retrieval_top_k, steps)
        if count == steps:
            indices = torch.arange(steps, device=semantic.device).view(1, 1, -1)
            indices = indices.expand(batch, candidates, -1)
        else:
            query = torch.nn.functional.normalize(semantic, dim=-1)
            keys = torch.nn.functional.normalize(memory.keys, dim=-1)
            score = torch.einsum("bcd,bsd->bcs", query, keys)
            score = score.masked_fill(~memory.valid.unsqueeze(1), -torch.inf)
            indices = score.topk(count, dim=-1, sorted=True).indices
        selected_valid = torch.gather(
            memory.valid.unsqueeze(1).expand(-1, candidates, -1), 2, indices)
        return indices, selected_valid

    def score_candidates(self, memory, candidate_features,
                         return_attention=False):
        semantic = self.type_projector(
            candidate_features.to(memory.values.device)).unsqueeze(0)
        semantic = semantic.expand(memory.values.shape[0], -1, -1)
        transition_score = (self.transition_prior(semantic, memory.keys, memory.valid,
                                                  memory.days)
                            if self.transition_prior is not None else None)
        if self.no_evolution:
            batch,candidates,dim = semantic.shape
            steps = memory.values.shape[1]
            history = memory.values.unsqueeze(1).expand(-1,candidates,-1,-1)
            history = history.reshape(batch*candidates,steps,dim)
            padding = (~memory.valid).unsqueeze(1).expand(-1,candidates,-1)
            padding = padding.reshape(batch*candidates,steps)
            context,attention = self.set_cross(
                semantic.reshape(batch*candidates,1,dim),history,history,
                key_padding_mask=padding,need_weights=return_attention,
                average_attn_weights=False)
            context = context.reshape(batch,candidates,dim)
            value = self.set_norm(self.set_fusion(torch.cat(
                [semantic,context,semantic*context],dim=-1)))
            logits = self.head(value).squeeze(-1)
            if transition_score is not None:
                logits = logits + transition_score
            if attention is not None:
                attention = attention.reshape(
                    batch,candidates,self.set_cross.num_heads,steps)
            router = semantic.new_ones((batch,candidates,1))
            return EvoEPOutput(logits,router,attention,[])
        graph = self.graph_seed(semantic)
        retrieval_indices, retrieval_valid = self.retrieval_plan(semantic, memory)
        history = memory.values
        routing_details = []
        attention = None
        for layer in self.layers:
            history, semantic, graph, sem_alpha, graph_alpha, attention = layer(
                semantic, graph, history, memory.valid, memory.days,
                retrieval_indices, retrieval_valid, return_attention)
            routing_details.append((sem_alpha, graph_alpha))
        if routing_details:
            router_probs = .5 * (routing_details[-1][0] + routing_details[-1][1])
        else:
            router_probs = semantic.new_ones((*semantic.shape[:2], 1))
        value = self.final_norm(self.final_fusion(torch.cat(
            [semantic, graph, semantic * graph], dim=-1)))
        logits = self.head(value).squeeze(-1)
        if transition_score is not None:
            logits = logits + transition_score
        return EvoEPOutput(logits, router_probs,
                            attention if return_attention else None,
                            routing_details)

    def routing_balance_loss(self, output):
        if not output.routing_details:
            return output.logits.sum() * 0
        losses = []
        for semantic_alpha, graph_alpha in output.routing_details:
            losses.extend([load_balance_loss(semantic_alpha),
                           load_balance_loss(graph_alpha)])
        return torch.stack(losses).mean()

    def forward(self, history_batch, seen_type_features, candidate_features):
        memory = self.encode_history(history_batch, seen_type_features)
        return self.score_candidates(memory, candidate_features)
