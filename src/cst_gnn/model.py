# ============================================================
# CruxSight.ai — CST-GNN Architecture
# Causal Spatio-Temporal Graph Neural Network
# ============================================================

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ═══════════════════════════════════════════════════════════
# LAYER 1: TOC-Informed Graph Attention (Spatial Encoder)
# ═══════════════════════════════════════════════════════════

class TOCGATLayer(nn.Module):
    """
    GAT layer where node features are pre-scaled by TOC capacity
    before attention is computed — biasing attention toward nodes
    that are structurally more likely to become constraints.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 4, toc_lambda: float = 2.0,
                 dropout: float = 0.2):
        super().__init__()
        self.toc_lambda = toc_lambda
        self.toc_scale  = nn.Parameter(torch.tensor(1.0))
        self.gat = GATConv(in_channels, out_channels,
                           heads=heads, dropout=dropout,
                           add_self_loops=True)
        full_dim  = out_channels * heads
        self.proj = nn.Linear(full_dim, full_dim)
        self.norm = nn.LayerNorm(full_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                toc_capacity: torch.Tensor) -> torch.Tensor:
        capacity_weight = (1.0 + self.toc_lambda *
                           self.toc_scale * toc_capacity)
        x_toc = x * capacity_weight.unsqueeze(-1)
        h = self.gat(x_toc, edge_index)
        h = self.proj(h)
        h = self.norm(h)
        return F.elu(h)


class SpatialEncoder(nn.Module):
    """
    Stacks TOC-GAT layers. Processes ALL timesteps of ALL batch
    items in ONE forward pass using block-diagonal batching.
    """

    def __init__(self, in_feats: int, hidden: int,
                 n_layers: int = 2, heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TOCGATLayer(in_feats, hidden // heads, heads))
        for _ in range(n_layers - 1):
            self.layers.append(TOCGATLayer(hidden, hidden // heads, heads))
        self.out_dim = hidden

    @staticmethod
    def build_batched_edge_index(edge_index: torch.Tensor,
                                  n_nodes: int,
                                  n_graphs: int) -> torch.Tensor:
        E = edge_index.shape[1]
        device = edge_index.device
        offsets = (torch.arange(n_graphs, device=device)
                   .repeat_interleave(E) * n_nodes)
        repeated = edge_index.repeat(1, n_graphs)
        return repeated + offsets.unsqueeze(0)

    def forward(self, x_seq: torch.Tensor,
                edge_index: torch.Tensor,
                toc_capacity: torch.Tensor) -> torch.Tensor:
        B, T, N, Fdim = x_seq.shape
        x_flat    = x_seq.reshape(B * T * N, Fdim)
        batched_ei = self.build_batched_edge_index(edge_index, N, B * T)
        cap_flat  = toc_capacity.repeat(B * T)
        h = x_flat
        for layer in self.layers:
            h = layer(h, batched_ei, cap_flat)
        return h.reshape(B, T, N, self.out_dim)


# ═══════════════════════════════════════════════════════════
# LAYER 2: Temporal Fusion Transformer (Temporal Encoder)
# ═══════════════════════════════════════════════════════════

class GatedResidual(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_model)
        self.fc2  = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class VariableSelectionNetwork(nn.Module):
    """Learns which of the 7 latency features matter most."""

    def __init__(self, d_model: int, n_vars: int):
        super().__init__()
        self.n_vars    = n_vars
        self.var_grns  = nn.ModuleList(
            [GatedResidual(d_model) for _ in range(n_vars)])
        self.weight_net = nn.Linear(d_model * n_vars, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        processed = [self.var_grns[i](x[..., i, :])
                     for i in range(self.n_vars)]
        concat  = torch.cat(processed, dim=-1)
        weights = torch.softmax(
            self.weight_net(concat), dim=-1).unsqueeze(-1)
        stacked = torch.stack(processed, dim=2)
        return (stacked * weights).sum(dim=2)


class TemporalEncoder(nn.Module):
    """
    LSTM + Multi-Head Attention over the 12-step window,
    applied independently per node via batch flattening.
    """

    def __init__(self, d_spatial: int, d_model: int = 64,
                 n_heads: int = 4, n_lstm_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.input_proj    = nn.Linear(d_spatial, d_model)
        self.feature_split = nn.Linear(d_model, d_model * 5)
        self.vsn  = VariableSelectionNetwork(d_model, n_vars=5)
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=n_lstm_layers, batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True)
        self.grn  = GatedResidual(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, h_spatial: torch.Tensor) -> torch.Tensor:
        B, T, N, D = h_spatial.shape
        x = h_spatial.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x = self.input_proj(x)
        x_split = self.feature_split(x).view(B * N, T, 5, -1)
        x_vsn   = self.vsn(x_split)
        x_lstm, _ = self.lstm(x_vsn)
        x_attn, _ = self.attn(x_lstm, x_lstm, x_lstm)
        x_out     = self.grn(self.norm(x_attn + x_lstm))
        return x_out[:, -1, :].reshape(B, N, -1)


# ═══════════════════════════════════════════════════════════
# LAYER 3: Causal Inference (NOTEARS-inspired DAG learning)
# ═══════════════════════════════════════════════════════════

class CausalInferenceLayer(nn.Module):
    """
    Learns a sparse DAG over node embeddings.
    Root Cause Score (RCS) = out_degree × ToC capacity prior.

    Note: RCS is currently dominated by the static capacity prior.
    Future work: normalize by prior to isolate the learned signal.
    See Limitations section in README.
    """

    def __init__(self, d_model: int, n_nodes: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.W_raw   = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, n_nodes),
        )

    def acyclicity_constraint(self, W: torch.Tensor) -> torch.Tensor:
        """h(W) = tr(e^(W∘W)) - d via truncated matrix series."""
        d  = W.shape[0]
        WW = W * W
        I  = torch.eye(d, device=W.device)
        M  = I + WW / d + (WW @ WW) / (2 * d * d)
        return M.trace() - d

    def forward(self, h_temporal: torch.Tensor,
                toc_capacity: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = self.n_nodes
        W = torch.sigmoid(self.W_raw)
        W = W * (1 - torch.eye(N, device=W.device))
        causal_contrib = torch.sigmoid(self.encoder(h_temporal))
        causal_graph   = W.unsqueeze(0) * causal_contrib
        causal_mean    = causal_graph.mean(0)
        out_degree     = causal_graph.sum(dim=-1)
        rcs = out_degree * toc_capacity.unsqueeze(0)
        dag_penalty = self.acyclicity_constraint(causal_mean)
        return causal_mean, rcs, dag_penalty


# ═══════════════════════════════════════════════════════════
# LAYER 4: N-Agnostic Prediction Heads
# ═══════════════════════════════════════════════════════════

def rcs_summary(rcs: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Converts variable-length (B, N) RCS into fixed-size (B, k+3):
        [top-1, top-2, ..., top-k, mean, max, std]
    This makes prediction heads shareable across graph sizes.
    """
    topk = torch.topk(rcs, k=min(k, rcs.shape[-1]), dim=-1).values
    if topk.shape[-1] < k:
        pad  = torch.zeros(rcs.shape[0], k - topk.shape[-1],
                           device=rcs.device)
        topk = torch.cat([topk, pad], dim=-1)
    mean = rcs.mean(dim=-1, keepdim=True)
    mx   = rcs.max(dim=-1, keepdim=True).values
    std  = rcs.std(dim=-1, keepdim=True)
    return torch.cat([topk, mean, mx, std], dim=-1)


class PredictionHeads(nn.Module):
    """
    Shared across graph sizes — uses fixed-size RCS summary.
    Four outputs: bottleneck probability, pattern, TTB, root-cause.
    """

    def __init__(self, d_model: int, n_patterns: int = 8,
                 summary_dim: int = 6):
        super().__init__()
        self.pool = nn.Linear(d_model, d_model)
        head_in   = d_model + summary_dim
        self.head_bn = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1))
        self.head_pattern = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(),
            nn.Linear(64, n_patterns))
        self.head_ttb = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Softplus())

    def forward(self, h_temporal: torch.Tensor,
                rcs: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_graph  = self.pool(h_temporal).mean(dim=1)
        summary  = rcs_summary(rcs, k=3)
        combined = torch.cat([h_graph, summary], dim=-1)
        return (self.head_bn(combined),
                self.head_pattern(combined),
                self.head_ttb(combined))


# ═══════════════════════════════════════════════════════════
# FULL MODEL
# ═══════════════════════════════════════════════════════════

class CSTGNN(nn.Module):
    """
    Causal Spatio-Temporal Graph Neural Network.

    N-agnostic: works for any graph size.
    Tested on compose (N=30) and home (N=7) workflows.
    Causal layer is lazily instantiated per N.
    Prediction heads are shared across all N.
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.spatial = SpatialEncoder(
            in_feats=m.gat_in_feats, hidden=m.gat_hidden,
            n_layers=m.gat_layers, heads=m.gat_heads)
        self.temporal = TemporalEncoder(
            d_spatial=m.gat_hidden, d_model=m.tft_hidden,
            n_heads=m.tft_heads, n_lstm_layers=m.lstm_layers,
            dropout=m.tft_dropout)
        self.heads = PredictionHeads(m.tft_hidden, m.n_patterns)
        self._causal_cache: Dict[int, CausalInferenceLayer] = {}
        self.tft_hidden = m.tft_hidden

    def _get_causal(self, n_nodes: int,
                    device) -> CausalInferenceLayer:
        if n_nodes not in self._causal_cache:
            layer = CausalInferenceLayer(
                self.tft_hidden, n_nodes).to(device)
            self._causal_cache[n_nodes] = layer
            self.add_module(f'causal_{n_nodes}', layer)
        return self._causal_cache[n_nodes]

    def forward(self, x_seq: torch.Tensor,
                edge_index: torch.Tensor,
                toc_capacity: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        B, T, N, _ = x_seq.shape
        device = x_seq.device
        h_spatial  = self.spatial(x_seq, edge_index, toc_capacity)
        h_temporal = self.temporal(h_spatial)
        causal     = self._get_causal(N, device)
        causal_graph, rcs, dag_penalty = causal(h_temporal,
                                                 toc_capacity)
        bn_logit, pattern_logit, ttb = self.heads(h_temporal, rcs)
        return {
            'bn_logit':      bn_logit,
            'pattern_logit': pattern_logit,
            'ttb':           ttb,
            'causal_graph':  causal_graph,
            'rcs':           rcs,
            'dag_penalty':   dag_penalty,
        }
