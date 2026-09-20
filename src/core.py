import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATConv


# ----------------------------- Config -----------------------------

@dataclass
class DataConfig:
    window_steps:  int   = 12
    horizon_steps: int   = 6
    step_sec:      int   = 10
    min_samples:   int   = 500
    n_features:    int   = 7
    train_ratio:   float = 0.70
    val_ratio:     float = 0.15
    random_seed:   int   = 42
    batch_size:    int   = 64
    num_workers:   int   = 2


@dataclass
class ModelConfig:
    gat_in_feats:   int   = 7
    gat_hidden:     int   = 64
    gat_heads:      int   = 4
    gat_layers:     int   = 2
    gat_dropout:    float = 0.1
    toc_lambda:     float = 2.0
    toc_gamma:      float = 0.5
    tft_hidden:     int   = 128
    tft_heads:      int   = 4
    lstm_layers:    int   = 2
    tft_dropout:    float = 0.1
    causal_hidden:  int   = 64
    dag_reg:        float = 1.0
    n_patterns:     int   = 8


@dataclass
class TrainingConfig:
    epochs:         int   = 60
    lr:             float = 1e-3
    weight_decay:   float = 1e-4
    grad_clip:      float = 1.0
    patience:       int   = 10
    lambda_pattern: float = 0.5
    lambda_ttb:     float = 0.3
    lambda_causal:  float = 0.2
    lambda_sub:     float = 0.1
    fn_weight:      float = 5.0
    fp_weight:      float = 1.0
    constraint_mult: float = 3.0
    lambda_rcs_sup: float = 0.0


@dataclass
class Config:
    data:     DataConfig     = field(default_factory=DataConfig)
    model:    ModelConfig    = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device:   str            = 'cuda'
    results_dir: str         = ''
    model_dir:   str         = ''

    @classmethod
    def load(cls, path: str) -> 'Config':
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        cfg          = cls()
        cfg.data     = DataConfig(**d['data'])
        cfg.model    = ModelConfig(**d['model'])
        cfg.training = TrainingConfig(**d['training'])
        cfg.device   = d.get('device', 'cuda')
        return cfg


# ----------------------------- TOC priors -----------------------------

class TOCPriorLoader:
    PATTERNS = {
        'A': frozenset([4,5,7,8,11,12,13,14,18,19,20,21,26,27,28]),
        'B': frozenset([4,5,13,14,20,21,26,27,28]),
        'C': frozenset([0,1,2,22]),
        'D': frozenset([0,1,2,13,14,20,21,22,26,27,28]),
        'E': frozenset([0,1,2,4,5,7,8,11,12,13,14,18,19,20,21,22,26,27,28]),
        'F': frozenset([0,1,2,4,5,7,8,11,12,18,19]),
        'G': frozenset([3,4]),
        'none': frozenset(),
    }
    PATTERN_TO_IDX = {p: i for i, p in
                      enumerate(['A','B','C','D','E','F','G','none'])}
    IDX_TO_PATTERN = {v: k for k, v in PATTERN_TO_IDX.items()}
    CRITICAL_PATH = frozenset([0,1,2,4,5,7,8,11,12,13,14,
                                18,19,20,21,26,27,28])
    STORAGE_CORE  = frozenset([13,14,20,21,26,27,28])
    SILENT_NODES  = frozenset([3,6,9,10,15,16,17,23,24,25,29])

    def __init__(self, capacity_compose, capacity_home):
        self.capacity_compose = capacity_compose
        self.capacity_home    = capacity_home

    def identify_pattern(self, flagged: set) -> str:
        flagged_fs = frozenset(int(n) for n in flagged)
        for name, nodes in self.PATTERNS.items():
            if flagged_fs == nodes:
                return name
        best, best_iou = 'none', 0.0
        for name, nodes in self.PATTERNS.items():
            if name == 'none' or not nodes:
                continue
            iou = (len(flagged_fs & nodes) /
                   max(len(flagged_fs | nodes), 1))
            if iou > best_iou:
                best, best_iou = name, iou
        return best if best_iou > 0.7 else 'none'

    def get_tensor(self, n_nodes: int = 30) -> torch.Tensor:
        cap = (self.capacity_compose if n_nodes == 30
               else self.capacity_home)
        return torch.tensor(cap, dtype=torch.float32)


# ----------------------------- Model -----------------------------

class TOCGATLayer(nn.Module):
    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 heads:        int   = 4,
                 toc_lambda:   float = 2.0,
                 dropout:      float = 0.1):
        super().__init__()
        self.toc_lambda = toc_lambda
        self.toc_scale  = nn.Parameter(torch.tensor(1.0))

        self.gat = GATConv(in_channels, out_channels,
                           heads=heads, dropout=dropout,
                           add_self_loops=True)

        full_dim  = out_channels * heads
        self.proj = nn.Linear(full_dim, full_dim)
        self.norm = nn.LayerNorm(full_dim)

    def forward(self,
                x:            torch.Tensor,
                edge_index:   torch.Tensor,
                toc_capacity: torch.Tensor
                ) -> torch.Tensor:
        capacity_weight = (1.0 + self.toc_lambda *
                           self.toc_scale * toc_capacity)
        x_toc = x * capacity_weight.unsqueeze(-1)

        h = self.gat(x_toc, edge_index)
        h = self.proj(h)
        h = self.norm(h)
        return F.elu(h)


class SpatialEncoder(nn.Module):
    def __init__(self,
                 in_feats: int,
                 hidden:   int,
                 n_layers: int = 2,
                 heads:    int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TOCGATLayer(in_feats, hidden // heads, heads))
        for _ in range(n_layers - 1):
            self.layers.append(TOCGATLayer(hidden, hidden // heads, heads))
        self.out_dim = hidden

    @staticmethod
    def build_batched_edge_index(edge_index: torch.Tensor,
                                  n_nodes:    int,
                                  n_graphs:   int) -> torch.Tensor:
        E = edge_index.shape[1]
        device = edge_index.device
        offsets = (torch.arange(n_graphs, device=device)
                   .repeat_interleave(E) * n_nodes)
        repeated = edge_index.repeat(1, n_graphs)
        return repeated + offsets.unsqueeze(0)

    def forward(self,
                x_seq:        torch.Tensor,
                edge_index:   torch.Tensor,
                toc_capacity: torch.Tensor
                ) -> torch.Tensor:
        B, T, N, Fdim = x_seq.shape

        x_flat = x_seq.reshape(B * T * N, Fdim)

        batched_ei = self.build_batched_edge_index(
            edge_index, N, B * T
        )
        cap_flat = toc_capacity.repeat(B * T)

        h = x_flat
        for layer in self.layers:
            h = layer(h, batched_ei, cap_flat)

        return h.reshape(B, T, N, self.out_dim)


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
    def __init__(self, d_model: int, n_vars: int):
        super().__init__()
        self.n_vars   = n_vars
        self.var_grns = nn.ModuleList(
            [GatedResidual(d_model) for _ in range(n_vars)]
        )
        self.weight_net = nn.Linear(d_model * n_vars, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        processed = [self.var_grns[i](x[..., i, :])
                     for i in range(self.n_vars)]
        concat  = torch.cat(processed, dim=-1)
        weights = torch.softmax(
            self.weight_net(concat), dim=-1
        ).unsqueeze(-1)
        stacked = torch.stack(processed, dim=2)
        return (stacked * weights).sum(dim=2)


class TemporalEncoder(nn.Module):
    def __init__(self,
                 d_spatial:     int,
                 d_model:       int = 128,
                 n_heads:       int = 4,
                 n_lstm_layers: int = 2,
                 dropout:       float = 0.1):
        super().__init__()

        self.input_proj = nn.Linear(d_spatial, d_model)
        self.feature_split = nn.Linear(d_model, d_model * 5)
        self.vsn = VariableSelectionNetwork(d_model, n_vars=5)

        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=n_lstm_layers, batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.grn  = GatedResidual(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, h_spatial: torch.Tensor) -> torch.Tensor:
        B, T, N, D = h_spatial.shape

        x = h_spatial.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x = self.input_proj(x)

        x_split = self.feature_split(x)
        x_split = x_split.view(B * N, T, 5, -1)
        x_vsn   = self.vsn(x_split)

        x_lstm, _ = self.lstm(x_vsn)
        x_attn, _ = self.attn(x_lstm, x_lstm, x_lstm)
        x_out     = self.grn(self.norm(x_attn + x_lstm))

        x_final = x_out[:, -1, :]
        return x_final.reshape(B, N, -1)


class CausalInferenceLayer(nn.Module):
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
        d  = W.shape[0]
        WW = W * W
        I  = torch.eye(d, device=W.device)
        M  = I + WW / d + (WW @ WW) / (2 * d * d)
        return M.trace() - d

    def forward(self,
                h_temporal:   torch.Tensor,
                toc_capacity: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = self.n_nodes
        W = torch.sigmoid(self.W_raw)
        W = W * (1 - torch.eye(N, device=W.device))

        causal_contrib = torch.sigmoid(self.encoder(h_temporal))
        causal_graph   = W.unsqueeze(0) * causal_contrib
        causal_graph_mean = causal_graph.mean(0)

        out_degree = causal_graph.sum(dim=-1)
        rcs = out_degree * toc_capacity.unsqueeze(0)

        dag_penalty = self.acyclicity_constraint(causal_graph_mean)
        return causal_graph_mean, rcs, dag_penalty


def rcs_summary(rcs: torch.Tensor, k: int = 3) -> torch.Tensor:
    topk = torch.topk(rcs, k=min(k, rcs.shape[-1]), dim=-1).values
    if topk.shape[-1] < k:
        pad = torch.zeros(rcs.shape[0], k - topk.shape[-1],
                          device=rcs.device)
        topk = torch.cat([topk, pad], dim=-1)

    mean = rcs.mean(dim=-1, keepdim=True)
    mx   = rcs.max(dim=-1, keepdim=True).values
    std  = rcs.std(dim=-1, keepdim=True)

    return torch.cat([topk, mean, mx, std], dim=-1)


class PredictionHeads(nn.Module):
    def __init__(self, d_model: int, n_patterns: int = 8,
                 summary_dim: int = 6):
        super().__init__()
        self.pool = nn.Linear(d_model, d_model)
        head_in = d_model + summary_dim

        self.head_bn = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.head_pattern = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(),
            nn.Linear(64, n_patterns),
        )
        self.head_ttb = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Softplus(),
        )

    def forward(self, h_temporal, rcs):
        h_graph  = self.pool(h_temporal).mean(dim=1)
        summary  = rcs_summary(rcs, k=3)
        combined = torch.cat([h_graph, summary], dim=-1)

        return (self.head_bn(combined),
                self.head_pattern(combined),
                self.head_ttb(combined))


class CSTGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model

        self.spatial = SpatialEncoder(
            in_feats=m.gat_in_feats, hidden=m.gat_hidden,
            n_layers=m.gat_layers, heads=m.gat_heads,
        )
        self.temporal = TemporalEncoder(
            d_spatial=m.gat_hidden, d_model=m.tft_hidden,
            n_heads=m.tft_heads, n_lstm_layers=m.lstm_layers,
            dropout=m.tft_dropout,
        )
        self.heads = PredictionHeads(m.tft_hidden, m.n_patterns)

        self._causal_cache: Dict[int, CausalInferenceLayer] = {}
        self.tft_hidden = m.tft_hidden

    def _get_causal(self, n_nodes: int, device) -> CausalInferenceLayer:
        if n_nodes not in self._causal_cache:
            layer = CausalInferenceLayer(self.tft_hidden, n_nodes).to(device)
            self._causal_cache[n_nodes] = layer
            self.add_module(f'causal_{n_nodes}', layer)
        return self._causal_cache[n_nodes]

    def forward(self, x_seq, edge_index, toc_capacity):
        B, T, N, _ = x_seq.shape
        device = x_seq.device

        h_spatial  = self.spatial(x_seq, edge_index, toc_capacity)
        h_temporal = self.temporal(h_spatial)

        causal = self._get_causal(N, device)
        causal_graph, rcs, dag_penalty = causal(h_temporal, toc_capacity)

        bn_logit, pattern_logit, ttb = self.heads(h_temporal, rcs)

        return {
            'bn_logit': bn_logit, 'pattern_logit': pattern_logit,
            'ttb': ttb, 'causal_graph': causal_graph,
            'rcs': rcs, 'dag_penalty': dag_penalty,
        }


# ----------------------------- Data -----------------------------

class CachedWindowDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['x'], s['label'], s['pattern_idx'], s['ttb']


def make_loader(split, batch_size, shuffle, cache_dir, generator=None):
    samples = torch.load(f'{cache_dir}/{split}.pt', weights_only=False)
    return DataLoader(CachedWindowDataset(samples), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=True,
                      generator=generator)


# ----------------------------- Loss -----------------------------

class _TOCWeightedLossBase(nn.Module):
    def __init__(self, cfg, toc, pattern_class_weights):
        super().__init__()
        t = cfg.training
        self.fn_weight       = t.fn_weight
        self.fp_weight       = t.fp_weight
        self.constraint_mult = t.constraint_mult
        self.lambda_pattern  = t.lambda_pattern
        self.lambda_ttb      = t.lambda_ttb
        self.lambda_causal   = t.lambda_causal
        self.lambda_sub      = t.lambda_sub

        node_weights = torch.ones(30)
        for n in toc.CRITICAL_PATH:
            node_weights[n] = t.constraint_mult
        self.register_buffer('node_weights', node_weights)

        silent_mask = torch.zeros(30)
        for n in toc.SILENT_NODES:
            silent_mask[n] = 1.0
        self.register_buffer('silent_mask', silent_mask)

        self.register_buffer('pattern_class_weights',
                             pattern_class_weights)

    def detection_loss(self, bn_logit, bn_label, rcs):
        bn_prob = torch.sigmoid(bn_logit.squeeze(-1))
        label   = bn_label.float()

        bce = F.binary_cross_entropy(bn_prob, label, reduction='none')

        pos_w = label * self.fn_weight
        neg_w = (1 - label) * self.fp_weight
        asym  = pos_w + neg_w

        crit = self.node_weights.unsqueeze(0)
        rcs_crit = (rcs * crit).max(dim=-1).values
        rcs_crit = rcs_crit / (rcs_crit.max() + 1e-8)

        missed = label * (1 - (bn_prob > 0.5).float())
        amp = 1.0 + missed * rcs_crit

        return (bce * asym * amp).mean()

    def pattern_loss(self, pattern_logit, pattern_idx):
        return F.cross_entropy(pattern_logit, pattern_idx,
                               weight=self.pattern_class_weights)

    def ttb_loss(self, ttb_pred, ttb_true, bn_label):
        mask = bn_label.float().unsqueeze(-1)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=ttb_pred.device)
        return F.huber_loss(ttb_pred * mask,
                            ttb_true.unsqueeze(-1) * mask,
                            delta=1.0)

    def subordination_loss(self, rcs):
        silent = self.silent_mask.unsqueeze(0)
        rcs_silent = rcs * silent
        violation  = F.relu(rcs_silent - 0.1)
        return (violation ** 2).mean()

    def causal_loss(self, causal_graph, dag_penalty):
        sparsity = causal_graph.abs().mean()
        return sparsity + (dag_penalty ** 2)

    def forward(self, outputs, targets):
        L_det = self.detection_loss(outputs['bn_logit'],
                                    targets['label'], outputs['rcs'])
        L_pat = self.pattern_loss(outputs['pattern_logit'],
                                  targets['pattern_idx'])
        L_ttb = self.ttb_loss(outputs['ttb'],
                              targets['ttb'], targets['label'])
        L_sub = self.subordination_loss(outputs['rcs'])
        L_cau = self.causal_loss(outputs['causal_graph'],
                                 outputs['dag_penalty'])

        total = (L_det
                 + self.lambda_pattern * L_pat
                 + self.lambda_ttb     * L_ttb
                 + self.lambda_causal  * L_cau
                 + self.lambda_sub     * L_sub)

        return {'total': total, 'detection': L_det, 'pattern': L_pat,
                'ttb': L_ttb, 'causal': L_cau, 'subordination': L_sub}


class TOCWeightedLoss(_TOCWeightedLossBase):
    def __init__(self, cfg, toc, pattern_class_weights):
        super().__init__(cfg, toc, pattern_class_weights)
        self.toc_ref = toc
        self.lambda_rcs_sup = cfg.training.lambda_rcs_sup

    def rcs_supervision_loss(self, rcs, pattern_idx, label, margin=0.1):
        n_nodes = rcs.shape[-1]
        losses = []
        for i in range(rcs.shape[0]):
            if label[i] == 0:
                continue
            pat_name = self.toc_ref.IDX_TO_PATTERN.get(int(pattern_idx[i]), 'none')
            flagged = [n for n in self.toc_ref.PATTERNS.get(pat_name, frozenset())
                       if n < n_nodes]
            if not flagged or len(flagged) >= n_nodes:
                continue
            unflagged = [n for n in range(n_nodes) if n not in flagged]
            rcs_f = rcs[i, flagged].mean()
            rcs_u = rcs[i, unflagged].mean()
            losses.append(F.relu(margin - (rcs_f - rcs_u)))
        if not losses:
            return torch.tensor(0.0, device=rcs.device)
        return torch.stack(losses).mean()

    def forward(self, outputs, targets):
        base = super().forward(outputs, targets)
        L_rcs = self.rcs_supervision_loss(
            outputs['rcs'], targets['pattern_idx'], targets['label'])
        base['total'] = base['total'] + self.lambda_rcs_sup * L_rcs
        base['rcs_supervision'] = L_rcs
        return base


# ----------------------------- Evaluator -----------------------------

class TOCEvaluator:
    def __init__(self, toc: 'TOCPriorLoader', n_nodes: int = 30):
        self.toc = toc
        self.n_nodes = n_nodes
        self.reset()

    def reset(self):
        self.labels, self.probs = [], []
        self.pattern_pred, self.pattern_true = [], []
        self.ttb_pred, self.ttb_true = [], []
        self.rcs_all = []

    def update(self, outputs, targets):
        probs = torch.sigmoid(outputs['bn_logit']).squeeze(-1)
        self.labels.extend(targets['label'].cpu().tolist())
        self.probs.extend(probs.detach().cpu().tolist())

        pat_pred = outputs['pattern_logit'].argmax(dim=-1)
        self.pattern_pred.extend(pat_pred.cpu().tolist())
        self.pattern_true.extend(targets['pattern_idx'].cpu().tolist())

        self.ttb_pred.extend(outputs['ttb'].squeeze(-1).detach().cpu().tolist())
        self.ttb_true.extend(targets['ttb'].cpu().tolist())

        self.rcs_all.extend(outputs['rcs'].detach().cpu().tolist())

    def compute(self) -> dict:
        from sklearn.metrics import roc_auc_score, f1_score

        labels = np.array(self.labels)
        probs  = np.array(self.probs)
        preds  = (probs > 0.5).astype(int)

        res = {}
        res['auc'] = (roc_auc_score(labels, probs)
                      if len(set(labels)) > 1 else 0.0)
        res['f1']  = f1_score(labels, preds, zero_division=0)
        res['precision'] = ((preds*labels).sum() / (preds.sum()+1e-8))
        res['recall']    = ((preds*labels).sum() / (labels.sum()+1e-8))

        pat_pred = np.array(self.pattern_pred)
        pat_true = np.array(self.pattern_true)
        res['pattern_accuracy'] = float((pat_pred == pat_true).mean())

        if self.n_nodes == 30:
            rcs_arr = np.array(self.rcs_all)
            crit = self.toc.CRITICAL_PATH
            correct, total = 0, 0
            for i, lab in enumerate(self.labels):
                if lab == 0:
                    continue
                total += 1
                top3 = set(np.argsort(rcs_arr[i])[-3:].tolist())
                if top3 & crit:
                    correct += 1
            res['cp_recall'] = correct / max(total, 1)

            silent = list(self.toc.SILENT_NODES)
            rcs_norm = rcs_arr / (rcs_arr.max(axis=1, keepdims=True)+1e-8)
            res['subordination_score'] = float(
                1.0 - rcs_norm[:, silent].mean()
            )
        else:
            res['cp_recall'] = None
            res['subordination_score'] = None

        ttb_pred = np.array(self.ttb_pred)
        tp = (labels == 1) & (preds == 1)
        res['lead_time_min'] = (float(ttb_pred[tp].mean())
                                if tp.sum() > 0 else 0.0)

        return res