# ============================================================
# CruxSight.ai — Loss Function
# TOC-Weighted Loss + Direct RCS Supervision
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class TOCWeightedLoss(nn.Module):
    """
    Combined training loss with five terms:

    L = L_detection + λ_pat·L_pattern + λ_ttb·L_ttb
      + λ_cau·L_causal + λ_sub·L_subordination
      + λ_rcs·L_rcs_supervision

    Detection loss is asymmetric: missing a bottleneck (FN)
    costs fn_weight × more than a false alarm (FP), amplified
    further if the missed constraint lies on the critical path.

    Subordination loss penalizes high RCS on silent nodes —
    ToC principle: idle buffer capacity should never appear
    to be the root cause.

    RCS supervision uses the empirically-derived pattern taxonomy
    (A-G) as weak labels: for positive samples, RCS on the
    pattern's flagged nodes should exceed RCS on non-flagged
    nodes by a margin. This is the key novel training signal
    connecting ToC theory to the causal layer.
    """

    def __init__(self, cfg, toc, pattern_class_weights: torch.Tensor):
        super().__init__()
        t = cfg.training
        self.fn_weight       = t.fn_weight
        self.fp_weight       = t.fp_weight
        self.constraint_mult = t.constraint_mult
        self.lambda_pattern  = t.lambda_pattern
        self.lambda_ttb      = t.lambda_ttb
        self.lambda_causal   = t.lambda_causal
        self.lambda_sub      = t.lambda_sub
        self.lambda_rcs_sup  = t.lambda_rcs_sup
        self.toc_ref         = toc

        # Critical path nodes get higher weight in detection loss
        node_weights = torch.ones(30)
        for n in toc.CRITICAL_PATH:
            node_weights[n] = t.constraint_mult
        self.register_buffer('node_weights', node_weights)

        # Silent nodes should never have high RCS
        silent_mask = torch.zeros(30)
        for n in toc.SILENT_NODES:
            silent_mask[n] = 1.0
        self.register_buffer('silent_mask', silent_mask)

        self.register_buffer('pattern_class_weights',
                             pattern_class_weights)

    def detection_loss(self, bn_logit: torch.Tensor,
                       bn_label: torch.Tensor,
                       rcs: torch.Tensor) -> torch.Tensor:
        bn_prob = torch.sigmoid(bn_logit.squeeze(-1))
        label   = bn_label.float()
        bce     = F.binary_cross_entropy(bn_prob, label,
                                          reduction='none')
        pos_w   = label * self.fn_weight
        neg_w   = (1 - label) * self.fp_weight
        asym    = pos_w + neg_w
        crit    = self.node_weights.unsqueeze(0)
        rcs_crit = (rcs * crit).max(dim=-1).values
        rcs_crit = rcs_crit / (rcs_crit.max() + 1e-8)
        missed  = label * (1 - (bn_prob > 0.5).float())
        amp     = 1.0 + missed * rcs_crit
        return (bce * asym * amp).mean()

    def pattern_loss(self, pattern_logit: torch.Tensor,
                     pattern_idx: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(pattern_logit, pattern_idx,
                               weight=self.pattern_class_weights)

    def ttb_loss(self, ttb_pred: torch.Tensor,
                 ttb_true: torch.Tensor,
                 bn_label: torch.Tensor) -> torch.Tensor:
        mask = bn_label.float().unsqueeze(-1)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=ttb_pred.device)
        return F.huber_loss(ttb_pred * mask,
                            ttb_true.unsqueeze(-1) * mask,
                            delta=1.0)

    def subordination_loss(self, rcs: torch.Tensor) -> torch.Tensor:
        silent    = self.silent_mask.unsqueeze(0)
        rcs_silent = rcs * silent
        violation  = F.relu(rcs_silent - 0.1)
        return (violation ** 2).mean()

    def causal_loss(self, causal_graph: torch.Tensor,
                    dag_penalty: torch.Tensor) -> torch.Tensor:
        sparsity = causal_graph.abs().mean()
        return sparsity + (dag_penalty ** 2)

    def rcs_supervision_loss(self, rcs: torch.Tensor,
                              pattern_idx: torch.Tensor,
                              label: torch.Tensor,
                              margin: float = 0.1) -> torch.Tensor:
        """
        For positive (bottleneck) samples: RCS on the pattern's
        flagged nodes should exceed RCS on non-flagged nodes by
        `margin`. Uses the empirical pattern taxonomy as weak labels.

        Note: currently limited by static capacity prior dominance.
        See README Limitations for future decoupling work.
        """
        n_nodes = rcs.shape[-1]
        losses  = []
        for i in range(rcs.shape[0]):
            if label[i] == 0:
                continue
            pat_name = self.toc_ref.IDX_TO_PATTERN.get(
                int(pattern_idx[i]), 'none')
            flagged  = [n for n in
                        self.toc_ref.PATTERNS.get(pat_name, frozenset())
                        if n < n_nodes]
            if not flagged or len(flagged) >= n_nodes:
                continue
            unflagged = [n for n in range(n_nodes)
                         if n not in flagged]
            rcs_f = rcs[i, flagged].mean()
            rcs_u = rcs[i, unflagged].mean()
            losses.append(F.relu(margin - (rcs_f - rcs_u)))
        if not losses:
            return torch.tensor(0.0, device=rcs.device)
        return torch.stack(losses).mean()

    def forward(self, outputs: dict,
                targets: dict) -> dict:
        L_det = self.detection_loss(
            outputs['bn_logit'], targets['label'], outputs['rcs'])
        L_pat = self.pattern_loss(
            outputs['pattern_logit'], targets['pattern_idx'])
        L_ttb = self.ttb_loss(
            outputs['ttb'], targets['ttb'], targets['label'])
        L_sub = self.subordination_loss(outputs['rcs'])
        L_cau = self.causal_loss(
            outputs['causal_graph'], outputs['dag_penalty'])
        L_rcs = self.rcs_supervision_loss(
            outputs['rcs'], targets['pattern_idx'], targets['label'])

        total = (L_det
                 + self.lambda_pattern * L_pat
                 + self.lambda_ttb     * L_ttb
                 + self.lambda_causal  * L_cau
                 + self.lambda_sub     * L_sub
                 + self.lambda_rcs_sup * L_rcs)

        return {
            'total':          total,
            'detection':      L_det,
            'pattern':        L_pat,
            'ttb':            L_ttb,
            'causal':         L_cau,
            'subordination':  L_sub,
            'rcs_supervision': L_rcs,
        }
