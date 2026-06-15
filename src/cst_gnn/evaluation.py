# ============================================================
# CruxSight.ai — Evaluator
# Tracks predictions across an epoch, computes all metrics
# ============================================================

import numpy as np
import torch


class TOCEvaluator:
    """
    Computes standard + ToC-specific metrics after each epoch.

    Standard metrics:
        AUC, F1, Precision, Recall

    ToC-specific metrics:
        pattern_accuracy    — correct identification of A-G pattern
        cp_recall           — do top-3 RCS nodes intersect the
                              critical path? (compose graph only)
        subordination_score — 1 - mean RCS on silent nodes
                              (measures ToC subordination principle)
        lead_time_min       — average predicted time-to-breach
                              on true positive samples

    Note on cp_recall: this metric is currently near its random
    baseline (0.944 for top-3 of 30 nodes) because RCS rankings
    are dominated by the static ToC capacity prior rather than
    the per-sample causal signal. See README Limitations.
    """

    def __init__(self, toc, n_nodes: int = 30):
        self.toc     = toc
        self.n_nodes = n_nodes
        self.reset()

    def reset(self):
        self.labels       = []
        self.probs        = []
        self.pattern_pred = []
        self.pattern_true = []
        self.ttb_pred     = []
        self.ttb_true     = []
        self.rcs_all      = []

    def update(self, outputs: dict, targets: dict):
        probs = torch.sigmoid(outputs['bn_logit']).squeeze(-1)
        self.labels.extend(targets['label'].cpu().tolist())
        self.probs.extend(probs.detach().cpu().tolist())
        pat_pred = outputs['pattern_logit'].argmax(dim=-1)
        self.pattern_pred.extend(pat_pred.cpu().tolist())
        self.pattern_true.extend(targets['pattern_idx'].cpu().tolist())
        self.ttb_pred.extend(
            outputs['ttb'].squeeze(-1).detach().cpu().tolist())
        self.ttb_true.extend(targets['ttb'].cpu().tolist())
        self.rcs_all.extend(outputs['rcs'].detach().cpu().tolist())

    def compute(self) -> dict:
        from sklearn.metrics import roc_auc_score, f1_score

        labels = np.array(self.labels)
        probs  = np.array(self.probs)
        preds  = (probs > 0.5).astype(int)

        res = {}

        # Standard metrics
        res['auc'] = (roc_auc_score(labels, probs)
                      if len(set(labels)) > 1 else 0.0)
        res['f1']  = f1_score(labels, preds, zero_division=0)
        res['precision'] = float(
            (preds * labels).sum() / (preds.sum() + 1e-8))
        res['recall'] = float(
            (preds * labels).sum() / (labels.sum() + 1e-8))

        # Pattern accuracy
        pat_pred = np.array(self.pattern_pred)
        pat_true = np.array(self.pattern_true)
        res['pattern_accuracy'] = float((pat_pred == pat_true).mean())

        # ToC-specific metrics (compose graph only)
        if self.n_nodes == 30:
            rcs_arr = np.array(self.rcs_all)
            crit    = self.toc.CRITICAL_PATH

            # CP Recall — top-3 RCS nodes vs critical path
            correct, total = 0, 0
            for i, lab in enumerate(self.labels):
                if lab == 0:
                    continue
                total += 1
                top3 = set(np.argsort(rcs_arr[i])[-3:].tolist())
                if top3 & crit:
                    correct += 1
            res['cp_recall'] = correct / max(total, 1)

            # Subordination score — silent nodes should have low RCS
            silent   = list(self.toc.SILENT_NODES)
            rcs_norm = rcs_arr / (
                rcs_arr.max(axis=1, keepdims=True) + 1e-8)
            res['subordination_score'] = float(
                1.0 - rcs_norm[:, silent].mean())
        else:
            res['cp_recall']           = None
            res['subordination_score'] = None

        # Lead time — average TTB on true positives
        ttb_pred = np.array(self.ttb_pred)
        tp = (labels == 1) & (preds == 1)
        res['lead_time_min'] = (float(ttb_pred[tp].mean())
                                if tp.sum() > 0 else 0.0)

        return res
