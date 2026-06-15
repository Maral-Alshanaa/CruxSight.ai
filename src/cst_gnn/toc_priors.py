# ============================================================
# CruxSight.ai — Theory of Constraints Priors
# Patterns A-G, critical path, capacity priors
# ============================================================

import numpy as np
import torch


class TOCPriorLoader:
    """
    Encodes the seven structural bottleneck patterns discovered
    from the 196-file DeathStarBench analysis, along with the
    ToC-derived node classifications (critical path, storage
    core, silent nodes).

    Pattern taxonomy (compose workflow, N=30):
        A — Entry layer only
        B — Storage core only
        C — Middle tier only
        D — Entry + Storage core (hybrid, 49.7% of files)
        E — Full cascade
        F — Partial storage
        G — Home workflow pattern (N=7)
        none — No bottleneck / unclassified
    """

    PATTERNS = {
        'A':    frozenset([4,5,7,8,11,12,13,14,18,19,20,21,26,27,28]),
        'B':    frozenset([4,5,13,14,20,21,26,27,28]),
        'C':    frozenset([0,1,2,22]),
        'D':    frozenset([0,1,2,13,14,20,21,22,26,27,28]),
        'E':    frozenset([0,1,2,4,5,7,8,11,12,13,14,
                           18,19,20,21,22,26,27,28]),
        'F':    frozenset([0,1,2,4,5,7,8,11,12,18,19]),
        'G':    frozenset([3,4]),
        'none': frozenset(),
    }

    PATTERN_TO_IDX = {
        p: i for i, p in
        enumerate(['A','B','C','D','E','F','G','none'])
    }
    IDX_TO_PATTERN = {v: k for k, v in PATTERN_TO_IDX.items()}

    # Nodes that must always be active in a constrained path
    CRITICAL_PATH = frozenset([
        0,1,2,4,5,7,8,11,12,13,14,18,19,20,21,26,27,28
    ])

    # The irreducible storage core — appears in Patterns A, B, D, E
    # Architectural finding: constraint is topology-driven,
    # not resource-driven (same subgraph under CPU, memory,
    # and combined stress)
    STORAGE_CORE = frozenset([13,14,20,21,26,27,28])

    # Nodes that should never carry high RCS (subordination)
    SILENT_NODES = frozenset([3,6,9,10,15,16,17,23,24,25,29])

    def __init__(self,
                 capacity_compose: np.ndarray,
                 capacity_home:    np.ndarray):
        self.capacity_compose = capacity_compose
        self.capacity_home    = capacity_home

    def identify_pattern(self, flagged: set) -> str:
        """Match a set of flagged nodes to the closest pattern."""
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
        """Return capacity prior as a float tensor for the given graph."""
        cap = (self.capacity_compose if n_nodes == 30
               else self.capacity_home)
        return torch.tensor(cap, dtype=torch.float32)

    @classmethod
    def from_files(cls,
                   compose_path: str,
                   home_path:    str) -> 'TOCPriorLoader':
        """Load capacity arrays from .npy files saved during analysis."""
        return cls(
            capacity_compose=np.load(compose_path),
            capacity_home=np.load(home_path),
        )
