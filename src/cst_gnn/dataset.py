# ============================================================
# CruxSight.ai — Dataset & Graph Builder
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CachedWindowDataset(Dataset):
    """
    Thin wrapper around the cached list of windowed samples.

    Each sample is a dict with keys:
        x           : (T, N, F) latency feature tensor
        label       : 0 or 1 (bottleneck / no bottleneck)
        pattern_idx : int (0-7, maps to patterns A-G + none)
        ttb         : float (time to breach in minutes)

    The cache (train.pt, val.pt, test.pt) is pre-built by the
    analysis pipeline and stored on Google Drive. Each file
    contains a list of these sample dicts.
    """

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['x'], s['label'], s['pattern_idx'], s['ttb']


def make_loader(cache_dir: str, split: str,
                batch_size: int, shuffle: bool) -> DataLoader:
    """
    Load a cached split and return a DataLoader.

    Args:
        cache_dir  : path to directory containing train/val/test.pt
        split      : 'train', 'val', or 'test'
        batch_size : number of samples per batch
        shuffle    : whether to shuffle (True for training)
    """
    samples = torch.load(f'{cache_dir}/{split}.pt')
    ds = CachedWindowDataset(samples)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


class GraphBuilder:
    """
    Builds edge_index (COO format) for the two known topologies:
      - Compose workflow : 30 nodes
      - Home workflow    : 7 nodes

    Edge lists are based on the DeathStarBench social-network
    call graph structure observed in the 196-file analysis.
    Reverse edges are added for bidirectional message passing
    (DeathStarBench RPCs have request/response flow).
    """

    # Compose workflow (30 nodes) — directed service call edges
    COMPOSE_EDGES = [
        (0,1),(0,2),(0,22),
        (1,3),(1,4),(2,3),(2,4),
        (3,5),(3,6),(4,5),(4,6),
        (5,7),(5,8),(5,13),
        (6,7),(6,8),
        (7,9),(7,10),(8,11),(8,12),
        (13,14),(13,20),(14,21),
        (20,26),(21,27),(26,28),(27,28),
        (22,13),(22,14),
        (15,16),(16,17),(17,18),(18,19),(19,20),
    ]

    # Home workflow (7 nodes) — simple timeline-read chain
    HOME_EDGES = [
        (0,1),(0,2),(1,3),(2,3),(3,4),(4,5),(4,6),
    ]

    def __init__(self, capacity: np.ndarray):
        self.capacity = capacity

    def get_edge_index(self, n_nodes: int) -> torch.Tensor:
        """
        Returns (2, E) edge_index for PyG, filtered to valid
        node indices. Adds reverse edges for bidirectional flow.
        """
        edges = self.COMPOSE_EDGES if n_nodes == 30 else self.HOME_EDGES
        valid = [(s, d) for s, d in edges
                 if s < n_nodes and d < n_nodes]
        if not valid:
            return torch.zeros(2, 0, dtype=torch.long)
        all_edges = valid + [(d, s) for s, d in valid]
        return torch.tensor(all_edges, dtype=torch.long).t().contiguous()

    def build_dense(self, n_nodes: int) -> torch.Tensor:
        """Dense adjacency (N, N) weighted by destination capacity."""
        A = torch.zeros(n_nodes, n_nodes)
        edges = self.COMPOSE_EDGES if n_nodes == 30 else self.HOME_EDGES
        for src, dst in edges:
            if src < n_nodes and dst < n_nodes:
                cap = (float(self.capacity[dst])
                       if dst < len(self.capacity) else 0.0)
                A[src, dst] = cap + 0.1
        return A
