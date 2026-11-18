from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torch_cluster import radius_graph as tc_radius_graph
except Exception:
    tc_radius_graph = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    x: (N, F)
    batch: (N,) with values in [0, B-1]
    returns: (B, F)
    """
    n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    out = torch.zeros(n_graphs, x.size(-1), device=x.device, dtype=x.dtype)
    count = torch.zeros(n_graphs, 1, device=x.device, dtype=x.dtype)

    out.index_add_(0, batch, x)
    ones = torch.ones(batch.size(0), 1, device=x.device, dtype=x.dtype)
    count.index_add_(0, batch, ones)
    out = out / count.clamp_min(1.0)
    return out


def global_sum_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    out = torch.zeros(n_graphs, x.size(-1), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    return out


def build_radius_graph(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    max_neighbors: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edges within each molecule using brute-force pair distances.

    Returns
    -------
    edge_index: LongTensor of shape (2, E)
        edge_index[0] = src
        edge_index[1] = dst
    edge_vectors: FloatTensor of shape (E, 3)
        vector from dst to src: pos[src] - pos[dst]
    """
    device = pos.device

    # Fast path: torch-cluster radius graph (GPU/CPU optimized neighbor search).
    if tc_radius_graph is not None:
        n_nodes = int(pos.shape[0])
        if n_nodes <= 1:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_vectors = torch.empty((0, 3), dtype=pos.dtype, device=device)
            return edge_index, edge_vectors

        # torch-cluster requires an integer; use large cap when no explicit limit is set.
        max_num_neighbors = max_neighbors if max_neighbors is not None else max(1, n_nodes - 1)
        edge_index = tc_radius_graph(
            x=pos,
            r=cutoff,
            batch=batch,
            loop=False,
            max_num_neighbors=max_num_neighbors,
            flow="source_to_target",
        )
        src = edge_index[0]
        dst = edge_index[1]
        edge_vectors = pos[src] - pos[dst]
        return edge_index, edge_vectors

    # Fallback path: brute-force pair distances.
    edge_src = []
    edge_dst = []
    edge_vecs = []

    n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    for g in range(n_graphs):
        idx = torch.where(batch == g)[0]
        if idx.numel() <= 1:
            continue

        p = pos[idx]  # (n_g, 3)
        diff = p[:, None, :] - p[None, :, :]  # (n_g, n_g, 3)
        dist = torch.norm(diff, dim=-1)  # (n_g, n_g)

        mask = (dist <= cutoff) & (dist > 0.0)

        for i_local in range(idx.numel()):
            nbr_local = torch.where(mask[i_local])[0]

            if max_neighbors is not None and nbr_local.numel() > max_neighbors:
                d = dist[i_local, nbr_local]
                order = torch.argsort(d)
                nbr_local = nbr_local[order[:max_neighbors]]

            src_global = idx[nbr_local]
            dst_global = idx[i_local].repeat(nbr_local.numel())
            vec = pos[src_global] - pos[dst_global]

            edge_src.append(src_global)
            edge_dst.append(dst_global)
            edge_vecs.append(vec)

    if len(edge_src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_vectors = torch.empty((0, 3), dtype=pos.dtype, device=device)
    else:
        edge_index = torch.stack(
            [torch.cat(edge_src, dim=0), torch.cat(edge_dst, dim=0)], dim=0
        )
        edge_vectors = torch.cat(edge_vecs, dim=0)

    return edge_index, edge_vectors


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, n_rbf)
        widths = torch.full((n_rbf,), (cutoff / max(n_rbf - 1, 1)))
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        distances: (E,)
        returns: (E, n_rbf)
        """
        x = distances[:, None]
        rbf = torch.exp(-0.5 * ((x - self.centers[None, :]) / self.widths[None, :]) ** 2)

        cutoff_fn = 0.5 * (
            torch.cos(math.pi * distances / self.cutoff).clamp(min=-1.0, max=1.0) + 1.0
        )
        cutoff_fn = torch.where(distances <= self.cutoff, cutoff_fn, torch.zeros_like(cutoff_fn))
        return rbf * cutoff_fn[:, None]
