from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch>=2.6 defaults torch.load(weights_only=True), but current e3nn/mace stacks
# still load internal constants via plain torch.load(...). Keep compatibility.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from mace_layer import MACE_layer

from .utils import GaussianRBF, build_radius_graph, global_mean_pool, global_sum_pool


class AtomicNumberEmbedding(nn.Module):
    """Convert atomic number to one-hot node attributes for MACE."""

    def __init__(self, max_z: int):
        super().__init__()
        self.max_z = max_z

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.one_hot(z - 1, num_classes=self.max_z).float()


class MACEGlobalSpectrumModel(nn.Module):
    """
    Predict whole-molecule fixed-length spectrum from:
      z:   (N,)
      pos: (N, 3)
      batch: (N,)
    """

    def __init__(
        self,
        max_z: int,
        n_energy: int,
        cutoff: float = 5.0,
        max_neighbors: Optional[int] = None,
        n_rbf: int = 16,
        hidden_dim: int = 64,
        n_layers: int = 4,
        correlation: int = 3,
        max_ell: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
        multi_scale_readout: bool = False,
    ):
        super().__init__()
        self.max_z = max_z
        self.n_energy = n_energy
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pooling = pooling
        self.task_type = "global"
        self.multi_scale_readout = multi_scale_readout

        self.node_attr_encoder = AtomicNumberEmbedding(max_z=max_z)
        self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

        hidden_irreps = f"{hidden_dim}x0e"
        edge_feats_irreps = f"{n_rbf}x0e"
        node_feats_irreps = hidden_irreps

        self.input_proj = nn.Linear(max_z, hidden_dim)

        self.layers = nn.ModuleList(
            [
                MACE_layer(
                    max_ell=max_ell,
                    correlation=correlation,
                    n_dims_in=max_z,
                    hidden_irreps=hidden_irreps,
                    node_feats_irreps=node_feats_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    avg_num_neighbors=10.0,
                    use_sc=True,
                )
                for _ in range(n_layers)
            ]
        )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        decoder_in_dim = hidden_dim * (n_layers if multi_scale_readout else 1)

        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, n_energy),
        )

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_attrs = self.node_attr_encoder(z)
        node_feats = self.input_proj(node_attrs)

        if (edge_index is None) != (edge_vectors is None):
            raise ValueError("edge_index and edge_vectors must be provided together.")
        if edge_index is None:
            edge_index, edge_vectors = build_radius_graph(
                pos=pos,
                batch=batch,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
            )

        if edge_index.numel() == 0:
            raise RuntimeError("No edges were constructed. Increase cutoff or check coordinates.")

        edge_dist = torch.norm(edge_vectors, dim=-1)
        edge_feats = self.rbf(edge_dist)

        avg_num_neighbors = edge_index.shape[1] / max(1, z.shape[0])

        x = node_feats
        pooled_features = []
        for layer, norm in zip(self.layers, self.norms):
            if hasattr(layer, "avg_num_neighbors"):
                layer.avg_num_neighbors = float(avg_num_neighbors)

            x_res = x
            x = layer(
                edge_vectors,
                x,
                node_attrs,
                edge_feats,
                edge_index,
            )
            x = norm(x + x_res)

            if self.pooling == "mean":
                pooled_features.append(global_mean_pool(x, batch))
            elif self.pooling == "sum":
                pooled_features.append(global_sum_pool(x, batch))
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

        mol_feat = (
            torch.cat(pooled_features, dim=-1)
            if self.multi_scale_readout
            else pooled_features[-1]
        )

        return self.decoder(mol_feat)


class MACEAbsorbingAtomSpectrumModel(nn.Module):
    """
    Predict absorbing-atom-conditioned fixed-length spectrum from:
      z:   (N,)
      pos: (N, 3)
      batch: (N,)
      absorbing_atom_index: (B,)
    """

    def __init__(
        self,
        max_z: int,
        n_energy: int,
        cutoff: float = 5.0,
        max_neighbors: Optional[int] = None,
        n_rbf: int = 16,
        hidden_dim: int = 64,
        n_layers: int = 4,
        correlation: int = 3,
        max_ell: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
        multi_scale_readout: bool = False,
    ):
        super().__init__()
        self.max_z = max_z
        self.n_energy = n_energy
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pooling = pooling
        self.task_type = "absorbing_atom"
        self.multi_scale_readout = multi_scale_readout

        self.node_attr_encoder = AtomicNumberEmbedding(max_z=max_z)
        self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

        hidden_irreps = f"{hidden_dim}x0e"
        edge_feats_irreps = f"{n_rbf}x0e"
        node_feats_irreps = hidden_irreps

        # One extra node feature indicates the selected absorbing atom.
        self.input_proj = nn.Linear(max_z + 1, hidden_dim)

        self.layers = nn.ModuleList(
            [
                MACE_layer(
                    max_ell=max_ell,
                    correlation=correlation,
                    n_dims_in=max_z + 1,
                    hidden_irreps=hidden_irreps,
                    node_feats_irreps=node_feats_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    avg_num_neighbors=10.0,
                    use_sc=True,
                )
                for _ in range(n_layers)
            ]
        )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        decoder_in_dim = 2 * hidden_dim * (n_layers if multi_scale_readout else 1)

        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, n_energy),
        )

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        absorbing_atom_index: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if absorbing_atom_index.ndim != 1:
            raise ValueError("absorbing_atom_index must have shape (B,)")

        node_attrs = self.node_attr_encoder(z)
        absorbing_flag = torch.zeros(
            (z.shape[0], 1), dtype=node_attrs.dtype, device=node_attrs.device
        )
        absorbing_flag[absorbing_atom_index, 0] = 1.0
        node_inputs = torch.cat([node_attrs, absorbing_flag], dim=-1)
        node_feats = self.input_proj(node_inputs)

        if (edge_index is None) != (edge_vectors is None):
            raise ValueError("edge_index and edge_vectors must be provided together.")
        if edge_index is None:
            edge_index, edge_vectors = build_radius_graph(
                pos=pos,
                batch=batch,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
            )

        if edge_index.numel() == 0:
            raise RuntimeError("No edges were constructed. Increase cutoff or check coordinates.")

        edge_dist = torch.norm(edge_vectors, dim=-1)
        edge_feats = self.rbf(edge_dist)

        avg_num_neighbors = edge_index.shape[1] / max(1, z.shape[0])

        x = node_feats
        readout_features = []
        for layer, norm in zip(self.layers, self.norms):
            if hasattr(layer, "avg_num_neighbors"):
                layer.avg_num_neighbors = float(avg_num_neighbors)

            x_res = x
            x = layer(
                edge_vectors,
                x,
                node_inputs,
                edge_feats,
                edge_index,
            )
            x = norm(x + x_res)

            if self.pooling == "mean":
                mol_feat = global_mean_pool(x, batch)
            elif self.pooling == "sum":
                mol_feat = global_sum_pool(x, batch)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            atom_feat = x[absorbing_atom_index]
            readout_features.append(torch.cat([atom_feat, mol_feat], dim=-1))

        readout_feat = (
            torch.cat(readout_features, dim=-1)
            if self.multi_scale_readout
            else readout_features[-1]
        )
        return self.decoder(readout_feat)


def build_model_from_checkpoint(ckpt: Dict[str, Any], device: str) -> nn.Module:
    required = {
        "max_z",
        "n_energy",
        "cutoff",
        "max_neighbors",
        "n_rbf",
        "hidden_dim",
        "n_layers",
        "correlation",
        "max_ell",
        "pooling",
        "model_state_dict",
    }
    missing = required - set(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")

    model_cls = MACEGlobalSpectrumModel
    if ckpt.get("task_type") == "absorbing_atom":
        model_cls = MACEAbsorbingAtomSpectrumModel

    model = model_cls(
        max_z=int(ckpt["max_z"]),
        n_energy=int(ckpt["n_energy"]),
        cutoff=float(ckpt["cutoff"]),
        max_neighbors=None if ckpt["max_neighbors"] is None else int(ckpt["max_neighbors"]),
        n_rbf=int(ckpt["n_rbf"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        n_layers=int(ckpt["n_layers"]),
        correlation=int(ckpt["correlation"]),
        max_ell=int(ckpt["max_ell"]),
        pooling=str(ckpt["pooling"]),
        multi_scale_readout=bool(ckpt.get("multi_scale_readout", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
