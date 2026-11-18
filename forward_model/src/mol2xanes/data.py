from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .io_utils import load_torch_file
from .utils import build_radius_graph


class MoleculeSpectrumDataset(Dataset):
    """
    Each item:
    sample = {
        "z": np.ndarray,        # (n_atoms,)
        "pos": np.ndarray,      # (n_atoms, 3)
        "spectrum": np.ndarray, # (n_E,)
        "absorbing_atom_index": int,  # optional
    }
    """

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        dtype: torch.dtype = torch.float32,
        require_spectrum: bool = True,
    ):
        self.samples = list(samples)
        self.dtype = dtype
        self.require_spectrum = require_spectrum

        if len(self.samples) == 0:
            raise ValueError("samples is empty")

        has_spectrum = "spectrum" in self.samples[0]
        if require_spectrum and not has_spectrum:
            raise ValueError("samples must include 'spectrum' when require_spectrum=True")
        spec_len = len(self.samples[0]["spectrum"]) if has_spectrum else None
        required_keys = {"z", "pos"} | ({"spectrum"} if require_spectrum else set())
        optional_keys = {"spectrum", "edge_index", "edge_vectors", "absorbing_atom_index"}
        has_absorbing_atom_index = "absorbing_atom_index" in self.samples[0]
        for i, s in enumerate(self.samples):
            keys = set(s.keys())
            if not required_keys.issubset(keys):
                raise ValueError(
                    f"sample[{i}] must include keys {required_keys}, got {keys}"
                )
            unknown = keys - required_keys - optional_keys
            if unknown:
                raise ValueError(f"sample[{i}] has unknown keys: {unknown}")
            if ("edge_index" in keys) != ("edge_vectors" in keys):
                raise ValueError(f"sample[{i}] must include both edge_index and edge_vectors")
            if ("absorbing_atom_index" in keys) != has_absorbing_atom_index:
                raise ValueError(
                    "Either all or none of samples must include 'absorbing_atom_index'."
                )
            if ("spectrum" in keys) != has_spectrum:
                raise ValueError("Either all or none of samples must include 'spectrum'.")
            z = np.asarray(s["z"])
            pos = np.asarray(s["pos"])

            if z.ndim != 1:
                raise ValueError(f"sample[{i}]['z'] must be 1D")
            if pos.ndim != 2 or pos.shape[1] != 3:
                raise ValueError(f"sample[{i}]['pos'] must be (n_atoms, 3)")
            if len(z) != len(pos):
                raise ValueError(f"sample[{i}] z and pos length mismatch")
            if has_spectrum:
                spec = np.asarray(s["spectrum"])
                if spec.ndim != 1:
                    raise ValueError(f"sample[{i}]['spectrum'] must be 1D")
                if len(spec) != spec_len:
                    raise ValueError(
                        f"sample[{i}] spectrum length mismatch: expected {spec_len}, got {len(spec)}"
                    )
            if has_absorbing_atom_index:
                absorbing_atom_index = int(s["absorbing_atom_index"])
                if absorbing_atom_index < 0 or absorbing_atom_index >= len(z):
                    raise ValueError(
                        f"sample[{i}]['absorbing_atom_index'] out of range: {absorbing_atom_index}"
                    )
            if "edge_index" in keys:
                edge_index = np.asarray(s["edge_index"])
                edge_vectors = np.asarray(s["edge_vectors"])
                if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                    raise ValueError(f"sample[{i}]['edge_index'] must be (2, E)")
                if edge_vectors.ndim != 2 or edge_vectors.shape[1] != 3:
                    raise ValueError(f"sample[{i}]['edge_vectors'] must be (E, 3)")
                if edge_index.shape[1] != edge_vectors.shape[0]:
                    raise ValueError(f"sample[{i}] edge_index/edge_vectors size mismatch")

        self.n_energy = spec_len
        self.max_z = int(max(np.max(np.asarray(s["z"])) for s in self.samples))
        self.has_absorbing_atom_index = has_absorbing_atom_index
        self.has_spectrum = has_spectrum

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        out: Dict[str, torch.Tensor] = {
            "z": torch.as_tensor(np.asarray(s["z"]), dtype=torch.long),
            "pos": torch.as_tensor(np.asarray(s["pos"]), dtype=self.dtype),
        }
        if "spectrum" in s:
            out["spectrum"] = torch.as_tensor(np.asarray(s["spectrum"]), dtype=self.dtype)
        if "absorbing_atom_index" in s:
            out["absorbing_atom_index"] = torch.as_tensor(
                int(s["absorbing_atom_index"]), dtype=torch.long
            )
        if "edge_index" in s:
            out["edge_index"] = torch.as_tensor(np.asarray(s["edge_index"]), dtype=torch.long)
            out["edge_vectors"] = torch.as_tensor(np.asarray(s["edge_vectors"]), dtype=self.dtype)
        return out


@dataclass
class BatchData:
    z: torch.Tensor
    pos: torch.Tensor
    batch: torch.Tensor
    spectrum: Optional[torch.Tensor] = None
    absorbing_atom_index: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    edge_vectors: Optional[torch.Tensor] = None


def collate_molecules(batch_list: Sequence[Dict[str, torch.Tensor]]) -> BatchData:
    z_all = []
    pos_all = []
    batch_idx = []
    spec_all = []
    edge_index_all = []
    edge_vectors_all = []
    has_spectrum = "spectrum" in batch_list[0]
    has_precomputed_graphs = "edge_index" in batch_list[0]
    has_absorbing_atom_index = "absorbing_atom_index" in batch_list[0]
    absorbing_atom_index_all = []
    atom_offset = 0

    for i, item in enumerate(batch_list):
        n = item["z"].shape[0]
        z_all.append(item["z"])
        pos_all.append(item["pos"])
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        if has_spectrum:
            if "spectrum" not in item:
                raise ValueError("Either all or none of batch samples must include spectrum.")
            spec_all.append(item["spectrum"])
        elif "spectrum" in item:
            raise ValueError("Either all or none of batch samples must include spectrum.")
        if has_absorbing_atom_index:
            if "absorbing_atom_index" not in item:
                raise ValueError(
                    "Either all or none of batch samples must include absorbing_atom_index."
                )
            absorbing_atom_index_all.append(item["absorbing_atom_index"] + atom_offset)
        if has_precomputed_graphs:
            if "edge_index" not in item or "edge_vectors" not in item:
                raise ValueError("Either all or none of batch samples must include precomputed edges.")
            edge_index_all.append(item["edge_index"] + atom_offset)
            edge_vectors_all.append(item["edge_vectors"])
        atom_offset += n

    edge_index = None
    edge_vectors = None
    absorbing_atom_index = None
    spectrum = None
    if has_precomputed_graphs:
        edge_index = torch.cat(edge_index_all, dim=1)
        edge_vectors = torch.cat(edge_vectors_all, dim=0)
    if has_absorbing_atom_index:
        absorbing_atom_index = torch.stack(absorbing_atom_index_all, dim=0)
    if has_spectrum:
        spectrum = torch.stack(spec_all, dim=0)

    return BatchData(
        z=torch.cat(z_all, dim=0),
        pos=torch.cat(pos_all, dim=0),
        batch=torch.cat(batch_idx, dim=0),
        spectrum=spectrum,
        absorbing_atom_index=absorbing_atom_index,
        edge_index=edge_index,
        edge_vectors=edge_vectors,
    )


def split_tensor_dict_to_samples(
    split_data: Dict[str, torch.Tensor],
    positions_key: str = "positions",
    charges_key: str = "charges",
    spectrum_key: str = "spectrum",
    exs_key: str = "exs",
    num_atoms_key: str = "num_atoms",
    absorbing_atom_index_key: str = "absorbing_atom_index",
    task_mode: str = "auto",
    allow_missing_spectrum: bool = False,
    default_n_energy: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert one split tensor-dict into list[dict] format expected by dataset:
      positions -> pos
      charges   -> z
      spectrum  -> spectrum

    Expected tensor shapes (typical):
      positions: (B, A, 3)
      charges:   (B, A)
      spectrum:  (B, E) for whole-molecule targets, or (B, A, E) for
                 absorbing-atom-conditioned targets
      exs:       (B, A) optional one-hot / multi-hot absorbing atom indicators
      num_atoms: (B,)  # optional but recommended when padded
      absorbing_atom_index: (B,) optional direct absorbing-atom indices
    """
    if task_mode not in {"auto", "global", "absorbing_atom"}:
        raise ValueError("task_mode must be one of: auto, global, absorbing_atom")

    required_keys = (positions_key, charges_key)
    if not allow_missing_spectrum:
        required_keys = required_keys + (spectrum_key,)
    for key in required_keys:
        if key not in split_data:
            raise ValueError(f"Missing key in split_data: '{key}'")

    positions = split_data[positions_key]
    charges = split_data[charges_key]
    spectra = split_data.get(spectrum_key, None)
    exs = split_data.get(exs_key, None)
    num_atoms = split_data.get(num_atoms_key, None)
    absorbing_atom_indices = split_data.get(absorbing_atom_index_key, None)

    if task_mode == "global":
        exs = None
        absorbing_atom_indices = None

    if (
        not isinstance(positions, torch.Tensor)
        or not isinstance(charges, torch.Tensor)
    ):
        raise ValueError("positions and charges must be torch.Tensor")
    if spectra is not None and not isinstance(spectra, torch.Tensor):
        raise ValueError("spectrum must be torch.Tensor when provided")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (B, A, 3)")
    if charges.ndim != 2:
        raise ValueError("charges must have shape (B, A)")
    if spectra is not None and spectra.ndim not in (2, 3):
        raise ValueError("spectrum must have shape (B, E) or (B, A, E)")

    n_samples = positions.shape[0]
    if charges.shape[0] != n_samples:
        raise ValueError("Batch dimension mismatch between positions and charges")
    if spectra is not None and spectra.shape[0] != n_samples:
        raise ValueError("Batch dimension mismatch among positions/charges/spectrum")
    if charges.shape[1] != positions.shape[1]:
        raise ValueError("Atom dimension mismatch between positions and charges")
    if spectra is not None and spectra.ndim == 3 and spectra.shape[1] != charges.shape[1]:
        raise ValueError("When spectrum is 3D, its atom dimension must match charges/positions")
    if exs is not None:
        if not isinstance(exs, torch.Tensor):
            raise ValueError("exs must be torch.Tensor when provided")
        if exs.ndim != 2 or exs.shape != charges.shape:
            raise ValueError("exs must have shape (B, A)")
        if spectra is None or spectra.ndim != 3:
            raise ValueError("exs is only supported when spectrum has shape (B, A, E)")
    if absorbing_atom_indices is not None:
        if not isinstance(absorbing_atom_indices, torch.Tensor):
            raise ValueError("absorbing_atom_index must be torch.Tensor when provided")
        if absorbing_atom_indices.ndim != 1 or absorbing_atom_indices.shape[0] != n_samples:
            raise ValueError("absorbing_atom_index must have shape (B,)")
    if task_mode == "global" and spectra is not None and spectra.ndim != 2:
        raise ValueError("task_mode='global' requires spectrum to have shape (B, E)")
    if task_mode == "absorbing_atom":
        if spectra is None:
            if not allow_missing_spectrum or absorbing_atom_indices is None:
                raise ValueError(
                    "task_mode='absorbing_atom' without spectrum requires "
                    "absorbing_atom_index with shape (B,)"
                )
        elif spectra.ndim == 2:
            if absorbing_atom_indices is None:
                raise ValueError(
                    "task_mode='absorbing_atom' with spectrum shape (B, E) requires "
                    "absorbing_atom_index with shape (B,)"
                )
        elif exs is None and absorbing_atom_indices is None:
            raise ValueError(
                "task_mode='absorbing_atom' with spectrum shape (B, A, E) requires "
                "exs or absorbing_atom_index"
            )
    if spectra is None and default_n_energy is None:
        raise ValueError("default_n_energy must be provided when spectrum is missing")

    if num_atoms is not None:
        if not isinstance(num_atoms, torch.Tensor):
            raise ValueError("num_atoms must be torch.Tensor when provided")
        if num_atoms.ndim != 1 or num_atoms.shape[0] != n_samples:
            raise ValueError("num_atoms must have shape (B,)")

    samples: List[Dict[str, Any]] = []
    for i in range(n_samples):
        if num_atoms is not None:
            n_atoms = int(num_atoms[i].item())
        else:
            n_atoms = int((charges[i] != 0).sum().item())
            if n_atoms <= 0:
                n_atoms = int(charges.shape[1])

        if n_atoms <= 0 or n_atoms > int(charges.shape[1]):
            raise ValueError(f"Invalid num_atoms at index {i}: {n_atoms}")

        z = charges[i, :n_atoms].detach().cpu().to(torch.long).numpy()
        pos = positions[i, :n_atoms, :].detach().cpu().to(torch.float32).numpy()
        absorbing_atom_index = None
        if absorbing_atom_indices is not None:
            absorbing_atom_index = int(absorbing_atom_indices[i].item())
            if absorbing_atom_index < 0 or absorbing_atom_index >= n_atoms:
                raise ValueError(
                    f"absorbing_atom_index out of range at sample {i}: "
                    f"{absorbing_atom_index} for {n_atoms} atoms"
                )

        if spectra is None:
            spectrum = np.zeros((int(default_n_energy),), dtype=np.float32)
            sample = {"z": z, "pos": pos, "spectrum": spectrum}
            if absorbing_atom_index is not None:
                sample["absorbing_atom_index"] = absorbing_atom_index
            samples.append(sample)
            continue

        if spectra.ndim == 2:
            spectrum = spectra[i].detach().cpu().to(torch.float32).numpy()
            sample = {"z": z, "pos": pos, "spectrum": spectrum}
            if absorbing_atom_index is not None:
                sample["absorbing_atom_index"] = absorbing_atom_index
            samples.append(sample)
            continue

        if exs is None:
            if absorbing_atom_index is None:
                raise ValueError(
                    "3D spectrum input requires exs or absorbing_atom_index to identify "
                    "which atom spectrum to supervise."
                )
            spectrum = spectra[i, absorbing_atom_index].detach().cpu().to(torch.float32).numpy()
            samples.append(
                {
                    "z": z,
                    "pos": pos,
                    "spectrum": spectrum,
                    "absorbing_atom_index": absorbing_atom_index,
                }
            )
            continue

        exs_i = exs[i, :n_atoms]
        active_indices = torch.nonzero(exs_i > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            raise ValueError(f"No absorbing atom indicated by exs at index {i}.")

        for atom_idx_t in active_indices:
            atom_idx = int(atom_idx_t.item())
            spectrum = spectra[i, atom_idx].detach().cpu().to(torch.float32).numpy()
            samples.append(
                {
                    "z": z,
                    "pos": pos,
                    "spectrum": spectrum,
                    "absorbing_atom_index": atom_idx,
                }
            )

    return samples


def precompute_graphs_for_samples(
    samples: Sequence[Dict[str, Any]],
    cutoff: float,
    max_neighbors: Optional[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sample in samples:
        s = dict(sample)
        pos_t = torch.as_tensor(np.asarray(s["pos"]), dtype=torch.float32)
        batch = torch.zeros(pos_t.shape[0], dtype=torch.long)
        edge_index, edge_vectors = build_radius_graph(
            pos=pos_t,
            batch=batch,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
        )
        s["edge_index"] = edge_index.cpu().numpy().astype(np.int64, copy=False)
        s["edge_vectors"] = edge_vectors.cpu().numpy().astype(np.float32, copy=False)
        out.append(s)
    return out


def load_torch_splits_as_samples(
    path: str,
    split_keys: Tuple[str, str, str] = ("train", "valid", "test"),
    task_mode: str = "auto",
    edge: Optional[str] = None,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    max_samples_per_split: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load a .pt/.pth file and convert split-wise tensor dicts into list[dict] samples.
    """
    raw = load_torch_file(path, map_location="cpu")
    if isinstance(raw, tuple):
        if edge is None:
            raise ValueError("--edge is required when loading a raw QM9 XANES data file.")
        from .raw_qm9_xanes import load_raw_qm9_xanes_splits

        return load_raw_qm9_xanes_splits(
            path,
            edge=edge,
            task_mode=task_mode,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            max_samples_per_split=max_samples_per_split,
            raw=raw,
        )
    if not isinstance(raw, dict):
        raise ValueError("Loaded object must be a dict with train/val/test splits")

    out: Dict[str, List[Dict[str, Any]]] = {}
    for split in split_keys:
        if split not in raw:
            raise ValueError(f"Missing split '{split}' in loaded file")
        if not isinstance(raw[split], dict):
            raise ValueError(f"Split '{split}' must be a dict")
        out[split] = split_tensor_dict_to_samples(raw[split], task_mode=task_mode)
    return out
