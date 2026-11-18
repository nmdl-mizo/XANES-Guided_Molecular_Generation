from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .io_utils import load_torch_file


EDGE_TO_ATOMIC_NUMBER = {"CK": 6, "NK": 7, "OK": 8}
EDGE_TO_GLOBAL_FIELD = {edge: f"xanes_{edge}" for edge in EDGE_TO_ATOMIC_NUMBER}
EDGE_TO_SITE_MASK_FIELD = {edge: f"spectrum_{edge}" for edge in EDGE_TO_ATOMIC_NUMBER}


def _slice_field(data: Any, slices: Dict[str, torch.Tensor], field: str, index: int) -> torch.Tensor:
    values = getattr(data, field)
    offsets = slices[field]
    start = int(offsets[index].item())
    end = int(offsets[index + 1].item())
    return values[start:end]


def _molecule_arrays(
    data: Any, slices: Dict[str, torch.Tensor], index: int
) -> Tuple[np.ndarray, np.ndarray]:
    z = _slice_field(data, slices, "charge", index).detach().cpu().to(torch.long).numpy()
    pos = _slice_field(data, slices, "pos", index).detach().cpu().to(torch.float32).numpy()
    return z, pos


def _split_ranges(n_items: int, train_ratio: float, valid_ratio: float) -> Dict[str, range]:
    if n_items < 3:
        raise ValueError("At least three eligible molecules are required for train/valid/test splits.")
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Require 0 < train_ratio, valid_ratio and train_ratio + valid_ratio < 1.")
    train_end = int(n_items * train_ratio)
    valid_end = train_end + int(n_items * valid_ratio)
    if train_end == 0 or valid_end == train_end or valid_end == n_items:
        raise ValueError("Split ratios produced an empty train, valid, or test split.")
    return {
        "train": range(0, train_end),
        "valid": range(train_end, valid_end),
        "test": range(valid_end, n_items),
    }


def _limit(samples: List[Dict[str, Any]], max_samples: Optional[int]) -> List[Dict[str, Any]]:
    if max_samples is None:
        return samples
    if max_samples <= 0:
        raise ValueError("max_samples_per_split must be positive when specified.")
    return samples[:max_samples]


def _validate_raw_tuple(raw: Any) -> Tuple[Any, Dict[str, torch.Tensor]]:
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise ValueError("Raw QM9 XANES input must be the (data, slices) tuple used by PyG.")
    data, slices = raw
    if not isinstance(slices, dict):
        raise ValueError("The second raw-data tuple item must be a slices dictionary.")
    for field in ("charge", "pos"):
        if not hasattr(data, field) or field not in slices:
            raise ValueError(f"Raw QM9 XANES data is missing required field '{field}'.")
    return data, slices


def _global_samples(
    data: Any,
    slices: Dict[str, torch.Tensor],
    edge: str,
    train_ratio: float,
    valid_ratio: float,
    max_samples_per_split: Optional[int],
) -> Dict[str, List[Dict[str, Any]]]:
    target_z = EDGE_TO_ATOMIC_NUMBER[edge]
    spectrum_field = EDGE_TO_GLOBAL_FIELD[edge]
    if not hasattr(data, spectrum_field) or spectrum_field not in slices:
        raise ValueError(f"Input does not contain '{spectrum_field}'.")

    n_molecules = len(slices["charge"]) - 1
    eligible: List[int] = []
    for molecule_index in range(n_molecules):
        charges = _slice_field(data, slices, "charge", molecule_index)
        if bool((charges == target_z).any()):
            eligible.append(molecule_index)

    ranges = _split_ranges(len(eligible), train_ratio, valid_ratio)
    output: Dict[str, List[Dict[str, Any]]] = {}
    for split, split_range in ranges.items():
        samples: List[Dict[str, Any]] = []
        for selected_index in split_range:
            molecule_index = eligible[selected_index]
            z, pos = _molecule_arrays(data, slices, molecule_index)
            spectrum = _slice_field(data, slices, spectrum_field, molecule_index)
            spectrum = spectrum.reshape(-1).detach().cpu().to(torch.float32).numpy()
            samples.append({"z": z, "pos": pos, "spectrum": spectrum})
        output[split] = _limit(samples, max_samples_per_split)
    return output


def _site_entries(
    data: Any, slices: Dict[str, torch.Tensor], edge: str
) -> Tuple[List[List[int]], Optional[torch.Tensor]]:
    target_z = EDGE_TO_ATOMIC_NUMBER[edge]
    target_field = EDGE_TO_GLOBAL_FIELD[edge]
    mask_field = EDGE_TO_SITE_MASK_FIELD[edge]
    for field in (target_field, mask_field):
        if not hasattr(data, field) or field not in slices:
            raise ValueError(
                f"Site-conditioned input does not contain '{field}'. "
                "Use data_qm9_xanes_site.pt for --task absorbing_atom."
            )

    n_molecules = len(slices["charge"]) - 1
    entries: List[List[int]] = []
    for molecule_index in range(n_molecules):
        charges = _slice_field(data, slices, "charge", molecule_index).to(torch.long)
        mask_rows = _slice_field(data, slices, mask_field, molecule_index).to(torch.float32)
        usable = min(int(charges.numel()), int(mask_rows.shape[0]))
        active = (charges[:usable] == target_z) & (mask_rows[:usable].abs().sum(dim=1) > 0)
        entries.append(torch.nonzero(active, as_tuple=False).flatten().tolist())

    split_ids = getattr(data, "hole_split_id", None)
    if split_ids is not None:
        split_ids = split_ids.reshape(-1).detach().cpu().to(torch.long)
        if split_ids.numel() != n_molecules:
            split_ids = None
    return entries, split_ids


def _site_samples(
    data: Any,
    slices: Dict[str, torch.Tensor],
    edge: str,
    train_ratio: float,
    valid_ratio: float,
    max_samples_per_split: Optional[int],
) -> Dict[str, List[Dict[str, Any]]]:
    entries, split_ids = _site_entries(data, slices, edge)
    target_field = EDGE_TO_GLOBAL_FIELD[edge]
    molecule_indices = [index for index, atom_indices in enumerate(entries) if atom_indices]

    split_molecules: Dict[str, Sequence[int]]
    if split_ids is not None:
        candidate = {
            "train": [index for index in molecule_indices if int(split_ids[index]) == 0],
            "valid": [index for index in molecule_indices if int(split_ids[index]) == 1],
            "test": [index for index in molecule_indices if int(split_ids[index]) == 2],
        }
        split_molecules = candidate if all(candidate.values()) else {}
    else:
        split_molecules = {}

    if not split_molecules:
        ranges = _split_ranges(len(molecule_indices), train_ratio, valid_ratio)
        split_molecules = {
            split: [molecule_indices[i] for i in split_range]
            for split, split_range in ranges.items()
        }

    output: Dict[str, List[Dict[str, Any]]] = {}
    for split, indices in split_molecules.items():
        samples: List[Dict[str, Any]] = []
        for molecule_index in indices:
            z, pos = _molecule_arrays(data, slices, molecule_index)
            target_rows = _slice_field(data, slices, target_field, molecule_index)
            for atom_index in entries[molecule_index]:
                spectrum = target_rows[atom_index].detach().cpu().to(torch.float32).numpy()
                samples.append(
                    {
                        "z": z,
                        "pos": pos,
                        "spectrum": spectrum,
                        "absorbing_atom_index": atom_index,
                    }
                )
                if max_samples_per_split is not None and len(samples) >= max_samples_per_split:
                    break
            if max_samples_per_split is not None and len(samples) >= max_samples_per_split:
                break
        output[split] = samples
    return output


def load_raw_qm9_xanes_splits(
    path: str,
    edge: str,
    task_mode: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    max_samples_per_split: Optional[int] = None,
    raw: Any = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load the public DiffSpectra files directly, without derived .pt files."""
    edge = edge.upper()
    if edge not in EDGE_TO_ATOMIC_NUMBER:
        raise ValueError(f"edge must be one of {sorted(EDGE_TO_ATOMIC_NUMBER)}, got {edge!r}")
    if task_mode == "auto":
        task_mode = "absorbing_atom" if path.endswith("_site.pt") else "global"
    if task_mode not in {"global", "absorbing_atom"}:
        raise ValueError("Raw QM9 XANES loading requires task_mode global or absorbing_atom.")

    if raw is None:
        raw = load_torch_file(path, map_location="cpu")
    data, slices = _validate_raw_tuple(raw)
    if task_mode == "global":
        return _global_samples(
            data, slices, edge, train_ratio, valid_ratio, max_samples_per_split
        )
    return _site_samples(
        data, slices, edge, train_ratio, valid_ratio, max_samples_per_split
    )
