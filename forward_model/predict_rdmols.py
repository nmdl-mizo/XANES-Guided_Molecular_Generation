#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*TorchScript type system doesn't support instance-level annotations.*",
)

from mol2xanes import build_model_from_checkpoint, load_torch_file, predict_spectra


EDGE_TO_ATOMIC_NUMBER = {"CK": 6, "NK": 7, "OK": 8}


def parse_args() -> argparse.Namespace:
    release_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Predict molecule-level and absorbing-site XANES spectra directly "
            "from DiffSpectra sample_rdmols_3d.pkl output."
        )
    )
    parser.add_argument("input_pkl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=release_root / "pretrained" / "forward",
    )
    parser.add_argument(
        "--edges",
        nargs="+",
        choices=sorted(EDGE_TO_ATOMIC_NUMBER),
        default=sorted(EDGE_TO_ATOMIC_NUMBER),
    )
    parser.add_argument("--mode", choices=["global", "site", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rdmols(path: Path) -> List[Chem.Mol]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"{path} must contain a list or tuple of RDKit molecules.")

    mols: List[Chem.Mol] = []
    for index, mol in enumerate(payload):
        if not isinstance(mol, Chem.Mol):
            raise ValueError(f"Entry {index} is not an RDKit Mol: {type(mol)}")
        if mol.GetNumConformers() == 0:
            raise ValueError(f"Entry {index} has no 3D conformer.")
        mols.append(mol)
    if not mols:
        raise ValueError(f"{path} contains no molecules.")
    return mols


def molecule_arrays(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    conformer = mol.GetConformer()
    pos = np.asarray(
        [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
    return z, pos


def load_model(path: Path, device: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = load_torch_file(str(path), map_location="cpu")
    return build_model_from_checkpoint(checkpoint, device)


def save_predictions(
    path: Path,
    spectra: np.ndarray,
    molecule_indices: Sequence[int],
    atom_indices: Sequence[int] | None = None,
) -> None:
    payload = {
        "spectra": spectra,
        "molecule_indices": np.asarray(molecule_indices, dtype=np.int64),
    }
    if atom_indices is not None:
        payload["absorbing_atom_indices"] = np.asarray(atom_indices, dtype=np.int64)
    np.savez_compressed(path, **payload)


def predict_global(
    mol_arrays: Sequence[Tuple[np.ndarray, np.ndarray]],
    edge: str,
    model_path: Path,
    output_path: Path,
    batch_size: int,
    device: str,
) -> int:
    target_z = EDGE_TO_ATOMIC_NUMBER[edge]
    model = load_model(model_path, device)
    samples = []
    molecule_indices = []
    for molecule_index, (z, pos) in enumerate(mol_arrays):
        if target_z not in z or z.max(initial=0) > int(model.max_z):
            continue
        samples.append({"z": z, "pos": pos})
        molecule_indices.append(molecule_index)

    if samples:
        spectra = predict_spectra(model, samples, batch_size=batch_size, device=device)
    else:
        spectra = np.empty((0, int(model.n_energy)), dtype=np.float32)
    save_predictions(output_path, spectra, molecule_indices)
    return len(samples)


def predict_sites(
    mol_arrays: Sequence[Tuple[np.ndarray, np.ndarray]],
    edge: str,
    model_path: Path,
    output_path: Path,
    batch_size: int,
    device: str,
) -> int:
    target_z = EDGE_TO_ATOMIC_NUMBER[edge]
    model = load_model(model_path, device)
    samples = []
    molecule_indices = []
    atom_indices = []
    for molecule_index, (z, pos) in enumerate(mol_arrays):
        if z.max(initial=0) > int(model.max_z):
            continue
        for atom_index in np.flatnonzero(z == target_z).tolist():
            samples.append(
                {
                    "z": z,
                    "pos": pos,
                    "absorbing_atom_index": int(atom_index),
                }
            )
            molecule_indices.append(molecule_index)
            atom_indices.append(atom_index)

    if samples:
        spectra = predict_spectra(model, samples, batch_size=batch_size, device=device)
    else:
        spectra = np.empty((0, int(model.n_energy)), dtype=np.float32)
    save_predictions(output_path, spectra, molecule_indices, atom_indices)
    return len(samples)


def main() -> None:
    args = parse_args()
    mols = load_rdmols(args.input_pkl)
    mol_arrays = [molecule_arrays(mol) for mol in mols]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "input_pkl": str(args.input_pkl),
        "number_of_molecules": len(mols),
        "smiles": [Chem.MolToSmiles(mol) for mol in mols],
        "device": args.device,
        "predictions": {},
    }

    for edge in args.edges:
        edge_summary: Dict[str, object] = {}
        if args.mode in {"global", "both"}:
            output_name = f"global_{edge}.npz"
            count = predict_global(
                mol_arrays,
                edge,
                args.models_dir / "global" / edge / "best_model.pt",
                args.output_dir / output_name,
                args.batch_size,
                args.device,
            )
            edge_summary["global"] = {"file": output_name, "number_of_predictions": count}
        if args.mode in {"site", "both"}:
            output_name = f"site_{edge}.npz"
            count = predict_sites(
                mol_arrays,
                edge,
                args.models_dir / "site" / edge / "best_model.pt",
                args.output_dir / output_name,
                args.batch_size,
                args.device,
            )
            edge_summary["site"] = {"file": output_name, "number_of_predictions": count}
        summary["predictions"][edge] = edge_summary

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved predictions to: {args.output_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
