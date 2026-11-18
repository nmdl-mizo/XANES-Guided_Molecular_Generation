import math
from collections import deque

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


BOND_TYPE_TO_INT = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}


def _to_numpy_positions(pos):
    if isinstance(pos, torch.Tensor):
        return pos.detach().cpu().numpy()
    return np.asarray(pos)


def _to_numpy_atom_types(atom_types):
    if isinstance(atom_types, torch.Tensor):
        return atom_types.detach().cpu().numpy()
    return np.asarray(atom_types)


def _gt_edge_matrix(rdmol):
    num_atoms = rdmol.GetNumAtoms()
    edge_matrix = np.zeros((num_atoms, num_atoms), dtype=np.int64)
    for bond in rdmol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = BOND_TYPE_TO_INT[bond.GetBondType()]
        edge_matrix[begin, end] = bond_type
        edge_matrix[end, begin] = bond_type
    return edge_matrix


def _gt_atom_types(rdmol, dataset_info):
    atom_decoder = dataset_info['atom_decoder']
    if isinstance(atom_decoder, dict):
        decoder_items = atom_decoder.items()
    else:
        decoder_items = enumerate(atom_decoder)
    atom_encoder = {symbol: idx for idx, symbol in decoder_items}
    return np.array([atom_encoder[atom.GetSymbol()] for atom in rdmol.GetAtoms()], dtype=np.int64)


def _neighbors_from_edge_matrix(edge_matrix):
    neighbors = {}
    for idx in range(edge_matrix.shape[0]):
        neighbors[idx] = np.nonzero(edge_matrix[idx] > 0)[0].tolist()
    return neighbors


def _two_hop_nodes(edge_matrix, center_idx=0):
    neighbors = _neighbors_from_edge_matrix(edge_matrix)
    visited = {center_idx}
    queue = deque([(center_idx, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth == 2:
            continue
        for nxt in neighbors[node]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, depth + 1))
    return sorted(visited)


def _kabsch_rmsd(pred_pos, true_pos):
    pred_centered = pred_pos - pred_pos.mean(axis=0, keepdims=True)
    true_centered = true_pos - true_pos.mean(axis=0, keepdims=True)
    cov = pred_centered.T @ true_centered
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    aligned = pred_centered @ rot
    return float(np.sqrt(np.mean(np.sum((aligned - true_centered) ** 2, axis=1))))


def _bond_length(pos, i, j):
    return float(np.linalg.norm(pos[i] - pos[j]))


def _bond_angle(pos, center, i, j):
    v1 = pos[i] - pos[center]
    v2 = pos[j] - pos[center]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_theta)))


def compute_local_structure_metrics(processed_mols, groundtruth_pos, groundtruth_rdmols, dataset_info):
    rmsd_values = []
    bond_mae_1hop_values = []
    bond_mae_2hop_values = []
    angle_mae_values = []
    local_graph_matches = []

    n_pairs = min(len(processed_mols), len(groundtruth_pos), len(groundtruth_rdmols))
    for idx in range(n_pairs):
        pred = processed_mols[idx]
        gt_pos = _to_numpy_positions(groundtruth_pos[idx])
        gt_rdmol = groundtruth_rdmols[idx]
        if pred is None or gt_rdmol is None:
            continue

        pred_pos, pred_atom_types, pred_edge_types, _ = pred
        pred_pos = _to_numpy_positions(pred_pos)
        pred_atom_types = _to_numpy_atom_types(pred_atom_types)
        pred_edge_types = _to_numpy_atom_types(pred_edge_types)

        gt_edge_types = _gt_edge_matrix(gt_rdmol)
        gt_atom_types = _gt_atom_types(gt_rdmol, dataset_info)
        two_hop_nodes = _two_hop_nodes(gt_edge_types, center_idx=0)
        one_hop_nodes = [node for node in np.nonzero(gt_edge_types[0] > 0)[0].tolist()]

        if len(two_hop_nodes) > 0 and max(two_hop_nodes) < pred_pos.shape[0]:
            rmsd_values.append(_kabsch_rmsd(pred_pos[two_hop_nodes], gt_pos[two_hop_nodes]))

        one_hop_bond_errors = []
        for node in one_hop_nodes:
            if node < pred_pos.shape[0]:
                one_hop_bond_errors.append(abs(_bond_length(pred_pos, 0, node) - _bond_length(gt_pos, 0, node)))
        if one_hop_bond_errors:
            bond_mae_1hop_values.append(float(np.mean(one_hop_bond_errors)))

        local_edges = []
        for i in two_hop_nodes:
            for j in two_hop_nodes:
                if i < j and gt_edge_types[i, j] > 0:
                    local_edges.append((i, j))
        two_hop_bond_errors = []
        for i, j in local_edges:
            if max(i, j) < pred_pos.shape[0]:
                two_hop_bond_errors.append(abs(_bond_length(pred_pos, i, j) - _bond_length(gt_pos, i, j)))
        if two_hop_bond_errors:
            bond_mae_2hop_values.append(float(np.mean(two_hop_bond_errors)))

        angle_errors = []
        for i_idx in range(len(one_hop_nodes)):
            for j_idx in range(i_idx + 1, len(one_hop_nodes)):
                i = one_hop_nodes[i_idx]
                j = one_hop_nodes[j_idx]
                if max(i, j) >= pred_pos.shape[0]:
                    continue
                pred_angle = _bond_angle(pred_pos, 0, i, j)
                gt_angle = _bond_angle(gt_pos, 0, i, j)
                if pred_angle is None or gt_angle is None:
                    continue
                angle_errors.append(abs(pred_angle - gt_angle))
        if angle_errors:
            angle_mae_values.append(float(np.mean(angle_errors)))

        if len(two_hop_nodes) > 0 and max(two_hop_nodes) < pred_atom_types.shape[0]:
            gt_local_atoms = gt_atom_types[two_hop_nodes]
            pred_local_atoms = pred_atom_types[two_hop_nodes]
            gt_local_edges = gt_edge_types[np.ix_(two_hop_nodes, two_hop_nodes)]
            pred_local_edges = pred_edge_types[np.ix_(two_hop_nodes, two_hop_nodes)]
            local_graph_matches.append(float(
                np.array_equal(gt_local_atoms, pred_local_atoms)
                and np.array_equal(gt_local_edges, pred_local_edges)
            ))

    def _safe_mean(values):
        return float(np.mean(values)) if values else float('nan')

    return {
        'Local RMSD@2hop': _safe_mean(rmsd_values),
        '1-hop Bond Length MAE': _safe_mean(bond_mae_1hop_values),
        '2-hop Bond Length MAE': _safe_mean(bond_mae_2hop_values),
        'Absorber-centered Bond Angle MAE': _safe_mean(angle_mae_values),
        'Local Graph Match Rate': _safe_mean(local_graph_matches),
        'Num Pairs Evaluated': n_pairs,
        'Num RMSD Pairs': len(rmsd_values),
        'Num 1-hop Bond Pairs': len(bond_mae_1hop_values),
        'Num 2-hop Bond Pairs': len(bond_mae_2hop_values),
        'Num Angle Pairs': len(angle_mae_values),
        'Num Local Graph Pairs': len(local_graph_matches),
    }


def save_local_structure_metrics(processed_mols, groundtruth_pos, groundtruth_rdmols,
                                 dataset_info, eval_dir, ckpt, ns=None):
    metrics = compute_local_structure_metrics(processed_mols, groundtruth_pos, groundtruth_rdmols, dataset_info)
    suffix = f"_ckpt_{ckpt}" if ns is None else f"_ckpt_{ckpt}_{ns}"
    csv_path = f"{eval_dir}/local_structure_metrics{suffix}.csv"
    df = pd.DataFrame({
        'Evaluation Metric': list(metrics.keys()),
        'Value': list(metrics.values()),
    })
    df.to_csv(csv_path, index=False)
    return metrics, csv_path
