from __future__ import annotations

import copy
import gc
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import MoleculeSpectrumDataset, collate_molecules, precompute_graphs_for_samples
from .model import MACEAbsorbingAtomSpectrumModel, MACEGlobalSpectrumModel
from .utils import set_seed


def spectrum_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    use_derivative_loss: bool = False,
    derivative_loss_weight: float = 1.0,
) -> torch.Tensor:
    loss = F.mse_loss(pred, target)
    if not use_derivative_loss:
        return loss

    if pred.shape[-1] < 2:
        return loss

    pred_diff = pred[..., 1:] - pred[..., :-1]
    target_diff = target[..., 1:] - target[..., :-1]
    derivative_loss = F.mse_loss(pred_diff, target_diff)
    return loss + derivative_loss_weight * derivative_loss


def _samples_use_absorbing_atom(
    samples: Sequence[Dict[str, Any]],
) -> bool:
    return len(samples) > 0 and "absorbing_atom_index" in samples[0]


def _build_model(
    samples: Sequence[Dict[str, Any]],
    *,
    max_z: int,
    n_energy: int,
    cutoff: float,
    max_neighbors: Optional[int],
    n_rbf: int,
    hidden_dim: int,
    n_layers: int,
    correlation: int,
    max_ell: int,
    pooling: str,
    multi_scale_readout: bool = True,
) -> nn.Module:
    model_cls = (
        MACEAbsorbingAtomSpectrumModel
        if _samples_use_absorbing_atom(samples)
        else MACEGlobalSpectrumModel
    )
    return model_cls(
        max_z=max_z,
        n_energy=n_energy,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        n_rbf=n_rbf,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        correlation=correlation,
        max_ell=max_ell,
        pooling=pooling,
        multi_scale_readout=multi_scale_readout,
    )


@torch.no_grad()
def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(F.mse_loss(pred, target)).item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    scaler: Optional[Any] = None,
    use_amp: bool = False,
    use_derivative_loss: bool = False,
    derivative_loss_weight: float = 1.0,
    timing: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    amp_enabled = use_amp and device.type == "cuda"
    autocast_dtype = torch.float16

    total_loss = 0.0
    total_rmse = 0.0
    total_samples = 0
    total_batches = 0
    t_transfer = 0.0
    t_forward = 0.0
    t_backward_step = 0.0
    t_metrics = 0.0

    for batch in loader:
        t0 = time.perf_counter()
        z = batch.z.to(device)
        pos = batch.pos.to(device)
        b = batch.batch.to(device)
        y = batch.spectrum.to(device)
        absorbing_atom_index = (
            batch.absorbing_atom_index.to(device)
            if batch.absorbing_atom_index is not None
            else None
        )
        edge_index = batch.edge_index.to(device) if batch.edge_index is not None else None
        edge_vectors = batch.edge_vectors.to(device) if batch.edge_vectors is not None else None
        t_transfer += time.perf_counter() - t0

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        t1 = time.perf_counter()
        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=autocast_dtype):
                model_kwargs = {
                    "z": z,
                    "pos": pos,
                    "batch": b,
                    "edge_index": edge_index,
                    "edge_vectors": edge_vectors,
                }
                if absorbing_atom_index is not None:
                    model_kwargs["absorbing_atom_index"] = absorbing_atom_index
                pred = model(**model_kwargs)
                loss = spectrum_loss(
                    pred,
                    y,
                    use_derivative_loss=use_derivative_loss,
                    derivative_loss_weight=derivative_loss_weight,
                )
        t_forward += time.perf_counter() - t1

        if train_mode:
            t2 = time.perf_counter()
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            t_backward_step += time.perf_counter() - t2

        bs = y.shape[0]
        t3 = time.perf_counter()
        total_loss += loss.item() * bs
        total_rmse += compute_rmse(pred, y) * bs
        t_metrics += time.perf_counter() - t3
        total_samples += bs
        total_batches += 1

    if timing is not None:
        timing["batches"] = float(total_batches)
        timing["samples"] = float(total_samples)
        timing["transfer_s"] = t_transfer
        timing["forward_s"] = t_forward
        timing["backward_step_s"] = t_backward_step
        timing["metrics_s"] = t_metrics
        timing["epoch_total_s"] = t_transfer + t_forward + t_backward_step + t_metrics

    return total_loss / total_samples, total_rmse / total_samples


def _create_grad_scaler(enabled: bool) -> Any:
    # torch.cuda.amp.GradScaler is deprecated in newer PyTorch releases.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _emit_timing(line: str, timing_log_path: Optional[str]) -> None:
    print(line, flush=True)
    if timing_log_path is None:
        return
    os.makedirs(os.path.dirname(timing_log_path) or ".", exist_ok=True)
    with open(timing_log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_log_line(line: str, log_path: Optional[str]) -> None:
    if log_path is None:
        return
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def split_dataset(
    samples: Sequence[Dict[str, Any]],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    idx = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_idx = idx[:n_train]
    valid_idx = idx[n_train : n_train + n_valid]
    test_idx = idx[n_train + n_valid :]

    train_samples = [samples[i] for i in train_idx]
    valid_samples = [samples[i] for i in valid_idx]
    test_samples = [samples[i] for i in test_idx]
    return train_samples, valid_samples, test_samples


def _fit_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device_t: torch.device,
    epochs: int,
    early_stopping_patience: Optional[int],
    early_stopping_min_delta: float,
    use_amp: bool = True,
    use_derivative_loss: bool = False,
    derivative_loss_weight: float = 1.0,
    log_timing: bool = False,
    timing_log_path: Optional[str] = None,
    metrics_log_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    amp_enabled = use_amp and device_t.type == "cuda"
    scaler = _create_grad_scaler(enabled=amp_enabled)
    best_valid_loss = float("inf")
    best_state = None
    no_improve_count = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        train_timing: Dict[str, float] = {}
        valid_timing: Dict[str, float] = {}
        train_loss, train_rmse = run_epoch(
            model,
            train_loader,
            optimizer,
            device_t,
            scaler=scaler,
            use_amp=amp_enabled,
            use_derivative_loss=use_derivative_loss,
            derivative_loss_weight=derivative_loss_weight,
            timing=train_timing if log_timing else None,
        )
        valid_loss, valid_rmse = run_epoch(
            model,
            valid_loader,
            None,
            device_t,
            scaler=None,
            use_amp=amp_enabled,
            use_derivative_loss=use_derivative_loss,
            derivative_loss_weight=derivative_loss_weight,
            timing=valid_timing if log_timing else None,
        )

        improved = valid_loss < (best_valid_loss - early_stopping_min_delta)
        if improved:
            best_valid_loss = valid_loss
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "valid_loss": valid_loss,
                "valid_rmse": valid_rmse,
            }
            no_improve_count = 0
        else:
            no_improve_count += 1

        if verbose:
            metrics_line = (
                f"[{epoch:03d}/{epochs:03d}] "
                f"train_loss={train_loss:.6f} train_rmse={train_rmse:.6f} "
                f"valid_loss={valid_loss:.6f} valid_rmse={valid_rmse:.6f}"
            )
            _write_log_line(metrics_line, metrics_log_path)
            if log_timing:
                _emit_timing(
                    "  timing "
                    f"train(total={train_timing['epoch_total_s']:.2f}s, "
                    f"transfer={train_timing['transfer_s']:.2f}s, "
                    f"forward={train_timing['forward_s']:.2f}s, "
                    f"backward_step={train_timing['backward_step_s']:.2f}s, "
                    f"metrics={train_timing['metrics_s']:.2f}s) "
                    f"valid(total={valid_timing['epoch_total_s']:.2f}s, "
                    f"transfer={valid_timing['transfer_s']:.2f}s, "
                    f"forward={valid_timing['forward_s']:.2f}s, "
                    f"metrics={valid_timing['metrics_s']:.2f}s)",
                    timing_log_path,
                )

        if early_stopping_patience is not None and no_improve_count >= early_stopping_patience:
            stopped_early = True
            if verbose:
                _write_log_line(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta}).",
                    metrics_log_path,
                )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state["model"])
    return {
        "model": model,
        "best_epoch": best_state["epoch"],
        "best_valid_loss": best_state["valid_loss"],
        "best_valid_rmse": best_state["valid_rmse"],
        "epochs_ran": epoch,
        "stopped_early": stopped_early,
    }


def train_model(
    samples: Sequence[Dict[str, Any]],
    batch_size: int = 16,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    cutoff: float = 5.0,
    max_neighbors: Optional[int] = None,
    hidden_dim: int = 64,
    n_layers: int = 4,
    n_rbf: int = 16,
    correlation: int = 3,
    max_ell: int = 2,
    pooling: str = "mean",
    multi_scale_readout: bool = True,
    use_derivative_loss: bool = False,
    derivative_loss_weight: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    precompute_graphs: bool = True,
    use_amp: bool = True,
    log_timing: bool = False,
    timing_log_path: Optional[str] = None,
    metrics_log_path: Optional[str] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    device_t = torch.device(device)

    train_samples, valid_samples, test_samples = split_dataset(samples, seed=seed)
    if precompute_graphs:
        train_samples = precompute_graphs_for_samples(train_samples, cutoff=cutoff, max_neighbors=max_neighbors)
        valid_samples = precompute_graphs_for_samples(valid_samples, cutoff=cutoff, max_neighbors=max_neighbors)
        test_samples = precompute_graphs_for_samples(test_samples, cutoff=cutoff, max_neighbors=max_neighbors)

    train_ds = MoleculeSpectrumDataset(train_samples)
    valid_ds = MoleculeSpectrumDataset(valid_samples)
    test_ds = MoleculeSpectrumDataset(test_samples)

    max_z = max(train_ds.max_z, valid_ds.max_z, test_ds.max_z)
    n_energy = train_ds.n_energy

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_molecules)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules)

    model = _build_model(
        train_samples,
        max_z=max_z,
        n_energy=n_energy,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        n_rbf=n_rbf,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        correlation=correlation,
        max_ell=max_ell,
        pooling=pooling,
        multi_scale_readout=multi_scale_readout,
    ).to(device_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    fit_result = _fit_with_early_stopping(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device_t=device_t,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        use_amp=use_amp,
        use_derivative_loss=use_derivative_loss,
        derivative_loss_weight=derivative_loss_weight,
        log_timing=log_timing,
        timing_log_path=timing_log_path,
        metrics_log_path=metrics_log_path,
        verbose=True,
    )
    test_timing: Dict[str, float] = {}
    test_loss, test_rmse = run_epoch(
        model,
        test_loader,
        None,
        device_t,
        use_amp=use_amp,
        use_derivative_loss=use_derivative_loss,
        derivative_loss_weight=derivative_loss_weight,
        timing=test_timing if log_timing else None,
    )
    if log_timing:
        _emit_timing(
            "Test timing "
            f"total={test_timing['epoch_total_s']:.2f}s "
            f"transfer={test_timing['transfer_s']:.2f}s "
            f"forward={test_timing['forward_s']:.2f}s "
            f"metrics={test_timing['metrics_s']:.2f}s",
            timing_log_path,
        )

    return {
        "model": fit_result["model"],
        "best_epoch": fit_result["best_epoch"],
        "best_valid_loss": fit_result["best_valid_loss"],
        "best_valid_rmse": fit_result["best_valid_rmse"],
        "epochs_ran": fit_result["epochs_ran"],
        "stopped_early": fit_result["stopped_early"],
        "test_loss": test_loss,
        "test_rmse": test_rmse,
        "max_z": max_z,
        "n_energy": n_energy,
    }


def train_model_from_splits(
    split_samples: Dict[str, Sequence[Dict[str, Any]]],
    batch_size: int = 16,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    cutoff: float = 5.0,
    max_neighbors: Optional[int] = None,
    hidden_dim: int = 64,
    n_layers: int = 4,
    n_rbf: int = 16,
    correlation: int = 3,
    max_ell: int = 2,
    pooling: str = "mean",
    multi_scale_readout: bool = True,
    use_derivative_loss: bool = False,
    derivative_loss_weight: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    precompute_graphs: bool = True,
    use_amp: bool = True,
    log_timing: bool = False,
    timing_log_path: Optional[str] = None,
    metrics_log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train with explicit data splits:
      split_samples = {"train": [...], "valid": [...], "test": [...]}.
    """
    for key in ("train", "valid", "test"):
        if key not in split_samples:
            raise ValueError(f"split_samples must include '{key}'")
        if len(split_samples[key]) == 0:
            raise ValueError(f"split_samples['{key}'] is empty")

    set_seed(seed)
    device_t = torch.device(device)

    train_samples = list(split_samples["train"])
    valid_samples = list(split_samples["valid"])
    test_samples = list(split_samples["test"])
    if precompute_graphs:
        train_samples = precompute_graphs_for_samples(train_samples, cutoff=cutoff, max_neighbors=max_neighbors)
        valid_samples = precompute_graphs_for_samples(valid_samples, cutoff=cutoff, max_neighbors=max_neighbors)
        test_samples = precompute_graphs_for_samples(test_samples, cutoff=cutoff, max_neighbors=max_neighbors)

    train_ds = MoleculeSpectrumDataset(train_samples)
    valid_ds = MoleculeSpectrumDataset(valid_samples)
    test_ds = MoleculeSpectrumDataset(test_samples)

    max_z = max(train_ds.max_z, valid_ds.max_z, test_ds.max_z)
    n_energy = train_ds.n_energy

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_molecules)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules)

    model = _build_model(
        train_samples,
        max_z=max_z,
        n_energy=n_energy,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        n_rbf=n_rbf,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        correlation=correlation,
        max_ell=max_ell,
        pooling=pooling,
        multi_scale_readout=multi_scale_readout,
    ).to(device_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    fit_result = _fit_with_early_stopping(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device_t=device_t,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        use_amp=use_amp,
        use_derivative_loss=use_derivative_loss,
        derivative_loss_weight=derivative_loss_weight,
        log_timing=log_timing,
        timing_log_path=timing_log_path,
        metrics_log_path=metrics_log_path,
        verbose=True,
    )
    test_timing: Dict[str, float] = {}
    test_loss, test_rmse = run_epoch(
        model,
        test_loader,
        None,
        device_t,
        use_amp=use_amp,
        use_derivative_loss=use_derivative_loss,
        derivative_loss_weight=derivative_loss_weight,
        timing=test_timing if log_timing else None,
    )
    if log_timing:
        _emit_timing(
            "Test timing "
            f"total={test_timing['epoch_total_s']:.2f}s "
            f"transfer={test_timing['transfer_s']:.2f}s "
            f"forward={test_timing['forward_s']:.2f}s "
            f"metrics={test_timing['metrics_s']:.2f}s",
            timing_log_path,
        )

    return {
        "model": fit_result["model"],
        "best_epoch": fit_result["best_epoch"],
        "best_valid_loss": fit_result["best_valid_loss"],
        "best_valid_rmse": fit_result["best_valid_rmse"],
        "epochs_ran": fit_result["epochs_ran"],
        "stopped_early": fit_result["stopped_early"],
        "test_loss": test_loss,
        "test_rmse": test_rmse,
        "max_z": max_z,
        "n_energy": n_energy,
    }


def tune_hyperparameters_with_optuna(
    split_samples: Dict[str, Sequence[Dict[str, Any]]],
    n_trials: int = 20,
    epochs: int = 100,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    early_stopping_patience: Optional[int] = 20,
    early_stopping_min_delta: float = 0.0,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    timeout: Optional[int] = None,
    precompute_graphs: bool = True,
    use_amp: bool = True,
    use_derivative_loss: bool = False,
    derivative_loss_weight: float = 1.0,
    tune_derivative_loss_weight: bool = False,
    derivative_loss_weight_min: float = 0.1,
    derivative_loss_weight_max: float = 10.0,
    log_timing: bool = False,
    timing_log_path: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for tuning. Install with: pip install optuna"
        ) from exc

    fixed_cutoff = 6.0
    fixed_max_neighbors = 29

    for key in ("train", "valid", "test"):
        if key not in split_samples:
            raise ValueError(f"split_samples must include '{key}'")
    base_train_samples = list(split_samples["train"])
    base_valid_samples = list(split_samples["valid"])
    base_test_samples = list(split_samples["test"])
    if precompute_graphs:
        # cutoff/max_neighbors are fixed during tuning, so build graphs once.
        base_train_samples = precompute_graphs_for_samples(
            base_train_samples,
            cutoff=fixed_cutoff,
            max_neighbors=fixed_max_neighbors,
        )
        base_valid_samples = precompute_graphs_for_samples(
            base_valid_samples,
            cutoff=fixed_cutoff,
            max_neighbors=fixed_max_neighbors,
        )
        base_test_samples = precompute_graphs_for_samples(
            base_test_samples,
            cutoff=fixed_cutoff,
            max_neighbors=fixed_max_neighbors,
        )
    base_train_ds = MoleculeSpectrumDataset(base_train_samples)
    base_valid_ds = MoleculeSpectrumDataset(base_valid_samples)
    base_test_ds = MoleculeSpectrumDataset(base_test_samples)
    max_z = max(base_train_ds.max_z, base_valid_ds.max_z, base_test_ds.max_z)
    n_energy = base_train_ds.n_energy

    def _cleanup_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def objective(trial: Any) -> float:
        t_trial_start = time.perf_counter()
        t_precompute = 0.0
        t_data = 0.0
        t_model_opt = 0.0
        set_seed(seed + int(trial.number))
        device_t = torch.device(device)
        amp_enabled = use_amp and device_t.type == "cuda"
        scaler: Optional[Any] = _create_grad_scaler(enabled=amp_enabled)
        model: Optional[nn.Module] = None
        optimizer: Optional[torch.optim.Optimizer] = None
        train_loader: Optional[DataLoader] = None
        valid_loader: Optional[DataLoader] = None

        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
        cutoff = fixed_cutoff
        max_neighbors = fixed_max_neighbors
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
        n_layers = trial.suggest_int("n_layers", 2, 6)
        n_rbf = trial.suggest_categorical("n_rbf", [8, 16, 24, 32])
        correlation = trial.suggest_categorical("correlation", [2, 3])
        max_ell = trial.suggest_int("max_ell", 1, 3)
        pooling = trial.suggest_categorical("pooling", ["mean", "sum"])
        trial_derivative_loss_weight = derivative_loss_weight
        if use_derivative_loss and tune_derivative_loss_weight:
            trial_derivative_loss_weight = trial.suggest_float(
                "derivative_loss_weight",
                derivative_loss_weight_min,
                derivative_loss_weight_max,
                log=True,
            )

        train_samples = base_train_samples
        valid_samples = base_valid_samples

        t1 = time.perf_counter()
        train_ds = MoleculeSpectrumDataset(train_samples)
        valid_ds = MoleculeSpectrumDataset(valid_samples)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_molecules)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules)
        t_data = time.perf_counter() - t1

        t2 = time.perf_counter()
        model = _build_model(
            train_samples,
            max_z=max_z,
            n_energy=n_energy,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            n_rbf=n_rbf,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            correlation=correlation,
            max_ell=max_ell,
            pooling=pooling,
            multi_scale_readout=True,
        ).to(device_t)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        t_model_opt = time.perf_counter() - t2

        best_valid_loss = float("inf")
        no_improve_count = 0

        try:
            for _epoch in range(1, epochs + 1):
                train_timing: Dict[str, float] = {}
                valid_timing: Dict[str, float] = {}
                run_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device_t,
                    scaler=scaler,
                    use_amp=amp_enabled,
                    use_derivative_loss=use_derivative_loss,
                    derivative_loss_weight=trial_derivative_loss_weight,
                    timing=train_timing if log_timing else None,
                )
                valid_loss, _ = run_epoch(
                    model,
                    valid_loader,
                    None,
                    device_t,
                    scaler=None,
                    use_amp=amp_enabled,
                    use_derivative_loss=use_derivative_loss,
                    derivative_loss_weight=trial_derivative_loss_weight,
                    timing=valid_timing if log_timing else None,
                )
                if log_timing:
                    _emit_timing(
                        f"[trial {trial.number:03d} epoch {_epoch:03d}] "
                        f"train(total={train_timing['epoch_total_s']:.2f}s, "
                        f"transfer={train_timing['transfer_s']:.2f}s, "
                        f"forward={train_timing['forward_s']:.2f}s, "
                        f"backward_step={train_timing['backward_step_s']:.2f}s) "
                        f"valid(total={valid_timing['epoch_total_s']:.2f}s, "
                        f"transfer={valid_timing['transfer_s']:.2f}s, "
                        f"forward={valid_timing['forward_s']:.2f}s)",
                        timing_log_path,
                    )

                improved = valid_loss < (best_valid_loss - early_stopping_min_delta)
                if improved:
                    best_valid_loss = valid_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                trial.report(best_valid_loss, _epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if early_stopping_patience is not None and no_improve_count >= early_stopping_patience:
                    break
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            msg = str(exc).lower()
            if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in msg:
                model = None
                optimizer = None
                train_loader = None
                valid_loader = None
                scaler = None
                _cleanup_cuda()
                raise optuna.TrialPruned("Pruned due to CUDA OOM.") from exc
            raise
        finally:
            if log_timing:
                _emit_timing(
                    f"[trial {trial.number:03d}] timing "
                    f"precompute={t_precompute:.2f}s "
                    f"data={t_data:.2f}s "
                    f"model_opt={t_model_opt:.2f}s "
                    f"trial_total={time.perf_counter() - t_trial_start:.2f}s",
                    timing_log_path,
                )
            model = None
            optimizer = None
            train_loader = None
            valid_loader = None
            scaler = None
            _cleanup_cuda()

        return best_valid_loss

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)

    best = study.best_trial
    best_params = dict(best.params)
    best_params["cutoff"] = fixed_cutoff
    best_params["max_neighbors"] = fixed_max_neighbors
    return {
        "best_value": float(best.value),
        "best_params": best_params,
        "best_trial_number": int(best.number),
        "n_trials": len(study.trials),
    }


@torch.no_grad()
def predict_spectra(
    model: nn.Module,
    samples: Sequence[Dict[str, Any]],
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    precompute_graphs: bool = True,
) -> np.ndarray:
    use_samples: Sequence[Dict[str, Any]] = samples
    if precompute_graphs and hasattr(model, "cutoff") and hasattr(model, "max_neighbors"):
        use_samples = precompute_graphs_for_samples(
            samples,
            cutoff=float(model.cutoff),
            max_neighbors=None if model.max_neighbors is None else int(model.max_neighbors),
        )

    ds = MoleculeSpectrumDataset(use_samples, require_spectrum=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules)

    model.eval()
    model.to(device)

    preds = []
    for batch in loader:
        z = batch.z.to(device)
        pos = batch.pos.to(device)
        b = batch.batch.to(device)
        model_kwargs = {"z": z, "pos": pos, "batch": b}
        if batch.absorbing_atom_index is not None:
            model_kwargs["absorbing_atom_index"] = batch.absorbing_atom_index.to(device)
        pred = model(**model_kwargs)
        preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0)


def make_dummy_samples(
    n_samples: int = 100,
    n_energy: int = 300,
    max_atoms: int = 12,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples = []

    possible_z = np.array([1, 6, 7, 8, 9], dtype=np.int64)

    x = np.linspace(-5, 5, n_energy)
    for _ in range(n_samples):
        n_atoms = rng.integers(4, max_atoms + 1)
        z = rng.choice(possible_z, size=n_atoms, replace=True)
        pos = rng.normal(size=(n_atoms, 3)).astype(np.float32)

        center1 = 0.5 * np.mean(z) - 2.0
        center2 = 0.2 * np.std(pos) + 1.0
        spectrum = (
            0.8 * np.exp(-0.5 * ((x - center1) / 0.8) ** 2)
            + 0.5 * np.exp(-0.5 * ((x - center2) / 1.2) ** 2)
        ).astype(np.float32)

        samples.append({"z": z, "pos": pos, "spectrum": spectrum})
    return samples
