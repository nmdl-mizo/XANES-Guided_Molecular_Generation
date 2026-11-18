from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import time
import warnings
from typing import Any, Dict

import numpy as np
import torch

def configure_runtime_warnings(suppress_third_party_warnings: bool) -> None:
    if not suppress_third_party_warnings:
        return

    # e3nn/mace emits compatibility warnings when loading constants.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"e3nn\.o3\._wigner",
        message=r".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"e3nn\.o3\._wigner",
    )
    # Repeated warning emitted from TorchScript checks inside dependencies.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*TorchScript type system doesn't support instance-level annotations.*",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/predict XANES models from raw QM9 XANES data or a split dictionary."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train with train/val/test splits.")
    train_parser.add_argument(
        "--data_pt",
        type=str,
        required=True,
        help="Path to raw data_qm9_xanes*.pt or a converted split dictionary.",
    )
    train_parser.add_argument("--outdir", type=str, required=True, help="Output directory for checkpoint/results.")
    train_parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "global", "absorbing_atom"],
        help="Task mode. 'auto' infers from input tensors, or specify explicitly.",
    )
    train_parser.add_argument(
        "--edge",
        choices=["CK", "NK", "OK"],
        default=None,
        help="Absorption edge. Required for raw data_qm9_xanes*.pt input.",
    )
    train_parser.add_argument("--train_ratio", type=float, default=0.8)
    train_parser.add_argument("--valid_ratio", type=float, default=0.1)
    train_parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Optional debug limit; omit for full training.",
    )
    train_parser.add_argument(
        "--best_params_json",
        type=str,
        default=None,
        help="Path to Optuna result JSON. If provided, train hyperparameters are loaded from it.",
    )
    train_parser.add_argument("--train_key", type=str, default="train")
    train_parser.add_argument("--val_key", type=str, default="valid")
    train_parser.add_argument("--test_key", type=str, default="test")
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight_decay", type=float, default=1e-5)
    train_parser.add_argument("--cutoff", type=float, default=5.0)
    train_parser.add_argument("--max_neighbors", type=int, default=None)
    train_parser.add_argument("--hidden_dim", type=int, default=64)
    train_parser.add_argument("--n_layers", type=int, default=4)
    train_parser.add_argument("--n_rbf", type=int, default=16)
    train_parser.add_argument("--correlation", type=int, default=3)
    train_parser.add_argument("--max_ell", type=int, default=2)
    train_parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "sum"])
    train_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP). AMP is enabled by default on CUDA.",
    )
    train_parser.add_argument(
        "--log_timing",
        action="store_true",
        help="Print per-epoch timing breakdown (transfer/forward/backward/metrics).",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--use_derivative_loss",
        action="store_true",
        help="Add first-derivative MSE term to the spectrum loss (default: disabled).",
    )
    train_parser.add_argument(
        "--derivative_loss_weight",
        type=float,
        default=1.0,
        help="Weight for derivative loss term when --use_derivative_loss is enabled.",
    )
    train_parser.add_argument(
        "--disable_precompute_graphs",
        action="store_true",
        help="Disable precomputing molecular graphs (default: precompute enabled).",
    )
    train_parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Early stopping patience. Set <0 to disable.",
    )
    train_parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum valid_loss improvement to reset patience.",
    )
    train_parser.add_argument(
        "--show_third_party_warnings",
        action="store_true",
        help="Show repetitive warnings from dependencies (default: hidden).",
    )
    train_parser.add_argument(
        "--save_pred_split",
        type=str,
        default="test",
        help="Which split prediction to save after training (train/valid/test).",
    )
    train_parser.add_argument(
        "--save_pred_name",
        type=str,
        default="pred.npy",
        help="Filename for saved prediction array in outdir.",
    )
    train_parser.add_argument(
        "--runtime_json",
        type=str,
        default=None,
        help="Optional path to save runtime summary JSON.",
    )

    pred_parser = subparsers.add_parser("predict", help="Predict from saved checkpoint.")
    pred_parser.add_argument("--model_ckpt", type=str, required=True, help="Path to checkpoint .pt.")
    pred_parser.add_argument("--data_pt", type=str, required=True, help="Path to .pt/.pth split dictionary.")
    pred_parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "global", "absorbing_atom"],
        help="Task mode. 'auto' infers from input tensors, or specify explicitly.",
    )
    pred_parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split key to run prediction on. Use 'all' to merge all available splits.",
    )
    pred_parser.add_argument("--batch_size", type=int, default=32)
    pred_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    pred_parser.add_argument("--out_npy", type=str, required=True, help="Output .npy path for predictions.")
    pred_parser.add_argument(
        "--runtime_json",
        type=str,
        default=None,
        help="Optional path to save runtime summary JSON.",
    )
    pred_parser.add_argument(
        "--show_third_party_warnings",
        action="store_true",
        help="Show repetitive warnings from dependencies (default: hidden).",
    )

    tune_parser = subparsers.add_parser("tune", help="Hyperparameter tuning with Optuna.")
    tune_parser.add_argument("--data_pt", type=str, required=True, help="Path to .pt/.pth split dictionary.")
    tune_parser.add_argument("--outdir", type=str, required=True, help="Output directory for tuning results.")
    tune_parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "global", "absorbing_atom"],
        help="Task mode. 'auto' infers from input tensors, or specify explicitly.",
    )
    tune_parser.add_argument("--train_key", type=str, default="train")
    tune_parser.add_argument("--val_key", type=str, default="valid")
    tune_parser.add_argument("--test_key", type=str, default="test")
    tune_parser.add_argument("--n_trials", type=int, default=20)
    tune_parser.add_argument("--epochs", type=int, default=80)
    tune_parser.add_argument("--batch_size", type=int, default=16)
    tune_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    tune_parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP). AMP is enabled by default on CUDA.",
    )
    tune_parser.add_argument(
        "--log_timing",
        action="store_true",
        help="Print per-trial/per-epoch timing breakdown.",
    )
    tune_parser.add_argument("--seed", type=int, default=42)
    tune_parser.add_argument(
        "--use_derivative_loss",
        action="store_true",
        help="Add first-derivative MSE term to the spectrum loss during tuning (default: disabled).",
    )
    tune_parser.add_argument(
        "--derivative_loss_weight",
        type=float,
        default=1.0,
        help="Weight for derivative loss term when --use_derivative_loss is enabled.",
    )
    tune_parser.add_argument(
        "--tune_derivative_loss_weight",
        action="store_true",
        help="Include derivative_loss_weight in the Optuna search space when derivative loss is enabled.",
    )
    tune_parser.add_argument(
        "--derivative_loss_weight_min",
        type=float,
        default=0.1,
        help="Minimum derivative_loss_weight for Optuna search.",
    )
    tune_parser.add_argument(
        "--derivative_loss_weight_max",
        type=float,
        default=10.0,
        help="Maximum derivative_loss_weight for Optuna search.",
    )
    tune_parser.add_argument(
        "--disable_precompute_graphs",
        action="store_true",
        help="Disable precomputing molecular graphs per trial (default: enabled).",
    )
    tune_parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="Early stopping patience for each trial. Set <0 to disable.",
    )
    tune_parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum valid_loss improvement to reset patience within each trial.",
    )
    tune_parser.add_argument("--timeout_sec", type=int, default=None, help="Optuna timeout in seconds.")
    tune_parser.add_argument("--study_name", type=str, default="mol2xanes_tuning")
    tune_parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///study.db). If omitted, in-memory study.",
    )
    tune_parser.add_argument(
        "--show_third_party_warnings",
        action="store_true",
        help="Show repetitive warnings from dependencies (default: hidden).",
    )
    tune_parser.add_argument(
        "--runtime_json",
        type=str,
        default=None,
        help="Optional path to save runtime summary JSON.",
    )

    return parser


def _load_best_params_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "best_params" in raw:
        params = raw["best_params"]
    else:
        params = raw

    if not isinstance(params, dict):
        raise ValueError("best_params_json must contain a dict or a dict with 'best_params' key.")
    return params


def _apply_best_params_to_train_args(args: argparse.Namespace) -> None:
    if args.best_params_json is None:
        return

    params = _load_best_params_json(args.best_params_json)
    key_map = {
        "lr": "lr",
        "weight_decay": "weight_decay",
        "cutoff": "cutoff",
        "max_neighbors": "max_neighbors",
        "hidden_dim": "hidden_dim",
        "n_layers": "n_layers",
        "n_rbf": "n_rbf",
        "correlation": "correlation",
        "max_ell": "max_ell",
        "pooling": "pooling",
        "derivative_loss_weight": "derivative_loss_weight",
    }

    applied_keys = []
    ignored_keys = []
    for src_key, value in params.items():
        if src_key not in key_map:
            ignored_keys.append(src_key)
            continue
        dst_key = key_map[src_key]
        setattr(args, dst_key, value)
        applied_keys.append(src_key)

    print(f"Loaded best params from: {args.best_params_json}")
    if applied_keys:
        print("Applied params:", ", ".join(applied_keys))
    if ignored_keys:
        print("Ignored params:", ", ".join(ignored_keys))


def _save_runtime_json(
    path: str,
    command: str,
    started_at: datetime,
    finished_at: datetime,
    extra: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "command": command,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": (finished_at - started_at).total_seconds(),
    }
    payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Saved runtime:", path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_runtime_warnings(suppress_third_party_warnings=not args.show_third_party_warnings)
    from .data import load_torch_splits_as_samples, split_tensor_dict_to_samples
    from .io_utils import load_torch_file
    from .model import build_model_from_checkpoint
    from .train import predict_spectra, train_model_from_splits, tune_hyperparameters_with_optuna

    if args.command == "train":
        started_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        os.makedirs(args.outdir, exist_ok=True)
        timing_log_path = os.path.join(args.outdir, "timing_train.log") if args.log_timing else None
        metrics_log_path = os.path.join(args.outdir, "train_metrics.log")
        if timing_log_path is not None:
            with open(timing_log_path, "w", encoding="utf-8") as f:
                f.write(f"# timing log started at {started_at.isoformat()}\n")
        with open(metrics_log_path, "w", encoding="utf-8") as f:
            f.write(f"# train metrics log started at {started_at.isoformat()}\n")
        _apply_best_params_to_train_args(args)
        early_stopping_patience = None if args.early_stopping_patience < 0 else args.early_stopping_patience
        split_samples = load_torch_splits_as_samples(
            args.data_pt,
            split_keys=(args.train_key, args.val_key, args.test_key),
            task_mode=args.task,
            edge=args.edge,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            max_samples_per_split=args.max_samples_per_split,
        )
        split_samples = {
            "train": split_samples[args.train_key],
            "valid": split_samples[args.val_key],
            "test": split_samples[args.test_key],
        }

        result = train_model_from_splits(
            split_samples=split_samples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_rbf=args.n_rbf,
            correlation=args.correlation,
            max_ell=args.max_ell,
            pooling=args.pooling,
            multi_scale_readout=True,
            use_derivative_loss=args.use_derivative_loss,
            derivative_loss_weight=args.derivative_loss_weight,
            device=args.device,
            seed=args.seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            precompute_graphs=not args.disable_precompute_graphs,
            use_amp=not args.disable_amp,
            log_timing=args.log_timing,
            timing_log_path=timing_log_path,
            metrics_log_path=metrics_log_path,
        )

        ckpt = {
            "task_type": getattr(result["model"], "task_type", "global"),
            "max_z": result["max_z"],
            "n_energy": result["n_energy"],
            "cutoff": args.cutoff,
            "max_neighbors": args.max_neighbors,
            "n_rbf": args.n_rbf,
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "correlation": args.correlation,
            "max_ell": args.max_ell,
            "pooling": args.pooling,
            "multi_scale_readout": True,
            "use_derivative_loss": args.use_derivative_loss,
            "derivative_loss_weight": args.derivative_loss_weight,
            "best_epoch": result["best_epoch"],
            "best_valid_loss": result["best_valid_loss"],
            "best_valid_rmse": result["best_valid_rmse"],
            "epochs_ran": result["epochs_ran"],
            "stopped_early": result["stopped_early"],
            "test_loss": result["test_loss"],
            "test_rmse": result["test_rmse"],
            "model_state_dict": result["model"].state_dict(),
        }
        ckpt_path = os.path.join(args.outdir, "best_model.pt")
        torch.save(ckpt, ckpt_path)

        if args.save_pred_split not in split_samples:
            raise ValueError(f"--save_pred_split must be one of {list(split_samples.keys())}")
        pred = predict_spectra(
            result["model"],
            split_samples[args.save_pred_split],
            batch_size=args.batch_size,
            device=args.device,
        )
        pred_path = os.path.join(args.outdir, args.save_pred_name)
        np.save(pred_path, pred)

        print("\nBest epoch:", result["best_epoch"])
        print("Best valid loss:", result["best_valid_loss"])
        print("Best valid RMSE:", result["best_valid_rmse"])
        print("Epochs ran:", result["epochs_ran"])
        print("Stopped early:", result["stopped_early"])
        print("Test loss:", result["test_loss"])
        print("Test RMSE:", result["test_rmse"])
        print("Saved checkpoint:", ckpt_path)
        print(f"Saved {args.save_pred_split} prediction:", pred_path)
        print("Saved train metrics log:", metrics_log_path)
        finished_at = datetime.now(timezone.utc)
        runtime_path = args.runtime_json or os.path.join(args.outdir, "runtime_train.json")
        _save_runtime_json(
            path=runtime_path,
            command="train",
            started_at=started_at,
            finished_at=finished_at,
            extra={
                "elapsed_seconds_perf_counter": time.perf_counter() - t0,
                "best_epoch": result["best_epoch"],
                "epochs_ran": result["epochs_ran"],
                "stopped_early": result["stopped_early"],
                "best_valid_loss": result["best_valid_loss"],
                "test_loss": result["test_loss"],
                "test_rmse": result["test_rmse"],
                "checkpoint_path": ckpt_path,
                "train_metrics_log_path": metrics_log_path,
            },
        )

    elif args.command == "predict":
        started_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        raw = load_torch_file(args.data_pt, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError("Loaded data file must be dict with split keys.")
        ckpt = load_torch_file(args.model_ckpt, map_location="cpu")
        model = build_model_from_checkpoint(ckpt, args.device)
        allowed_max_z = int(getattr(model, "max_z"))
        default_n_energy = int(getattr(model, "n_energy"))

        def _is_split_like(payload: Any) -> bool:
            return isinstance(payload, dict) and {"positions", "charges"}.issubset(payload.keys())

        split_lower = args.split.lower()
        if split_lower == "all":
            used_splits = []
            merged_samples = []
            if _is_split_like(raw):
                merged_samples.extend(
                    split_tensor_dict_to_samples(
                        raw,
                        task_mode=args.task,
                        allow_missing_spectrum=True,
                        default_n_energy=default_n_energy,
                    )
                )
                used_splits.append("__root__")
            preferred_split_order = ["train", "valid", "test"]
            ordered_split_names = [
                name for name in preferred_split_order if name in raw
            ] + [
                name for name in raw.keys() if name not in preferred_split_order
            ]
            for split_name in ordered_split_names:
                split_payload = raw[split_name]
                # Accept split-like dicts that contain the minimum tensor keys for prediction.
                if not _is_split_like(split_payload):
                    continue
                merged_samples.extend(
                    split_tensor_dict_to_samples(
                        split_payload,
                        task_mode=args.task,
                        allow_missing_spectrum=True,
                        default_n_energy=default_n_energy,
                    )
                )
                used_splits.append(split_name)
            if not merged_samples:
                raise ValueError(
                    "No split-like entries found. Expected dict entries containing "
                    "'positions' and 'charges' (and 'spectrum' for labeled data)."
                )
            samples = merged_samples
            print("Predict mode: merged splits ->", ", ".join(used_splits))
            split_label = "all"
        else:
            if args.split in raw:
                split_payload = raw[args.split]
            elif _is_split_like(raw):
                split_payload = raw
                print(
                    f"Predict mode: top-level input is split-like; using it directly for requested split '{args.split}'."
                )
            else:
                raise ValueError(f"Split '{args.split}' not found in data file.")
            samples = split_tensor_dict_to_samples(
                split_payload,
                task_mode=args.task,
                allow_missing_spectrum=True,
                default_n_energy=default_n_energy,
            )
            used_splits = [args.split if args.split in raw else "__root__"]
            split_label = args.split
        filtered_samples = []
        kept_indices = []
        dropped_indices = []
        dropped = 0
        for idx, s in enumerate(samples):
            z = np.asarray(s["z"])
            if z.size == 0:
                dropped += 1
                dropped_indices.append(idx)
                continue
            z_min = int(z.min())
            z_max = int(z.max())
            if z_min < 1 or z_max > allowed_max_z:
                dropped += 1
                dropped_indices.append(idx)
                continue
            filtered_samples.append(s)
            kept_indices.append(idx)
        if dropped > 0:
            print(
                f"Filtered out {dropped} samples due to atomic numbers outside "
                f"[1, {allowed_max_z}] from checkpoint."
            )
        if len(filtered_samples) == 0:
            raise ValueError(
                f"All samples were filtered out. Check data atomic numbers and checkpoint max_z={allowed_max_z}."
            )
        samples = filtered_samples

        out_dir = os.path.dirname(args.out_npy)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.basename(args.out_npy)
        filtered_input_path = os.path.join(out_dir or ".", f"{base_name}.filtered_input.pt")
        torch.save(
            {
                "samples": samples,
                "used_splits": used_splits,
                "predict_split": split_label,
                "model_max_z": allowed_max_z,
                "n_samples_before_filter": len(kept_indices) + len(dropped_indices),
                "n_samples_after_filter": len(samples),
                "n_samples_filtered_out_of_range_z": dropped,
                "kept_indices": kept_indices,
                "dropped_indices": dropped_indices,
            },
            filtered_input_path,
        )
        print("Saved filtered input:", filtered_input_path)
        pred = predict_spectra(model, samples, batch_size=args.batch_size, device=args.device)

        np.save(args.out_npy, pred)
        print("Saved prediction:", args.out_npy)
        finished_at = datetime.now(timezone.utc)
        runtime_path = args.runtime_json or os.path.join(out_dir or ".", "runtime_predict.json")
        _save_runtime_json(
            path=runtime_path,
            command="predict",
            started_at=started_at,
            finished_at=finished_at,
            extra={
                "elapsed_seconds_perf_counter": time.perf_counter() - t0,
                "n_predictions": int(pred.shape[0]),
                "output_npy": args.out_npy,
                "predict_split": split_label,
                "used_splits": used_splits,
                "model_max_z": allowed_max_z,
                "n_samples_after_filter": len(samples),
                "n_samples_filtered_out_of_range_z": dropped,
                "filtered_input_path": filtered_input_path,
            },
        )

    elif args.command == "tune":
        started_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        os.makedirs(args.outdir, exist_ok=True)
        timing_log_path = os.path.join(args.outdir, "timing_tune.log") if args.log_timing else None
        if timing_log_path is not None:
            with open(timing_log_path, "w", encoding="utf-8") as f:
                f.write(f"# timing log started at {started_at.isoformat()}\n")
        early_stopping_patience = None if args.early_stopping_patience < 0 else args.early_stopping_patience
        split_samples = load_torch_splits_as_samples(
            args.data_pt,
            split_keys=(args.train_key, args.val_key, args.test_key),
            task_mode=args.task,
        )
        split_samples = {
            "train": split_samples[args.train_key],
            "valid": split_samples[args.val_key],
            "test": split_samples[args.test_key],
        }

        tuning_result = tune_hyperparameters_with_optuna(
            split_samples=split_samples,
            n_trials=args.n_trials,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            study_name=args.study_name,
            storage=args.storage,
            timeout=args.timeout_sec,
            precompute_graphs=not args.disable_precompute_graphs,
            use_amp=not args.disable_amp,
            use_derivative_loss=args.use_derivative_loss,
            derivative_loss_weight=args.derivative_loss_weight,
            tune_derivative_loss_weight=args.tune_derivative_loss_weight,
            derivative_loss_weight_min=args.derivative_loss_weight_min,
            derivative_loss_weight_max=args.derivative_loss_weight_max,
            log_timing=args.log_timing,
            timing_log_path=timing_log_path,
        )

        out_json = os.path.join(args.outdir, "best_params_optuna.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(tuning_result, f, indent=2)

        print("Best trial number:", tuning_result["best_trial_number"])
        print("Best valid loss:", tuning_result["best_value"])
        print("Best params:", tuning_result["best_params"])
        print("Saved tuning result:", out_json)
        finished_at = datetime.now(timezone.utc)
        runtime_path = args.runtime_json or os.path.join(args.outdir, "runtime_tune.json")
        _save_runtime_json(
            path=runtime_path,
            command="tune",
            started_at=started_at,
            finished_at=finished_at,
            extra={
                "elapsed_seconds_perf_counter": time.perf_counter() - t0,
                "best_trial_number": tuning_result["best_trial_number"],
                "best_value": tuning_result["best_value"],
                "n_trials": tuning_result["n_trials"],
                "tuning_result_json": out_json,
            },
        )


if __name__ == "__main__":
    main()
