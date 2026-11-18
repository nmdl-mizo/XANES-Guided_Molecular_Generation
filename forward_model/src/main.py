from __future__ import annotations

import os

# PyTorch>=2.6 compatibility for dependency imports (e3nn/mace constants load).
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

if __name__ == "__main__":
    from mol2xanes.cli import main as cli_main

    cli_main()
else:
    from mol2xanes import (
        AtomicNumberEmbedding,
        BatchData,
        GaussianRBF,
        MACEAbsorbingAtomSpectrumModel,
        MACEGlobalSpectrumModel,
        MoleculeSpectrumDataset,
        build_model_from_checkpoint,
        build_radius_graph,
        collate_molecules,
        compute_rmse,
        global_mean_pool,
        global_sum_pool,
        load_torch_splits_as_samples,
        make_dummy_samples,
        precompute_graphs_for_samples,
        predict_spectra,
        run_epoch,
        set_seed,
        spectrum_loss,
        split_dataset,
        split_tensor_dict_to_samples,
        train_model,
        train_model_from_splits,
        tune_hyperparameters_with_optuna,
    )

    __all__ = [
        "set_seed",
        "global_mean_pool",
        "global_sum_pool",
        "build_radius_graph",
        "GaussianRBF",
        "MoleculeSpectrumDataset",
        "BatchData",
        "collate_molecules",
        "precompute_graphs_for_samples",
        "split_tensor_dict_to_samples",
        "load_torch_splits_as_samples",
        "AtomicNumberEmbedding",
        "MACEAbsorbingAtomSpectrumModel",
        "MACEGlobalSpectrumModel",
        "build_model_from_checkpoint",
        "spectrum_loss",
        "compute_rmse",
        "run_epoch",
        "split_dataset",
        "train_model",
        "train_model_from_splits",
        "tune_hyperparameters_with_optuna",
        "predict_spectra",
        "make_dummy_samples",
    ]
