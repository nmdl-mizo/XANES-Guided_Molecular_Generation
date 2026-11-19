# XANES-Guided_Molecular_Generation

[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24339709.svg)](https://doi.org/10.5281/zenodo.24339709)

This repository contains an end-to-end XANES demonstration:

```text
input XANES spectra
        |
        v
DiffSpectra inverse model
        |
        v
generated 3D RDKit molecules (sample_rdmols_3d.pkl)
        |
        v
Mol2XANES forward models
        |
        v
predicted CK / NK / OK spectra
```

The repository is organized as follows:

- `inverse_model/`: DiffSpectra spectra-to-structure model.
- `forward_model/`: Mol2XANES structure-to-spectra model.
- `data/processed/`: ten-molecule demonstration subsets.
- `pretrained/`: one inverse and six forward checkpoints.
- `run_demo.sh`: complete inverse-to-forward inference pipeline.
- `train_inverse.sh`: DiffSpectra training entry point.
- `train_forward.sh`: Mol2XANES training entry point.

The original Mol2XANES integration code and repository documentation are
released under the MIT License. DiffSpectra-derived and MACE code retain their
respective third-party licenses; see the License section below.

## Installation

Python 3.10 is recommended for compatibility with the scientific Python
dependencies used by both models.

Use a Python environment with a PyTorch/PyG build appropriate for the local
CPU or CUDA version. Then install the Python dependencies and bundled MACE
layer:

```bash
python -m pip install -r requirements.txt
python -m pip install -e forward_model/mace-layer
```

`torch-geometric` and `torch-scatter` may require wheels matching the installed
PyTorch and CUDA versions. Follow their official installation instructions if
a normal `pip` installation cannot resolve a compatible build.

## End-to-end inference

Run the full pretrained pipeline from the repository root:

```bash
./run_demo.sh
```

By default this generates up to ten structures using the training-time
1000-step sampler and runs all CK, NK, and OK global/site forward models.
Results are saved under `outputs/demo/`.

Useful environment variables:

```bash
# Small plumbing check; DEVICE controls the forward model. The inverse model
# selects CUDA automatically when available.
NUM_SAMPLES=1 SAMPLING_STEPS=1 DEVICE=cpu ./run_demo.sh

# Explicit Python executables are supported when inverse and forward dependencies
# are installed in separate environments.
INVERSE_PYTHON=/path/to/inverse/python \
FORWARD_PYTHON=/path/to/forward/python \
./run_demo.sh
```

The principal outputs are:

```text
outputs/demo/inverse/molecules_ckpt_12/sample_rdmols_3d.pkl
outputs/demo/forward/global_CK.npz
outputs/demo/forward/global_NK.npz
outputs/demo/forward/global_OK.npz
outputs/demo/forward/site_CK.npz
outputs/demo/forward/site_NK.npz
outputs/demo/forward/site_OK.npz
outputs/demo/forward/summary.json
```

Each global NPZ contains `spectra` with shape `(number of eligible molecules,
300)` and `molecule_indices`. Each site NPZ additionally contains
`absorbing_atom_indices`. Molecules without the corresponding C, N, or O atom
are omitted for that edge and mapped back through `molecule_indices`.

## Forward prediction from an existing DiffSpectra pickle

`sample_rdmols_3d.pkl` can also be passed directly to Mol2XANES:

```bash
cd forward_model
PYTHONPATH=src:mace-layer python predict_rdmols.py \
  /path/to/sample_rdmols_3d.pkl \
  ../outputs/forward_only \
  --models_dir ../pretrained/forward \
  --mode both \
  --edges CK NK OK
```

## Training

The training scripts are run from the repository root. To train the inverse
model with all three global spectra, matching the conditioning used by the
included inverse checkpoint:

```bash
./train_inverse.sh global ALL
```

Single-edge global training and absorbing-site training are also supported:

```bash
./train_inverse.sh global CK
./train_inverse.sh site CK
```

DiffSpectra uses the iteration counts and batch sizes in its configuration by
default. They can be overridden without editing the Python files:

```bash
N_ITERS=200000 BATCH_SIZE=64 NUM_WORKERS=4 \
  OUTPUT_DIR=outputs/my_inverse_run \
  ./train_inverse.sh global ALL
```

Train a global or absorbing-site Mol2XANES model for one absorption edge with:

```bash
./train_forward.sh global CK
./train_forward.sh site CK
```

Replace `CK` with `NK` or `OK` to train the other edges. Common forward-model
settings are configurable in the same way:

```bash
EPOCHS=500 BATCH_SIZE=32 DEVICE=cuda \
  OUTPUT_DIR=outputs/my_forward_run \
  ./train_forward.sh global NK
```

Training outputs are written under `outputs/training/` unless `OUTPUT_DIR` is
specified. DiffSpectra writes checkpoints to the run's `checkpoints/`
directory. Mol2XANES writes `best_model.pt`, predictions, metrics, and runtime
information directly to its run directory. Existing output directories are
resumed by DiffSpectra when a compatible meta-checkpoint is present; use a new
`OUTPUT_DIR` to start an independent run.

### Quick training-path check

The following commands reduce DiffSpectra to two optimizer updates and
Mol2XANES to one epoch. They are intended only to verify loading,
forward/backward propagation, and checkpoint writing:

```bash
SMOKE_TEST=1 ./train_inverse.sh global ALL
SMOKE_TEST=1 ./train_forward.sh global CK
```

`INVERSE_PYTHON` and `FORWARD_PYTHON` select alternative Python executables if
the models are installed in separate environments.

## Full training dataset

The complete version 1.0.0 training dataset, containing 27,112 QM9-derived
molecules with molecule-level and absorbing-site-resolved CK, NK, and OK XANES
spectra, is available from Zenodo:

- Dataset: [XANES-Guided Molecular Generation Dataset: QM9 Structures and
  Simulated XANES Spectra](https://doi.org/10.5281/zenodo.24339709)
- DOI: `10.5281/zenodo.24339709`
- License: CC BY 4.0; incorporated QM9 source records retain their CC0 status

Download the following four files from the Zenodo record and place them under
`data/processed/`, replacing the demonstration files where applicable:

```text
data_qm9_xanes.pt
data_qm9_xanes_site.pt
split_dict_data_qm9_xanes.pt
split_dict_data_qm9_xanes_site.pt
```

The Zenodo record also provides `SHA256SUMS` for integrity verification. From
the directory containing the downloaded files, run:

```bash
sha256sum -c SHA256SUMS
```

When using the full dataset, please cite:

> Fujikata, Yu, & Mizoguchi, Teruyasu. (2026). *XANES-Guided Molecular
> Generation Dataset: QM9 Structures and Simulated XANES Spectra* (Version
> 1.0.0) [Data set]. Zenodo.
> <https://doi.org/10.5281/zenodo.24339709>

## Demonstration data

The included `.pt` files contain only the first ten molecules from the full
datasets. They are intended for loading and inference tests, not for meaningful
training. Replace them with the full files before training:

```text
data/processed/data_qm9_xanes.pt
data/processed/data_qm9_xanes_site.pt
```

Both models read these same PyG `(data, slices)` files. Mol2XANES converts them
to samples in memory, so derived `data_true_qm9_xanes_*` files are unnecessary.
The ten-molecule files can run the smoke tests above, but full training
requires replacing them with the complete datasets at the same paths.

## License

Original source code and documentation authored for this repository, together
with the trained model weights under `pretrained/`, are licensed under the MIT
License; see `LICENSE` and `pretrained/LICENSE`.

- `inverse_model/` is derived from DiffSpectra and remains under Apache-2.0;
  see `inverse_model/LICENSE`.
- `forward_model/mace-layer/` remains under the ACEsuit MIT License; see
  `forward_model/mace-layer/LICENSE`.
- `data/` is licensed under Creative Commons Attribution 4.0 International
  (CC BY 4.0); see `data/LICENSE`. Incorporated QM9 source records retain
  their original CC0 terms.

See `THIRD_PARTY_LICENSES.md` for provenance and scope details.

## Git LFS

The 526 MB inverse checkpoint must be committed through Git LFS. The repository
includes a `.gitattributes` entry for `.pth` files. The demonstration data and
forward `.pt` checkpoints are below GitHub's normal per-file limit and remain
ordinary Git objects. Confirm `git lfs install` and the account's LFS
storage/bandwidth quota before pushing.
