# XANES-Guided Molecular Generation Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24339709.svg)](https://doi.org/10.5281/zenodo.24339709)

Version: 1.0.0

DOI: <https://doi.org/10.5281/zenodo.24339709>

This dataset contains QM9-derived molecular structures paired with processed,
simulated carbon, nitrogen, and oxygen K-edge X-ray absorption near-edge
structure (XANES) spectra. It is the full training dataset for the inverse
DiffSpectra and forward Mol2XANES models distributed in the
[XANES-Guided_Molecular_Generation](https://github.com/nmdl-mizo/XANES-Guided_Molecular_Generation)
repository.

The dataset contains 27,112 molecules. Each spectrum is represented by 300
intensity values on the common preprocessed spectral grid used by the training
code. An explicit energy-axis array is not stored in these files.

## Files

| File | Size (bytes) | Description |
| --- | ---: | --- |
| `data_qm9_xanes.pt` | 1,019,069,294 | Molecule-level CK, NK, and OK XANES spectra and molecular graphs. |
| `data_qm9_xanes_site.pt` | 6,587,337,010 | Absorbing-site-resolved CK, NK, and OK XANES spectra and molecular graphs. |
| `split_dict_data_qm9_xanes.pt` | 232,217 | Train/validation/test indices for the molecule-level dataset. |
| `split_dict_data_qm9_xanes_site.pt` | 232,247 | Train/validation/test indices for the site-resolved dataset. |

Both split files contain 16,267 training, 5,422 validation, and 5,423 test
molecule indices. The molecule-level and site-resolved datasets use the same
split sizes and molecule ordering.

The smaller ten-molecule files in the GitHub repository are demonstration
subsets. They are not the complete files described here.

## Data format

The two main files use the serialized PyTorch Geometric `InMemoryDataset`
representation:

```python
(data, slices)
```

`data` is a `torch_geometric.data.Data` object containing concatenated fields,
and `slices` maps each field to the offsets needed to reconstruct individual
molecules.

Common molecular fields include:

| Field | Description |
| --- | --- |
| `idx` | Source molecule identifier. |
| `num_atom` | Number of atoms in the molecule. |
| `charge` | Atomic numbers. |
| `pos` | Three-dimensional Cartesian coordinates. |
| `atom_type` | Atom-type encoding used by DiffSpectra. |
| `fc` | Formal-charge encoding. |
| `edge_index` | Molecular graph connectivity. |
| `edge_type` | Bond/edge-type encoding. |
| `rdmol` | Serialized RDKit molecule. |
| `y` | QM9 molecular-property values retained from the processed source data. |

`data_qm9_xanes.pt` contains `xanes_CK`, `xanes_NK`, and `xanes_OK`. Each
molecule has one 300-point molecule-level spectrum for each absorption edge.

`data_qm9_xanes_site.pt` contains padded per-atom `xanes_CK`, `xanes_NK`, and
`xanes_OK` arrays, together with `spectrum_CK`, `spectrum_NK`, `spectrum_OK`,
`exs`, `hole_index`, and `hole_split_id` fields used by the site-conditioned
loaders. Only rows corresponding to valid atoms and the requested absorbing
element are selected by the public training code.

The files also retain UV, IR, and Raman fields inherited from the processed
QM9 source records. They are not required by the XANES-only training scripts.

## Software requirements and loading

Python 3.10 is recommended. Loading the files requires compatible versions of
PyTorch, PyTorch Geometric, and RDKit. See the GitHub repository's
`requirements.txt` files for the tested software dependencies.

```python
import torch

data, slices = torch.load(
    "data_qm9_xanes.pt",
    map_location="cpu",
    weights_only=False,
)

num_molecules = len(slices["charge"]) - 1
print(num_molecules)  # 27112
print(data.xanes_CK.shape[-1])  # 300
```

These `.pt` files contain serialized Python and RDKit objects and therefore
require `weights_only=False`. Only load files obtained from this Zenodo record
or another trusted source, and verify their checksums first.

## Use with the public code

Place the downloaded files at the following paths in a clone of the public
repository:

```text
data/processed/data_qm9_xanes.pt
data/processed/data_qm9_xanes_site.pt
data/processed/split_dict_data_qm9_xanes.pt
data/processed/split_dict_data_qm9_xanes_site.pt
```

The inverse and forward training entry points then read the same data files:

```bash
./train_inverse.sh global ALL
./train_inverse.sh site CK

./train_forward.sh global CK
./train_forward.sh site CK
```

Replace `CK` with `NK` or `OK` where supported. See the repository README for
installation, smoke-test, and training options.

## Integrity checks

SHA-256 checksums for version 1.0.0:

```text
577b013c1ea00afd87cf40fbc4d304cc389a0f60e11df40479ad7991acce130e  data_qm9_xanes.pt
f200f685c608cc4858a6217a7461a0e8dfc0f454fb0f99e05ebf7eb56ca56fd0  data_qm9_xanes_site.pt
73bd926e1010c322fdbc8d4bb40b48677ca8724a9dab0e9a107088828f7d2004  split_dict_data_qm9_xanes.pt
b94c7eab4bcde6c26cbb24e6350914cb2243cfa055db47e4966b400ab47b6145  split_dict_data_qm9_xanes_site.pt
```

Save the four lines above as `SHA256SUMS` in the same directory as the files,
then verify them with:

```bash
sha256sum -c SHA256SUMS
```

## Provenance

The molecular records are derived from the QM9/GDB-9 dataset:

- *Data for 133885 GDB-9 molecules*
- DOI: <https://doi.org/10.6084/m9.figshare.978904_D12>
- Original data dedication: CC0 1.0

The XANES values and dataset packaging were generated for the
XANES-Guided_Molecular_Generation project. The dataset contains the 27,112 QM9
molecules for which the processed XANES data used in this project are
available.

## License

The generated XANES values and dataset packaging are released under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
The incorporated QM9 source records retain their original CC0 status.

## Citation

```text
Fujikata, Yu, & Mizoguchi, Teruyasu. (2026).
XANES-Guided Molecular Generation Dataset: QM9 Structures and Simulated XANES Spectra
(Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.24339709
```

## Contact and issues

For data-format questions or reproducibility issues, open an issue at:

<https://github.com/nmdl-mizo/XANES-Guided_Molecular_Generation/issues>
