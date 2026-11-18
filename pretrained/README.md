# Pretrained models

The trained model weights in this directory were trained from scratch by the
repository authors and are released under the MIT License. See `LICENSE` in
this directory. The licenses of the third-party source code used to train and
run the models remain unchanged; see `../THIRD_PARTY_LICENSES.md`.

- `inverse/checkpoints/checkpoint_12.pth`: DiffSpectra model trained with CK,
  NK, and OK XANES spectra.
- `forward/global/{CK,NK,OK}/best_model.pt`: molecule-level Mol2XANES models.
- `forward/site/{CK,NK,OK}/best_model.pt`: absorbing-site Mol2XANES models.

The inverse checkpoint is approximately 526 MB and must be stored with Git
LFS. The forward checkpoints are each below GitHub's ordinary 100 MB file
limit and can be stored as normal Git objects. Checksums for all seven files are
listed in `MANIFEST.sha256`.
