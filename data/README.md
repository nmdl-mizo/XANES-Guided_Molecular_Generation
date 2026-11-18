# Demonstration data

These demonstration datasets and their generated XANES values are licensed
under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
See `LICENSE` for the license notice and attribution requirements.

The molecular records are derived from the QM9 dataset, which was released
under CC0 1.0. Those incorporated source records retain their original CC0
status. Source: *Data for 133885 GDB-9 molecules*,
DOI: <https://doi.org/10.6084/m9.figshare.978904_D12>.

When redistributing or adapting these demonstration datasets, cite this
repository, indicate whether changes were made, and include a link to
<https://creativecommons.org/licenses/by/4.0/>. Academic users should also
cite the original QM9 publication and dataset.

The two files in `processed/` preserve the original PyG `(data, slices)` format
but contain only the first ten molecules:

- `data_qm9_xanes.pt`: molecule-level XANES data, 369,879 bytes.
- `data_qm9_xanes_site.pt`: absorbing-site-resolved XANES data, 2,427,158 bytes.

SHA-256 checksums:

```text
47105de0bcdd6c84f337220d6ecca6021e77efcb252ef757fd3988016c2771ca  data_qm9_xanes.pt
c2b4ca439a0d853bd9bfee3cf3ada127e97ae59cd891d620ed6dc2119d7ac2bd  data_qm9_xanes_site.pt
```

These subsets are for demonstrations and smoke tests only. They are too small
for meaningful model training. Both inverse and forward loaders are compatible
with the corresponding full datasets when the files are replaced in place.
