# Third-party licenses and scope

The root MIT License covers original source code and documentation authored for
this repository. It does not replace the licenses of the components below.

## DiffSpectra

- Local path: `inverse_model/`
- Upstream: <https://github.com/AzureLeon1/DiffSpectra>
- Upstream revision checked: `2c540c067f18fca683c27054050d2f54b75cf2ef`
- License: Apache License 2.0
- Local license text: `inverse_model/LICENSE`

The inverse model is derived from DiffSpectra and includes modifications for
XANES conditioning, site-conditioned data, training/evaluation integration,
and public-release packaging. Modified upstream files carry a modification
notice and remain distributed under Apache-2.0. The upstream repository did
not contain a `NOTICE` file at the revision checked during packaging.

## MACE layer

- Local path: `forward_model/mace-layer/`
- Upstream: <https://github.com/ACEsuit/mace-layer>
- Upstream revision checked: `ddc076bf6e9f24fa5007dfafeefea8e1d9cb2226`
- License: MIT
- Copyright: Copyright (c) 2022 ACEsuit
- Local license text: `forward_model/mace-layer/LICENSE`

The bundled MACE layer source and license match the upstream source revision
checked during packaging. Its original copyright and permission notice must
be retained.

## Demonstration data

- Local path: `data/`
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Local license notice: `data/LICENSE`

The demonstration datasets and generated XANES values are released under
CC BY 4.0. Molecular records incorporated from QM9 retain their original CC0
1.0 dedication. The QM9 source dataset is available as *Data for 133885 GDB-9
molecules*, DOI: <https://doi.org/10.6084/m9.figshare.978904_D12>.

## Trained model weights

- Local path: `pretrained/`
- License: MIT
- Local license text: `pretrained/LICENSE`

All included checkpoints were trained from scratch by the repository authors.
The MIT License applies to the distributed weight files. This does not alter
the licenses of DiffSpectra, MACE, or any other third-party software used to
train or run the models.

## Installed dependencies

Packages installed through `requirements.txt` are not copied into this
repository and remain governed by their respective licenses. Distributors of
bundled environments or binaries should review those dependency licenses as
part of their release process.
