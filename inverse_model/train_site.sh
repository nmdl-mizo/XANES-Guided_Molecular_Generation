#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
EDGE="${1:-CK}"
python main.py \
  --config configs/diffspectra_qm9s_site_xanes.py \
  --mode train \
  --workdir "outputs/site_${EDGE}" \
  --config.data.processed_file data_qm9_xanes_site.pt \
  --config.data.used_spectra "('xanes_${EDGE}',)" \
  --config.data.absorbing_edge "xanes_${EDGE}" \
  --config.data.site_spectra_field "xanes_${EDGE}" \
  --config.training.snapshot_sampling=false
