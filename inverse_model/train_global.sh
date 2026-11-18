#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
EDGE="${1:-CK}"
python main.py \
  --config configs/diffspectra_qm9s.py \
  --mode train \
  --workdir "outputs/global_${EDGE}" \
  --config.data.processed_file data_qm9_xanes.pt \
  --config.data.used_spectra "('xanes_${EDGE}',)" \
  --config.training.snapshot_sampling=false
