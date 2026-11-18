#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
EDGE="${1:-CK}"
PYTHONPATH="src:mace-layer${PYTHONPATH:+:$PYTHONPATH}" \
python src/main.py train \
  --data_pt ../data/processed/data_qm9_xanes.pt \
  --edge "$EDGE" \
  --task global \
  --outdir "outputs/global_${EDGE}" \
  --batch_size 128 \
  --epochs 2000 \
  --early_stopping_patience 30
