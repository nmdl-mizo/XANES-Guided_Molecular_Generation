#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

TASK="${1:-global}"
EDGE="${2:-CK}"
TASK="${TASK,,}"
EDGE="${EDGE^^}"

if [[ "${EDGE}" != "CK" && "${EDGE}" != "NK" && "${EDGE}" != "OK" ]]; then
  echo "EDGE must be CK, NK, or OK." >&2
  exit 2
fi

case "${TASK}" in
  global)
    TASK_MODE="global"
    DATA_FILE="${ROOT_DIR}/data/processed/data_qm9_xanes.pt"
    RUN_NAME="global_${EDGE}"
    ;;
  site)
    TASK_MODE="absorbing_atom"
    DATA_FILE="${ROOT_DIR}/data/processed/data_qm9_xanes_site.pt"
    RUN_NAME="site_${EDGE}"
    ;;
  *)
    echo "Usage: $0 {global|site} {CK|NK|OK}" >&2
    exit 2
    ;;
esac

FORWARD_PYTHON="${FORWARD_PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/training/forward/${RUN_NAME}}"
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${ROOT_DIR}/${OUTPUT_DIR}"
fi

# A one-epoch CPU run is sufficient to verify the public data-to-checkpoint path.
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  EPOCHS="${EPOCHS:-1}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:--1}"
  DEVICE="${DEVICE:-cpu}"
else
  EPOCHS="${EPOCHS:-2000}"
  BATCH_SIZE="${BATCH_SIZE:-128}"
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-30}"
fi

args=(
  "${ROOT_DIR}/forward_model/src/main.py"
  train
  --data_pt "${DATA_FILE}"
  --edge "${EDGE}"
  --task "${TASK_MODE}"
  --outdir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}"
)

[[ -n "${DEVICE:-}" ]] && args+=(--device "${DEVICE}")
[[ -n "${LEARNING_RATE:-}" ]] && args+=(--lr "${LEARNING_RATE}")
[[ -n "${WEIGHT_DECAY:-}" ]] && args+=(--weight_decay "${WEIGHT_DECAY}")
[[ -n "${TRAIN_RATIO:-}" ]] && args+=(--train_ratio "${TRAIN_RATIO}")
[[ -n "${VALID_RATIO:-}" ]] && args+=(--valid_ratio "${VALID_RATIO}")
[[ -n "${MAX_SAMPLES_PER_SPLIT:-}" ]] && args+=(--max_samples_per_split "${MAX_SAMPLES_PER_SPLIT}")

export PYTHONPATH="${ROOT_DIR}/forward_model/src:${ROOT_DIR}/forward_model/mace-layer${FORWARD_EXTRA_PYTHONPATH:+:${FORWARD_EXTRA_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Training Mol2XANES: task=${TASK}, edge=${EDGE}, output=${OUTPUT_DIR}"
exec "${FORWARD_PYTHON}" "${args[@]}"
