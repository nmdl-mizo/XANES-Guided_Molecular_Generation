#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

TASK="${1:-global}"
EDGE="${2:-ALL}"
TASK="${TASK,,}"
EDGE="${EDGE^^}"

case "${TASK}" in
  global)
    CONFIG_FILE="configs/diffspectra_qm9s.py"
    DATA_FILE="data_qm9_xanes.pt"
    if [[ "${EDGE}" == "ALL" ]]; then
      USED_SPECTRA="('xanes_CK', 'xanes_NK', 'xanes_OK')"
      RUN_NAME="global_all"
    elif [[ "${EDGE}" == "CK" || "${EDGE}" == "NK" || "${EDGE}" == "OK" ]]; then
      USED_SPECTRA="('xanes_${EDGE}',)"
      RUN_NAME="global_${EDGE}"
    else
      echo "For global training, EDGE must be ALL, CK, NK, or OK." >&2
      exit 2
    fi
    ;;
  site)
    if [[ "${EDGE}" != "CK" && "${EDGE}" != "NK" && "${EDGE}" != "OK" ]]; then
      echo "For site training, EDGE must be CK, NK, or OK." >&2
      exit 2
    fi
    CONFIG_FILE="configs/diffspectra_qm9s_site_xanes.py"
    DATA_FILE="data_qm9_xanes_site.pt"
    USED_SPECTRA="('xanes_${EDGE}',)"
    RUN_NAME="site_${EDGE}"
    ;;
  *)
    echo "Usage: $0 {global|site} {ALL|CK|NK|OK}" >&2
    exit 2
    ;;
esac

INVERSE_PYTHON="${INVERSE_PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/training/inverse/${RUN_NAME}}"
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${ROOT_DIR}/${OUTPUT_DIR}"
fi

# A smoke test exercises data loading, forward/backward propagation, checkpoint
# writing, and then stops. It is not a meaningful training run.
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  N_ITERS="${N_ITERS:-1}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
  LOG_FREQ="${LOG_FREQ:-1}"
fi

args=(
  main.py
  --config "${CONFIG_FILE}"
  --mode train
  --workdir "${OUTPUT_DIR}"
  --config.data.processed_file "${DATA_FILE}"
  --config.data.used_spectra "${USED_SPECTRA}"
  --config.training.snapshot_sampling=false
)

if [[ "${TASK}" == "site" ]]; then
  args+=(
    --config.data.absorbing_edge "xanes_${EDGE}"
    --config.data.site_spectra_field "xanes_${EDGE}"
  )
fi

[[ -n "${N_ITERS:-}" ]] && args+=(--config.training.n_iters "${N_ITERS}")
[[ -n "${BATCH_SIZE:-}" ]] && args+=(--config.training.batch_size "${BATCH_SIZE}")
[[ -n "${EVAL_BATCH_SIZE:-}" ]] && args+=(--config.training.eval_batch_size "${EVAL_BATCH_SIZE}")
[[ -n "${NUM_WORKERS:-}" ]] && args+=(--config.data.num_workers "${NUM_WORKERS}")
[[ -n "${LOG_FREQ:-}" ]] && args+=(--config.training.log_freq "${LOG_FREQ}")
[[ -n "${SNAPSHOT_FREQ:-}" ]] && args+=(--config.training.snapshot_freq "${SNAPSHOT_FREQ}")
[[ -n "${LEARNING_RATE:-}" ]] && args+=(--config.optim.lr "${LEARNING_RATE}")
[[ -n "${WARMUP_STEPS:-}" ]] && args+=(--config.optim.warmup "${WARMUP_STEPS}")

echo "Training DiffSpectra: task=${TASK}, edge=${EDGE}, output=${OUTPUT_DIR}"
cd "${ROOT_DIR}/inverse_model"
exec "${INVERSE_PYTHON}" "${args[@]}"
