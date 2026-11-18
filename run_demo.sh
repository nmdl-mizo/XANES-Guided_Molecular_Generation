#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/demo}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
SAMPLING_STEPS="${SAMPLING_STEPS:-1000}"
DEVICE="${DEVICE:-auto}"
INVERSE_PYTHON="${INVERSE_PYTHON:-python}"
FORWARD_PYTHON="${FORWARD_PYTHON:-python}"
FORWARD_EXTRA_PYTHONPATH="${FORWARD_EXTRA_PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

cd "${ROOT_DIR}/inverse_model"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/diffspectra-matplotlib}" \
"${INVERSE_PYTHON}" main.py \
  --config configs/diffspectra_qm9s.py \
  --mode eval \
  --workdir "${ROOT_DIR}/pretrained/inverse" \
  --eval_folder "${OUTPUT_DIR}/inverse" \
  --minimal_condition_eval \
  --preserve_eval_order \
  --config.data.processed_file data_qm9_xanes.pt \
  --config.data.spectra_version all_spectra \
  --config.data.used_spectra "('xanes_CK', 'xanes_NK', 'xanes_OK')" \
  --config.eval.ckpts "12" \
  --config.eval.num_samples "${NUM_SAMPLES}" \
  --config.eval.batch_size "${NUM_SAMPLES}" \
  --config.eval.use_full_dataset true \
  --config.eval.save_mols true \
  --config.sampling.steps "${SAMPLING_STEPS}" \
  --config.sampling_nums 1

RDMOL_PKL="${OUTPUT_DIR}/inverse/molecules_ckpt_12/sample_rdmols_3d.pkl"
if [[ ! -f "${RDMOL_PKL}" ]]; then
  echo "Inverse-model output was not created: ${RDMOL_PKL}" >&2
  exit 1
fi

cd "${ROOT_DIR}/forward_model"
DEVICE_ARGS=()
if [[ "${DEVICE}" != "auto" ]]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi
PYTHONPATH="src:mace-layer${FORWARD_EXTRA_PYTHONPATH:+:${FORWARD_EXTRA_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}" \
"${FORWARD_PYTHON}" predict_rdmols.py \
  "${RDMOL_PKL}" \
  "${OUTPUT_DIR}/forward" \
  --models_dir "${ROOT_DIR}/pretrained/forward" \
  --mode both \
  --edges CK NK OK \
  "${DEVICE_ARGS[@]}"

echo "End-to-end inference completed: ${OUTPUT_DIR}"
