#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-config/region_forecast_adrr.yaml}"
PREP_WORKERS="${PREP_WORKERS:-10}"
BUILD_WORKERS="${BUILD_WORKERS:-10}"
NUM_SHARDS="${NUM_SHARDS:-128}"
MAX_FLIGHTS="${MAX_FLIGHTS:-}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export OMP_NUM_THREADS
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS
export NUMEXPR_NUM_THREADS

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_step() {
  log "Running: $*"
  "$@"
}

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

PREP_ARGS=(
  tools/prepare_adrr_month.py
  --config "${CONFIG_PATH}"
  --workers "${PREP_WORKERS}"
  --num-shards "${NUM_SHARDS}"
)

if [[ -n "${MAX_FLIGHTS}" ]]; then
  PREP_ARGS+=(
    --max-flights "${MAX_FLIGHTS}"
  )
fi

log "Starting ADRR pipeline"
run_step "${PYTHON_BIN}" "${PREP_ARGS[@]}"
run_step "${PYTHON_BIN}" build_dataset.py --config "${CONFIG_PATH}" --workers "${BUILD_WORKERS}"
run_step "${PYTHON_BIN}" training.py --config "${CONFIG_PATH}"
run_step "${PYTHON_BIN}" evaluation.py --config "${CONFIG_PATH}"
log "ADRR pipeline finished successfully"
