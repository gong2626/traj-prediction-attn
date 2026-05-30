#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-logs/full_routerreg}"
FORCE="${FORCE:-0}"
BUILD_WORKERS="${BUILD_WORKERS:-10}"
DEVICE="${DEVICE:-cuda}"
VAL_SAMPLE_SIZE="${VAL_SAMPLE_SIZE:-50000}"
WEIGHT_STEP="${WEIGHT_STEP:-0.2}"
SEARCH_BATCH_SIZE="${SEARCH_BATCH_SIZE:-2048}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-4096}"
TEMPERATURES="${TEMPERATURES:-0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5}"

CONFIG_PATH="configs/region_forecast_adrr_full_horizon_router_distill_routerreg.yaml"
CHECKPOINT_PATH="artifacts/checkpoints/global_horizon_router_transformer_h3_distill_routerreg.pt"
TRAIN_SUMMARY_PATH="artifacts/checkpoints/global_horizon_router_transformer_h3_distill_routerreg_train_summary.json"
EVAL_SUMMARY_CSV="artifacts/evaluations/global_horizon_router_transformer_h3_distill_routerreg_summary.csv"
EVAL_SUMMARY_JSON="artifacts/evaluations/global_horizon_router_transformer_h3_distill_routerreg_summary.json"
TEACHER_REPORT="artifacts/evaluations/global_horizon_expert_blend_full_v1_report.json"
TUNED_CKPT="artifacts/checkpoints/global_multiscale_transformer_h3_tuned.pt"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_full_routerreg.sh [--force]

Environment overrides:
  PYTHON_BIN=python
  LOG_DIR=logs/full_routerreg
  FORCE=0|1
  BUILD_WORKERS=10
  DEVICE=cuda
  VAL_SAMPLE_SIZE=50000
  WEIGHT_STEP=0.2
  SEARCH_BATCH_SIZE=2048
  TEST_BATCH_SIZE=4096
  TEMPERATURES=0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

trap 'log "Failed at line ${LINENO}. Check logs in ${LOG_DIR}."' ERR

all_exist() {
  local path
  for path in "$@"; do
    [[ -f "${path}" ]] || return 1
  done
  return 0
}

run_step() {
  local step_id="$1"
  local label="$2"
  local log_file="$3"
  shift 3

  log "START ${step_id}: ${label}"
  log "Log: ${log_file}"
  {
    printf '[%s] START %s: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${step_id}" "${label}"
    printf '[%s] Command:' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf ' %q' "$@"
    printf '\n'
    time "$@"
    printf '[%s] DONE %s: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${step_id}" "${label}"
  } 2>&1 | tee -a "${log_file}"
  log "DONE ${step_id}: ${label}"
}

run_step_if_needed() {
  local step_id="$1"
  local label="$2"
  local log_file="$3"
  local outputs_raw="$4"
  shift 4

  IFS=';' read -r -a outputs <<< "${outputs_raw}"
  if [[ "${FORCE}" != "1" ]] && all_exist "${outputs[@]}"; then
    log "SKIP ${step_id}: ${label}"
    for output in "${outputs[@]}"; do
      log "  existing: ${output}"
    done
    return 0
  fi

  run_step "${step_id}" "${label}" "${log_file}" "$@"
}

ensure_dataset() {
  if all_exist artifacts/datasets/train.pt artifacts/datasets/val.pt artifacts/datasets/test.pt; then
    log "SKIP 00: dataset artifacts already exist"
    return 0
  fi

  run_step \
    "00" \
    "build dataset" \
    "${LOG_DIR}/00_build_dataset.log" \
    "${PYTHON_BIN}" build_dataset.py --config "${CONFIG_PATH}" --workers "${BUILD_WORKERS}"
}

preflight() {
  run_step "0a" "py_compile" "${LOG_DIR}/00_preflight.log" \
    "${PYTHON_BIN}" -m py_compile \
      regional_forecast/config.py \
      regional_forecast/pipeline/train.py \
      regional_forecast/pipeline/evaluate.py \
      regional_forecast/models/forecast_model.py \
      tools/horizon_expert_blend_full.py

  if command -v nvidia-smi >/dev/null 2>&1; then
    run_step "0b" "nvidia-smi" "${LOG_DIR}/00_preflight.log" nvidia-smi
  fi
}

ensure_teacher_report() {
  if all_exist "${TEACHER_REPORT}"; then
    log "SKIP 1a: full teacher blend report already exists"
    return 0
  fi

  if ! all_exist \
    artifacts/checkpoints/global_h3_movement_gru.pt \
    "${TUNED_CKPT}" \
    artifacts/checkpoints/global_traisformer_h3_baseline.pt \
    artifacts/checkpoints/global_multiscale_transformer_h3_gated_local_frozen_s6_b3.pt
  then
    cat >&2 <<'EOF'
Missing prerequisite checkpoints for the full teacher blend report.
Run `bash tools/run_full_current_scheme.sh` first, or ensure these files exist:
  - artifacts/checkpoints/global_h3_movement_gru.pt
  - artifacts/checkpoints/global_multiscale_transformer_h3_tuned.pt
  - artifacts/checkpoints/global_traisformer_h3_baseline.pt
  - artifacts/checkpoints/global_multiscale_transformer_h3_gated_local_frozen_s6_b3.pt
EOF
    return 1
  fi

  run_step \
    "1a" \
    "build full teacher blend report" \
    "${LOG_DIR}/01_build_teacher_blend.log" \
    "${PYTHON_BIN}" tools/horizon_expert_blend_full.py \
      --expert gru=configs/region_forecast_adrr_full_hierarchical_gru.yaml \
      --expert tuned=configs/region_forecast_adrr_full_multiscale_tuned.yaml \
      --expert trais=configs/region_forecast_adrr_full_traisformer_h3.yaml \
      --expert gated=configs/region_forecast_adrr_full_multiscale_gated_local_frozen_s6_b3_eval_parentk_temp.yaml \
      --device "${DEVICE}" \
      --val-sample-size "${VAL_SAMPLE_SIZE}" \
      --weight-step "${WEIGHT_STEP}" \
      --temperatures "${TEMPERATURES}" \
      --search-batch-size "${SEARCH_BATCH_SIZE}" \
      --test-batch-size "${TEST_BATCH_SIZE}" \
      --output-prefix artifacts/evaluations/global_horizon_expert_blend_full_v1
}

train_model() {
  if ! all_exist "${TUNED_CKPT}"; then
    cat >&2 <<'EOF'
Missing tuned checkpoint required for warm start:
  artifacts/checkpoints/global_multiscale_transformer_h3_tuned.pt
Run `bash tools/run_full_current_scheme.sh` first, or place the checkpoint at that path.
EOF
    return 1
  fi

  local train_args=("${PYTHON_BIN}" training.py --config "${CONFIG_PATH}")
  if [[ "${FORCE}" == "1" ]]; then
    train_args+=(--restart)
  fi

  run_step_if_needed \
    "2" \
    "train full horizon-router routerreg model" \
    "${LOG_DIR}/02_train_full_routerreg.log" \
    "${CHECKPOINT_PATH};${TRAIN_SUMMARY_PATH}" \
    "${train_args[@]}"
}

evaluate_model() {
  run_step_if_needed \
    "3" \
    "evaluate full horizon-router routerreg model" \
    "${LOG_DIR}/03_eval_full_routerreg.log" \
    "${EVAL_SUMMARY_CSV};${EVAL_SUMMARY_JSON}" \
    "${PYTHON_BIN}" evaluation.py --config "${CONFIG_PATH}"
}

main() {
  log "Full routerreg pipeline started"
  log "ROOT_DIR=${ROOT_DIR}"
  log "LOG_DIR=${LOG_DIR}"
  log "FORCE=${FORCE}"
  log "DEVICE=${DEVICE}"

  ensure_dataset
  preflight
  ensure_teacher_report
  train_model
  evaluate_model

  log "Full routerreg pipeline finished successfully"
  log "Key outputs:"
  log "  checkpoint: ${CHECKPOINT_PATH}"
  log "  train summary: ${TRAIN_SUMMARY_PATH}"
  log "  eval summary: ${EVAL_SUMMARY_JSON}"
}

main "$@"
