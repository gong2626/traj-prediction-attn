#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-logs/full_current_scheme}"
FORCE="${FORCE:-0}"
BUILD_WORKERS="${BUILD_WORKERS:-10}"
VAL_SAMPLE_SIZE="${VAL_SAMPLE_SIZE:-50000}"
WEIGHT_STEP="${WEIGHT_STEP:-0.2}"
SEARCH_BATCH_SIZE="${SEARCH_BATCH_SIZE:-2048}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-4096}"
DEVICE="${DEVICE:-cuda}"
TEMPERATURES="${TEMPERATURES:-0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5}"

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
  bash tools/run_full_current_scheme.sh [--force]

Environment overrides:
  PYTHON_BIN=python
  LOG_DIR=logs/full_current_scheme
  FORCE=0|1
  BUILD_WORKERS=10
  VAL_SAMPLE_SIZE=50000
  WEIGHT_STEP=0.2
  SEARCH_BATCH_SIZE=2048
  TEST_BATCH_SIZE=4096
  DEVICE=cuda
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

run_parallel_group_if_needed() {
  local group_id="$1"
  shift

  log "START ${group_id}"
  local pids=()
  local pid
  local fn
  for fn in "$@"; do
    (
      set -Eeuo pipefail
      "${fn}"
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
      for other_pid in "${pids[@]}"; do
        if [[ "${other_pid}" != "${pid}" ]]; then
          kill "${other_pid}" 2>/dev/null || true
        fi
      done
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    log "FAIL ${group_id}"
    return 1
  fi

  log "DONE ${group_id}"
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
    "${PYTHON_BIN}" build_dataset.py --config configs/region_forecast_adrr_full_multiscale_tuned.yaml --workers "${BUILD_WORKERS}"
}

preflight() {
  run_step "0a" "py_compile" "${LOG_DIR}/00_preflight.log" \
    "${PYTHON_BIN}" -m py_compile \
      regional_forecast/config.py \
      regional_forecast/pipeline/train.py \
      regional_forecast/pipeline/evaluate.py \
      regional_forecast/evaluation/metrics.py \
      regional_forecast/evaluation/spatial_refinement.py \
      regional_forecast/models/forecast_model.py \
      tools/horizon_expert_blend_full.py

  run_step "0b" "dataset sanity check" "${LOG_DIR}/00_preflight.log" \
    "${PYTHON_BIN}" - <<'PY'
import torch

for split in ["train", "val", "test"]:
    path = f"artifacts/datasets/{split}.pt"
    obj = torch.load(path, map_location="cpu", weights_only=False)
    actual = int(obj.get("num_samples", -1))
    chunk_count = len(obj.get("chunks", []))
    print(split, "samples=", actual, "chunks=", chunk_count)
    if actual <= 0:
        raise SystemExit(f"{split} has no samples")
    if chunk_count <= 0:
        raise SystemExit(f"{split} has no chunks")
PY

  if command -v nvidia-smi >/dev/null 2>&1; then
    run_step "0c" "nvidia-smi" "${LOG_DIR}/00_preflight.log" nvidia-smi
  fi
}

step_train_full_hierarchical_gru() {
  run_step_if_needed \
    "1" \
    "train full hierarchical GRU baseline" \
    "${LOG_DIR}/01_train_full_gru.log" \
    "artifacts/checkpoints/global_h3_movement_gru.pt;artifacts/checkpoints/global_h3_movement_gru_train_summary.json" \
    "${PYTHON_BIN}" training.py --config configs/region_forecast_adrr_full_hierarchical_gru.yaml
}

step_train_full_multiscale_tuned() {
  run_step_if_needed \
    "2" \
    "train full tuned multiscale model" \
    "${LOG_DIR}/02_train_full_multiscale_tuned.log" \
    "artifacts/checkpoints/global_multiscale_transformer_h3_tuned.pt;artifacts/checkpoints/global_multiscale_transformer_h3_tuned_train_summary.json" \
    "${PYTHON_BIN}" training.py --config configs/region_forecast_adrr_full_multiscale_tuned.yaml
}

step_train_full_traisformer() {
  run_step_if_needed \
    "3" \
    "train full TrAISformer-H3 baseline" \
    "${LOG_DIR}/03_train_full_traisformer.log" \
    "artifacts/checkpoints/global_traisformer_h3_baseline.pt;artifacts/checkpoints/global_traisformer_h3_baseline_train_summary.json" \
    "${PYTHON_BIN}" training.py --config configs/region_forecast_adrr_full_traisformer_h3.yaml
}

step_train_full_flightbertpp() {
  run_step_if_needed \
    "4" \
    "train full FlightBERT++-H3 baseline" \
    "${LOG_DIR}/04_train_full_flightbertpp.log" \
    "artifacts/checkpoints/global_flightbertpp_h3_baseline.pt;artifacts/checkpoints/global_flightbertpp_h3_baseline_train_summary.json" \
    "${PYTHON_BIN}" training.py --config configs/region_forecast_adrr_full_flightbertpp_h3.yaml
}

step_train_full_gated_local() {
  run_step_if_needed \
    "5" \
    "train full gated-local challenger" \
    "${LOG_DIR}/05_train_full_gated_local.log" \
    "artifacts/checkpoints/global_multiscale_transformer_h3_gated_local_frozen_s6_b3.pt;artifacts/checkpoints/global_multiscale_transformer_h3_gated_local_frozen_s6_b3_train_summary.json" \
    "${PYTHON_BIN}" training.py --config configs/region_forecast_adrr_full_multiscale_gated_local_frozen_s6_b3.yaml
}

phase1_train_tuned_vs_traisformer() {
  run_parallel_group_if_needed \
    "phase1_train_tuned_vs_traisformer" \
    step_train_full_multiscale_tuned \
    step_train_full_traisformer
}

phase2_train_gru_vs_flightbertpp() {
  run_parallel_group_if_needed \
    "phase2_train_gru_vs_flightbertpp" \
    step_train_full_hierarchical_gru \
    step_train_full_flightbertpp
}

phase3_train_gated_local() {
  step_train_full_gated_local
}

main() {
  log "Full current-scheme pipeline started"
  log "ROOT_DIR=${ROOT_DIR}"
  log "LOG_DIR=${LOG_DIR}"
  log "FORCE=${FORCE}"
  log "DEVICE=${DEVICE}"

  ensure_dataset
  preflight

  phase1_train_tuned_vs_traisformer
  phase2_train_gru_vs_flightbertpp
  phase3_train_gated_local

  run_step_if_needed \
    "6a" \
    "evaluate full GRU baseline" \
    "${LOG_DIR}/06a_eval_full_gru.log" \
    "artifacts/evaluations/global_h3_movement_gru_summary.csv;artifacts/evaluations/global_h3_movement_gru_summary.json" \
    "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_hierarchical_gru.yaml

  run_step_if_needed \
    "6b" \
    "evaluate full tuned multiscale model" \
    "${LOG_DIR}/06b_eval_full_multiscale_tuned.log" \
    "artifacts/evaluations/global_multiscale_transformer_h3_tuned_summary.csv;artifacts/evaluations/global_multiscale_transformer_h3_tuned_summary.json" \
    "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_multiscale_tuned.yaml

  run_step_if_needed \
    "6c" \
    "evaluate full TrAISformer-H3 baseline" \
    "${LOG_DIR}/06c_eval_full_traisformer.log" \
    "artifacts/evaluations/global_traisformer_h3_baseline_summary.csv;artifacts/evaluations/global_traisformer_h3_baseline_summary.json" \
    "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_traisformer_h3.yaml

  run_step_if_needed \
    "6d" \
    "evaluate full FlightBERT++-H3 baseline" \
    "${LOG_DIR}/06d_eval_full_flightbertpp.log" \
    "artifacts/evaluations/global_flightbertpp_h3_baseline_summary.csv;artifacts/evaluations/global_flightbertpp_h3_baseline_summary.json" \
    "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_flightbertpp_h3.yaml

  run_step_if_needed \
    "6e" \
    "evaluate full gated-local challenger" \
    "${LOG_DIR}/06e_eval_full_gated_local.log" \
    "artifacts/evaluations/global_multiscale_transformer_h3_gated_local_frozen_s6_b3_eval_parentk_temp_summary.csv;artifacts/evaluations/global_multiscale_transformer_h3_gated_local_frozen_s6_b3_eval_parentk_temp_summary.json" \
    "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_multiscale_gated_local_frozen_s6_b3_eval_parentk_temp.yaml

  run_step_if_needed \
    "7" \
    "full horizon-aware expert blend" \
    "${LOG_DIR}/07_full_horizon_expert_blend.log" \
    "artifacts/evaluations/global_horizon_expert_blend_full_v1_report.json;artifacts/evaluations/global_horizon_expert_blend_full_v1_summary.csv;artifacts/evaluations/global_horizon_expert_blend_full_v1_summary.json" \
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

  log "Full current-scheme pipeline finished successfully"
  log "Key outputs:"
  find artifacts/evaluations -maxdepth 1 -type f -name 'global_*summary.csv' | sort
  find artifacts/evaluations -maxdepth 1 -type f -name 'global_horizon_expert_blend_full_v1_*' | sort
}

main "$@"
