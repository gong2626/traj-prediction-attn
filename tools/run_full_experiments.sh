#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-logs/full_experiments}"
FORCE="${FORCE:-0}"
RUN_STRUCTURAL_ABLATION="${RUN_STRUCTURAL_ABLATION:-1}"
RUN_H3_SWEEP="${RUN_H3_SWEEP:-1}"
RUN_REACHABILITY_ABLATION="${RUN_REACHABILITY_ABLATION:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_full_experiments.sh [--force] [--no-structural-ablation] [--no-h3-sweep] [--no-reachability-ablation]

Environment overrides:
  PYTHON_BIN=python
  LOG_DIR=logs/full_experiments
  FORCE=0|1
  RUN_STRUCTURAL_ABLATION=0|1
  RUN_H3_SWEEP=0|1
  RUN_REACHABILITY_ABLATION=0|1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --no-structural-ablation)
      RUN_STRUCTURAL_ABLATION=0
      shift
      ;;
    --no-h3-sweep)
      RUN_H3_SWEEP=0
      shift
      ;;
    --no-reachability-ablation)
      RUN_REACHABILITY_ABLATION=0
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

preflight() {
  run_step "0a" "py_compile" "${LOG_DIR}/00_preflight.log" \
    "${PYTHON_BIN}" -m py_compile \
      regional_forecast/config.py \
      regional_forecast/pipeline/train.py \
      regional_forecast/pipeline/evaluate.py \
      regional_forecast/evaluation/metrics.py \
      regional_forecast/evaluation/spatial_refinement.py \
      regional_forecast/models/forecast_model.py

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

main() {
  log "Full experiment pipeline started"
  log "ROOT_DIR=${ROOT_DIR}"
  log "LOG_DIR=${LOG_DIR}"
  log "FORCE=${FORCE}"

  preflight

  run_step \
    "1" \
    "refresh existing full GRU baseline evaluation" \
    "${LOG_DIR}/01_eval_full_gru_baseline.log" \
    "${PYTHON_BIN}" evaluation.py --config config/region_forecast_adrr.yaml

  run_step_if_needed \
    "2" \
    "train full final multiscale model" \
    "${LOG_DIR}/02_train_full_multiscale_tuned.log" \
    "artifacts/checkpoints/global_multiscale_transformer_h3_tuned.pt;artifacts/checkpoints/global_multiscale_transformer_h3_tuned_train_summary.json" \
    "${PYTHON_BIN}" training.py --config configs/region_forecast_adrr_full_multiscale_tuned.yaml

  run_step_if_needed \
    "3" \
    "evaluate full final multiscale model" \
    "${LOG_DIR}/03_eval_full_multiscale_tuned.log" \
    "artifacts/evaluations/global_multiscale_transformer_h3_tuned_summary.csv;artifacts/evaluations/global_multiscale_transformer_h3_tuned_summary.json" \
    "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_multiscale_tuned.yaml

  if [[ "${RUN_H3_SWEEP}" == "1" ]]; then
    run_step_if_needed \
      "4a" \
      "evaluate H3 refinement alpha 0.05" \
      "${LOG_DIR}/04a_eval_full_multiscale_h3_refine_a005.log" \
      "artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a005_summary.csv;artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a005_summary.json" \
      "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_multiscale_eval_h3_refine_a005.yaml

    run_step_if_needed \
      "4b" \
      "evaluate H3 refinement alpha 0.10" \
      "${LOG_DIR}/04b_eval_full_multiscale_h3_refine_a010.log" \
      "artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a010_summary.csv;artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a010_summary.json" \
      "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_multiscale_eval_h3_refine_a010.yaml

    run_step_if_needed \
      "4c" \
      "evaluate H3 refinement alpha 0.20" \
      "${LOG_DIR}/04c_eval_full_multiscale_h3_refine_a020.log" \
      "artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a020_summary.csv;artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a020_summary.json" \
      "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_multiscale_eval_h3_refine_a020.yaml
  else
    log "SKIP 4: H3 sweep disabled"
  fi

  if [[ "${RUN_REACHABILITY_ABLATION}" == "1" ]]; then
    run_step_if_needed \
      "5a" \
      "evaluate no reachability" \
      "${LOG_DIR}/05a_eval_full_multiscale_no_reachability.log" \
      "artifacts/evaluations/global_multiscale_transformer_h3_tuned_no_reachability_summary.csv;artifacts/evaluations/global_multiscale_transformer_h3_tuned_no_reachability_summary.json" \
      "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_multiscale_eval_no_reachability.yaml

    run_step_if_needed \
      "5b" \
      "evaluate H3 refinement alpha 0.10 without reachability" \
      "${LOG_DIR}/05b_eval_full_multiscale_h3_refine_a010_no_reachability.log" \
      "artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a010_no_reachability_summary.csv;artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a010_no_reachability_summary.json" \
      "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_multiscale_eval_h3_refine_a010_no_reachability.yaml
  else
    log "SKIP 5: reachability ablation disabled"
  fi

  if [[ "${RUN_STRUCTURAL_ABLATION}" == "1" ]]; then
    run_step_if_needed \
      "6" \
      "train full transformer horizon structural ablation" \
      "${LOG_DIR}/06_train_full_transformer_horizon_ablation.log" \
      "artifacts/checkpoints/global_transformer_horizon_decoder_ablation.pt;artifacts/checkpoints/global_transformer_horizon_decoder_ablation_train_summary.json" \
      "${PYTHON_BIN}" training.py --config configs/region_forecast_adrr_full_transformer_horizon_ablation.yaml

    run_step_if_needed \
      "7" \
      "evaluate full transformer horizon structural ablation" \
      "${LOG_DIR}/07_eval_full_transformer_horizon_ablation.log" \
      "artifacts/evaluations/global_transformer_horizon_decoder_ablation_summary.csv;artifacts/evaluations/global_transformer_horizon_decoder_ablation_summary.json" \
      "${PYTHON_BIN}" evaluation.py --config configs/region_forecast_adrr_full_transformer_horizon_ablation.yaml
  else
    log "SKIP 6-7: structural ablation disabled"
  fi

  log "Full experiment pipeline finished successfully"
  log "Key outputs:"
  find artifacts/evaluations -maxdepth 1 -type f -name '*summary.csv' | sort
}

main "$@"
