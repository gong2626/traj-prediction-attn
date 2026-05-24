#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-full}"
mkdir -p logs/external_baselines

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

case "$MODE" in
  smoke)
    CONFIGS=(
      "configs/region_forecast_adrr_traisformer_h3_smoke.yaml"
      "configs/region_forecast_adrr_flightbertpp_h3_smoke.yaml"
    )
    ;;
  subset)
    CONFIGS=(
      "configs/region_forecast_adrr_traisformer_h3_subset.yaml"
      "configs/region_forecast_adrr_flightbertpp_h3_subset.yaml"
    )
    ;;
  full)
    CONFIGS=(
      "configs/region_forecast_adrr_full_traisformer_h3.yaml"
      "configs/region_forecast_adrr_full_flightbertpp_h3.yaml"
    )
    ;;
  *)
    echo "Usage: $0 [smoke|subset|full]" >&2
    exit 2
    ;;
esac

run_one() {
  local config="$1"
  local stem
  stem="$(basename "$config" .yaml)"
  echo "[`date '+%F %T'`] START train+eval $stem"
  python training.py --config "$config" | tee "logs/external_baselines/${stem}_train.log"
  python evaluation.py --config "$config" | tee "logs/external_baselines/${stem}_eval.log"
  echo "[`date '+%F %T'`] DONE train+eval $stem"
}

for config in "${CONFIGS[@]}"; do
  run_one "$config"
done
