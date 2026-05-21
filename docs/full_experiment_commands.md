# Full Experiment Run Commands

Run from `/root/autodl-tmp`. Start the next command only after the previous log reports completion.

## One-Command Script

The same sequence is available as a resumable script:

```bash
cd /root/autodl-tmp
nohup bash tools/run_full_experiments.sh > logs/full_experiments_driver.log 2>&1 &
echo $!
tail -f logs/full_experiments_driver.log
```

Behavior:

- refreshes the existing full GRU baseline evaluation
- trains the full final multiscale model if its checkpoint and train summary do not already exist
- evaluates final, H3 refinement, reachability, and structural ablation runs in order
- skips completed final/ablation outputs by default
- resumes interrupted training through the config `auto_resume: true`
- uses `epochs: 1000` for full training configs, with early stopping controlling the actual stopping point

Useful variants:

```bash
bash tools/run_full_experiments.sh --force
bash tools/run_full_experiments.sh --no-structural-ablation
bash tools/run_full_experiments.sh --no-h3-sweep
bash tools/run_full_experiments.sh --no-reachability-ablation
```

## 0. Preflight

```bash
cd /root/autodl-tmp
mkdir -p logs
export OMP_NUM_THREADS=1

python -m py_compile \
  regional_forecast/config.py \
  regional_forecast/pipeline/train.py \
  regional_forecast/pipeline/evaluate.py \
  regional_forecast/evaluation/metrics.py \
  regional_forecast/evaluation/spatial_refinement.py \
  regional_forecast/models/forecast_model.py

python - <<'PY'
import torch
for split in ["train", "val", "test"]:
    obj = torch.load(f"artifacts/datasets/{split}.pt", map_location="cpu", weights_only=False)
    print(split, "samples=", obj.get("num_samples"), "chunks=", len(obj.get("chunks", [])))
PY

nvidia-smi
```

Expected dataset samples in the current full artifact:

- train: `14505637`
- val: `3108933`
- test: `3098032`

## 1. Refresh Existing Full GRU Baseline Evaluation

This does not retrain. It re-evaluates the existing `global_h3_movement_gru.pt` with the upgraded metrics.

```bash
nohup bash -lc 'cd /root/autodl-tmp && export OMP_NUM_THREADS=1 && python evaluation.py --config config/region_forecast_adrr.yaml' \
  > logs/eval_full_gru_baseline.log 2>&1 &
echo $!
tail -f logs/eval_full_gru_baseline.log
```

Output:

- `artifacts/evaluations/global_h3_movement_gru_summary.csv`

## 2. Train Full Final Model

This is the main full-data training run.

The configured `epochs` is intentionally high (`1000`). Early stopping uses `min_epochs: 8`, `early_stopping_patience: 8`, and `early_stopping_min_delta: 0.0003`.

```bash
nohup bash -lc 'cd /root/autodl-tmp && export OMP_NUM_THREADS=1 && python training.py --config configs/region_forecast_adrr_full_multiscale_tuned.yaml' \
  > logs/train_full_multiscale_tuned.log 2>&1 &
echo $!
tail -f logs/train_full_multiscale_tuned.log
```

If interrupted, rerun the same command. The config has `auto_resume: true`.

Outputs:

- `artifacts/checkpoints/global_multiscale_transformer_h3_tuned.pt`
- `artifacts/checkpoints/global_multiscale_transformer_h3_tuned_last.pt`
- `artifacts/checkpoints/global_multiscale_transformer_h3_tuned_train_summary.json`

## 3. Evaluate Full Final Model

Base evaluation: reachability on, no H3 refinement.

```bash
nohup bash -lc 'cd /root/autodl-tmp && export OMP_NUM_THREADS=1 && python evaluation.py --config configs/region_forecast_adrr_full_multiscale_tuned.yaml' \
  > logs/eval_full_multiscale_tuned.log 2>&1 &
echo $!
tail -f logs/eval_full_multiscale_tuned.log
```

Output:

- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_summary.csv`

## 4. H3 Refinement Evaluation-Only Ablations

These reuse `global_multiscale_transformer_h3_tuned.pt`; no retraining.

```bash
nohup bash -lc 'cd /root/autodl-tmp && export OMP_NUM_THREADS=1 && python evaluation.py --config configs/region_forecast_adrr_full_multiscale_eval_h3_refine_a005.yaml' \
  > logs/eval_full_multiscale_h3_refine_a005.log 2>&1 &
echo $!
tail -f logs/eval_full_multiscale_h3_refine_a005.log
```

```bash
nohup bash -lc 'cd /root/autodl-tmp && export OMP_NUM_THREADS=1 && python evaluation.py --config configs/region_forecast_adrr_full_multiscale_eval_h3_refine_a010.yaml' \
  > logs/eval_full_multiscale_h3_refine_a010.log 2>&1 &
echo $!
tail -f logs/eval_full_multiscale_h3_refine_a010.log
```

```bash
nohup bash -lc 'cd /root/autodl-tmp && export OMP_NUM_THREADS=1 && python evaluation.py --config configs/region_forecast_adrr_full_multiscale_eval_h3_refine_a020.yaml' \
  > logs/eval_full_multiscale_h3_refine_a020.log 2>&1 &
echo $!
tail -f logs/eval_full_multiscale_h3_refine_a020.log
```

Outputs:

- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a005_summary.csv`
- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a010_summary.csv`
- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a020_summary.csv`

## 5. Reachability Filter Ablations

These also reuse the final checkpoint.

```bash
nohup bash -lc 'cd /root/autodl-tmp && export OMP_NUM_THREADS=1 && python evaluation.py --config configs/region_forecast_adrr_full_multiscale_eval_no_reachability.yaml' \
  > logs/eval_full_multiscale_no_reachability.log 2>&1 &
echo $!
tail -f logs/eval_full_multiscale_no_reachability.log
```

```bash
nohup bash -lc 'cd /root/autodl-tmp && export OMP_NUM_THREADS=1 && python evaluation.py --config configs/region_forecast_adrr_full_multiscale_eval_h3_refine_a010_no_reachability.yaml' \
  > logs/eval_full_multiscale_h3_refine_a010_no_reachability.log 2>&1 &
echo $!
tail -f logs/eval_full_multiscale_h3_refine_a010_no_reachability.log
```

Outputs:

- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_no_reachability_summary.csv`
- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a010_no_reachability_summary.csv`

## 6. Train Full Structural Ablation

This is the second full training run in the recommended plan. It removes the multiscale temporal branch but keeps the Transformer horizon decoder family and the same optimization recipe.

This config also uses `epochs: 1000`; early stopping determines the actual runtime.

```bash
nohup bash -lc 'cd /root/autodl-tmp && export OMP_NUM_THREADS=1 && python training.py --config configs/region_forecast_adrr_full_transformer_horizon_ablation.yaml' \
  > logs/train_full_transformer_horizon_ablation.log 2>&1 &
echo $!
tail -f logs/train_full_transformer_horizon_ablation.log
```

If interrupted, rerun the same command. The config has `auto_resume: true`.

Outputs:

- `artifacts/checkpoints/global_transformer_horizon_decoder_ablation.pt`
- `artifacts/checkpoints/global_transformer_horizon_decoder_ablation_last.pt`
- `artifacts/checkpoints/global_transformer_horizon_decoder_ablation_train_summary.json`

## 7. Evaluate Full Structural Ablation

```bash
nohup bash -lc 'cd /root/autodl-tmp && export OMP_NUM_THREADS=1 && python evaluation.py --config configs/region_forecast_adrr_full_transformer_horizon_ablation.yaml' \
  > logs/eval_full_transformer_horizon_ablation.log 2>&1 &
echo $!
tail -f logs/eval_full_transformer_horizon_ablation.log
```

Output:

- `artifacts/evaluations/global_transformer_horizon_decoder_ablation_summary.csv`

## Status Checks

```bash
ps -ef | grep -E 'training.py|evaluation.py' | grep -v grep
nvidia-smi
tail -n 80 logs/train_full_multiscale_tuned.log
tail -n 80 logs/train_full_transformer_horizon_ablation.log
ls -lh artifacts/checkpoints artifacts/evaluations
```
