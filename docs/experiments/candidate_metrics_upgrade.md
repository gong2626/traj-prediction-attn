# Experiment: Candidate Metrics Upgrade

## Goal

The previous evaluation reported `candidate_union_cover`, but it did not report how large the candidate set was. That made the metric ambiguous:

- a model can improve coverage by returning a much larger area
- a sharper model can reduce coverage under a probability-mass stopping rule by selecting fewer cells

This upgrade adds candidate-set size and fixed-budget coverage metrics so candidate quality can be interpreted academically.

## Code Changes

Files changed:

- `regional_forecast/evaluation/metrics.py`
- `tests/test_candidate_metrics.py`

New metrics:

- `candidate_union_cell_count_mean`
- `candidate_union_area_km2_mean`
- `candidate_union_probability_mass_mean`
- `candidate_budget_cells`
- `candidate_cover_at{k}_cells`
- `candidate_area_at{k}_cells_km2_mean`
- `candidate_cover_at_budget_cells`
- `candidate_area_at_budget_cells_km2_mean`

The old fields are preserved:

- `candidate_union_cover`
- `candidate_top{k}_cover`

Definitions:

- `candidate_union_*` uses the existing probability-mass candidate construction with `candidate_probability_mass` and `candidate_top_cells`.
- `candidate_cover_at{k}_cells` ignores probability mass and measures coverage under a fixed top-`k` valid H3-cell budget.
- `candidate_cover_at_budget_cells` uses the configured per-horizon `candidate_top_cells` as the fixed budget.
- area is computed from actual H3 cell area in square kilometers.

## Verification

```bash
OMP_NUM_THREADS=1 python -m unittest tests.test_candidate_metrics tests.test_subset_sampling tests.test_training_losses tests.test_attention_hierarchical_model tests.test_evaluate_sparse_probs -v
python -m py_compile regional_forecast/evaluation/metrics.py regional_forecast/data/dataset.py regional_forecast/data/subset_sampling.py tools/build_dev_subset_from_artifacts.py
```

Results:

- `15` tests passed
- py_compile passed

## Re-Evaluation Commands

No retraining was needed. Existing devsplit checkpoints were re-evaluated:

```bash
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_devsplit_baseline.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_devsplit_multiscale_tuned.yaml
```

Runtime:

- baseline eval: `11.79s`
- multiscale eval: `11.66s`

## Candidate Metrics on Devsplit

### GRU Baseline

| Horizon | Top5 | Union Cover | Union Cells | Union Area km2 | Union Mass | Cover at Budget Cells | Budget Cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.6990 | 0.9335 | 14.4550 | 179217.7629 | 0.7897 | 0.9750 | 24 |
| 30m | 0.5941 | 0.8666 | 16.4550 | 205615.5574 | 0.6849 | 0.8747 | 18 |
| 60m | 0.8138 | 0.8937 | 8.0594 | 701549.1230 | 0.8211 | 0.9758 | 18 |
| 120m | 0.6706 | 0.8654 | 13.9806 | 1237487.4767 | 0.7839 | 0.9160 | 24 |

### Multiscale Transformer

| Horizon | Top5 | Union Cover | Union Cells | Union Area km2 | Union Mass | Cover at Budget Cells | Budget Cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.7520 | 0.8320 | 6.8595 | 84955.2804 | 0.8326 | 0.9740 | 24 |
| 30m | 0.6299 | 0.7950 | 9.4111 | 116627.6606 | 0.8141 | 0.9007 | 18 |
| 60m | 0.8072 | 0.8189 | 5.0762 | 443974.9111 | 0.8380 | 0.9699 | 18 |
| 120m | 0.7051 | 0.7793 | 7.3143 | 646566.5156 | 0.8280 | 0.9429 | 24 |

### Delta: Multiscale - Baseline

| Horizon | Top5 | Union Cover | Union Cells | Union Area km2 | Union Mass | Cover at Budget Cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | +0.0530 | -0.1015 | -7.5955 | -94262.4825 | +0.0428 | -0.0010 |
| 30m | +0.0358 | -0.0716 | -7.0439 | -88987.8968 | +0.1292 | +0.0260 |
| 60m | -0.0066 | -0.0748 | -2.9831 | -257574.2118 | +0.0169 | -0.0059 |
| 120m | +0.0344 | -0.0861 | -6.6663 | -590920.9611 | +0.0441 | +0.0269 |

## Interpretation

The earlier candidate regression was partly a metric artifact.

Under the probability-mass candidate rule, the multiscale model reaches the target mass with fewer cells:

- `120m` baseline: `13.98` cells, `1.24M km2`
- `120m` multiscale: `7.31` cells, `0.65M km2`

This explains why `candidate_union_cover` drops. The model is sharper and produces smaller candidate sets.

Under a fixed cell budget:

- `30m` improves from `0.8747` to `0.9007`
- `120m` improves from `0.9160` to `0.9429`
- `10m` is essentially tied
- `60m` is slightly lower

So the next H3-aware refinement should not simply expand every candidate set. The more precise goal is:

- preserve the sharper probability distribution
- improve probability-mass candidate coverage at comparable area
- avoid inflating candidate area beyond baseline

## Next Step

The next model-side step should optimize the probability-mass candidate construction:

1. evaluate lightweight H3 neighbor smoothing after logits/probabilities
2. compare coverage at fixed area / fixed cells
3. accept a refinement only if it improves coverage without erasing the top-k, NLL, and ECE gains
