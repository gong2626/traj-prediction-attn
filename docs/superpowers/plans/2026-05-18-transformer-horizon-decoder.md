# Transformer Horizon Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `transformer_horizon_decoder` model and run a fair 60-minute-history comparison against a matching hierarchical GRU baseline.

**Architecture:** Reuse the existing hierarchical H3 coarse-to-fine output structure and round-2 horizon-query decoder, but replace the GRU sequence encoder with a lightweight Transformer encoder plus positional encoding and masked pooling. Keep all train/evaluate/predict and artifact interfaces compatible with existing runs.

**Tech Stack:** Python 3.12, PyTorch 2.8, YAML configs, unittest, existing ADRR dataset builders and evaluation pipeline.

---

### Task 1: Add failing tests for the Transformer horizon decoder

**Files:**
- Modify: `tests/test_attention_hierarchical_model.py`
- Test: `tests/test_attention_hierarchical_model.py`

- [ ] **Step 1: Write the failing test**

Add a new import and test case for `TransformerHorizonDecoderHierarchicalH3RegionalForecaster` that verifies:

- forward pass returns `coarse_logits`, `status_logits`, `decoder_features`, `attention_weights`
- shapes match the existing hierarchical variants
- `child_logits()` still works with per-horizon features

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_attention_hierarchical_model -v`
Expected: FAIL with `ImportError` or missing class for `TransformerHorizonDecoderHierarchicalH3RegionalForecaster`

- [ ] **Step 3: Commit**

```bash
git add tests/test_attention_hierarchical_model.py
git commit -m "test: add transformer horizon decoder model coverage"
```

### Task 2: Implement the Transformer horizon decoder

**Files:**
- Modify: `regional_forecast/config.py`
- Modify: `regional_forecast/models/forecast_model.py`
- Modify: `regional_forecast/models/__init__.py`
- Test: `tests/test_attention_hierarchical_model.py`

- [ ] **Step 1: Add model config fields**

Add transformer config fields to `ModelConfig`:

- `transformer_num_layers: int = 2`
- `transformer_num_heads: int = 4`
- `transformer_ffn_dim: int = 512`
- `transformer_dropout: float = 0.1`

- [ ] **Step 2: Implement minimal model**

In `regional_forecast/models/forecast_model.py`, add:

- a positional encoding helper
- `TransformerHorizonDecoderHierarchicalH3RegionalForecaster`

The class should:

- reuse the same input embeddings/projections as round 2
- run `nn.TransformerEncoder` with `batch_first=True`
- use the temporal mask for Transformer padding
- build a masked mean-pooled summary plus context features
- reuse horizon query attention and hierarchical heads

- [ ] **Step 3: Export the model**

Add the new class to `regional_forecast/models/__init__.py`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_attention_hierarchical_model -v`
Expected: PASS

- [ ] **Step 5: Run compile verification**

Run: `python -m py_compile regional_forecast/models/forecast_model.py regional_forecast/models/__init__.py regional_forecast/config.py`
Expected: no output

- [ ] **Step 6: Commit**

```bash
git add regional_forecast/config.py regional_forecast/models/forecast_model.py regional_forecast/models/__init__.py tests/test_attention_hierarchical_model.py
git commit -m "feat: add transformer horizon decoder model"
```

### Task 3: Wire the model into train, evaluate, and predict

**Files:**
- Modify: `regional_forecast/pipeline/train.py`
- Modify: `regional_forecast/pipeline/evaluate.py`
- Modify: `regional_forecast/pipeline/predict.py`
- Test: `tests/test_evaluate_sparse_probs.py`

- [ ] **Step 1: Add model factory support in training**

Update `_build_model()` in `regional_forecast/pipeline/train.py` to support `model_type == "transformer_horizon_decoder"`

- [ ] **Step 2: Add checkpoint loading support in evaluation**

Update `_load_model()` and hierarchical model dispatch sets in `regional_forecast/pipeline/evaluate.py`

- [ ] **Step 3: Add checkpoint loading support in prediction**

Update `_load_model_bundle()` in `regional_forecast/pipeline/predict.py`

- [ ] **Step 4: Run regression tests**

Run: `python -m unittest tests.test_attention_hierarchical_model tests.test_evaluate_sparse_probs -v`
Expected: PASS

- [ ] **Step 5: Run compile verification**

Run: `python -m py_compile regional_forecast/pipeline/train.py regional_forecast/pipeline/evaluate.py regional_forecast/pipeline/predict.py`
Expected: no output

- [ ] **Step 6: Commit**

```bash
git add regional_forecast/pipeline/train.py regional_forecast/pipeline/evaluate.py regional_forecast/pipeline/predict.py
git commit -m "feat: wire transformer horizon decoder into pipelines"
```

### Task 4: Create fair-history configs

**Files:**
- Create: `configs/region_forecast_adrr_history60_baseline_smoke.yaml`
- Create: `configs/region_forecast_adrr_history60_baseline_subset.yaml`
- Create: `configs/region_forecast_adrr_transformer_horizon_smoke.yaml`
- Create: `configs/region_forecast_adrr_transformer_horizon_subset.yaml`

- [ ] **Step 1: Add 60-minute baseline configs**

Create smoke and subset baseline configs with:

- `model_type: hierarchical_gru`
- `history_minutes: 60`
- smoke `max_trajectories: 32`
- subset `max_trajectories: 256`

- [ ] **Step 2: Add Transformer configs**

Create smoke and subset Transformer configs with:

- `model_type: transformer_horizon_decoder`
- `history_minutes: 60`
- the same training/evaluation settings as the fair-history baseline where practical

- [ ] **Step 3: Sanity check config parsing**

Run:

```bash
python - <<'PY'
from regional_forecast.config import load_config
for path in [
    "configs/region_forecast_adrr_history60_baseline_smoke.yaml",
    "configs/region_forecast_adrr_history60_baseline_subset.yaml",
    "configs/region_forecast_adrr_transformer_horizon_smoke.yaml",
    "configs/region_forecast_adrr_transformer_horizon_subset.yaml",
]:
    cfg = load_config(path)
    print(path, cfg.model.model_type, cfg.dataset.history_minutes)
PY
```

Expected: each config prints successfully with `history_minutes` equal to `60`

- [ ] **Step 4: Commit**

```bash
git add configs/region_forecast_adrr_history60_baseline_smoke.yaml configs/region_forecast_adrr_history60_baseline_subset.yaml configs/region_forecast_adrr_transformer_horizon_smoke.yaml configs/region_forecast_adrr_transformer_horizon_subset.yaml
git commit -m "chore: add round 3 fair-history configs"
```

### Task 5: Run smoke experiments

**Files:**
- Uses existing configs and artifact directories

- [ ] **Step 1: Run fair-history baseline smoke build**

Run: `python build_dataset.py --config configs/region_forecast_adrr_history60_baseline_smoke.yaml --workers 8`
Expected: dataset artifacts under `artifacts/research_history60_baseline_smoke/`

- [ ] **Step 2: Run fair-history baseline smoke training and evaluation**

Run:

```bash
python training.py --config configs/region_forecast_adrr_history60_baseline_smoke.yaml
python evaluation.py --config configs/region_forecast_adrr_history60_baseline_smoke.yaml
```

Expected: checkpoint and evaluation summary files exist

- [ ] **Step 3: Run Transformer smoke build**

Run: `python build_dataset.py --config configs/region_forecast_adrr_transformer_horizon_smoke.yaml --workers 8`
Expected: dataset artifacts under `artifacts/research_transformer_horizon_smoke/`

- [ ] **Step 4: Run Transformer smoke training and evaluation**

Run:

```bash
python training.py --config configs/region_forecast_adrr_transformer_horizon_smoke.yaml
python evaluation.py --config configs/region_forecast_adrr_transformer_horizon_smoke.yaml
```

Expected: checkpoint and evaluation summary files exist

### Task 6: Run subset experiments

**Files:**
- Uses existing configs and artifact directories

- [ ] **Step 1: Run fair-history baseline subset build**

Run: `python build_dataset.py --config configs/region_forecast_adrr_history60_baseline_subset.yaml --workers 8`

- [ ] **Step 2: Run fair-history baseline subset training and evaluation**

Run:

```bash
python training.py --config configs/region_forecast_adrr_history60_baseline_subset.yaml
python evaluation.py --config configs/region_forecast_adrr_history60_baseline_subset.yaml
```

- [ ] **Step 3: Run Transformer subset build**

Run: `python build_dataset.py --config configs/region_forecast_adrr_transformer_horizon_subset.yaml --workers 8`

- [ ] **Step 4: Run Transformer subset training and evaluation**

Run:

```bash
python training.py --config configs/region_forecast_adrr_transformer_horizon_subset.yaml
python evaluation.py --config configs/region_forecast_adrr_transformer_horizon_subset.yaml
```

- [ ] **Step 5: Benchmark inference time**

Run a timing script on the subset test split for both history-60 runs and report:

- mean wall time
- ms/sample
- samples/sec

### Task 7: Update research documentation

**Files:**
- Modify: `docs/method.md`
- Modify: `docs/ablation.md`
- Modify: `docs/paper_outline.md`
- Create: `docs/experiments/transformer_horizon_decoder_round3.md`
- Modify: `docs/research_plan.md`

- [ ] **Step 1: Record round-3 experiment log**

Create `docs/experiments/transformer_horizon_decoder_round3.md` with:

- configs
- git diff summary
- runtime
- metrics
- deltas vs fair-history baseline
- deltas vs round-2 horizon decoder where useful
- problems observed
- next-step recommendation

- [ ] **Step 2: Update method description**

Extend `docs/method.md` to describe the Transformer encoder variant and its relation to round 2

- [ ] **Step 3: Update ablation notes**

Add the round-3 model as the next ablation stage

- [ ] **Step 4: Update paper outline**

Shift the working title and main-method section toward Transformer horizon decoding if the results support it

- [ ] **Step 5: Update research plan status**

Mark the newly completed branch and note whether round 4 should prioritize:

- multiscale temporal modeling
- H3 graph refinement
- consistency losses

### Task 8: Final verification

**Files:**
- No new files beyond the above

- [ ] **Step 1: Run focused regression tests**

Run: `python -m unittest tests.test_attention_hierarchical_model tests.test_evaluate_sparse_probs -v`

- [ ] **Step 2: Run compile verification**

Run: `python -m py_compile regional_forecast/models/forecast_model.py regional_forecast/models/__init__.py regional_forecast/pipeline/train.py regional_forecast/pipeline/evaluate.py regional_forecast/pipeline/predict.py`

- [ ] **Step 3: Inspect git status**

Run: `git status --short`
Expected: only intended tracked modifications and newly added configs/docs/tests
