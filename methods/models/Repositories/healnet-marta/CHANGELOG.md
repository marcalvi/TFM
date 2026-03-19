# Changelog — Adaptations by MartaBuetas for multimodal projects

> 🚧 **Disclaimer:** This is work in progress. There may be bugs 🐛.

All notable changes made in this branch (compared to [konst-int-i/healnet](https://github.com/konst-int-i/healnet/)) are documented below so colleagues can understand, review and reuse the modifications. These changes were implemented to (1) integrate a custom dataset pipeline, (2) support sample-level missing modalities inside a batch, and (3) make the repo easier to use and share. 


```
Version: 0.1.0-martabuetas
🔴 Date: 2025-12-03 
Author: MartaBuetas
Based on: konst-int-i/healnet (NeurIPS 2024)
```

## Summary of top-level changes
- **Add robust per-sample missing-modality support**: collate+pipeline+model changes let single samples in a batch be missing modalities (previously model handled whole-batch missing modalities only).
- **New dataloader Class** (MMDataset) to support CSV-based multimodal inputs, cleaning/imputation, and per-split feature scaling.
- Add **Synthetic dataset** (for quick sanity checks).
- Add a 'monitor' for early-stopping.
- Utils: calibration curve plotting, box plot and KM curves.

## Important compatibility notes
- Censorship encoding (pipeline convention): 1 == censored, 0 == event observed. This convention is preserved across the fork and used by all survival losses here. It is computed as 1 - event_column from the target data in /healnet/etl/loaders.py: `self.censorship.append(1-int(row[self.event_col]))`
- discretized survival bin index `y_disc` (0..K-1): to be specified in the target data. The dataset(s) compute y_disc at data-load time if not provided in the target csv. In the original Healnet repository, it is computed binning the whole set `df["y_disc"] = pd.qcut(df[label_col], q=self.n_bins, labels=False).values`. However, this should be done on the train set, if the bin edges are not previously fixed. CoxPH loss path does not require y_disc, but Cross Entropy or NLL losses do.

## Detailed per-file changes
(Only files that were modified in this fork are listed. The description includes the intent, what changed, and the user-visible effect.)
### 1) healnet/etl/loaders.py
- New or updated:
  - Added `MMDataset`: a configurable CSV-driven dataset that:
    - Reads tabular data modalities (clinical/pathology/radiology CSVs) and a target CSV.
    - Builds modality maps (id -> token tensor).
    - Optionally filters to overlapping modalities (filter_overlap).
    - Creates `y_disc` (discretized survival bins) if missing and saves bin edges / distributions when `log_dir` is provided.  should be created only from the train data!!
    - Implements cleaning & imputation: drops columns with >70% NaNs; fills binary NaNs with 0.5 and numeric NaNs with mean; attempts numeric coercion for other columns.
    - Provides `compute_standardization_from_indices(train_indices)` that computes mean/std only on training indices and applies standardization to dataframes and modality maps (store params in `standardization_params`).
  - Synthetic dataset: `SyntheticMultiModalSurvival` for quick sanity checks and unit tests (returns ([mods], censorship, event_time, y_disc)).


### 2) main.py (pipeline)
- New or updated:
  - Replaced many print statements with structured `logging` calls (logger.info/debug/warning/exception).
  - Collate function (`make_collate_fn`):
    - Accepts per-sample missing modalities (each modality can be None for a sample).
    - Produces: (modalities_batched, masks, censorship, event_time, y_disc), where modalities_batched is a list length M and masks is a list of boolean masks (or None for modality absent entirely).
  - Training/evaluation:
    - Training loop adapted to accept per-sample missingness: builds a `present` boolean list per modality and passes `masks`/`present` into `HealNet`.
    - Safer, single-in-memory best-state checkpointing (store in memory, write once per fold).
- Bug fixes:
  - Deterministic splitting uses torch.Generator seeded per fold. 
- User-visible behavior:
  - The pipeline now supports batches with sample-level missing modalities.
  - To use WandB sweeps, supply --api_key or set env var. Sweep handling is preserved.
  - The collate function expects dataset __getitem__ to return ([mods], censorship, event_time, y_disc) where each mod element may be None.

### 3) healnet/models/healnet.py (model)
- New or updated:
  - Extended HealNet.forward to handle sample-level missing modalities (bucketed processing):
    - For each modality, determine which samples in the batch actually have data (via `present` or `masks`).
    - Run cross-attention only on the present subset, scatter updated latents back to full-batch positions.
    - Fall back gracefully if a modality is absent for the whole batch (this scenario would be the original behaviour).
- User-visible behavior:
  - Models can now be fed batches where some samples lack a modality, without needing to split the batch or make special per-batch calls.

### 4) healnet/models/survival_loss.py
- Minor robustness changes:
  - Ensure weights normalization is performed on the same device and dtype as hazards (device-safe handling).
  - Wrapped some reductions and added small clarifying comments.
- No change in mathematical definitions (nll, ce style loss and Cox wrapper preserved). CE loss uses `y_disc` from dataset; CoxPH path does not require discretization.

## Notes regarding original HealNet repo

1.  y_disc when not available in the dataset, it is binned using the whole dataset (should be done only on the train set if the bi edges are not fixed).
2.  Seed not fixed in the cross validation

## Tests & examples added or recommended
- Added `SyntheticMultiModalSurvival` dataset for quick sanity checks and to enable unit tests that exercise missing-modality behavior.
- Sanity-check mode in `main.py` (mode == "sanity_check") uses SyntheticMultiModalSurvival to run a short training loop and compute simple metrics.

## Available run modes (`--mode`)
The command-line `--mode` argument in `main.py` controls high-level execution. These modes are now documented so users can choose the appropriate workflow.

- `single_run`
  - Purpose: Run the full pipeline once with the configuration in `config_path`.
  - Use when you want a controlled single experiment (no wandb sweep).
  - Example:
    - python healnet/main.py --mode single_run --config_path config/main_gpu.yml

- `sweep`
  - Purpose: Launch a WandB hyperparameter sweep. The pipeline reads the sweep configuration file (`--sweep_config`) and uses WandB to control hyperparameters.
  - Requires: `--sweep_config` pointing to a sweep YAML and `--api_key` or env var `WANDNB_API_KEY`.
  - Behavior: creates a sweep with wandb and launches `wandb.agent` (or expects agents to be started).
  - Example:
    - python healnet/main.py --mode sweep --sweep_config config/sweep.yml --api_key $WANDNB_API_KEY

- `test_trained_model`
  - Purpose: Load a trained checkpoint/artifact and evaluate it on the test split.
  - Requires: `--artifact_dir` pointing to a checkpoint file or directory containing `.pt`/`.pth`.
  - Example:
    - python healnet/main.py --mode test_trained_model --artifact_dir /path/to/checkpoint_dir

- `sanity_check`
  - Purpose: Quick synthetic sanity tests using `SyntheticMultiModalSurvival`. Runs a short training/eval workflow to verify end-to-end behavior.
  - Very useful for local smoke-testing after code changes or to verify environment setup.
  - Example:
    - python healnet/main.py --mode sanity_check

**Notes about WandB and sweeps:**
- Sweeps are integrated but require proper WandB configuration (API key). 
- When running `--mode sweep` you can still configure how the sweep updates the pipeline by editing the sweep YAML.

## Migration / usage notes for users
- WandB:
  - No default API key in CLI now. Use --api_key or set env var `WANDNB_API_KEY` / `WANDB_API_KEY`. 
- Standardization:
  - compute_standardization_from_indices must be called (Pipeline calls it when available). If you bypass Pipeline, call this method on the dataset to compute and apply train-only stats.

## Useful notes

### Data structure:


### Config files:
- mean_gpu.yaml
- best_hyperparameters.yaml
- sweep_grid.yaml
- best_config_sweep_seeds.yaml: fixed best config obtained in the previous step, only variying the seed

### Results visualisation utils:


## TO-DO list

  1. Output dims does not need to be specified if it can be infered from the y_discrete number of bins.
  2. Pass 'endpoint' as argument.
  3. Ablation study debugging
  4. Explainability

### Contact
- This branch is based on konst-int-i/healnet (NeurIPS 2024). Please cite the original work per its license.
