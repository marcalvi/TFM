<p align="center">
  <img src="assets/logo.png" alt="M3TRICS" width="360">
</p>

M3TRICS is a multimodal training and analysis framework for comparing missing-modality learning methods across clinical datasets. It handles dataset preprocessing, modality alignment, nested cross-validation, progressive missingness studies, fixed-dataset training, binary classification, survival modelling, and downstream statistical analysis.

This README describes the workflow from environment setup to result analysis. The project root referred to below is this directory, the one containing `scripts/m3trics.py` and the `run_M3TRICS_*.sh` scripts.

## What This Project Can Do

- Train multimodal models from per-modality CSV files.
- Align endpoint and modality tables by patient ID.
- Drop dataset-specific columns before training.
- Aggregate duplicated patient rows with mean pooling or supervised attention pooling.
- Impute missing values within each modality using configurable numeric/categorical strategies.
- Compare multiple missing-modality methods under the same nested-CV setup.
- Run either a synthetic missingness decay/progressive missingness study or a fixed observed dataset experiment.
- Evaluate binary classification and discrete-time survival tasks.
- Save inner-CV, outer-test, epoch-history, split, prediction, and processed-data outputs.
- Analyze trained results with notebooks in `analysis/`.

## Repository Layout

```text
m3trics/
├── run_M3TRICS_MIMM.sh                # MIMM training launcher
├── run_M3TRICS_mmCRC.sh               # mmCRC training launcher
├── run_M3TRICS_1001P.sh               # 1001Prostate training launcher
├── env/                               # Conda environment definitions and install guide
├── hyperparams/                       # Hyperparameter grids per method
├── dataset/                           # Dataset preprocessing, loaders, missingness simulator, pooling, imputation
├── scripts/                           # CLI entrypoint, nested-CV training logic, training loops, shared utilities
├── models/                            # Model implementations
├── analysis/                          # Result analysis notebooks and helper code
└── results/                           # Generated processed data, model outputs, W&B logs
```

## Available Methods

Use these names in `RUN_MODELS` inside the `.sh` launchers.

| Method | Purpose |
| --- | --- |
| `ZI_MLP` | MultimodalMLP with zero imputation for missing modalities. |
| `KNN_MLP` | MultimodalMLP with KNN imputation for missing modalities. |
| `VAE_MLP` | MultimodalMLP with VAE-based imputation for missing modalities. |
| `pAM` | Attention masking over unimodal predictions model. |
| `Di-PAM` | Distilled version of pAM; teacher uses complete modalities, student receives the configured missingness mask. |
| `Di-MMLP` | Distilled multimodal MLP; teacher uses complete modalities, student receives the configured missingness mask. |
| `HealNet` | HealNet wrapper for modality-level embeddings. |
| `SMILe` | SMIL generalization for n>=2 modalities with mask-aware latent reconstruction. |

Hyperparameter grids live in `hyperparams/`. Edit those files when you want to change the search space for a method.

SMILe learns its modality priors only from modality slots visible in the current training split and missingness condition. It does not recover synthetic hidden modalities from the complete base dataset.

## 1. Install The Environment

Environment files and VHIO/OSIRIS-specific installation instructions are in:

```text
env/README.md
```

Typical setup:

```bash
cd /home/osiris-user/Desktop/TFM/m3trics
conda env create -f env/m3trics_4090.yml
conda activate m3trics_4090
```

or for 5090 workers:

```bash
conda env create -f env/m3trics_5090.yml
conda activate m3trics_5090
```

Verify CUDA after activation:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## 2. Configure A Run Script

The recommended workflow is to edit one of the dataset launchers:

```text
run_M3TRICS_MIMM.sh
run_M3TRICS_mmCRC.sh
run_M3TRICS_1001P.sh
```

Each script is organized into the same sections.

### 0. Environment Setup

Set the conda activation and optional Weights & Biases logging.

```bash
source /home/osiris-user/anaconda3/etc/profile.d/conda.sh
conda activate m3trics_5090

WANDB_ENABLED="true"
WANDB_MODE="online"
```

Use `WANDB_ENABLED="false"` if you do not want W&B logging.

### 1. General Paths

Set where the project code and input data live.

```bash
PROJECT_ROOT="/home/osiris-user/Desktop/TFM/m3trics"
DATA_ROOT="/nfs/rnas/projects/M3TRICS/data/inputs"
```

### 2. Dataset And Endpoint

Define the dataset label, patient ID column, and endpoint CSV.

```bash
DATASET="mmCRC"
PATIENT_ID_COL="sap"
ENDPOINTS_CSV="mmCRC_endpoints.csv"
```

For `1001P`, `DATASET` is the output label and `DATASET_DIR` is the real input folder name.

### 3. Modalities Configuration

Each modality needs a name and a CSV file. Example:

```bash
RADIO_NAME="radio"
RADIO_CSV="mmCRC_radiology_data.csv"
```

Optional per-modality settings:

| Variable pattern | Meaning |
| --- | --- |
| `*_DROP_COLS` | Comma-separated columns removed before training. |
| `*_AGGREGATION_METHOD` | How to collapse duplicate patient rows: `mean` or `attention`. |
| `*_CATEGORICAL_COLS` | Comma-separated categorical feature columns. |
| `*_CATEGORICAL_IMPUTATION_METHOD` | `column_mode` or `knn_mode`. |
| `*_NUMERIC_IMPUTATION_METHOD` | `mean`, `median`, or `knn_mean`. |
| `*_KNN_NEIGHBORS` | Number of neighbors for KNN imputation. |

Input requirements:

- Endpoint CSV must contain `PATIENT_ID_COL` and the target columns.
- Each modality CSV must contain the same patient ID column.
- Feature columns should be numeric unless declared as categorical.
- Multiple rows per patient are allowed and are collapsed by the selected aggregation method.

### 4. Training Configuration

Select models, CV folds, seeds, scheduler and hyperparameter-selection behavior.

```bash
RUN_MODELS="ZI_MLP, KNN_MLP, VAE_MLP, pAM, Di-PAM, Di-MMLP, HealNet, SMILe"
RETRAIN_OUTER="true"
SAVE_INNER="true"
k=5
INNER_SPLITS=${k}
OUTER_SPLITS=${k}
HP_SELECTION_EPSILON="0.02"
SCHEDULER_TYPE="cosine_annealing"
MIN_LR="1e-6"
SEEDS="22,2002,4,18473,55602"
MISSING_PATTERN_SEED=2026
```

Scheduler options:

| Scheduler | Required argument |
| --- | --- |
| `cosine_annealing` | `MIN_LR` |
| `reduce_lr_on_plateau` | `LR_PATIENCE` |

Nested-CV behavior:

- `RETRAIN_OUTER="true"`: select HPs in inner CV, then refit on the full outer-train split and evaluate on outer-test.
- `RETRAIN_OUTER="false"`: retain selected inner-fold models and evaluate them on outer-test.
- `SAVE_INNER="true"`: when `RETRAIN_OUTER=true`, also saves retained-inner outputs under the matching `retrainfalse` directory.

### 5. Task Configuration

Binary classification:

```bash
TASK_TYPE="binary_classification"
ENDPOINT_COL="OS_27_label"
```

Survival:

```bash
TASK_TYPE="survival"
ENDPOINT_COL="OS"
SURVIVAL_LOSS="nll"
SURVIVAL_TIME_COL="OS"
SURVIVAL_EVENT_COL="Patient Status"
SURVIVAL_N_BINS=4
```

Implemented survival losses:

```text
nll, ce_survival, cox
```

For survival, models output `SURVIVAL_N_BINS` logits and the analysis uses task-specific metrics such as C-index and loss.

### 6. Progressive Missingness Study

Decay/missingness-study mode simulates missing modalities at train and test time.

```bash
MISSINGNESS_STUDY="true"
MISSING_LOCATION="global"
TRAIN_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"
```

Use this when you want to study robustness as missingness increases. The process requires a subset with all selected modalities available before synthetic missingness is applied.

Fixed dataset mode disables synthetic missingness and trains on the observed dataset as-is for standard methods. For distillation methods (`Di-PAM`, `Di-MMLP`), M3TRICS first estimates the observed patient-modality missingness proportion in the original cohort, then trains on the complete-case subset and applies an equivalent synthetic missingness mask to the student branch only.

```bash
MISSINGNESS_STUDY="false"
# MISSING_LOCATION, TRAIN_MISSING_PROP and TEST_MISSING_PROP are not passed.
```

## 3. Run Training

From the project root:

```bash
bash run_M3TRICS_MIMM.sh
```

or:

```bash
bash run_M3TRICS_mmCRC.sh
bash run_M3TRICS_1001P.sh
```

The scripts call `scripts/m3trics.py` with all configured arguments.

Direct CLI execution is also possible, but the `.sh` launchers are the source of truth because they keep dataset paths, modalities, task settings, and experiment mode in one place.

## 4. Output Structure

For each run, outputs are written under:

```text
results/<DATASET>_<ENDPOINT_COL>/
```

Main folders:

| Path | Content |
| --- | --- |
| `processed_data/` | Aligned endpoints and processed modality CSVs. |
| `training_runs/` | Nested-CV outputs per method, seed, and missingness setting. |
| `wandb/` | Local W&B files when logging is enabled. |

Missingness-study mode output:

```text
results/<DATASET>_<ENDPOINT_COL>/training_runs/<MODEL>_retrain<true|false>_k<K>/TRAIN_MISSING/<LOCATION>/<TRAIN_MISSING_PERCENT>/seed_<SEED>/
```

Fixed dataset output:

```text
results/<DATASET>_<ENDPOINT_COL>/training_runs/<MODEL>_retrain<true|false>_k<K>/FIXED/seed_<SEED>/
```

Typical CSVs inside each seed folder:

| File | Meaning |
| --- | --- |
| `inner_hp_eval.csv` | Inner-CV HP evaluation rows. |
| `inner_epoch_history.csv` | Per-epoch learning curves. |
| `outer_test_metrics.csv` | Outer-test metrics. |
| `outer_test_summary.csv` | Aggregated outer-test summary. |
| `test_predictions.csv` | Patient-level predictions and model outputs. |
| `splits_manifest.csv` | Outer/inner split membership. |

## 5. Analyze Results

Analysis notebooks live in:

```text
analysis/
```

### Missing-Modality Decay Analysis

Use this for `MISSINGNESS_STUDY=true` runs:

```text
analysis/MM_decay_analysis.ipynb
```

It loads missingness-study outputs, computes replicate AUC tables, global Friedman tests, Wilcoxon pairwise comparisons, heatmaps, method-level performance/resilience metrics, and normalized degradation coefficients. Distillation methods are excluded from Training intuition and Train degradation coefficient.

Outputs are saved to:

```text
analysis/MM_decay_analysis_outputs/
```

### Fixed Dataset Analysis

Use this for `MISSINGNESS_STUDY=false` runs:

```text
analysis/fixed_dataset_analysis.ipynb
```

It compares methods on the fixed observed dataset, ranks them by mean AUC, performs global and pairwise statistical tests, and builds pairwise heatmaps.

Outputs are saved to:

```text
analysis/fixed_dataset_analysis_outputs/
```

### Data Exploration

Dataset exploration and conversion notebooks live in:

```text
analysis/data_exploration/
```

Current notebooks include:

| Notebook | Purpose |
| --- | --- |
| `h5_to_csvs.ipynb` | Convert `.h5` files into modality CSVs. |
| `os_distribution_1001P.ipynb` | Inspect 1001Prostate OS distribution. |
| `os_distribution_MIMM.ipynb` | Inspect MIMM OS distribution. |
| `os_distribution_mmCRC.ipynb` | Inspect mmCRC OS distribution. |

### Notebook Configuration

At the top of each analysis notebook, set the dataset/run parameters, for example:

```python
DATASET_NAME = "mmCRC"
LABEL_NAME = "OS_21_label"
TRAIN_MISSING_LOCATION = "GLOBAL"
RETRAIN_OUTER = True
```

For fixed-dataset analysis, make sure the notebook is pointed to `results_mode='fixed_dataset'` or uses the provided fixed-dataset helper cells.

## 6. Recommended Workflow

1. Create and activate the environment from `env/README.md`.
2. Choose the dataset launcher closest to your experiment.
3. Edit paths, endpoint, modality CSVs, dropped columns, and imputation settings.
4. Choose `RUN_MODELS`, CV folds, seeds, task type, and scheduler.
5. Set `MISSINGNESS_STUDY=true` for decay analysis or `false` for fixed-dataset training.
6. Run the `.sh` launcher.
7. Check `results/<DATASET>_<ENDPOINT_COL>/processed_data/` to verify preprocessing.
8. Check `training_runs/` to verify every model/seed/missingness configuration completed.
9. Open the matching notebook in `analysis/` and generate tables/figures.

## 7. Practical Checks Before Long Runs

Before launching a full experiment, run each model with a relatively large HP grid separately:

```bash
RUN_MODELS="ZI_MLP"
SEEDS="22,..."
```

Then:

- Study recurrence of each HP and reduce grid
- Validate learning curves shape
- See whether you get better results with or without retraining on outer train set

Only then expand to all models and seeds.

## 8. Current Implemented Results Notebooks

The current result-analysis notebooks are implemented for classification outputs only. Training supports survival configurations, but the survival-specific analysis notebooks and tables are not implemented yet.

| Task | Mode | Launcher settings | Analysis notebook | Output folder |
| --- | --- | --- | --- | --- |
| Classification | Progressive missingness study | `TASK_TYPE="binary_classification"` + `MISSINGNESS_STUDY="true"` | `analysis/MM_decay_analysis.ipynb` | `analysis/MM_decay_analysis_outputs/` |
| Classification | Fixed observed dataset | `TASK_TYPE="binary_classification"` + `MISSINGNESS_STUDY="false"` | `analysis/fixed_dataset_analysis.ipynb` | `analysis/fixed_dataset_analysis_outputs/` |

Not currently implemented:

- Survival task analysis notebooks.
- Modality-specific decay analysis notebooks.
