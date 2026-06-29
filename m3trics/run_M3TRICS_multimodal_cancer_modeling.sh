#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------------------
# 0. ENVIRONMENT SETUP
# Set source for your environment manager and activate the Python environment
# Optional: configure Weights & Biases for tracking purposes
# -----------------------------------------------------------------------------------------

# Osiris cluster setup (VHIO)
source /home/osiris-user/anaconda3/etc/profile.d/conda.sh
conda activate TFM_5090

# Optional WandB configuration
WANDB_LOGIN_KEY="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"
WANDB_ENABLED="true"
WANDB_MODE="online"

# -----------------------------------------------------------------------------------------
# 1. GENERAL PATHS
# Define project root and data root directories
# -----------------------------------------------------------------------------------------

# Osiris cluster paths (VHIO)
PROJECT_ROOT="/home/osiris-user/Desktop/TFM/m3trics"
DATA_ROOT="/nfs/rnas/projects/M3TRICS/data/inputs"

# -----------------------------------------------------------------------------------------
# 2. DATASET AND ENDPOINT
# -----------------------------------------------------------------------------------------

DATASET="multimodal_cancer_modeling"
ENDPOINTS_CSV="endpoints.csv"
PATIENT_ID_COL="patient_id"

# -----------------------------------------------------------------------------------------
# 3. TASK CONFIGURATION
# binary classification arguments:
#   TASK_TYPE="binary_classification"
#   PATIENT_ID_COL: patient identifier column shared by endpoint and modality files
#   ENDPOINT_COL: binary label column used as the prediction target
#
# survival arguments:
#   TASK_TYPE="survival"
#   Available survival losses: nll, ce_survival, cox
#   SURVIVAL_LOSS: survival loss to optimize
#   SURVIVAL_TIME_COL: continuous survival time column
#   SURVIVAL_EVENT_COL: binary event indicator column (1=event, 0=censored)
#   SURVIVAL_N_BINS: number of discrete survival bins
# -----------------------------------------------------------------------------------------

TASK_TYPE="binary_classification"
ENDPOINT_COL="48_month_OS"
# SURVIVAL_LOSS="nll"
# SURVIVAL_TIME_COL="survival_time_days"
# SURVIVAL_EVENT_COL="event_observed"
# SURVIVAL_N_BINS=4

# Output directory
RESULTS_ROOT="${PROJECT_ROOT}/results/${DATASET}_${ENDPOINT_COL}"

# -----------------------------------------------------------------------------------------
# 4. MODALITIES CONFIGURATION
# Available modalities expected in ${DATA_ROOT}/${DATASET}:
#   clinical         = structured covariates + categorical cancer_type
#   histology        = UNI2 histology embeddings aggregated per patient
#   pathology_report = TF-IDF pathology-report features
#
# Missing value preprocessing:
#   clinical.cancer_type uses KNN mode imputation if missing.
#   numeric features use KNN mean imputation if missing.
# -----------------------------------------------------------------------------------------

CLIN_NAME="clinical"
CLIN_CSV="clinical.csv"
CLIN_CATEGORICAL_COLS="cancer_type"
CLIN_CATEGORICAL_IMPUTATION_METHOD="knn_mode"
CLIN_NUMERIC_IMPUTATION_METHOD="knn_mean"
CLIN_KNN_NEIGHBORS=5

HIST_NAME="histology"
HIST_CSV="histology.csv"
HIST_DROP_COLS="tcga_project_source"
HIST_NUMERIC_IMPUTATION_METHOD="knn_mean"
HIST_KNN_NEIGHBORS=5

REPORT_NAME="pathology_report"
REPORT_CSV="pathology_report.csv"
REPORT_NUMERIC_IMPUTATION_METHOD="knn_mean"
REPORT_KNN_NEIGHBORS=5

# -----------------------------------------------------------------------------------------
# 5. TRAINING CONFIGURATION
# Method compatibility overview:
#   Binary-classification only: ZI_LR, KNN_LR, ZI_RF, KNN_RF
#   Survival only: ZI_CoxNet, KNN_CoxNet, ZI_RSF, KNN_RSF
#   Binary classification and survival compatible: ZI_MLP, KNN_MLP, VAE_MLP, pAM,
#     HealNet, SMILe
#
# Default binary-classification run includes sklearn baselines and all compatible MLMM methods.
# Override with RUN_MODELS="..." if you want a smaller run.
# Knowledge Distillation
# DISTILL_MODELS is a comma-separated list of base methods that are also trained as _KD variants.
# Base methods are launched automatically if missing from RUN_MODELS. The teacher is pretrained first;
# then the student is trained with the configured modality availability. In progressive missingness mode
# this means simulated missingness; in static-cohort mode this means the observed dataset as-is.
# DISTILL_ALPHA weights the inner-representation matching loss. DISTILL_BETA weights the logit matching loss.
# -----------------------------------------------------------------------------------------

DISTILL_MODELS="${DISTILL_MODELS:-}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.25}"
DISTILL_BETA="${DISTILL_BETA:-0.05}"

if [[ -z "${RUN_MODELS:-}" ]]; then
  if [[ "${TASK_TYPE}" == "survival" ]]; then
    RUN_MODELS="ZI_CoxNet,KNN_CoxNet,ZI_RSF,KNN_RSF,ZI_MLP,KNN_MLP"
  else
    RUN_MODELS="ZI_LR,KNN_LR,ZI_RF,KNN_RF,ZI_MLP,KNN_MLP,VAE_MLP,pAM,HealNet,SMILe"
  fi
fi

RETRAIN_OUTER="${RETRAIN_OUTER:-false}"
SAVE_INNER="${SAVE_INNER:-true}"
k="${K_FOLDS:-5}"
INNER_SPLITS="${INNER_SPLITS:-${k}}"
OUTER_SPLITS="${OUTER_SPLITS:-${k}}"

# HPs
# If true, compute a dataset fingerprint before training and use its suggested
# <=32-combination HP grid instead of the defaults in hyperparams/*.py.
FINGERPRINT="${FINGERPRINT:-false}"
FINGERPRINT_MAX_COMBINATIONS="${FINGERPRINT_MAX_COMBINATIONS:-32}"
HP_SELECTION_EPSILON="${HP_SELECTION_EPSILON:-0.02}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-reduce_lr_on_plateau}"
MIN_LR="${MIN_LR:-1e-6}"
LR_PATIENCE="${LR_PATIENCE:-6}"
SEEDS="${SEEDS:-55602}"
MISSING_PATTERN_SEED="${MISSING_PATTERN_SEED:-2026}"

# -----------------------------------------------------------------------------------------
# 6. PROGRESSIVE MISSINGNESS STUDY
# Progressive missingness is enabled by default for this dataset.
# If TRAIN_MISSING_PROP/TEST_MISSING_PROP are left as "auto", the grid is built as:
#   for i in range(0, n_modalities): missing_prop.append(i / n_modalities)
# With the current 3 modalities, this gives: 0.0, 0.333333, 0.666667.
# -----------------------------------------------------------------------------------------

MISSINGNESS_STUDY="${MISSINGNESS_STUDY:-true}"
DEGRADING_MODALITY="${DEGRADING_MODALITY:-global}"
TRAIN_MISSING_PROP="${TRAIN_MISSING_PROP:-auto}"
TEST_MISSING_PROP="${TEST_MISSING_PROP:-auto}"

# -----------------------------------------------------------------------------------------
# Wrap arguments and run M3TRICS
# -----------------------------------------------------------------------------------------

add_modality_args() {
  local modality_name="$1"
  local csv_filename="$2"
  local drop_cols="${3:-}"
  local categorical_cols="${4:-}"
  local aggregation_method="${5:-}"
  local categorical_imputation_method="${6:-}"
  local numeric_imputation_method="${7:-}"
  local knn_neighbors="${8:-}"

  MODALITY_ARGS+=(--modality_csv "${modality_name}=${DATA_ROOT}/${DATASET}/${csv_filename}")
  FINGERPRINT_MODALITY_ARGS+=(--modality_csv "${modality_name}=${DATA_ROOT}/${DATASET}/${csv_filename}")
  MODALITY_COUNT=$((MODALITY_COUNT + 1))
  if [[ -n "${drop_cols}" ]]; then
    MODALITY_ARGS+=(--drop_cols "${modality_name}=${drop_cols}")
    FINGERPRINT_MODALITY_ARGS+=(--drop_cols "${modality_name}=${drop_cols}")
  fi
  if [[ -n "${categorical_cols}" ]]; then
    MODALITY_ARGS+=(--categorical_cols "${modality_name}=${categorical_cols}")
    FINGERPRINT_MODALITY_ARGS+=(--categorical_cols "${modality_name}=${categorical_cols}")
  fi
  if [[ -n "${aggregation_method}" ]]; then
    MODALITY_ARGS+=(--aggregation_method "${modality_name}=${aggregation_method}")
  fi
  if [[ -n "${categorical_imputation_method}" ]]; then
    MODALITY_ARGS+=(--categorical_imputation "${modality_name}=${categorical_imputation_method}")
  fi
  if [[ -n "${numeric_imputation_method}" ]]; then
    MODALITY_ARGS+=(--numeric_imputation "${modality_name}=${numeric_imputation_method}")
  fi
  if [[ -n "${knn_neighbors}" ]]; then
    MODALITY_ARGS+=(--knn_neighbors "${modality_name}=${knn_neighbors}")
  fi
}

MODALITY_ARGS=()
FINGERPRINT_MODALITY_ARGS=()
MODALITY_COUNT=0
add_modality_args "${CLIN_NAME}" "${CLIN_CSV}" "${CLIN_DROP_COLS:-}" "${CLIN_CATEGORICAL_COLS:-}" "${CLIN_AGGREGATION_METHOD:-}" "${CLIN_CATEGORICAL_IMPUTATION_METHOD:-}" "${CLIN_NUMERIC_IMPUTATION_METHOD:-}" "${CLIN_KNN_NEIGHBORS:-}"
add_modality_args "${HIST_NAME}" "${HIST_CSV}" "${HIST_DROP_COLS:-}" "${HIST_CATEGORICAL_COLS:-}" "${HIST_AGGREGATION_METHOD:-}" "${HIST_CATEGORICAL_IMPUTATION_METHOD:-}" "${HIST_NUMERIC_IMPUTATION_METHOD:-}" "${HIST_KNN_NEIGHBORS:-}"
add_modality_args "${REPORT_NAME}" "${REPORT_CSV}" "${REPORT_DROP_COLS:-}" "${REPORT_CATEGORICAL_COLS:-}" "${REPORT_AGGREGATION_METHOD:-}" "${REPORT_CATEGORICAL_IMPUTATION_METHOD:-}" "${REPORT_NUMERIC_IMPUTATION_METHOD:-}" "${REPORT_KNN_NEIGHBORS:-}"

build_missingness_grid() {
  local n_modalities="$1"
  python - "${n_modalities}" <<'PY'
import sys
n = int(sys.argv[1])
if n <= 0:
    raise SystemExit("n_modalities must be positive")
values = [i / n for i in range(n)]
print(",".join(f"{value:.6f}".rstrip("0").rstrip(".") if value else "0.0" for value in values))
PY
}

if [[ "${MISSINGNESS_STUDY}" == "true" ]]; then
  if [[ "${TRAIN_MISSING_PROP}" == "auto" ]]; then
    TRAIN_MISSING_PROP="$(build_missingness_grid "${MODALITY_COUNT}")"
  fi
  if [[ "${TEST_MISSING_PROP}" == "auto" ]]; then
    TEST_MISSING_PROP="$(build_missingness_grid "${MODALITY_COUNT}")"
  fi
fi

M3TRICS_ARGS=(
  --dataset "${DATASET}"
  --results_root "${RESULTS_ROOT}"
  --endpoint_csv "${DATA_ROOT}/${DATASET}/${ENDPOINTS_CSV}"
  --patient_id_col "${PATIENT_ID_COL}"
  --endpoint_col "${ENDPOINT_COL}"
  --task_type "${TASK_TYPE}"
  --run_models "${RUN_MODELS}"
  --distill_models "${DISTILL_MODELS}"
  --distill_alpha "${DISTILL_ALPHA}"
  --distill_beta "${DISTILL_BETA}"
  --inner_splits "${INNER_SPLITS}"
  --outer_splits "${OUTER_SPLITS}"
  --retrain_outer "${RETRAIN_OUTER}"
  --save_inner "${SAVE_INNER}"
  --missingness_study "${MISSINGNESS_STUDY}"
  --hp_selection_epsilon "${HP_SELECTION_EPSILON}"
  --scheduler_type "${SCHEDULER_TYPE}"
  --seeds "${SEEDS}"
  --missing_pattern_seed "${MISSING_PATTERN_SEED}"
)

if [[ "${SCHEDULER_TYPE}" == "cosine_annealing" ]]; then
  M3TRICS_ARGS+=(--min_lr "${MIN_LR}")
elif [[ "${SCHEDULER_TYPE}" == "reduce_lr_on_plateau" ]]; then
  M3TRICS_ARGS+=(--lr_patience "${LR_PATIENCE}")
fi

if [[ "${TASK_TYPE}" == "survival" ]]; then
  M3TRICS_ARGS+=(
    --survival_loss "${SURVIVAL_LOSS}"
    --survival_time_col "${SURVIVAL_TIME_COL}"
    --survival_event_col "${SURVIVAL_EVENT_COL}"
    --survival_n_bins "${SURVIVAL_N_BINS}"
  )
fi

if [[ "${MISSINGNESS_STUDY}" == "true" ]]; then
  M3TRICS_ARGS+=(
    --degrading_modality "${DEGRADING_MODALITY}"
    --train_missing_prop "${TRAIN_MISSING_PROP}"
    --test_missing_prop "${TEST_MISSING_PROP}"
  )
fi

if [[ "${FINGERPRINT}" == "true" ]]; then
  FINGERPRINT_DIR="${RESULTS_ROOT}/fingerprint"
  mkdir -p "${FINGERPRINT_DIR}"

  FINGERPRINT_RUN_MODELS="${RUN_MODELS}"
  if [[ -n "${DISTILL_MODELS}" ]]; then
    FINGERPRINT_RUN_MODELS="${FINGERPRINT_RUN_MODELS},${DISTILL_MODELS}"
  fi

  FINGERPRINT_ARGS=(
    --dataset_dir "${DATA_ROOT}/${DATASET}"
    --endpoint_csv "${DATA_ROOT}/${DATASET}/${ENDPOINTS_CSV}"
    --patient_id_col "${PATIENT_ID_COL}"
    --task_type "${TASK_TYPE}"
    --run_models "${FINGERPRINT_RUN_MODELS}"
    --max_combinations "${FINGERPRINT_MAX_COMBINATIONS}"
    --output_json "${FINGERPRINT_DIR}/fingerprint.json"
    --output_shell "${FINGERPRINT_DIR}/fingerprint_hp_suggestions.sh"
  )

  if [[ "${TASK_TYPE}" == "survival" ]]; then
    FINGERPRINT_ARGS+=(
      --survival_time_col "${SURVIVAL_TIME_COL}"
      --survival_event_col "${SURVIVAL_EVENT_COL}"
    )
  else
    FINGERPRINT_ARGS+=(--endpoint_col "${ENDPOINT_COL}")
  fi

  FINGERPRINT_ARGS+=("${FINGERPRINT_MODALITY_ARGS[@]}")
  python "${PROJECT_ROOT}/scripts/fingerprint.py" "${FINGERPRINT_ARGS[@]}"
  M3TRICS_ARGS+=(--fingerprint_hp_json "${FINGERPRINT_DIR}/fingerprint.json")
fi

WANDB_DIR_RESOLVED="${WANDB_DIR:-${RESULTS_ROOT}/wandb}"
mkdir -p "${WANDB_DIR_RESOLVED}"
export WANDB_DIR="${WANDB_DIR_RESOLVED}"
if [[ "${WANDB_ENABLED}" == "true" && -n "${WANDB_LOGIN_KEY}" ]]; then
  wandb login "${WANDB_LOGIN_KEY}"
fi

M3TRICS_ARGS+=(--wandb_mode "${WANDB_MODE}")
if [[ "${WANDB_ENABLED}" == "true" ]]; then
  M3TRICS_ARGS+=(--wandb)
fi

M3TRICS_ARGS+=("${MODALITY_ARGS[@]}")
python "${PROJECT_ROOT}/scripts/m3trics.py" "${M3TRICS_ARGS[@]}"
