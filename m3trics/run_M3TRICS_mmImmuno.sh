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
# Define dataset name and endpoint CSV file
# -----------------------------------------------------------------------------------------

DATASET="mmImmuno"
ENDPOINTS_CSV="patients_mmImmuno.csv"

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
PATIENT_ID_COL="patient"
ENDPOINT_COL="OS_9_label"
# SURVIVAL_LOSS="nll"
# SURVIVAL_TIME_COL=""
# SURVIVAL_EVENT_COL=""
# SURVIVAL_N_BINS=4

# Output directory
RESULTS_ROOT="${PROJECT_ROOT}/results/${DATASET}_${ENDPOINT_COL}"

# -----------------------------------------------------------------------------------------
# 4. MODALITIES CONFIGURATION
# Define modality names and corresponding CSV files

# Optional
#   Per-modality column dropping
#       *_DROP_COLS
#   Preprocessing row aggregation for duplicated patient rows
#       *_AGGREGATION_METHOD                            # options: mean, attention
#   Per-modality missing value handling
#       *_CATEGORICAL_IMPUTATION_METHOD="knn_mode"      # options: column_mode, knn_mode
#       *_NUMERIC_IMPUTATION_METHOD="knn_mean"          # options: mean, median, knn_mean
#       *_CATEGORICAL_COLS="col1,col2"
#       *_KNN_NEIGHBORS=5
# -----------------------------------------------------------------------------------------

# Template:
# MODALITY_NAME="modality_name"
# MODALITY_CSV="modality_file.csv"
# MODALITY_DROP_COLS=""
# MODALITY_AGGREGATION_METHOD=""
# MODALITY_CATEGORICAL_IMPUTATION_METHOD="knn_mode"
# MODALITY_NUMERIC_IMPUTATION_METHOD="knn_mean"
# MODALITY_CATEGORICAL_COLS=""
# MODALITY_KNN_NEIGHBORS=5

# Path modality
PATH_NAME="path"
PATH_CSV="pathology_mmImmuno.csv"

# Radiology modality
RADIO_NAME="radio"
RADIO_CSV="radiology_mmImmuno.csv"
RADIO_DROP_COLS="image_path,lesion_tag"
RADIO_AGGREGATION_METHOD="mean"

# Clinical modality
CLIN_NAME="clin"
CLIN_CSV="clinical_mmImmuno.csv"

# Blood modality
BLOOD_NAME="blood"
BLOOD_CSV="blood_mmImmuno.csv"

# Radio-report modality
RADIO_REPORT_NAME="radio_report"
RADIO_REPORT_CSV="radioreports_mmImmuno.csv"

# -----------------------------------------------------------------------------------------
# 5. TRAINING CONFIGURATION
# Select the models to run after preprocessing.

# Method compatibility overview:
#   Binary-classification only:
#     ZI_LR       = zero-imputed concatenated embeddings + LogisticRegression
#     KNN_LR      = KNN-imputed concatenated embeddings + LogisticRegression
#     ZI_RF       = zero-imputed concatenated embeddings + RandomForestClassifier
#     KNN_RF      = KNN-imputed concatenated embeddings + RandomForestClassifier
#     Note: RF means binary-classification Random Forest; RSF is the survival model.
#   Survival only:
#     ZI_CoxNet   = zero-imputed concatenated embeddings + regularized Cox elastic-net
#     KNN_CoxNet  = KNN-imputed concatenated embeddings + regularized Cox elastic-net
#     ZI_RSF      = zero-imputed concatenated embeddings + Random Survival Forest
#     KNN_RSF     = KNN-imputed concatenated embeddings + Random Survival Forest
#     Note: RSF means Random Survival Forest; it is not a binary-classification random forest.
#   Binary classification and survival compatible:
#     ZI_MLP, KNN_MLP, VAE_MLP, pAM, HealNet, SMILe

# Example BC run:
#   RUN_MODELS="ZI_LR,KNN_LR,ZI_RF,KNN_RF,ZI_MLP,KNN_MLP,pAM"
# Example survival run:
#   TASK_TYPE="survival"
#   RUN_MODELS="ZI_CoxNet,KNN_CoxNet,ZI_RSF,KNN_RSF,ZI_MLP,KNN_MLP,HealNet"

# Available scheduler types: cosine_annealing, reduce_lr_on_plateau
#   cosine_annealing requires MIN_LR
#   reduce_lr_on_plateau requires LR_PATIENCE

# Missing pattern seed is fixed to ensure the same missingness patterns across seeds
# -----------------------------------------------------------------------------------------

# Methods
RUN_MODELS="ZI_LR,KNN_LR,ZI_RF,KNN_RF,ZI_MLP,KNN_MLP,VAE_MLP,pAM,HealNet,SMILe"

# Knowledge Distillation
# DISTILL_MODELS is a comma-separated list of base methods that are also trained as _KD variants.
# Base methods are launched automatically if missing from RUN_MODELS. The teacher is pretrained first;
# then the student is trained with the configured modality availability. In progressive missingness mode
# this means simulated missingness; in static-cohort mode this means the observed dataset as-is.
# DISTILL_ALPHA weights the inner-representation matching loss. DISTILL_BETA weights the logit matching loss.
DISTILL_MODELS="${DISTILL_MODELS:-ZI_MLP,HealNet}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.25}"
DISTILL_BETA="${DISTILL_BETA:-0.05}"

# Nested CV
RETRAIN_OUTER="false"
SAVE_INNER="true"
k=5
INNER_SPLITS=${k}
OUTER_SPLITS=${k}

# HPs
# If true, compute a dataset fingerprint before training and use its suggested
# <=32-combination HP grid instead of the defaults in hyperparams/*.py.
FINGERPRINT="${FINGERPRINT:-true}"
FINGERPRINT_MAX_COMBINATIONS="${FINGERPRINT_MAX_COMBINATIONS:-32}"
HP_SELECTION_EPSILON="0.02"
SCHEDULER_TYPE="reduce_lr_on_plateau"
MIN_LR="1e-6"
LR_PATIENCE=6

# Seeds
SEEDS="22,2002,4,18473,55602"
# SEEDS = "22,55602"
MISSING_PATTERN_SEED=2026

# -----------------------------------------------------------------------------------------
# 6. PROGRESSIVE MISSINGNESS STUDY
# Specify these values to perform the proposed progressive missingness study and downstream analysis.

# Notes: this process needs a subset with all modalities available, so make sure N is large
# enough to support meaningful statistical comparisons.

# Static-cohort mode: if MISSINGNESS_STUDY=false, synthetic missingness is disabled
# and the input dataset is trained as-is, preserving its natural modality availability.

# For _KD variants in static-cohort mode, the teacher is pretrained first and the
# student is trained on the full observed cohort without extra synthetic missingness.
# -----------------------------------------------------------------------------------------

MISSINGNESS_STUDY="true"
DEGRADING_MODALITY="global"
TRAIN_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"

# -----------------------------------------------------------------------------------------
# Wrap arguments and run m3trics script
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
add_modality_args "${PATH_NAME}" "${PATH_CSV}" "${PATH_DROP_COLS:-}" "${PATH_CATEGORICAL_COLS:-}" "${PATH_AGGREGATION_METHOD:-}" "${PATH_CATEGORICAL_IMPUTATION_METHOD:-}" "${PATH_NUMERIC_IMPUTATION_METHOD:-}" "${PATH_KNN_NEIGHBORS:-}"
add_modality_args "${RADIO_NAME}" "${RADIO_CSV}" "${RADIO_DROP_COLS:-}" "${RADIO_CATEGORICAL_COLS:-}" "${RADIO_AGGREGATION_METHOD:-}" "${RADIO_CATEGORICAL_IMPUTATION_METHOD:-}" "${RADIO_NUMERIC_IMPUTATION_METHOD:-}" "${RADIO_KNN_NEIGHBORS:-}"
add_modality_args "${CLIN_NAME}" "${CLIN_CSV}" "${CLIN_DROP_COLS:-}" "${CLIN_CATEGORICAL_COLS:-}" "${CLIN_AGGREGATION_METHOD:-}" "${CLIN_CATEGORICAL_IMPUTATION_METHOD:-}" "${CLIN_NUMERIC_IMPUTATION_METHOD:-}" "${CLIN_KNN_NEIGHBORS:-}"
add_modality_args "${BLOOD_NAME}" "${BLOOD_CSV}" "${BLOOD_DROP_COLS:-}" "${BLOOD_CATEGORICAL_COLS:-}" "${BLOOD_AGGREGATION_METHOD:-}" "${BLOOD_CATEGORICAL_IMPUTATION_METHOD:-}" "${BLOOD_NUMERIC_IMPUTATION_METHOD:-}" "${BLOOD_KNN_NEIGHBORS:-}"
add_modality_args "${RADIO_REPORT_NAME}" "${RADIO_REPORT_CSV}" "${RADIO_REPORT_DROP_COLS:-}" "${RADIO_REPORT_CATEGORICAL_COLS:-}" "${RADIO_REPORT_AGGREGATION_METHOD:-}" "${RADIO_REPORT_CATEGORICAL_IMPUTATION_METHOD:-}" "${RADIO_REPORT_NUMERIC_IMPUTATION_METHOD:-}" "${RADIO_REPORT_KNN_NEIGHBORS:-}"

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

WANDB_ENABLED_RESOLVED="${WANDB_ENABLED:-false}"
WANDB_MODE_RESOLVED="${WANDB_MODE:-online}"
WANDB_DIR_RESOLVED="${WANDB_DIR:-${RESULTS_ROOT}/wandb}"

mkdir -p "${WANDB_DIR_RESOLVED}"
export WANDB_DIR="${WANDB_DIR_RESOLVED}"
if [[ "${WANDB_ENABLED_RESOLVED}" == "true" && -n "${WANDB_LOGIN_KEY:-}" ]]; then
  wandb login "${WANDB_LOGIN_KEY}"
fi

M3TRICS_ARGS+=(--wandb_mode "${WANDB_MODE_RESOLVED}")
if [[ "${WANDB_ENABLED_RESOLVED}" == "true" ]]; then
  M3TRICS_ARGS+=(--wandb)
fi

M3TRICS_ARGS+=("${MODALITY_ARGS[@]}")
python "${PROJECT_ROOT}/scripts/m3trics.py" "${M3TRICS_ARGS[@]}"
