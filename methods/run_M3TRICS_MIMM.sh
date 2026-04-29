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

# Local setup (macOS)
# source /opt/miniconda3/etc/profile.d/conda.sh
# conda activate TFM

# Optional WandB configuration
WANDB_LOGIN_KEY="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"
WANDB_ENABLED="true"
WANDB_MODE="online"

# -----------------------------------------------------------------------------------------
# 1. GENERAL PATHS
# Define project root and data root directories
# -----------------------------------------------------------------------------------------

# Osiris cluster paths (VHIO)
PROJECT_ROOT="/home/osiris-user/Desktop/TFM/methods"
DATA_ROOT="/nfs/rnas/projects/M3TRICS/data/inputs"

# Local paths (macOS)
# PROJECT_ROOT="/Users/marcalbesa/Desktop/TFM/git_exp/methods"
# DATA_ROOT="/Users/marcalbesa/Desktop/TFM/data"

# -----------------------------------------------------------------------------------------
# 2. DATASET AND ENDPOINT
# Define dataset name, patient ID column, endpoint CSV file, and endpoint column
# -----------------------------------------------------------------------------------------

DATASET="MIMM"
PATIENT_ID_COL="patient"
ENDPOINTS_CSV="patients_mimm.csv"
ENDPOINT_COL="OS_9_label"

# Output directory
RESULTS_ROOT="${PROJECT_ROOT}/results/${DATASET}_${ENDPOINT_COL}"

# -----------------------------------------------------------------------------------------
# 3. MODALITIES CONFIGURATION
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
# PATH_NAME="PATH"
# PATH_CSV="pathology_mimm.csv"
# PATH_DROP_COLS=""
# PATH_AGGREGATION_METHOD=""
# PATH_CATEGORICAL_IMPUTATION_METHOD="knn_mode"
# PATH_NUMERIC_IMPUTATION_METHOD="knn_mean"
# PATH_CATEGORICAL_COLS=""
# PATH_KNN_NEIGHBORS=5

# Path modality
PATH_NAME="path"
PATH_CSV="pathology_mimm.csv"

# Radiology modality
RADIO_NAME="radio"
RADIO_CSV="radiology_mimm.csv"
RADIO_DROP_COLS="image_path,lesion_tag"
RADIO_AGGREGATION_METHOD="mean"

# Clinical modality
CLIN_NAME="clin"
CLIN_CSV="clinical_mimm.csv"

# Blood modality
BLOOD_NAME="blood"
BLOOD_CSV="blood_mimm.csv"

# Radio-report modality
RADIO_REPORT_NAME="radio_report"
RADIO_REPORT_CSV="radioreports_mimm.csv"

# -----------------------------------------------------------------------------------------
# 4. TRAINING CONFIGURATION
# Select the models to run after preprocessing.
# Available methods: ZI_MLP, KNN_MLP, VAE_MLP, pAM, PAMDiPAM, MLPDiPAM, HealNet, SMILe
# -----------------------------------------------------------------------------------------

RUN_MODELS="SMILe"
RETRAIN_OUTER="false"
k=5
INNER_SPLITS=${k}
OUTER_SPLITS=${k}
MISSING_LOCATION="global"
TRAIN_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"
# "0.0,0.2,0.4,0.6,0.8"

# Proposed seeds for reproducibility
SEEDS="18473,55602"

# Missing pattern seed is fixed to ensure the same ablation patterns across seeds
MISSING_PATTERN_SEED=2026

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
  if [[ -n "${drop_cols}" ]]; then
    MODALITY_ARGS+=(--drop_cols "${modality_name}=${drop_cols}")
  fi
  if [[ -n "${categorical_cols}" ]]; then
    MODALITY_ARGS+=(--categorical_cols "${modality_name}=${categorical_cols}")
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
  --run_models "${RUN_MODELS}"
  --inner_splits "${INNER_SPLITS}"
  --outer_splits "${OUTER_SPLITS}"
  --retrain_outer "${RETRAIN_OUTER}"
  --missing_location "${MISSING_LOCATION}"
  --train_missing_prop "${TRAIN_MISSING_PROP}"
  --test_missing_prop "${TEST_MISSING_PROP}"
  --seeds "${SEEDS}"
  --missing_pattern_seed "${MISSING_PATTERN_SEED}"
)

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
python "${PROJECT_ROOT}/m3trics.py" "${M3TRICS_ARGS[@]}"
