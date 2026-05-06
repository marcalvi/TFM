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


# -----------------------------------------------------------------------------------------
# 2. DATASET AND ENDPOINT
# DATASET is the label used in outputs; DATASET_DIR is the actual input folder name.
# -----------------------------------------------------------------------------------------

DATASET="1001P"
DATASET_DIR="1001Prostate"
PATIENT_ID_COL="patient"
ENDPOINTS_CSV="endpoints_1001prostate.csv"
ENDPOINT_COL="SET_ME"

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

# Radiology modality
RADIO_NAME="radio"
RADIO_CSV="radiology_1001prostate.csv"
RADIO_DROP_COLS="study_date,time_to_diagnosis,source_file"

# Blood modality
BLOOD_NAME="blood"
BLOOD_CSV="blood_1001prostate.csv"
BLOOD_DROP_COLS="study_date,time_to_diagnosis,source_file"
BLOOD_NUMERIC_IMPUTATION_METHOD="knn_mean"
BLOOD_KNN_NEIGHBORS=5

# Radio-report modality
RADIO_REPORT_NAME="radio_report"
RADIO_REPORT_CSV="radioreports_1001prostate.csv"
RADIO_REPORT_DROP_COLS="study_date,time_to_diagnosis,source_file,report_text"

# -----------------------------------------------------------------------------------------
# 4. TRAINING CONFIGURATION
# Available methods: ZI_MLP, KNN_MLP, VAE_MLP, pAM, Di-PAM, Di-MMLP, HealNet, SMILe
# -----------------------------------------------------------------------------------------

RUN_MODELS="ZI_MLP, KNN_MLP, VAE_MLP, pAM, Di-PAM, Di-MMLP, HealNet, SMILe"
RETRAIN_OUTER="true"
HP_SELECTION_EPSILON="0.02"
k=5
INNER_SPLITS=${k}
OUTER_SPLITS=${k}
MISSING_LOCATION="global"
TRAIN_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"

SEEDS="22,2002,4,18473,55602"
MISSING_PATTERN_SEED=2026

# -----------------------------------------------------------------------------------------
# 5. BASIC VALIDATION
# -----------------------------------------------------------------------------------------

ENDPOINTS_PATH="${DATA_ROOT}/${DATASET_DIR}/${ENDPOINTS_CSV}"
if [[ ! -f "${ENDPOINTS_PATH}" ]]; then
  echo "Missing endpoint CSV: ${ENDPOINTS_PATH}" >&2
  echo "Create endpoints_1001prostate.csv first, then set ENDPOINT_COL to a real label column." >&2
  exit 1
fi

if [[ "${ENDPOINT_COL}" == "SET_ME" ]]; then
  echo "ENDPOINT_COL is still SET_ME. Point it to a real endpoint label column before running." >&2
  exit 1
fi

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

  MODALITY_ARGS+=(--modality_csv "${modality_name}=${DATA_ROOT}/${DATASET_DIR}/${csv_filename}")
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
add_modality_args "${RADIO_NAME}" "${RADIO_CSV}" "${RADIO_DROP_COLS:-}" "${RADIO_CATEGORICAL_COLS:-}" "${RADIO_AGGREGATION_METHOD:-}" "${RADIO_CATEGORICAL_IMPUTATION_METHOD:-}" "${RADIO_NUMERIC_IMPUTATION_METHOD:-}" "${RADIO_KNN_NEIGHBORS:-}"
add_modality_args "${BLOOD_NAME}" "${BLOOD_CSV}" "${BLOOD_DROP_COLS:-}" "${BLOOD_CATEGORICAL_COLS:-}" "${BLOOD_AGGREGATION_METHOD:-}" "${BLOOD_CATEGORICAL_IMPUTATION_METHOD:-}" "${BLOOD_NUMERIC_IMPUTATION_METHOD:-}" "${BLOOD_KNN_NEIGHBORS:-}"
add_modality_args "${RADIO_REPORT_NAME}" "${RADIO_REPORT_CSV}" "${RADIO_REPORT_DROP_COLS:-}" "${RADIO_REPORT_CATEGORICAL_COLS:-}" "${RADIO_REPORT_AGGREGATION_METHOD:-}" "${RADIO_REPORT_CATEGORICAL_IMPUTATION_METHOD:-}" "${RADIO_REPORT_NUMERIC_IMPUTATION_METHOD:-}" "${RADIO_REPORT_KNN_NEIGHBORS:-}"

M3TRICS_ARGS=(
  --dataset "${DATASET}"
  --results_root "${RESULTS_ROOT}"
  --endpoint_csv "${ENDPOINTS_PATH}"
  --patient_id_col "${PATIENT_ID_COL}"
  --endpoint_col "${ENDPOINT_COL}"
  --run_models "${RUN_MODELS}"
  --inner_splits "${INNER_SPLITS}"
  --outer_splits "${OUTER_SPLITS}"
  --retrain_outer "${RETRAIN_OUTER}"
  --hp_selection_epsilon "${HP_SELECTION_EPSILON}"
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
