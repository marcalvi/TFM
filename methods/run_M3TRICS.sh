#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------------
# 0. Environment setup:
# Set source for your environment manager and activate the Python environment
# -------------------------------------------------------------------------

source /home/osiris-user/anaconda3/etc/profile.d/conda.sh
conda activate TFM

# -------------------------------------------------------------------------
# 1. Paths configuration:
# Define project root and data root directories
# -------------------------------------------------------------------------

PROJECT_ROOT="/home/osiris-user/Desktop/TFM/methods"
DATA_ROOT="/nfs/rnas/projects/M3BENCH/data/inputs"

# -------------------------------------------------------------------------
# 2.Dataset and endpoint configuration:
# Define dataset name, patient ID column, endpoint CSV file, and endpoint column
# -------------------------------------------------------------------------

DATASET="MIMM"
PATIENT_ID_COL="patient"
ENDPOINTS_CSV="patients_mimm.csv"
ENDPOINT_COL="OS_9_label"

# Output directory
RESULTS_ROOT="${PROJECT_ROOT}/results/${DATASET}_${ENDPOINT_COL}"

# -------------------------------------------------------------------------
# 3. Missing-value handling configuration:
# Define imputation methods in case some modalities have missing values
# -------------------------------------------------------------------------

# Missing-value handling
CATEGORICAL_IMPUTATION_METHOD="knn_mode"  # column_mode | knn_mode
NUMERIC_IMPUTATION_METHOD="knn_mean"     # mean | median | knn_mean
KNN_NEIGHBORS=5

# -------------------------------------------------------------------------
# Modality configuration:
# Define one block per modality.
# Optional: *_DROP_COLS, *_CATEGORICAL_COLS and *_AGGREGATION_METHOD
# -------------------------------------------------------------------------

# Template:
# PATH_NAME="PATH"
# PATH_CSV="pathology_mimm.csv"
# PATH_DROP_COLS=""
# PATH_CATEGORICAL_COLS=""
# PATH_AGGREGATION_METHOD=""

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

# -------------------------------------------------------------------------
# Wrap arguments and run m3trics script
# -------------------------------------------------------------------------

add_modality_args() {
  local modality_name="$1"
  local csv_filename="$2"
  local drop_cols="${3:-}"
  local categorical_cols="${4:-}"
  local aggregation_method="${5:-}"

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
}

MODALITY_ARGS=()
add_modality_args "${PATH_NAME}" "${PATH_CSV}" "${PATH_DROP_COLS:-}" "${PATH_CATEGORICAL_COLS:-}" "${PATH_AGGREGATION_METHOD:-}"
add_modality_args "${RADIO_NAME}" "${RADIO_CSV}" "${RADIO_DROP_COLS:-}" "${RADIO_CATEGORICAL_COLS:-}" "${RADIO_AGGREGATION_METHOD:-}"
add_modality_args "${CLIN_NAME}" "${CLIN_CSV}" "${CLIN_DROP_COLS:-}" "${CLIN_CATEGORICAL_COLS:-}" "${CLIN_AGGREGATION_METHOD:-}"
add_modality_args "${BLOOD_NAME}" "${BLOOD_CSV}" "${BLOOD_DROP_COLS:-}" "${BLOOD_CATEGORICAL_COLS:-}" "${BLOOD_AGGREGATION_METHOD:-}"
add_modality_args "${RADIO_REPORT_NAME}" "${RADIO_REPORT_CSV}" "${RADIO_REPORT_DROP_COLS:-}" "${RADIO_REPORT_CATEGORICAL_COLS:-}" "${RADIO_REPORT_AGGREGATION_METHOD:-}"

M3TRICS_ARGS=(
  --dataset "${DATASET}"
  --results_root "${RESULTS_ROOT}"
  --endpoint_csv "${DATA_ROOT}/${DATASET}/${ENDPOINTS_CSV}"
  --patient_id_col "${PATIENT_ID_COL}"
  --endpoint_col "${ENDPOINT_COL}"
  --numeric_imputation "${NUMERIC_IMPUTATION_METHOD}"
  --categorical_imputation "${CATEGORICAL_IMPUTATION_METHOD}"
  --knn_neighbors "${KNN_NEIGHBORS}"
)

M3TRICS_ARGS+=("${MODALITY_ARGS[@]}")
python "${PROJECT_ROOT}/m3trics.py" "${M3TRICS_ARGS[@]}"
