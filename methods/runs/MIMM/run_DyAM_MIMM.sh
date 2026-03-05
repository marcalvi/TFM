#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment
#source /opt/miniconda3/etc/profile.d/conda.sh
source /home/osiris-user/anaconda3/etc/profile.d/conda.sh
conda activate TFM

# Optional WandB login from environment variable
wandb login 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

# Define paths
#PROJECT_ROOT="/Users/marcalbesa/Desktop/TFM/git_exp/methods"
#DATA_ROOT="/Users/marcalbesa/Desktop/TFM/data/MIMM"

PROJECT_ROOT="/home/osiris-user/Desktop/TFM/methods"
DATA_ROOT="/nfs/rnas/projects/M3BENCH/data/inputs/MIMM/"
RESULTS_ROOT="${PROJECT_ROOT}/results"

# Define endpoint
ENDPOINT="OS_6"

# Proposed tuning grid (DyAM)
SEEDS="22,2002,4,18473,55602"
#SEEDS="22,2002,4,18473,55602"
MISSING_PATTERN_SEED=2026

INNER_SPLITS=5
OUTER_SPLITS=5
EPOCHS=80

# Shared optimization hparams
BATCH_SIZE_GRID="16,32"
LR_GRID="5e-5,1e-4"

# DyAM-specific hparams
DYAM_DROPOUT_GRID="0.2,0.4"
DYAM_TEMPERATURE_GRID="1.0,2.0"

# Missingness experiments
TRAIN_MISSING_LOCATION_GRID="global, path, radio, clin, blood, radio_report"
TRAIN_MISSING_PROB_GRID="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_LOCATION_GRID="global, path, radio, clin, blood, radio_report"
TEST_MISSING_PROB_GRID="0.0,0.2,0.4,0.6,0.8"

# Run training
python "${PROJECT_ROOT}/main.py" \
  --dataset "MIMM" \
  --odir "${RESULTS_ROOT}" \
  --endpoint "${ENDPOINT}" \
  --model "DyAM" \
  --inst_data "${DATA_ROOT}/patients_mimm.csv" \
  --patient_ids_col "patient" \
  --patho_data "${DATA_ROOT}/pathology_mimm.csv" \
  --radio_data "${DATA_ROOT}/radiology_mimm.csv" \
  --clin_data "${DATA_ROOT}/clinical_mimm.csv" \
  --radio_report_data "${DATA_ROOT}/radioreports_mimm.csv" \
  --blood_data "${DATA_ROOT}/blood_mimm.csv" \
  --inner_splits "${INNER_SPLITS}" \
  --outer_splits "${OUTER_SPLITS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE_GRID}" \
  --learning_rate "${LR_GRID}" \
  --dyam_dropout "${DYAM_DROPOUT_GRID}" \
  --dyam_temperature "${DYAM_TEMPERATURE_GRID}" \
  --train_missing_prob "${TRAIN_MISSING_PROB_GRID}" \
  --train_missing_location "${TRAIN_MISSING_LOCATION_GRID}" \
  --test_missing_prob "${TEST_MISSING_PROB_GRID}" \
  --test_missing_location "${TEST_MISSING_LOCATION_GRID}" \
  --seeds "${SEEDS}" \
  --missing_pattern_seed "${MISSING_PATTERN_SEED}" \
  --wandb \
  --wandb_project "DyAMs" \
  --wandb_mode "online"
