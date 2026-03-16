#!/usr/bin/env bash
export WANDB_DIR=/nfs/rnas/workspaces/malbesa/TFM/methods/runs/wandb
set -euo pipefail

# Activate conda environment
#source /opt/miniconda3/etc/profile.d/conda.sh
source /home/osiris-user/anaconda3/etc/profile.d/conda.sh
conda activate TFM

# WandB login token
WANDB_LOGIN_KEY="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"
if [[ -n "${WANDB_LOGIN_KEY}" ]]; then
  wandb login "${WANDB_LOGIN_KEY}"
else
  echo "WANDB_LOGIN_KEY not set; skipping wandb login."
fi

WANDB_ARGS=()
if [[ -n "${WANDB_LOGIN_KEY}" ]]; then
  WANDB_ARGS+=(--wandb --wandb_project "DyAM" --wandb_mode "online")
fi

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
MISSING_LOCATION_GRID="global, path, radio, clin, blood, radio_report"
TRAIN_MISSING_PROP_GRID="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_PROP_GRID="0.0,0.2,0.4,0.6,0.8"

# Run training
python "${PROJECT_ROOT}/main.py" \
  --model "DyAM" \
  --dataset "MIMM" \
  --odir "${RESULTS_ROOT}" \
  --dataset_dir "${DATA_ROOT}" \
  --endpoint "${ENDPOINT}" \
  --inner_splits "${INNER_SPLITS}" \
  --outer_splits "${OUTER_SPLITS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE_GRID}" \
  --learning_rate "${LR_GRID}" \
  --dyam_dropout "${DYAM_DROPOUT_GRID}" \
  --dyam_temperature "${DYAM_TEMPERATURE_GRID}" \
  --train_missing_prop "${TRAIN_MISSING_PROP_GRID}" \
  --missing_location "${MISSING_LOCATION_GRID}" \
  --test_missing_prop "${TEST_MISSING_PROP_GRID}" \
  --seeds "${SEEDS}" \
  --missing_pattern_seed "${MISSING_PATTERN_SEED}" \
  "${WANDB_ARGS[@]}"
