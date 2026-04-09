#!/usr/bin/env bash
export WANDB_DIR=/nfs/rnas/workspaces/malbesa/TFM/methods/runs
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
  WANDB_ARGS+=(--wandb --wandb_project "ZI_MLP_reduced" --wandb_mode "online")
fi

# Radio aggregation method
RADIO_AGGREGATION_METHOD="mean"
RETRAIN_OUTER="true"

# Define endpoint
ENDPOINT="OS_9"

# Define paths
PROJECT_ROOT="/home/osiris-user/Desktop/TFM/methods"
DATA_ROOT="/nfs/rnas/projects/M3BENCH/data/inputs/MIMM/"
RESULTS_ROOT="${PROJECT_ROOT}/results/results_reduced_${ENDPOINT}"

# Proposed tuning grid
SEEDS="22,2002,4,18473,55602"
MISSING_PATTERN_SEED=2026

INNER_SPLITS=5
OUTER_SPLITS=5
EPOCHS=80

BATCH_SIZE_GRID="16,32"
LR_GRID="1e-5,5e-5,1e-4"
FUSION_HIDDEN_DIM_GRID="32,64"
FUSION_HIDDEN_LAYERS_GRID="1"
FUSION_BATCHNORM_GRID="false"
MODALITY_HIDDEN_LAYERS_GRID="1"
DROPOUT_GRID="0.2,0.1"
IMPUTATION_METHOD="zero"

# Missingness experiments
MISSING_LOCATION_GRID="global"
TRAIN_MISSING_PROP_GRID="0.0"
TEST_MISSING_PROP_GRID="0.0"

# Run training
python "${PROJECT_ROOT}/main.py" \
  --model "MLP" \
  --dataset "MIMM" \
  --odir "${RESULTS_ROOT}" \
  --dataset_dir "${DATA_ROOT}" \
  --endpoint "${ENDPOINT}" \
  --inner_splits "${INNER_SPLITS}" \
  --outer_splits "${OUTER_SPLITS}" \
  --epochs "${EPOCHS}" \
  --retrain_outer "${RETRAIN_OUTER}" \
  --radio_aggregation_method "${RADIO_AGGREGATION_METHOD}" \
  --batch_size "${BATCH_SIZE_GRID}" \
  --learning_rate "${LR_GRID}" \
  --fusion_hidden_dim "${FUSION_HIDDEN_DIM_GRID}" \
  --fusion_hidden_layers "${FUSION_HIDDEN_LAYERS_GRID}" \
  --fusion_batchnorm "${FUSION_BATCHNORM_GRID}" \
  --modality_hidden_layers "${MODALITY_HIDDEN_LAYERS_GRID}" \
  --dropout "${DROPOUT_GRID}" \
  --imputation_method "${IMPUTATION_METHOD}" \
  --train_missing_prop "${TRAIN_MISSING_PROP_GRID}" \
  --missing_location "${MISSING_LOCATION_GRID}" \
  --test_missing_prop "${TEST_MISSING_PROP_GRID}" \
  --seeds "${SEEDS}" \
  --missing_pattern_seed "${MISSING_PATTERN_SEED}" \
  "${WANDB_ARGS[@]}"
