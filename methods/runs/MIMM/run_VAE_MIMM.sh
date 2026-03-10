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

# Proposed tuning grid
SEEDS="4,18473,55602"
#SEEDS="2002,4,18473,55602"
MISSING_PATTERN_SEED=2026
INNER_SPLITS=5
OUTER_SPLITS=5
EPOCHS=80

BATCH_SIZE_GRID="16,32"
LR_GRID="5e-5,1e-4"
FUSION_HIDDEN_DIM_GRID="32,64"
FUSION_HIDDEN_LAYERS_GRID="1"
MODALITY_HIDDEN_LAYERS_GRID="1"
DROPOUT_GRID="0.2,0.1"
IMPUTATION_METHOD="vae"

# VAE imputer hyperparameters
VAE_IMPUTER_LATENT_DIM=16
VAE_IMPUTER_HIDDEN_DIM=128
VAE_IMPUTER_EPOCHS=30
VAE_IMPUTER_BATCH_SIZE=64
VAE_IMPUTER_LR=1e-3
VAE_IMPUTER_BETA=1e-3

# Missingness experiments
MISSING_LOCATION_GRID="global, path, radio, clin, blood, radio_report"
TRAIN_MISSING_PROP_GRID="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_PROP_GRID="0.0,0.2,0.4,0.6,0.8"

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
  --batch_size "${BATCH_SIZE_GRID}" \
  --learning_rate "${LR_GRID}" \
  --fusion_hidden_dim "${FUSION_HIDDEN_DIM_GRID}" \
  --fusion_hidden_layers "${FUSION_HIDDEN_LAYERS_GRID}" \
  --modality_hidden_layers "${MODALITY_HIDDEN_LAYERS_GRID}" \
  --dropout "${DROPOUT_GRID}" \
  --imputation_method "${IMPUTATION_METHOD}" \
  --vae_imputer_latent_dim "${VAE_IMPUTER_LATENT_DIM}" \
  --vae_imputer_hidden_dim "${VAE_IMPUTER_HIDDEN_DIM}" \
  --vae_imputer_epochs "${VAE_IMPUTER_EPOCHS}" \
  --vae_imputer_batch_size "${VAE_IMPUTER_BATCH_SIZE}" \
  --vae_imputer_lr "${VAE_IMPUTER_LR}" \
  --vae_imputer_beta "${VAE_IMPUTER_BETA}" \
  --train_missing_prop "${TRAIN_MISSING_PROP_GRID}" \
  --missing_location "${MISSING_LOCATION_GRID}" \
  --test_missing_prop "${TEST_MISSING_PROP_GRID}" \
  --seeds "${SEEDS}" \
  --missing_pattern_seed "${MISSING_PATTERN_SEED}" \
  --wandb \
  --wandb_project "VAE_MLP" \
  --wandb_mode "offline"
