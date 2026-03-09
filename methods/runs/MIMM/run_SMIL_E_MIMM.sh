#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment
#source /opt/miniconda3/etc/profile.d/conda.sh
source /home/osiris-user/anaconda3/etc/profile.d/conda.sh
conda activate TFM

# Optional WandB login from environment variable
wandb login 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

PROJECT_ROOT="/home/osiris-user/Desktop/TFM/methods"
DATA_ROOT="/nfs/rnas/projects/M3BENCH/data/inputs/MIMM/"
RESULTS_ROOT="${PROJECT_ROOT}/results"

ENDPOINT="OS_6"
SEEDS="22,2002,4,18473,55602"
MISSING_PATTERN_SEED=2026
INNER_SPLITS=5
OUTER_SPLITS=5
EPOCHS=80

# Shared optimization
BATCH_SIZE_GRID="16,32"
LR_GRID="5e-5,1e-4"

# SMIL-E-specific
SMIL_E_LATENT_DIM_GRID="32,64"
SMIL_E_NUM_PRIORS_GRID="32,64"
SMIL_E_NUM_HEADS_GRID="4"
SMIL_E_DROPOUT_GRID="0.1,0.2"
SMIL_E_ALPHA_GRID="1e-2"
SMIL_E_BETA_GRID="1e-2"

# Missingness experiments
TRAIN_MISSING_LOCATION_GRID="global, path, radio, clin, blood, radio_report"
TRAIN_MISSING_PROB_GRID="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_LOCATION_GRID="global, path, radio, clin, blood, radio_report"
TEST_MISSING_PROB_GRID="0.0,0.2,0.4,0.6,0.8"

python "${PROJECT_ROOT}/main.py" \
  --model "SMIL_E" \
  --dataset "MIMM" \
  --odir "${RESULTS_ROOT}" \
  --dataset_dir "${DATA_ROOT}" \
  --endpoint "${ENDPOINT}" \
  --inner_splits "${INNER_SPLITS}" \
  --outer_splits "${OUTER_SPLITS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE_GRID}" \
  --learning_rate "${LR_GRID}" \
  --smil_e_latent_dim "${SMIL_E_LATENT_DIM_GRID}" \
  --smil_e_num_priors "${SMIL_E_NUM_PRIORS_GRID}" \
  --smil_e_num_heads "${SMIL_E_NUM_HEADS_GRID}" \
  --smil_e_dropout "${SMIL_E_DROPOUT_GRID}" \
  --smil_e_alpha "${SMIL_E_ALPHA_GRID}" \
  --smil_e_beta "${SMIL_E_BETA_GRID}" \
  --train_missing_prob "${TRAIN_MISSING_PROB_GRID}" \
  --train_missing_location "${TRAIN_MISSING_LOCATION_GRID}" \
  --test_missing_prob "${TEST_MISSING_PROB_GRID}" \
  --test_missing_location "${TEST_MISSING_LOCATION_GRID}" \
  --seeds "${SEEDS}" \
  --missing_pattern_seed "${MISSING_PATTERN_SEED}" \
  --wandb \
  --wandb_project "SMIL_E" \
  --wandb_mode "online"
