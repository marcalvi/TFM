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
BATCH_SIZE_GRID="4"
LR_GRID="1e-4,5e-5"

# HealNet-specific
HEALNET_DEPTH_GRID="3"
HEALNET_NUM_FREQ_BANDS_GRID="2"
HEALNET_NUM_LATENTS_GRID="128"
HEALNET_LATENT_DIM_GRID="64"
HEALNET_CROSS_HEADS_GRID="1"
HEALNET_LATENT_HEADS_GRID="4"
HEALNET_CROSS_DIM_HEAD_GRID="64"
HEALNET_LATENT_DIM_HEAD_GRID="64"
HEALNET_ATTN_DROPOUT_GRID="0.0"
HEALNET_FF_DROPOUT_GRID="0.0"
HEALNET_SELF_PER_CROSS_ATTN_GRID="0"

# Missingness experiments
TRAIN_MISSING_LOCATION_GRID="global, path, radio, clin, blood, radio_report"
TRAIN_MISSING_PROB_GRID="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_LOCATION_GRID="global, path, radio, clin, blood, radio_report"
TEST_MISSING_PROB_GRID="0.0,0.2,0.4,0.6,0.8"

python "${PROJECT_ROOT}/main.py" \
  --dataset "MIMM" \
  --odir "${RESULTS_ROOT}" \
  --endpoint "${ENDPOINT}" \
  --model "HealNet" \
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
  --healnet_depth "${HEALNET_DEPTH_GRID}" \
  --healnet_num_freq_bands "${HEALNET_NUM_FREQ_BANDS_GRID}" \
  --healnet_num_latents "${HEALNET_NUM_LATENTS_GRID}" \
  --healnet_latent_dim "${HEALNET_LATENT_DIM_GRID}" \
  --healnet_cross_heads "${HEALNET_CROSS_HEADS_GRID}" \
  --healnet_latent_heads "${HEALNET_LATENT_HEADS_GRID}" \
  --healnet_cross_dim_head "${HEALNET_CROSS_DIM_HEAD_GRID}" \
  --healnet_latent_dim_head "${HEALNET_LATENT_DIM_HEAD_GRID}" \
  --healnet_attn_dropout "${HEALNET_ATTN_DROPOUT_GRID}" \
  --healnet_ff_dropout "${HEALNET_FF_DROPOUT_GRID}" \
  --healnet_self_per_cross_attn "${HEALNET_SELF_PER_CROSS_ATTN_GRID}" \
  --train_missing_prob "${TRAIN_MISSING_PROB_GRID}" \
  --train_missing_location "${TRAIN_MISSING_LOCATION_GRID}" \
  --test_missing_prob "${TEST_MISSING_PROB_GRID}" \
  --test_missing_location "${TEST_MISSING_LOCATION_GRID}" \
  --seeds "${SEEDS}" \
  --missing_pattern_seed "${MISSING_PATTERN_SEED}" \
  --wandb \
  --wandb_project "HEALNet" \
  --wandb_mode "online"
