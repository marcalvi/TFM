#!/usr/bin/env bash
export WANDB_DIR=/nfs/rnas/workspaces/malbesa/TFM/methods/runs/wandb
set -euo pipefail

# Activate conda environment
#source /opt/miniconda3/etc/profile.d/conda.sh
source /home/osiris-user/anaconda3/etc/profile.d/conda.sh
conda activate TFM_5090

# WandB login token
WANDB_LOGIN_KEY="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"
if [[ -n "${WANDB_LOGIN_KEY}" ]]; then
  wandb login "${WANDB_LOGIN_KEY}"
else
  echo "WANDB_LOGIN_KEY not set; skipping wandb login."
fi

WANDB_ARGS=()
if [[ -n "${WANDB_LOGIN_KEY}" ]]; then
  WANDB_ARGS+=(--wandb --wandb_project "HealNet" --wandb_mode "online")
fi

PROJECT_ROOT="/home/osiris-user/Desktop/TFM/methods"
DATA_ROOT="/nfs/rnas/projects/M3BENCH/data/inputs/MIMM/"
RESULTS_ROOT="${PROJECT_ROOT}/results"

ENDPOINT="OS_6"
SEEDS="22,2002,4,18473,55602"
MISSING_PATTERN_SEED=2026
INNER_SPLITS=5
OUTER_SPLITS=5
EPOCHS=80

# HP grid
LR_GRID="2e-5,5e-5"
BATCH_SIZE_GRID="4,8,16"
HEALNET_DEPTH_GRID="1,2"
HEALNET_NUM_LATENTS_GRID="16,32,64"
HEALNET_LATENT_DIM_GRID="64"
HEALNET_LATENT_HEADS_GRID="2"
HEALNET_CROSS_HEADS_GRID="1"
HEALNET_CROSS_DIM_HEAD_GRID="64"
HEALNET_LATENT_DIM_HEAD_GRID="64"
HEALNET_NUM_FREQ_BANDS_GRID="2"
HEALNET_DROPOUT_GRID="0.2,0.3"
HEALNET_SELF_PER_CROSS_ATTN_GRID="1"

# Missingness experiments
MISSING_LOCATION_GRID="global, path, radio, clin, blood, radio_report"
TRAIN_MISSING_PROP_GRID="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_PROP_GRID="0.0,0.2,0.4,0.6,0.8"

python "${PROJECT_ROOT}/main.py" \
  --model "HealNet" \
  --dataset "MIMM" \
  --odir "${RESULTS_ROOT}" \
  --dataset_dir "${DATA_ROOT}" \
  --endpoint "${ENDPOINT}" \
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
  --healnet_attn_dropout "${HEALNET_DROPOUT_GRID}" \
  --healnet_ff_dropout "${HEALNET_DROPOUT_GRID}" \
  --healnet_self_per_cross_attn "${HEALNET_SELF_PER_CROSS_ATTN_GRID}" \
  --train_missing_prop "${TRAIN_MISSING_PROP_GRID}" \
  --missing_location "${MISSING_LOCATION_GRID}" \
  --test_missing_prop "${TEST_MISSING_PROP_GRID}" \
  --seeds "${SEEDS}" \
  --missing_pattern_seed "${MISSING_PATTERN_SEED}" \
  "${WANDB_ARGS[@]}"
