#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate TFM

# Optional WandB login from environment variable
wandb login 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

# Define paths
PROJECT_ROOT="/Users/marcalbesa/Desktop/TFM/git_exp/methods"
DATA_ROOT="/Users/marcalbesa/Desktop/TFM/data/MIMM"
RESULTS_ROOT="${PROJECT_ROOT}/results"

# Define endpoint
ENDPOINT="OS_6"

# Proposed tuning grid
SEEDS="22,33"
INNER_SPLITS=5
OUTER_SPLITS=5
EPOCHS=80

BATCH_SIZE_GRID="16,32"
LR_GRID="5e-5,1e-4"
FUSION_HIDDEN_DIM_GRID="32,64"
FUSION_HIDDEN_LAYERS_GRID="1,2"
MODALITY_HIDDEN_LAYERS_GRID="1,2"
DROPOUT_GRID="0.2,0.1"

# Missingness experiments
MISSING_SCOPE_GRID="train,test,both,none"
MISSING_LOCATION_GRID="global, path, radio, clin, blood, radio_report"
MISSING_PROB_GRID="0.2,0.4,0.6,0.8"

# Run training
python "${PROJECT_ROOT}/MLPs/main.py" \
  --dataset "MIMM" \
  --odir "${RESULTS_ROOT}" \
  --endpoint "${ENDPOINT}" \
  --model "ZI_MLP" \
  --inst_data "${DATA_ROOT}/patients_mimm.csv" \
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
  --fusion_hidden_dim "${FUSION_HIDDEN_DIM_GRID}" \
  --fusion_hidden_layers "${FUSION_HIDDEN_LAYERS_GRID}" \
  --modality_hidden_layers "${MODALITY_HIDDEN_LAYERS_GRID}" \
  --dropout "${DROPOUT_GRID}" \
  --missing_prob "${MISSING_PROB_GRID}" \
  --missing_scope "${MISSING_SCOPE_GRID}" \
  --missing_location "${MISSING_LOCATION_GRID}" \
  --seeds "${SEEDS}" \
  --wandb \
  --wandb_project "ZI_MLPs" \
  --wandb_mode "online"
