#!/bin/bash
export WANDB_DIR=/nfs/rnas/workspaces/malbesa/TFM/m3trics/runs/wandb


API_KEY="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"

MODE="sweep"
PROJECT_NAME="mmCRC_CV"
DATASET="mmCRC"
SWEEP_CONFIG="config/sweep_grid.yaml"

source /home/osiris-user/anaconda3/envs/healnet/bin/activate
python /nfs/rnas/projects/mmCRC/git/healnet-adoption/healnet/main.py --mode $MODE --sweep_config $SWEEP_CONFIG --project_name $PROJECT_NAME --dataset $DATASET --api_key $API_KEY
