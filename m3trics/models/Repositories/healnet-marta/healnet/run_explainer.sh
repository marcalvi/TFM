export WANDB_DIR=/nfs/rnas/workspaces/malbesa/TFM/m3trics/runs/wandb

log_dir="/nfs/rnas/projects/mmCRC/git/healnet-adoption/logs/hp_53fdcc7725/r07ta2rg"

source activate /home/osiris-user/anaconda3/envs/healnet
python /nfs/rnas/projects/mmCRC/git/healnet-adoption/healnet/models/mmcrc_explainer.py --log_dir ${log_dir} --show --n 2