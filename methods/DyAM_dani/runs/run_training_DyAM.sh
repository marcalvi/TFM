#!/bin/bash

# Activate the conda environment
source activate /home/osiris-user/miniconda3/envs/healnet

# Run the Python script with variables
python /nfs/rnas/projects/MIMM/code/3-2-train_DL/DyAM/main.py \
--odir /nfs/rnas/projects/MIMM/results/new_results/DL/DyAM/ \
--inst_data /nfs/rnas/projects/MIMM/data/inputs_ML/all_internal/patients_all_internal.csv \
--model healnet \
--endpoint PFS_6 \
--patho_data /nfs/rnas/projects/MIMM/data/inputs_ML/all_internal/titan_df_all_internal.csv \
--radio_data /nfs/rnas/projects/MIMM/data/inputs_ML/all_internal/fmcib_all_internal.csv \
--clin_data /nfs/rnas/projects/MIMM/data/inputs_ML/all_internal/clinical_df_all_internal.csv \
--body_composition_data /nfs/rnas/projects/MIMM/data/inputs_ML/all_internal/body_composition_all_internal.csv \
--lung_data /nfs/rnas/projects/MIMM/data/inputs_ML/all_internal/fmcib_lungs_all_internal.csv \
--liver_data /nfs/rnas/projects/MIMM/data/inputs_ML/all_internal/fmcib_liver_all_internal.csv \
--radio_report_data /nfs/rnas/projects/MIMM/data/inputs_ML/all_internal/reports_df_32_all_internal.csv \
--blood_data /nfs/rnas/projects/MIMM/data/inputs_ML/all_internal/blood_df_all_internal.csv \
--radiology-agg avg \
--seeds 1