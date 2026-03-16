# Generate clean CSV files for the mmCRC dataset

import pandas as pd
import os
import sys

project_path = os.path.abspath(os.path.join(os.getcwd(), "../mmCRC_project/"))
sys.path.append(project_path)

try:
    import utils
except ImportError as e:
    print(f"Failed to import utils: {e}")

from utils.merge_clinical_rad_path_data import merge_clinical_rad_path_data

# Specify radiology FM used
FM = 'FM-CIB'

#Path to the data folder to save the clean csv files
healnet_data_path = os.path.join(os.getcwd(), 'data', 'mmCRC')

y_disc_bin_edges_path_mOS = '/nfs/rnas/projects/mmCRC/git/healnet-adoption/data/mmCRC/y_disc_bin_edges_mOS.csv'
y_disc_bin_edges_path_PFS = '/nfs/rnas/projects/mmCRC/git/healnet-adoption/data/mmCRC/y_disc_bin_edges_PFS.csv' 

# Read raw data files
clinical_data_path = '/nfs/rnas/projects/mmCRC/git/mmCRC_project/data/target_cohort_mmCRC_BL_CTs_HE_first_line_30122025_categorised.csv'
sap_patient_to_include = '/nfs/rnas/projects/mmCRC/git/mmCRC_project/data/cohort_tracking/v2_cohort_sap.csv'
list_patients = pd.read_csv(sap_patient_to_include)['sap'].astype(str).unique().tolist()

if FM == 'CT_FM':
    radiology_data_path = '/nfs/rnas/projects/mmCRC/git/mmCRC_project/data/checkme/embeddings/CT-FM-whole-liver-with-dilation/v2_cohort_mmCRC_whole_liver_with_dilation_radiology_embeddings.csv'
elif FM == 'FM-CIB':
    radiology_data_path = '/nfs/rnas/projects/mmCRC/git/mmCRC_project/data/checkme/embeddings/FM-CIB-whole-liver-with-dilation/v2_cohort_mmCRC_whole_liver_with_dilation_radiology_embeddings.csv'
pathology_data_path  ='/nfs/rnas/projects/mmCRC/git/mmCRC_project/data/checkme/embeddings/pathology_conch_v1_5_titan_512'

clinical, ct_fm_embeddings, ct_fm_embeddings_aggregated, pathology_features, feats = merge_clinical_rad_path_data(list_patients, clinical_data_path, radiology_data_path, pathology_embeddings_path =  pathology_data_path, lesions_aggregation_strategy = 'whole_liver', pathology_aggregation_strategy = 'titan')

def discretize_event_times(event_times, bin_edges):
    """
    Discretize continuous event times into bins defined by bin_edges.
    Parameters
    ----------
    event_times: pd.Series
        Continuous event times.
    bin_edges: pd.Series
        Edges of the bins for discretization.
    Returns
    -------
    pd.Series
        Discretized event time indices.
    """
    bin_indices = pd.cut(event_times, bins=[-float('inf')] + bin_edges.tolist() + [float('inf')], labels=False)
    return bin_indices

# Clean clinical data
model_input = feats.drop(columns=['sap', 'n_treatment_line', 'CT_BL_date', 'CT_PD_date', 'CT_BL_relapse_date', 'CT_BL_date_filled','he_date', 'mOS_months', 'mOS_event', 'OS_months', 'PFS_months',
       'PFS_event', 'CT_BL_filename', 'pathology_feats_path'])
model_input.set_index('id', inplace=True)

clinical_columns = [col for col in model_input.columns if 'pred_' not in col and 'feat_' not in col]
radiology_columns = [col for col in model_input.columns if 'pred_' in col]
pathology_columns = [col for col in model_input.columns if 'feat_' in col]

clinical_input_data = model_input[clinical_columns]
radiology_input_data = model_input[radiology_columns]
pathology_input_data = model_input[pathology_columns]
target = feats[['id', 'mOS_months', 'mOS_event', 'PFS_months', 'PFS_event']].set_index('id')

# drop rows where ALL values are NaN in BOTH pathology and radiology
mask_path = pathology_input_data.isna().all(axis=1)
mask_rad = radiology_input_data.isna().all(axis=1)

print(f"Rows to drop in pathology: {mask_path.sum()}")
print(f"Rows to drop in radiology: {mask_rad.sum()}")

path_idx_to_drop = mask_path[mask_path].index
rad_idx_to_drop = mask_rad[mask_rad].index

pathology_input_data = pathology_input_data.drop(index=path_idx_to_drop)
radiology_input_data = radiology_input_data.drop(index=rad_idx_to_drop)

# The target needs to have the censorship column, so 1 if the case is censored and 0 otherwise
target['mOS_censorship'] = 1 - target['mOS_event']
target['PFS_censorship'] = 1 - target['PFS_event']

y_disc_bin_edges_mOS = pd.read_csv(y_disc_bin_edges_path_mOS)
target['y_disc_mOS'] = discretize_event_times(target['mOS_months'], y_disc_bin_edges_mOS.iloc[:,0])

y_disc_bin_edges_PFS = pd.read_csv(y_disc_bin_edges_path_PFS)
target['y_disc_PFS'] = discretize_event_times(target['PFS_months'], y_disc_bin_edges_PFS.iloc[:,0])

# Save clean csv files
print("clinical_input_data", clinical_input_data.columns, "Saved in: ", healnet_data_path + '/v2_cohort_clinical_input_data.csv')
clinical_input_data.to_csv(healnet_data_path + '/v2_cohort_clinical_input_data.csv')
radiology_input_data.to_csv(healnet_data_path + '/v2_cohort_' + FM + '_radiology_input_data.csv')
pathology_input_data.to_csv(healnet_data_path + '/v2_cohort_pathology_input_data.csv')
target.to_csv(healnet_data_path + '/v2_cohort_target_data.csv', index=True)
