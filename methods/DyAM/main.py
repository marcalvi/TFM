print("Importing packages")
import os
import pandas as pd
import numpy as np
import argparse
import time
import torch
import ast
from sklearn.model_selection import StratifiedKFold
from train_ncv import nested_cv

def get_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--odir", help = "Output directory to which the chosen modalities will be saved",
                        type = str, required=True)
    parser.add_argument("--model", help="Model healnet or dyam",
                        type=str, required=True)
    parser.add_argument("--endpoint", help = "The label to predict",
                        type = str, required=True)
    parser.add_argument("--inst_data", help = "Path to the .csv containing at least two cols: ID (of each instance) and the endpoint name",
                        type = str, required=True)
    
    # Multimodal data arguments
    parser.add_argument("--patho_data", help = "Path to the .csv containing the pathology embeddings",
                        type = str, default = None)
    parser.add_argument("--clin_data", help = "Path to the .csv containing the clinical data",
                        type = str, default = None)
    parser.add_argument("--blood_data", help = "Path to the .csv containing the blood data",
                        type = str, default = None)   
    parser.add_argument("--body_composition_data", help="Path to the .csv containing radiology embeddings for other lesions",
                        type=str, default=None)
    parser.add_argument("--lung_data", help="Path to the .csv containing radiology embeddings for lung lesions",
                        type=str, default=None)
    parser.add_argument("--liver_data", help="Path to the .csv containing radiology embeddings for liver lesions",
                        type=str, default=None)
    parser.add_argument("--radio_data", help = "Path to the .csv containing the radiology embeddings",
                        type = str, default = None)
    parser.add_argument("--radio_report_data", help = "Path to the .csv containing the radiology_report data",
                        type = str, default = None)
    parser.add_argument("--radiology-agg", help = "Aggregation method for the radiology data of different lesions. One of: max, avg3max, avg5max, avg, wavg",
                        type = str, default = 'max')
    
    # Hyperparameters
    parser.add_argument("--inner_splits", help = "Number of splits in the inner loop",
                        type = int, default = 5)
    parser.add_argument("--outer_splits", help = "Number of splits in the outer loop",
                        type = int, default = 5)
    parser.add_argument("--batch_size", help = "Batch size",
                        type = int, default = 16)
    parser.add_argument("--epochs", help = "Number of epochs in the inner loop",
                        type = int, default = 80)
    parser.add_argument("--learning_rate", help = "Learning rate",
                        type = float, default = 5e-5)
    parser.add_argument('--seeds', help = "List of seeds to be used (separated by comma)",
                        type = str, default = "123")
    parser.add_argument('--mode', help = "train or deploy",
                        type = str, default = "train")
    
    
    args = parser.parse_args()

    return (args.odir, args.inst_data, args.model, args.endpoint, args.patho_data, args.radio_data, args.clin_data, args.blood_data,
            args.body_composition_data, args.lung_data, args.liver_data, args.radio_report_data, args.radiology_agg, args.inner_splits, 
            args.outer_splits, args.batch_size, args.epochs, args.learning_rate, args.seeds, args.mode)


def main():

    # Start timer
    start_time = time.time()
    print("Running")

    # Load arguments
    odir, inst_dir, model_type, endpoint, patho_dir, radio_dir, clin_dir, blood_dir, body_composition_dir, lung_dir, liver_dir, radio_report_dir, agg, inner_splits, outer_splits, batch_size, epochs, lr, seeds, mode = get_args()
    
    # Create output directory
    use_modalities = [ 
        int(x is not None) for x in (
            patho_dir, radio_dir, clin_dir, blood_dir,
            body_composition_dir, lung_dir, liver_dir, radio_report_dir
        )
    ]
    use_patho, use_radio, use_clin, use_blood, use_body_composition, use_lung, use_liver, use_radio_report  = use_modalities

    modalities = []
    if use_clin:
        modalities.append('path')
    if use_patho:
        modalities.append('rad')
    if use_radio:
        modalities.append('clin')
    if use_blood:
        modalities.append('blood')
    if use_body_composition:
        modalities.append('body_composition')
    if use_lung:
        modalities.append('lung')
    if use_liver:
        modalities.append('liver')
    if use_radio_report:
        modalities.append('radio_report')
        
    odir = os.path.join(odir + '-'.join(modalities), endpoint, model_type)
    os.makedirs(odir, exist_ok = True)

    # Read dfs and select only patients matching with inst_df (patients.csv)
    inst_df = pd.read_csv(inst_dir)
    df = inst_df[['patient', f'{endpoint}_label']]
  
    if use_patho:
        pat_df = pd.read_csv(patho_dir)
        pat_df = pat_df.rename(columns = lambda x: x.replace('embedding_', 'patho_'))
        pat_df = pd.merge(pat_df, inst_df[['patient']], on='patient')[['patient'] + [col for col in pat_df.columns if col.startswith('patho_')]]
        df = pd.merge(df, pat_df, on = "patient", how = 'inner')

    if use_clin:
        clin_df = pd.read_csv(clin_dir)
        clin_df = clin_df.rename(columns=lambda x: 'clin_' + x if x != 'patient' else x)
        clin_df = pd.merge(clin_df, inst_df[['patient']], on='patient')
        df = pd.merge(df, clin_df, on = "patient", how = 'inner')

    if use_blood:
        blood_df = pd.read_csv(blood_dir)
        blood_df = blood_df.rename(columns=lambda x: 'blood_' + x if x != 'patient' else x)
        blood_df = pd.merge(blood_df, inst_df[['patient']], on = 'patient')
        df = pd.merge(df, blood_df, on = "patient", how = 'inner')

    if use_body_composition:
        body_composition_df = pd.read_csv(body_composition_dir)
        body_composition_df = body_composition_df.rename(columns=lambda x: 'body_composition_' + x if x != 'patient' else x)
        body_composition_df = pd.merge(body_composition_df, inst_df[['patient']], on = 'patient')
        df = pd.merge(df, body_composition_df, on = "patient", how = 'inner')

    if use_lung:
        lung_df = pd.read_csv(lung_dir)
        lung_df = lung_df.rename(columns=lambda x: 'lung_' + x if x != 'patient' else x)
        lung_df = pd.merge(lung_df, inst_df[['patient']], on = 'patient')
        df = pd.merge(df, lung_df, on = "patient", how = 'inner')

    if use_liver:
        liver_df = pd.read_csv(liver_dir)
        liver_df = liver_df.rename(columns=lambda x: 'liver_' + x if x != 'patient' else x)
        liver_df = pd.merge(liver_df, inst_df[['patient']], on = 'patient')
        df = pd.merge(df, liver_df, on = "patient", how = 'inner')

    if use_radio_report:
        radio_report_df = pd.read_csv(radio_report_dir)
        radio_report_df = radio_report_df.rename(columns=lambda x: 'radio_report_' + x if x != 'patient' else x)
        radio_report_df = pd.merge(radio_report_df, inst_df[['patient']], on = 'patient')
        df = pd.merge(df, radio_report_df, on = "patient", how = 'inner')

    if use_radio:
        rad_df = pd.read_csv(radio_dir).rename(columns = lambda x: x.replace('pred_', 'radio_')) 
        rad_df = rad_df.drop(columns=['image_path', 'lesion_tag'], errors='ignore')
        rad_df = rad_df.groupby('patient').mean().reset_index() # Group by patient and average all features - multiple observations merged into a single patient-level profile
        rad_df = pd.merge(rad_df, inst_df[['patient']], on='patient')[['patient'] + [col for col in rad_df.columns if col.startswith('radio_')]] #+ ['vol_bs', 'lesion_tag']]
        df = pd.merge(df, rad_df, on = "patient", how = 'inner')

    # Deletting unnamed cols and casting to float 
    df = df.loc[:, ~df.columns.str.contains('Unnamed')] 
    df = df.astype(float)
    
    # Create dict with individual dfs
    dfs = {}
    if use_patho: dfs["path"] = pat_df
    if use_radio: dfs["rad"] = rad_df
    if use_clin: dfs["clin"] = clin_df
    if use_blood: dfs["blood"] = blood_df
    if use_body_composition: dfs["body_composition"] = body_composition_df
    if use_lung: dfs["lung"] = lung_df                      
    if use_liver: dfs["liver"] = liver_df
    if use_radio_report: dfs["radio_report"] = radio_report_df  

    # Force patient IDs to int and set as index WITHOUT dropping the column
    for mod in dfs:
        dfs[mod]['patient'] = dfs[mod]['patient'].astype(int)
        dfs[mod] = dfs[mod].set_index('patient', drop=False) # drop=False keeps the 'patient' column in the dataframe

    inst_df['patient'] = inst_df['patient'].astype(int)
    inst_df = inst_df.set_index('patient', drop=False)

    print("Dataframes read. starting DL training.")

    # Modality availability dataframe
    summary_df = pd.DataFrame(index=inst_df.index)
    summary_df['patient'] = inst_df.index

    for mod in modalities:
        summary_df[mod] = summary_df.index.isin(dfs[mod].index).astype(int)

    summary_df.to_csv(os.path.join(odir, "modality_availability.csv"), index = False)
    total_patients = len(summary_df)
    availability_counts = summary_df[modalities].sum()
    
    print(f"Total Patients in inst_df: {total_patients}")
    print("\nNumber of patients with each modality available:")
    print(availability_counts)

    # Set seeds for reproducibility
    seeds_list = [int(s) for s in seeds.split(",")] if seeds else [123] 

    # Lisits to store results across seeds
    all_inner_results = []
    all_outer_results = []
    all_histories_train=[]
    all_seeds_bundles = {}

    for ind,seed in enumerate(seeds_list):
        # Run the nested cv
        inner_results_df, outer_results_df, all_histories_df, bundles = nested_cv(dfs, inst_df, use_modalities, batch_size, epochs, lr, seed, model_type)
        
        # Add seed column to results
        inner_results_df['seed'] = seed
        outer_results_df['seed'] = seed
        all_histories_df['seed'] = seed

        # Concatenate results
        all_inner_results.append(inner_results_df)
        all_outer_results.append(outer_results_df)
        all_histories_train.append(all_histories_df)

        all_seeds_bundles[f"seed_{seed}"] = bundles
        print(f"Inner and outer loops with seed {seed} completed.")
        print(f"Seed {seed} completed in {time.time() - start_time:.2f} seconds.\n")

    # Concatenate results across seeds and save to csv
    final_inner = pd.concat(all_inner_results)
    final_outer = pd.concat(all_outer_results)
    final_histories = pd.concat(all_histories_train)

    # Save results of nested cv    
    final_inner.to_csv(os.path.join(odir, "inner_loop_"+endpoint+("_"+agg if use_radio else '')+".csv"), index = False)
    final_outer.to_csv(os.path.join(odir, "outer_loop_"+endpoint+("_"+agg if use_radio else '')+".csv"), index = False)
    final_histories.to_csv(os.path.join(odir, endpoint,"ncv_training_logs.csv"), index=False)

    # Save the bundles of each seed (models, predictions, etc.) in a .pt file
    torch.save(all_seeds_bundles, os.path.join(odir, "all_seeds_ensemble_bundles.pt"))


if __name__ == "__main__":
    main()
    print("Finish!")
