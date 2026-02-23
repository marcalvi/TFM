print("Importing packages")
import os
import pandas as pd
import numpy as np
import argparse
import time
from sklearn.model_selection import StratifiedKFold
from train_ncv2 import nested_cv
import torch
import ast


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--odir", help = "Output directory to which the chosen modalities will be saved",
                        type = str, required=True)
    parser.add_argument("--inst_data", help = "Path to the .csv containing at least two cols: ID (of each instance) and the endpoint name",
                        type = str, required=True)
    parser.add_argument("--model", help="Model healnet or dyam",
                        type=str, required=True)
    parser.add_argument("--endpoint", help = "The label to predict",
                        type = str, required=True)
    parser.add_argument("--patho_data", help = "Path to the .csv containing the pathology embeddings",
                        type = str, default = None)
    parser.add_argument("--radio_data", help = "Path to the .csv containing the radiology embeddings",
                        type = str, default = None)
    parser.add_argument("--clin_data", help = "Path to the .csv containing the clinical data",
                        type = str, default = None)
    parser.add_argument("--blood_data", help = "Path to the .csv containing the blood data",
                        type = str, default = None)   

    parser.add_argument("--body_composition_data", help="Path to the .csv containing radiology embeddings for other lesions",
                        type=str, default=None)

    #Embeddings lung y liver
    parser.add_argument("--lung_data", help="Path to the .csv containing radiology embeddings for lung lesions",
                        type=str, default=None)
    parser.add_argument("--liver_data", help="Path to the .csv containing radiology embeddings for liver lesions",
                        type=str, default=None)

    parser.add_argument("--radio_report_data", help = "Path to the .csv containing the radiology_report data",
                        type = str, default = None)


    parser.add_argument("--radiology-agg", help = "Aggregation method for the radiology data of different lesions. One of: max, avg3max, avg5max, avg, wavg",
                        type = str, default = 'max')
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

    start_time = time.time()
    print("Running")

    # Load arguments
    odir, inst_dir, model_type, endpoint, patho_dir, radio_dir, clin_dir, blood_dir, body_composition_dir, lung_dir, liver_dir, radio_report_dir, agg, inner_splits, outer_splits, batch_size, epochs, lr, seeds, mode = get_args()
    
    # Define which modalities will be used based on input
    use_patho = 0 if patho_dir is None else 1
    use_radio = 0 if radio_dir is None else 1
    use_clin = 0 if clin_dir is None else 1
    use_blood = 0 if blood_dir is None else 1
    use_body_composition = 0 if body_composition_dir is None else 1
    use_lung = 0 if lung_dir is None else 1
    use_liver = 0 if liver_dir is None else 1
    use_radio_report = 0 if radio_report_dir is None else 1

    # Define the output directory according to the input modalities
    modalities = []
    if use_clin:
        modalities.append('clin')
    if use_patho:
        modalities.append('path')
    if use_radio:
        modalities.append('rad')
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
    odir = odir+'-'.join(modalities)
    odir = os.path.join(odir, endpoint)
    odir = os.path.join(odir, model_type)
    os.makedirs(odir, exist_ok = True)

    use_modalities = [use_radio, use_patho, use_clin, use_blood, use_body_composition, use_lung, use_liver, use_radio_report]

    # Read dfs and select only patients matching with inst_df
    inst_df = pd.read_csv(inst_dir)
    df = inst_df[['patient', f'{endpoint}_label']]

    if use_patho:
        pat_df = pd.read_csv(patho_dir).rename(columns = lambda x: x.replace('embedding_', 'patho_'))
        pat_df = pd.merge(pat_df, inst_df[['patient']], on='patient')[['patient'] + [col for col in pat_df.columns if col.startswith('patho_')]]
        df = pd.merge(df, pat_df, on = "patient", how = 'inner')

    if use_radio:
        rad_df = pd.read_csv(radio_dir).rename(columns = lambda x: x.replace('pred_', 'radio_')) 
        #rad_df['patient'] = rad_df['image_path'].apply(lambda x: x.split('_')[0]).astype('int64')
        #rad_df = rad_df.groupby(['image_path', 'lesion_tag', 'patient']).mean().reset_index()
        # 1. Drop the columns that are too specific (lesion and path)
        # We don't want these to be part of our "key" or our "data"
        cols_to_drop = ['image_path', 'lesion_tag']
        rad_df = rad_df.drop(columns=cols_to_drop, errors='ignore')

        # 2. Group by patient and average all features
        # This collapses multiple observations into a single patient-level profile
        rad_df = rad_df.groupby('patient').mean().reset_index()

        # Also get the df ready for lesion aggregation 
        rad_df = pd.merge(rad_df, inst_df[['patient']], on='patient')[['patient'] + [col for col in rad_df.columns if col.startswith('radio_')]] #+ ['vol_bs', 'lesion_tag']]
        lesion_volumes = rad_df[['patient']]
        # Perform the radiology lesion aggregation
        if agg == 'max':
            index = lesion_volumes.groupby('patient')['vol_bs'].nlargest(1).reset_index()['level_1'].tolist()
            rad_df = rad_df.iloc[index]
        elif agg == 'avg3max':
            index = lesion_volumes.groupby('patient')['vol_bs'].nlargest(3).reset_index()['level_1'].tolist()
            rad_df = rad_df.iloc[index].groupby('patient').mean().reset_index()
        elif agg == 'avg':
            rad_df = rad_df.groupby('patient').mean().reset_index()
        elif agg == 'wavg':
            totalvol = pd.merge(lesion_volumes[['patient']], lesion_volumes[['patient', 'vol_bs']].groupby('patient').sum().reset_index())
            rad_df[[f'radio_{k}' for k in range(4096)]] = lesion_volumes[['vol_bs']].to_numpy() / totalvol[['vol_bs']].to_numpy() * rad_df.to_numpy()[:,1:]
            rad_df = rad_df.groupby('patient').sum().reset_index()
        else:
            raise 'Aggregation method not supported'
        rad_df = rad_df[['patient'] + [col for col in rad_df.columns if col.startswith('radio_')]]
        df = pd.merge(df, rad_df, on = "patient", how = 'inner')

    if use_clin:
        clin_df = pd.read_csv(clin_dir)
        clin_df.rename(columns=lambda x: 'clin_' + x if x != 'patient' else x, inplace=True)
        clin_df = pd.merge(clin_df, inst_df[['patient']], on='patient')
        df = pd.merge(df, clin_df, on = "patient", how = 'inner')

    if use_blood:
        blood_df = pd.read_csv(blood_dir)
        blood_df.rename(columns=lambda x: 'blood_' + x if x != 'patient' else x, inplace=True)
        blood_df = pd.merge(blood_df, inst_df[['patient']], on = 'patient')
        df = pd.merge(df, blood_df, on = "patient", how = 'inner')

    if use_body_composition:
        body_composition_df = pd.read_csv(body_composition_dir)
        body_composition_df.rename(columns=lambda x: 'body_composition_' + x if x != 'patient' else x, inplace=True)
        body_composition_df = pd.merge(body_composition_df, inst_df[['patient']], on = 'patient')
        df = pd.merge(df, body_composition_df, on = "patient", how = 'inner')

    if use_lung:
        lung_df = pd.read_csv(lung_dir)
        lung_df.rename(columns=lambda x: 'lung_' + x if x != 'patient' else x, inplace=True)
        lung_df = pd.merge(lung_df, inst_df[['patient']], on = 'patient')
        df = pd.merge(df, lung_df, on = "patient", how = 'inner')

    if use_liver:
        liver_df = pd.read_csv(liver_dir)
        liver_df.rename(columns=lambda x: 'liver_' + x if x != 'patient' else x, inplace=True)
        liver_df = pd.merge(liver_df, inst_df[['patient']], on = 'patient')
        df = pd.merge(df, liver_df, on = "patient", how = 'inner')

    if use_radio_report:
        radio_report_df = pd.read_csv(radio_report_dir)
        radio_report_df.rename(columns=lambda x: 'radio_report_' + x if x != 'patient' else x, inplace=True)
        radio_report_df = pd.merge(radio_report_df, inst_df[['patient']], on = 'patient')
        df = pd.merge(df, radio_report_df, on = "patient", how = 'inner')


    # Deleting unnamed column if any
    df = df.loc[:, ~df.columns.str.contains('Unnamed')] 
    # Make sure all columns are in float
    df = df.astype(float)
    
    print("Dataframes read. starting DL training.")

    dfs= {
        "path": pat_df,
        "rad": rad_df,
        "clin": clin_df,
        "blood": blood_df,
        "body_composition": body_composition_df,
        "lung": lung_df,
        "liver": liver_df,
        "radio_report": radio_report_df
    }


    seeds_list = [int(s) for s in seeds.split(",")] if seeds else [123] 
    all_inner_results = []
    all_outer_results = []
    all_histories_train=[]

    # Force patient IDs to int and set as index WITHOUT dropping the column
    for mod in dfs:
        
        dfs[mod]['patient'] = dfs[mod]['patient'].astype(int)
        # drop=False keeps the 'patient' column in the dataframe
        dfs[mod] = dfs[mod].set_index('patient', drop=False)

    inst_df['patient'] = inst_df['patient'].astype(int)
    inst_df = inst_df.set_index('patient', drop=False)


    summary_df = pd.DataFrame(index=inst_df.index)
    summary_df['patient'] = inst_df.index
    modality_names = list(dfs.keys())
    for mod in modality_names:
        #dfs[mod].to_csv(os.path.join(odir, endpoint, f"df_state_{mod}.csv"), index=False)
        summary_df[mod] = summary_df.index.isin(dfs[mod].index).astype(int)

    summary_df.to_csv(os.path.join(odir, "modality_availability.csv"), index = False)
    total_patients = len(summary_df)
    availability_counts = summary_df[modality_names].sum()
    print(f"Total Patients in inst_df: {total_patients}")
    print("\nNumber of patients with each modality available:")
    print(availability_counts)
    all_seeds_bundles = {}

    for ind,seed in enumerate(seeds_list):
    # We define the inner and outer splits
        # Run the nested cv
        inner_results_df, outer_results_df, all_histories_df, bundles = nested_cv(dfs, inst_df, use_modalities, batch_size, epochs, lr, seed, model_type)
        
        #Add seed column to results
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

    final_inner = pd.concat(all_inner_results)
    final_outer = pd.concat(all_outer_results)
    final_histories = pd.concat(all_histories_train)

    # Save results of nested cv    
    final_inner.to_csv(os.path.join(odir, "inner_loop_"+endpoint+("_"+agg if use_radio else '')+".csv"), index = False)
    final_outer.to_csv(os.path.join(odir, "outer_loop_"+endpoint+("_"+agg if use_radio else '')+".csv"), index = False)

    final_histories.to_csv(os.path.join(odir, endpoint,"ncv_training_logs.csv"), index=False)
    torch.save(all_seeds_bundles, os.path.join(odir, "all_seeds_ensemble_bundles.pt"))


if __name__ == "__main__":
    main()
    print("Finish!")