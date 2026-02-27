print("Importing packages")
import argparse
import os
import time
import pandas as pd
import torch

from dataset import MissingModalitySimulator
from train_ncv import nested_cv


def get_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--odir", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="mlp", help="Model name for output path")
    parser.add_argument("--endpoint", type=str, required=True, help="Endpoint base name")
    parser.add_argument("--inst_data", type=str, required=True, help="CSV with patient and label")

    # Multimodal data arguments
    parser.add_argument("--patho_data", type=str, default=None)
    parser.add_argument("--clin_data", type=str, default=None)
    parser.add_argument("--blood_data", type=str, default=None)
    parser.add_argument("--body_composition_data", type=str, default=None)
    parser.add_argument("--lung_data", type=str, default=None)
    parser.add_argument("--liver_data", type=str, default=None)
    parser.add_argument("--radio_data", type=str, default=None)
    parser.add_argument("--radio_report_data", type=str, default=None)

    # Cross-validation and optimization hyperparameters
    parser.add_argument("--inner_splits", type=int, default=5)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--seeds", type=str, default="123")

    # MLP architecture hyperparameters
    parser.add_argument("--modality_hidden_dim", type=int, default=64)
    parser.add_argument("--fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Missing-modality simulation probabilities
    parser.add_argument("--missing_complete_prob", type=float, default=0.55)
    parser.add_argument("--missing_mild_prob", type=float, default=0.20)
    parser.add_argument("--missing_moderate_prob", type=float, default=0.15)
    parser.add_argument("--missing_severe_prob", type=float, default=0.10)

    return parser.parse_args()


# Find the label column in the inst_df
def _find_label_column(inst_df, endpoint):
    preferred = f"{endpoint}_label"
    if preferred in inst_df.columns:
        return preferred
    if endpoint in inst_df.columns:
        return endpoint
    raise ValueError(f"Label column not found. Tried '{preferred}' and '{endpoint}'.")

# Modalituy-specific loading functions
def _load_patho_df(patho_dir, inst_df):
    path_df = pd.read_csv(patho_dir)
    path_df = path_df.rename(columns=lambda x: x.replace("embedding_", "patho_"))
    path_df = pd.merge(path_df, inst_df[["patient"]], on="patient", how = "innner")
    keep = ["patient"] + [c for c in path_df.columns if c.startswith("patho_")]
    return path_df[keep]


def _load_prefixed_df(csv_path, inst_df, prefix):
    df = pd.read_csv(csv_path)
    df = df.rename(columns=lambda x: f"{prefix}_{x}" if x != "patient" else x)
    return pd.merge(df, inst_df[["patient"]], on="patient")


def _load_radio_df(radio_dir, inst_df):
    rad_df = pd.read_csv(radio_dir).rename(columns=lambda x: x.replace("pred_", "radio_"))
    rad_df = rad_df.drop(columns=["image_path", "lesion_tag"], errors="ignore")
    rad_df = rad_df.groupby("patient").mean().reset_index()
    rad_df = pd.merge(rad_df, inst_df[["patient"]], on="patient")
    keep = ["patient"] + [c for c in rad_df.columns if c.startswith("radio_")]
    return rad_df[keep]


def main():
    # Start timer
    args = get_args()
    start_time = time.time()
    print("Running")

    # Read labels dataframe and resolve label column name
    inst_df = pd.read_csv(args.inst_data)
    label_col = _find_label_column(inst_df, args.endpoint)
    inst_df = inst_df[["patient", label_col]].copy()

    # Read modality dataframes and keep only patients from inst_df
    dfs = {}
    if args.patho_data:
        dfs["path"] = _load_patho_df(args.patho_data, inst_df)
    if args.radio_data:
        dfs["rad"] = _load_radio_df(args.radio_data, inst_df)
    if args.clin_data:
        dfs["clin"] = _load_prefixed_df(args.clin_data, inst_df, "clin")
    if args.blood_data:
        dfs["blood"] = _load_prefixed_df(args.blood_data, inst_df, "blood")
    if args.body_composition_data:
        dfs["body_composition"] = _load_prefixed_df(args.body_composition_data, inst_df, "body_composition")
    if args.lung_data:
        dfs["lung"] = _load_prefixed_df(args.lung_data, inst_df, "lung")
    if args.liver_data:
        dfs["liver"] = _load_prefixed_df(args.liver_data, inst_df, "liver")
    if args.radio_report_data:
        dfs["radio_report"] = _load_prefixed_df(args.radio_report_data, inst_df, "radio_report")

    # Ensure at least one modality is selected
    if not dfs:
        raise ValueError("No modality provided. At least one modality CSV is required.")
    
    # Create output directory
    modality_names = list(dfs.keys())
    odir = os.path.join(f"{args.odir}{'-'.join(modality_names)}", args.endpoint, args.model)
    os.makedirs(odir, exist_ok=True)

    # Force patient IDs to int and set as index WITHOUT dropping the column
    for mod in modality_names:
        dfs[mod]["patient"] = dfs[mod]["patient"].astype(int)
        dfs[mod] = dfs[mod].set_index("patient", drop=False)

    inst_df["patient"] = inst_df["patient"].astype(int)
    inst_df = inst_df.set_index("patient", drop=False)

    print("Dataframes read. Starting MLP training.")

    # Set seeds for reproducibility
    seeds_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Define model hyperparameters
    model_kwargs = {
        "modality_hidden_dim": args.modality_hidden_dim,
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "dropout_p": args.dropout,
        "use_mask": True,
    }

    # Initialize missing-modality simulator for training augmentation
    missing_simulator = MissingModalitySimulator(
        num_modalities=len(modality_names),
        complete_prob=args.missing_complete_prob,
        mild_prob=args.missing_mild_prob,
        moderate_prob=args.missing_moderate_prob,
        severe_prob=args.missing_severe_prob,
    )

    # Lists to store results across seeds
    all_inner_results = []
    all_outer_results = []
    all_histories = []
    all_seeds_bundles = {}

    # Run nested cross-validation for each seed
    for seed in seeds_list:
        inner_df, outer_df, history_df, bundles = nested_cv(
            dfs=dfs,
            inst_df=inst_df,
            label_col=label_col,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.learning_rate,
            seed=seed,
            missing_simulator=missing_simulator,
            inner_splits=args.inner_splits,
            outer_splits=args.outer_splits,
            model_kwargs=model_kwargs,
        )

        # Add seed column to results
        inner_df["seed"] = seed
        outer_df["seed"] = seed
        history_df["seed"] = seed

        # Concatenate results
        all_inner_results.append(inner_df)
        all_outer_results.append(outer_df)
        all_histories.append(history_df)
        all_seeds_bundles[f"seed_{seed}"] = bundles

        print(f"Seed {seed} completed in {time.time() - start_time:.2f} seconds.")

    # Concatenate and save results across seeds
    final_inner = pd.concat(all_inner_results, ignore_index=True)
    final_outer = pd.concat(all_outer_results, ignore_index=True)
    final_histories = pd.concat(all_histories, ignore_index=True)

    final_inner.to_csv(os.path.join(odir, f"inner_loop_{args.endpoint}.csv"), index=False)
    final_outer.to_csv(os.path.join(odir, f"outer_loop_{args.endpoint}.csv"), index=False)
    final_histories.to_csv(os.path.join(odir, "ncv_training_logs.csv"), index=False)

    # Save the bundles of each seed (models, predictions, scalers, etc.) in a .pt file
    torch.save(all_seeds_bundles, os.path.join(odir, "all_seeds_ensemble_bundles.pt"))


if __name__ == "__main__":
    main()
    print("Finish!")
