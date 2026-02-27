print("Importing packages")
import argparse
import os
import time
import pandas as pd
from dataset import MissingModalitySimulator
from train_ncv import nested_cv
from utils import (
    build_hyperparameter_grid,
    parse_value_or_list,
)


def get_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--odir", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="MLP", help="Model name for output path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name suffix")
    parser.add_argument("--endpoint", type=str, required=True, help="Endpoint base name")
    parser.add_argument("--inst_data", type=str, required=True, help="CSV with patient and label")

    # Multimodal data arguments
    parser.add_argument("--patho_data", type=str, default=None)
    parser.add_argument("--clin_data", type=str, default=None)
    parser.add_argument("--blood_data", type=str, default=None)
    parser.add_argument("--radio_data", type=str, default=None)
    parser.add_argument("--radio_report_data", type=str, default=None)

    # Cross-validation and optimization hyperparameters
    parser.add_argument("--inner_splits", type=int, default=5)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=str, default="16") # Supports scalar or comma-separated list for tuning
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=str, default="5e-5") # Supports scalar or comma-separated list for tuning
    parser.add_argument("--seeds", type=str, default="123") # Supports scalar or comma-separated list for tuning

    # MLP architecture hyperparameters
    parser.add_argument("--fusion_hidden_dim", type=str, default="32") # Supports scalar or comma-separated list for tuning
    parser.add_argument("--fusion_hidden_layers", type=str, default="1") # Supports scalar or comma-separated list for tuning
    parser.add_argument("--modality_hidden_layers", type=str, default="1") # Supports scalar or comma-separated list for tuning
    parser.add_argument("--dropout", type=str, default="0.2") # Supports scalar or comma-separated list for tuning

    # Missing-modality setup
    parser.add_argument(
        "--missing_prob",
        type=str,
        default="0",
        help="Missing probability in [0, 1]. Supports scalar or comma-separated list.",
    )
    parser.add_argument(
        "--missing_scope",
        type=str,
        default="none",
        help="Where to apply missingness. Supports scalar or comma-separated list of: train,test,both,none",
    )
    parser.add_argument(
        "--missing_location",
        type=str,
        default="global",
        help="Missing location: global or modality key (path, radio, clin, blood, radio_report). Supports scalar or comma-separated list.",
    )

    # Weights & Biases logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="unknown")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return parser.parse_args()


# Find the label column in the inst_df
def _find_label_column(inst_df, endpoint):
    preferred = f"{endpoint}_label"
    if preferred in inst_df.columns:
        return preferred
    if endpoint in inst_df.columns:
        return endpoint
    raise ValueError(f"Label column not found. Tried '{preferred}' and '{endpoint}'.")

# Modality-specific loading functions
def _load_patho_df(patho_dir, inst_df):
    path_df = pd.read_csv(patho_dir)
    path_df = path_df.rename(columns=lambda x: x.replace("embedding_", "patho_"))
    path_df = pd.merge(path_df, inst_df[["patient"]], on="patient", how = "inner")
    keep = ["patient"] + [c for c in path_df.columns if c.startswith("patho_")]
    return path_df[keep]

def _load_prefixed_df(csv_path, inst_df, prefix):
    df = pd.read_csv(csv_path)
    df = df.rename(columns=lambda x: f"{prefix}_{x}" if x != "patient" else x)
    return pd.merge(df, inst_df[["patient"]], on="patient")

def _load_radio_df(radio_dir, inst_df):
    rad_df = pd.read_csv(radio_dir).rename(columns=lambda x: x.replace("pred_", "radio_"))
    rad_df = rad_df.drop(columns=["image_path", "lesion_tag"], errors="ignore")
    rad_df = pd.merge(rad_df, inst_df[["patient"]], on="patient")
    keep = ["patient"] + [c for c in rad_df.columns if c.startswith("radio_")]
    return rad_df[keep]

# Collapse duplicated rows by mean for each modality
def _check_and_collapse_modality_rows(dfs):
    """Ensure one row per patient in each modality dataframe."""
    cleaned = {}
    for mod_name, df in dfs.items():
        if "patient" not in df.columns:
            raise ValueError(f"Modality '{mod_name}' does not contain a 'patient' column.")

        work_df = df.copy()
        work_df["patient"] = work_df["patient"].astype(int)
        patient_counts = work_df["patient"].value_counts()
        duplicated = patient_counts[patient_counts > 1]

        if duplicated.empty:
            cleaned[mod_name] = work_df
            continue

        dup_ids = duplicated.index.astype(int).tolist()
        preview = dup_ids[:25]
        preview_txt = ", ".join(str(x) for x in preview)
        suffix = " ..." if len(dup_ids) > 25 else ""
        print(
            f"[{mod_name}] Duplicated patients detected: {len(dup_ids)}. "
            f"Collapsing by mean. Patient IDs: {preview_txt}{suffix}"
        )

        feature_cols = [c for c in work_df.columns if c != "patient"]

        # Build a dense temporary frame to avoid pandas fragmentation warnings.
        numeric_features = work_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        dense_df = pd.concat([work_df[["patient"]], numeric_features], axis=1).copy()
        collapsed = dense_df.groupby("patient", as_index=False)[feature_cols].mean()
        cleaned[mod_name] = collapsed

    return cleaned

# Function to map arg value to folder name
def _missing_scope_label(missing_scope):
    mapping = {
        "train": "MISSING_TRAIN",
        "test": "MISSING_VAL",
        "both": "MISSING_ALL",
        "none": "NONE",
    }
    return mapping[str(missing_scope).strip().lower()]

# Function to map missing location to folder name
def _missing_modality_label(missing_location):
    mapping = {
        "global": "GLOBAL",
        "radio": "RADIO",
        "path": "PATH",
        "clin": "CLIN",
        "radio_report": "RADIO_REPORT",
        "blood": "BLOOD",
    }
    key = str(missing_location).strip().lower()
    return mapping.get(key, key.upper())

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
        dfs["radio"] = _load_radio_df(args.radio_data, inst_df)
    if args.clin_data:
        dfs["clin"] = _load_prefixed_df(args.clin_data, inst_df, "clin")
    if args.blood_data:
        dfs["blood"] = _load_prefixed_df(args.blood_data, inst_df, "blood")
    if args.radio_report_data:
        dfs["radio_report"] = _load_prefixed_df(args.radio_report_data, inst_df, "radio_report")

    # Ensure at least one modality is selected
    if not dfs:
        raise ValueError("No modality provided. At least one modality CSV is required.")

    # Verify one row per patient in each modality and collapse duplicates by mean when needed.
    dfs = _check_and_collapse_modality_rows(dfs)
    modality_names = list(dfs.keys())
    num_modalities = len(modality_names)

    # Force patient IDs to int and set as index WITHOUT dropping the column
    for mod in modality_names:
        dfs[mod]["patient"] = dfs[mod]["patient"].astype(int)
        dfs[mod] = dfs[mod].set_index("patient", drop=False)

    inst_df["patient"] = inst_df["patient"].astype(int)
    inst_df = inst_df.set_index("patient", drop=False)

    print("Dataframes read. Starting MLP training.")
     
    # Parse run axes (supports scalar or comma-separated list)
    seeds_list = parse_value_or_list(args.seeds, int)
    missing_scopes = parse_value_or_list(args.missing_scope, str, to_lower=True)
    missing_locations = parse_value_or_list(args.missing_location, str, to_lower=True)
    missing_probs = parse_value_or_list(args.missing_prob, float)

    # Check valid values in scope, location, and probabilities
    valid_scopes = {"train", "test", "both", "none"}
    invalid_scopes = sorted(set(missing_scopes) - valid_scopes)
    if invalid_scopes:
        raise ValueError(
            f"Invalid --missing_scope values: {', '.join(invalid_scopes)}. "
            "Valid values: train, test, both, none."
        )

    invalid_locations = [
        loc for loc in missing_locations if loc != "global" and loc not in modality_names
    ]
    if invalid_locations:
        valid = ", ".join(["global"] + sorted(modality_names))
        raise ValueError(
            f"Invalid --missing_location values: {', '.join(sorted(set(invalid_locations)))}. "
            f"Valid values: {valid}"
        )

    invalid_probs = [p for p in missing_probs if p < 0.0 or p > 1.0]
    if invalid_probs:
        raise ValueError(
            f"Invalid --missing_prob values: {invalid_probs}. "
            "All values must be in [0, 1]."
        )

    # Construct base output labels
    model_label = str(args.model).strip().upper().replace("_", "-")
    dataset_label = str(args.dataset).strip().upper()
    combo_count = 0
    for missing_scope in missing_scopes:
        scope_probs = [0.0] if missing_scope == "none" else missing_probs
        scope_locations = ["global"] if missing_scope == "none" else missing_locations
        combo_count += len(scope_locations) * len(scope_probs) * len(seeds_list)
    print(f"Total runs to execute: {combo_count}")

    for seed in seeds_list:
        for missing_scope in missing_scopes:
            this_probs = [0.0] if missing_scope == "none" else missing_probs
            this_locations = ["global"] if missing_scope == "none" else missing_locations
            for missing_location in this_locations:
                for missing_prob in this_probs:
                    hp_configs = build_hyperparameter_grid(
                        args,
                        missing_prob=missing_prob,
                        missing_scope=missing_scope,
                        missing_location=missing_location,
                    )

                    missing_simulator = MissingModalitySimulator(
                        num_modalities=num_modalities,
                        modality_names=modality_names,
                        missing_prob=missing_prob,
                        missing_location=missing_location,
                    )

                    # Construct output directory path
                    missing_scope_label = _missing_scope_label(missing_scope)
                    missing_modality_label = _missing_modality_label(missing_location)
                    if missing_scope_label == "NONE":
                        odir = os.path.join(
                            args.odir,
                            model_label,
                            dataset_label,
                            missing_scope_label,
                            f"seed_{seed}",
                        )
                    else:
                        missing_pct = str(float(missing_prob) * 100.0)
                        odir = os.path.join(
                            args.odir,
                            model_label,
                            dataset_label,
                            missing_scope_label,
                            missing_modality_label,
                            missing_pct,
                            f"seed_{seed}",
                        )
                    os.makedirs(odir, exist_ok=True)

                    print(
                        "Running seed="
                        f"{seed}, missing_scope={missing_scope}, "
                        f"missing_location={missing_location}, missing_prob={missing_prob}"
                    )
                    print(f"Output directory: {odir}")
                    print(f"Hyperparameter combinations to evaluate: {len(hp_configs)}")

                    # Weights & Biases setup
                    wandb_project_name = f"{args.wandb_project}_{args.dataset}"
                    wandb_base_config = {
                        "endpoint": args.endpoint,
                        "model": args.model,
                        "dataset": args.dataset,
                        "modalities": modality_names,
                        "hp_grid_size": len(hp_configs),
                        "epochs": args.epochs,
                        "inner_splits": args.inner_splits,
                        "outer_splits": args.outer_splits,
                        "missing_prob": missing_prob,
                        "missing_scope": missing_scope,
                        "missing_location": missing_location,
                    }

                    # Run nested cross-validation
                    inner_df, outer_df, history_df, split_df = nested_cv(
                        dfs=dfs,
                        inst_df=inst_df,
                        label_col=label_col,
                        epochs=args.epochs,
                        seed=seed,
                        hp_configs=hp_configs,
                        missing_simulator=missing_simulator,
                        missing_scope=missing_scope,
                        inner_splits=args.inner_splits,
                        outer_splits=args.outer_splits,
                        wandb_enabled=(args.wandb and args.wandb_mode != "disabled"),
                        wandb_project=wandb_project_name,
                        wandb_mode=args.wandb_mode,
                        wandb_base_config=wandb_base_config,
                    )

                    # Save results and history to CSVs with run metadata
                    inner_df["seed"] = seed
                    inner_df["missing_scope"] = missing_scope
                    inner_df["missing_location"] = missing_location
                    inner_df["missing_prob"] = float(missing_prob)

                    outer_df["seed"] = seed
                    outer_df["missing_scope"] = missing_scope
                    outer_df["missing_location"] = missing_location
                    outer_df["missing_prob"] = float(missing_prob)

                    history_df["seed"] = seed
                    history_df["missing_scope"] = missing_scope
                    history_df["missing_location"] = missing_location
                    history_df["missing_prob"] = float(missing_prob)

                    split_df["seed"] = seed
                    split_df["missing_scope"] = missing_scope
                    split_df["missing_location"] = missing_location
                    split_df["missing_prob"] = float(missing_prob)

                    inner_df.to_csv(os.path.join(odir, "inner_hp_eval.csv"), index=False)
                    outer_df.to_csv(os.path.join(odir, "outer_test_metrics.csv"), index=False)
                    history_df.to_csv(os.path.join(odir, "inner_epoch_history.csv"), index=False)
                    split_df.to_csv(os.path.join(odir, "splits_manifest.csv"), index=False)

                    # Create outer test summary
                    metric_cols = [c for c in outer_df.columns if c.startswith("outer_train_")]
                    if metric_cols:
                        summary = {}
                        for col in metric_cols:
                            summary[f"{col}_mean"] = float(outer_df[col].mean())
                            summary[f"{col}_std"] = float(outer_df[col].std())
                        summary_df = pd.DataFrame([summary])
                        summary_df.to_csv(os.path.join(odir, "outer_test_summary.csv"), index=False)

                    print(f"Run finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
    print("Finish!")
