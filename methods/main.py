print("Importing packages")
import argparse
import os
import time
import pandas as pd
from dataset import MissingModalitySimulator, load_or_preprocess_dataset
from train_ncv import nested_cv
from utils import (
    build_hyperparameter_grid,
    parse_value_or_list,
    normalize_model_name,
)

def get_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--odir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["MLP", "DyAM", "HealNet", "Distill_DyAM"],
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name suffix")
    parser.add_argument("--endpoint", type=str, required=True, help="Endpoint base name")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Directory containing all CSV files for the selected dataset.",
    )

    # Cross-validation and optimization hyperparameters
    parser.add_argument("--inner_splits", type=int, default=5)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=str, default="16") # Supports scalar or comma-separated list for tuning
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=str, default="5e-5") # Supports scalar or comma-separated list for tuning
    parser.add_argument("--seeds", type=str, default="123") # Supports scalar or comma-separated list for tuning
    parser.add_argument(
        "--missing_pattern_seed",
        type=int,
        default=0,
        help="Deterministic seed used only for missing-modality mask simulation.",
    )
    # MLP architecture hyperparameters
    parser.add_argument("--fusion_hidden_dim", type=str, default="32") # Supports scalar or comma-separated list for tuning
    parser.add_argument("--fusion_hidden_layers", type=str, default="1") # Supports scalar or comma-separated list for tuning
    parser.add_argument("--modality_hidden_layers", type=str, default="1") # Supports scalar or comma-separated list for tuning
    parser.add_argument("--dropout", type=str, default="0.2") # Supports scalar or comma-separated list for tuning

    # DyAM architecture hyperparameters
    parser.add_argument("--dyam_dropout", type=str, default="0.4")  # scalar or comma-separated list
    parser.add_argument("--dyam_temperature", type=str, default="2.0")  # scalar or comma-separated list
    parser.add_argument(
        "--distill_alpha",
        type=str,
        default="1.0",
        help="Weight a for representation distillation loss in Distill-DyAM. Supports scalar or comma-separated list.",
    )
    parser.add_argument(
        "--distill_beta",
        type=str,
        default="0.3",
        help="Weight b for feature/logit distillation loss in Distill-DyAM. Supports scalar or comma-separated list.",
    )

    # HealNet architecture hyperparameters
    parser.add_argument("--healnet_depth", type=str, default="3")
    parser.add_argument("--healnet_num_freq_bands", type=str, default="2")
    parser.add_argument("--healnet_num_latents", type=str, default="128")
    parser.add_argument("--healnet_latent_dim", type=str, default="128")
    parser.add_argument("--healnet_cross_heads", type=str, default="1")
    parser.add_argument("--healnet_latent_heads", type=str, default="4")
    parser.add_argument("--healnet_cross_dim_head", type=str, default="64")
    parser.add_argument("--healnet_latent_dim_head", type=str, default="64")
    parser.add_argument("--healnet_attn_dropout", type=str, default="0.0")
    parser.add_argument("--healnet_ff_dropout", type=str, default="0.0")
    parser.add_argument("--healnet_self_per_cross_attn", type=str, default="0")

    # Missing-modality setup
    parser.add_argument(
        "--train_missing_prob",
        type=str,
        default="0",
        help="Train/validation missing probability in [0, 1]. Supports scalar or comma-separated list.",
    )
    parser.add_argument(
        "--train_missing_location",
        type=str,
        default="global",
        help="Train/validation missing location: global or modality key (path, radio, clin, blood, radio_report). Supports scalar or comma-separated list.",
    )
    parser.add_argument(
        "--test_missing_prob",
        type=str,
        default="0",
        help="Test missing probability in [0, 1]. Supports scalar or comma-separated list.",
    )
    parser.add_argument(
        "--test_missing_location",
        type=str,
        default="global",
        help="Test missing location: global or modality key (path, radio, clin, blood, radio_report). Supports scalar or comma-separated list.",
    )
    parser.add_argument(
        "--imputation_method",
        type=str,
        default="zero",
        choices=["zero", "knn", "vae"],
        help="Imputation method for missing modalities: zero, knn, or vae.",
    )
    parser.add_argument(
        "--vae_imputer_latent_dim",
        type=int,
        default=16,
        help="Latent size for the tabular VAE imputer.",
    )
    parser.add_argument(
        "--vae_imputer_hidden_dim",
        type=int,
        default=128,
        help="Hidden size for the tabular VAE imputer.",
    )
    parser.add_argument(
        "--vae_imputer_epochs",
        type=int,
        default=30,
        help="Training epochs for the tabular VAE imputer.",
    )
    parser.add_argument(
        "--vae_imputer_batch_size",
        type=int,
        default=64,
        help="Batch size for the tabular VAE imputer.",
    )
    parser.add_argument(
        "--vae_imputer_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the tabular VAE imputer.",
    )
    parser.add_argument(
        "--vae_imputer_beta",
        type=float,
        default=1e-3,
        help="KL weight for the tabular VAE imputer.",
    )

    # Weights & Biases logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="unknown")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return parser.parse_args()

# Function to build output directory path
def _build_output_dir(base_odir, model_label, dataset_label, train_missing_location, train_missing_prob, seed):
    mapping = {
        "global": "GLOBAL",
        "radio": "RADIO",
        "path": "PATH",
        "clin": "CLIN",
        "radio_report": "RADIO_REPORT",
        "blood": "BLOOD",
    }
    key = str(train_missing_location).strip().lower()
    missing_modality_label = mapping.get(key, key.upper())
    missing_pct = str(float(train_missing_prob) * 100.0)
    return os.path.join(
        base_odir,
        model_label,
        dataset_label,
        "TRAIN_MISSING",
        missing_modality_label,
        missing_pct,
        f"seed_{seed}",
    )

# Function to save outputs of a training run
def _save_run_outputs(
    odir,
    inner_df,
    outer_df,
    history_df,
    split_df,
    seed,
    train_missing_location,
    train_missing_prob,
):
    os.makedirs(odir, exist_ok=True)

    inner_df = inner_df.copy()
    outer_df = outer_df.copy()
    history_df = history_df.copy()
    split_df = split_df.copy()

    inner_df["seed"] = seed
    inner_df["train_missing_location"] = train_missing_location
    inner_df["train_missing_prob"] = float(train_missing_prob)

    outer_df["seed"] = seed
    outer_df["train_missing_location"] = train_missing_location
    outer_df["train_missing_prob"] = float(train_missing_prob)

    history_df["seed"] = seed
    history_df["train_missing_location"] = train_missing_location
    history_df["train_missing_prob"] = float(train_missing_prob)

    split_df["seed"] = seed
    split_df["train_missing_location"] = train_missing_location
    split_df["train_missing_prob"] = float(train_missing_prob)

    inner_df.to_csv(os.path.join(odir, "inner_hp_eval.csv"), index=False)
    outer_df.to_csv(os.path.join(odir, "outer_test_metrics.csv"), index=False)
    history_df.to_csv(os.path.join(odir, "inner_epoch_history.csv"), index=False)
    split_df.to_csv(os.path.join(odir, "splits_manifest.csv"), index=False)

    # Create outer test summary
    metric_cols = [c for c in outer_df.columns if c.startswith("outer_test_")]
    if metric_cols:
        summary = {}
        for col in metric_cols:
            summary[f"{col}_mean"] = float(outer_df[col].mean())
            summary[f"{col}_std"] = float(outer_df[col].std())
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(odir, "outer_test_summary.csv"), index=False)

def main():
    # Start timer
    args = get_args()
    start_time = time.time()
    print("Running")
    model_name_norm = normalize_model_name(args.model)
    patient_id_col = "patient"

    if str(args.imputation_method).strip().lower() != "zero" and model_name_norm != "mlp":
        raise ValueError(
            "imputation_method='knn' or 'vae' is currently supported only with model='MLP'."
        )

    # Load dataset-specific preprocessed bundle if available (`dataset/<dataset>.py`);
    # otherwise use the generic preprocessing pipeline.
    inst_df, dfs, label_col = load_or_preprocess_dataset(args)
    modality_names = list(dfs.keys())
    num_modalities = len(modality_names)

    print(f"Dataframes read. Starting {args.model} training.")
     
    # Parse run axes (supports scalar or comma-separated list)
    seeds_list = parse_value_or_list(args.seeds, int)
    train_missing_locations = parse_value_or_list(args.train_missing_location, str, to_lower=True)
    train_missing_probs = parse_value_or_list(args.train_missing_prob, float)
    test_missing_locations = parse_value_or_list(args.test_missing_location, str, to_lower=True)
    test_missing_probs = parse_value_or_list(args.test_missing_prob, float)

    # Validate train/test missingness locations against available modalities.
    invalid_train_locations = [
        loc for loc in train_missing_locations if loc != "global" and loc not in modality_names
    ]
    if invalid_train_locations:
        valid = ", ".join(["global"] + sorted(modality_names))
        raise ValueError(
            f"Invalid --train_missing_location values: {', '.join(sorted(set(invalid_train_locations)))}. "
            f"Valid values: {valid}"
        )

    invalid_test_locations = [
        loc for loc in test_missing_locations if loc != "global" and loc not in modality_names
    ]
    if invalid_test_locations:
        valid = ", ".join(["global"] + sorted(modality_names))
        raise ValueError(
            f"Invalid --test_missing_location values: {', '.join(sorted(set(invalid_test_locations)))}. "
            f"Valid values: {valid}"
        )

    invalid_train_probs = [p for p in train_missing_probs if p < 0.0 or p > 1.0]
    if invalid_train_probs:
        raise ValueError(
            f"Invalid --train_missing_prob values: {invalid_train_probs}. "
            "All values must be in [0, 1]."
        )

    invalid_test_probs = [p for p in test_missing_probs if p < 0.0 or p > 1.0]
    if invalid_test_probs:
        raise ValueError(
            f"Invalid --test_missing_prob values: {invalid_test_probs}. "
            "All values must be in [0, 1]."
        )

    # Construct base output labels
    if model_name_norm == "mlp":
        model_label = (
            f"{str(args.imputation_method).strip().upper()}_"
            f"{model_name_norm.upper().replace('_', '-')}"
        )
    else:
        model_label = model_name_norm.upper().replace("_", "-")
    dataset_label = str(args.dataset).strip().upper()
    combo_count = len(seeds_list) * len(train_missing_locations) * len(train_missing_probs)
    test_eval_count = len(test_missing_locations) * len(test_missing_probs)
    print(f"Total training runs to execute: {combo_count}")
    print(f"Test missingness evaluations per training run: {test_eval_count}")

    wandb_project_name = f"{args.wandb_project}_{args.dataset}"
    wandb_enabled_flag = (args.wandb and args.wandb_mode != "disabled")

    for seed in seeds_list:
        for train_missing_location in train_missing_locations:
            for train_missing_prob in train_missing_probs:
                hp_configs = build_hyperparameter_grid(
                    args,
                    train_missing_prob=train_missing_prob,
                    train_missing_location=train_missing_location,
                )

                train_missing_simulator = MissingModalitySimulator(
                    num_modalities=num_modalities,
                    modality_names=modality_names,
                    missing_prob=float(train_missing_prob),
                    missing_location=train_missing_location,
                )

                test_eval_setups = []
                for test_missing_location in test_missing_locations:
                    for test_missing_prob in test_missing_probs:
                        test_eval_setups.append(
                            {
                                "missing_location": str(test_missing_location).lower(),
                                "missing_prob": float(test_missing_prob),
                                "simulator": MissingModalitySimulator(
                                    num_modalities=num_modalities,
                                    modality_names=modality_names,
                                    missing_prob=float(test_missing_prob),
                                    missing_location=test_missing_location,
                                ),
                            }
                        )

                odir = _build_output_dir(
                    base_odir=args.odir,
                    model_label=model_label,
                    dataset_label=dataset_label,
                    train_missing_location=train_missing_location,
                    train_missing_prob=train_missing_prob,
                    seed=seed,
                )
                print(
                    "Running seed="
                    f"{seed}, train_missing_location={train_missing_location}, "
                    f"train_missing_prob={train_missing_prob}"
                )
                print(f"Missing pattern seed: {int(args.missing_pattern_seed)}")
                print(f"Output directory: {odir}")
                print(f"Hyperparameter combinations to evaluate: {len(hp_configs)}")
                print(f"Test missingness combinations to evaluate: {len(test_eval_setups)}")

                wandb_base_config = {
                    "endpoint": args.endpoint,
                    "model": model_name_norm,
                    "dataset": args.dataset,
                    "modalities": modality_names,
                    "hp_grid_size": len(hp_configs),
                    "epochs": args.epochs,
                    "inner_splits": args.inner_splits,
                    "outer_splits": args.outer_splits,
                    "train_missing_prob": float(train_missing_prob),
                    "train_missing_location": str(train_missing_location).lower(),
                    "missing_pattern_seed": int(args.missing_pattern_seed),
                    "imputation_method": args.imputation_method,
                    "test_eval_combinations": len(test_eval_setups),
                    "test_missing_probs_grid": ",".join(str(float(p)) for p in test_missing_probs),
                    "test_missing_locations_grid": ",".join(str(loc).lower() for loc in test_missing_locations),
                }
                if str(args.imputation_method).strip().lower() == "vae":
                    wandb_base_config.update(
                        {
                            "vae_imputer_latent_dim": int(args.vae_imputer_latent_dim),
                            "vae_imputer_hidden_dim": int(args.vae_imputer_hidden_dim),
                            "vae_imputer_epochs": int(args.vae_imputer_epochs),
                            "vae_imputer_batch_size": int(args.vae_imputer_batch_size),
                            "vae_imputer_lr": float(args.vae_imputer_lr),
                            "vae_imputer_beta": float(args.vae_imputer_beta),
                        }
                    )

                inner_df, outer_df, history_df, split_df = nested_cv(
                    dfs=dfs,
                    inst_df=inst_df,
                    label_col=label_col,
                    epochs=args.epochs,
                    seed=seed,
                    hp_configs=hp_configs,
                    train_missing_simulator=train_missing_simulator,
                    model_name=args.model,
                    imputation_method=args.imputation_method,
                    inner_splits=args.inner_splits,
                    outer_splits=args.outer_splits,
                    missing_pattern_seed=int(args.missing_pattern_seed),
                    patient_id_col=patient_id_col,
                    wandb_enabled=wandb_enabled_flag,
                    wandb_project=wandb_project_name,
                    wandb_mode=args.wandb_mode,
                    wandb_base_config=wandb_base_config,
                    test_eval_setups=test_eval_setups,
                    imputer_kwargs={
                        "latent_dim": int(args.vae_imputer_latent_dim),
                        "hidden_dim": int(args.vae_imputer_hidden_dim),
                        "epochs": int(args.vae_imputer_epochs),
                        "batch_size": int(args.vae_imputer_batch_size),
                        "lr": float(args.vae_imputer_lr),
                        "beta": float(args.vae_imputer_beta),
                    },
                )

                _save_run_outputs(
                    odir=odir,
                    inner_df=inner_df,
                    outer_df=outer_df,
                    history_df=history_df,
                    split_df=split_df,
                    seed=seed,
                    train_missing_location=train_missing_location,
                    train_missing_prob=float(train_missing_prob),
                )

                print(f"Run finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
    print("Finish!")
