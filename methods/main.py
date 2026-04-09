print("Importing packages")
import argparse
import os
import time
from collections import OrderedDict
import pandas as pd
from dataset import MissingModalitySimulator, load_or_preprocess_dataset
from train_ncv import nested_cv
from utils import (
    build_hyperparameter_grid,
    parse_value_or_list,
    normalize_model_name,
)

# ------------------------ HELPER FUNCTIONS --------------------------

def _parse_bool_flag(value):
    if isinstance(value, bool):
        return value
    value_l = str(value).strip().lower()
    if value_l in {"1", "true", "yes", "y"}:
        return True
    if value_l in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use one of: true, false, 1, 0, yes, no."
    )

def get_args():
    parser = argparse.ArgumentParser()

    # Main arguments
    parser.add_argument("--odir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name suffix")
    parser.add_argument("--endpoint", type=str, required=True, help="Endpoint base name")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["MLP", "DyAM", "HealNet", "Distill_DyAM", "SMIL_E"],
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Directory containing all CSV files for the selected dataset.",
    )
    parser.add_argument(
        "--radio_aggregation_method",
        type=str,
        default="mean",
        choices=["mean", "attention"],
        help="How to aggregate duplicated radiology rows into one patient-level embedding.",
    )

    # Cross-validation and optimization hyperparameters
    parser.add_argument("--inner_splits", type=int, default=5)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=str, default="16") # Supports scalar or comma-separated list for tuning
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--retrain_outer",
        type=_parse_bool_flag,
        default=True,
        help=(
            "If true, refit one final model on the full outer-train split using the robustly selected HPs. "
            "If false, keep the previous behavior: ensemble the retained inner-fold models on outer-test."
        ),
    )
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
    parser.add_argument(
        "--fusion_batchnorm",
        type=str,
        default="false",
        help="Whether to use BatchNorm in the shared fusion block of the MLP. Supports scalar or comma-separated list.",
    )
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

    # SMIL-E architecture hyperparameters
    parser.add_argument("--smil_e_latent_dim", type=str, default="64")
    parser.add_argument("--smil_e_num_priors", type=str, default="64")
    parser.add_argument("--smil_e_num_heads", type=str, default="4")
    parser.add_argument("--smil_e_dropout", type=str, default="0.1")
    parser.add_argument("--smil_e_alpha", type=str, default="1e-2")
    parser.add_argument("--smil_e_beta", type=str, default="1e-2")

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
    parser.add_argument("--healnet_self_per_cross_attn", type=str, default="1")

    # Missing-modality setup
    parser.add_argument(
        "--train_missing_prop",
        type=str,
        default="0",
        help="Train/validation missing proportion in [0, 1]. Supports scalar or comma-separated list.",
    )
    parser.add_argument(
        "--missing_location",
        type=str,
        default="global",
        help="Missing location: global or modality key (path, radio, clin, blood, radio_report). Supports scalar or comma-separated list.",
    )
    parser.add_argument(
        "--test_missing_prop",
        type=str,
        default="0",
        help="Test missing proportion in [0, 1]. Supports scalar or comma-separated list.",
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
    parser.add_argument(
        "--radio_attention_hidden_dim",
        type=int,
        default=128,
        help="Hidden size of the radiology attention pooling scorer.",
    )
    parser.add_argument(
        "--radio_attention_dropout",
        type=float,
        default=0.1,
        help="Dropout used inside the radiology attention pooling scorer.",
    )
    parser.add_argument(
        "--radio_attention_epochs",
        type=int,
        default=25,
        help="Training epochs for the radiology attention pooling module.",
    )
    parser.add_argument(
        "--radio_attention_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the radiology attention pooling module.",
    )
    parser.add_argument(
        "--radio_attention_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for the radiology attention pooling module.",
    )

    # Weights & Biases logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="unknown")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument(
        "--wandb_console",
        type=str,
        default="off",
        choices=["off", "on"],
        help="Whether to show W&B console chatter. 'off' keeps only the training prints from this pipeline.",
    )

    return parser.parse_args()

# Function to build output directory path
def _build_output_dir(
    base_odir,
    model_name,
    imputation_method,
    dataset_name,
    missing_location,
    train_missing_prop,
    seed,
    radio_aggregation_method="mean",
):
    mapping = {
        "global": "GLOBAL",
        "radio": "RADIO",
        "path": "PATH",
        "clin": "CLIN",
        "radio_report": "RADIO_REPORT",
        "blood": "BLOOD",
    }
    model_name_norm = normalize_model_name(model_name)
    if model_name_norm == "mlp":
        model_label = (
            f"{str(imputation_method).strip().upper()}_"
            f"{model_name_norm.upper().replace('_', '-')}"
        )
    else:
        model_label = model_name_norm.upper().replace("_", "-")
    if str(radio_aggregation_method).strip().lower() != "mean":
        model_label = f"{model_label}_RAD-{str(radio_aggregation_method).strip().upper()}"
    dataset_label = str(dataset_name).strip().upper()
    key = str(missing_location).strip().lower()
    missing_modality_label = mapping.get(key, key.upper())
    missing_pct = str(float(train_missing_prop) * 100.0)
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
    test_predictions_df,
    seed,
    missing_location,
    train_missing_prop,
):
    os.makedirs(odir, exist_ok=True)

    inner_df = inner_df.copy()
    outer_df = outer_df.copy()
    history_df = history_df.copy()
    split_df = split_df.copy()
    test_predictions_df = test_predictions_df.copy()

    inner_df["seed"] = seed
    inner_df["missing_location"] = missing_location
    inner_df["train_missing_prop"] = float(train_missing_prop)

    outer_df["seed"] = seed
    outer_df["missing_location"] = missing_location
    outer_df["train_missing_prop"] = float(train_missing_prop)

    history_df["seed"] = seed
    history_df["missing_location"] = missing_location
    history_df["train_missing_prop"] = float(train_missing_prop)

    split_df["seed"] = seed
    split_df["missing_location"] = missing_location
    split_df["train_missing_prop"] = float(train_missing_prop)

    test_predictions_df["seed"] = seed

    inner_df.to_csv(os.path.join(odir, "inner_hp_eval.csv"), index=False)
    outer_df.to_csv(os.path.join(odir, "outer_test_metrics.csv"), index=False)
    history_df.to_csv(os.path.join(odir, "inner_epoch_history.csv"), index=False)
    split_df.to_csv(os.path.join(odir, "splits_manifest.csv"), index=False)
    test_predictions_df.to_csv(os.path.join(odir, "test_predictions.csv"), index=False)

    # Create outer test summary
    metric_cols = [c for c in outer_df.columns if c.startswith("outer_test_")]
    if metric_cols:
        summary = {}
        for col in metric_cols:
            summary[f"{col}_mean"] = float(outer_df[col].mean())
            summary[f"{col}_std"] = float(outer_df[col].std())
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(odir, "outer_test_summary.csv"), index=False)


def _build_test_eval_setups_for_run(
    missing_location,
    train_missing_prop,
    test_missing_props,
    num_modalities,
    modality_names,
):
    missing_location = str(missing_location).strip().lower()
    train_missing_prop = float(train_missing_prop)

    if missing_location == "global":
        eval_props = [float(p) for p in test_missing_props]
    elif train_missing_prop == 0.0:
        eval_props = [float(p) for p in test_missing_props]
    else:
        eval_props = [0.0]

    setups = OrderedDict()
    for prop in eval_props:
        setups[float(prop)] = {
            "missing_location": missing_location,
            "missing_prop": float(prop),
            "simulator": MissingModalitySimulator(
                num_modalities=num_modalities,
                modality_names=modality_names,
                missing_prop=float(prop),
                missing_location=missing_location,
            ),
        }
    return list(setups.values())


# ------------------------ MAIN FUNCTION --------------------------

def main():
    # Parse command-line arguments and start timer
    args = get_args()
    if bool(args.wandb):
        if str(args.wandb_console).strip().lower() == "off":
            os.environ.setdefault("WANDB_SILENT", "true")
            os.environ.setdefault("WANDB_QUIET", "true")
            os.environ.setdefault("WANDB_CONSOLE", "off")
    start_time = time.time()
    model_name_norm = normalize_model_name(args.model)
    print("Running")

    # Validate imputation_methods are not used with non-MLP models
    if str(args.imputation_method).strip().lower() != "zero" and model_name_norm != "mlp":
        raise ValueError(
            "imputation_method='knn' or 'vae' is currently supported only with model='MLP'."
        )

    # Load dataset-specific preprocessed bundle
    inst_df, dfs, label_col, patient_id_col = load_or_preprocess_dataset(args)
    modality_names = list(dfs.keys())
    num_modalities = len(modality_names)
    print(f"Dataframes read. Starting {args.model} training.")
     
    # Parse run axes (supports scalar or comma-separated list)
    seeds_list = parse_value_or_list(args.seeds, int)
    missing_locations = parse_value_or_list(args.missing_location, str, to_lower=True)
    train_missing_props = parse_value_or_list(args.train_missing_prop, float)
    test_missing_props = parse_value_or_list(args.test_missing_prop, float)

    # Validate train/test missingness locations and probabilities against available values
    invalid_train_locations = [
        loc for loc in missing_locations if loc != "global" and loc not in modality_names
    ]
    invalid_test_props = [p for p in test_missing_props if p < 0.0 or p > 1.0]
    invalid_train_props = [p for p in train_missing_props if p < 0.0 or p > 1.0]
    if invalid_train_locations:
        valid = ", ".join(["global"] + sorted(modality_names))
        raise ValueError(
            f"Invalid --missing_location values: {', '.join(sorted(set(invalid_train_locations)))}. "
            f"Valid values: {valid}"
        )
    if invalid_train_props:
        raise ValueError(
            f"Invalid --train_missing_prop values: {invalid_train_props}. "
            "All values must be in [0, 1]."
        )
    if invalid_test_props:
        raise ValueError(
            f"Invalid --test_missing_prop values: {invalid_test_props}. "
            "All values must be in [0, 1]."
        )

    # Count total combinations to run and evaluate
    combo_count = len(seeds_list) * len(missing_locations) * len(train_missing_props)
    test_eval_total = len(seeds_list) * sum(
        len(
            _build_test_eval_setups_for_run(
                missing_location=loc,
                train_missing_prop=prop,
                test_missing_props=test_missing_props,
                num_modalities=num_modalities,
                modality_names=modality_names,
            )
        )
        for loc in missing_locations
        for prop in train_missing_props
    )
    print(f"Total training runs to execute: {combo_count}")
    print(f"Total test missingness evaluations across all runs: {test_eval_total}")

    wandb_project_name = f"{args.wandb_project}_{args.dataset}"
    wandb_enabled_flag = (args.wandb and args.wandb_mode != "disabled")

    for seed in seeds_list:
        for missing_location in missing_locations:
            for train_missing_prop in train_missing_props:
                hp_configs = build_hyperparameter_grid(
                    args,
                    train_missing_prop=train_missing_prop,
                    missing_location=missing_location,
                )

                train_missing_simulator = MissingModalitySimulator(
                    num_modalities=num_modalities,
                    modality_names=modality_names,
                    missing_prop=float(train_missing_prop),
                    missing_location=missing_location,
                )

                test_eval_setups = _build_test_eval_setups_for_run(
                    missing_location=missing_location,
                    train_missing_prop=train_missing_prop,
                    test_missing_props=test_missing_props,
                    num_modalities=num_modalities,
                    modality_names=modality_names,
                )

                odir = _build_output_dir(
                    base_odir=args.odir,
                    model_name=args.model,
                    imputation_method=args.imputation_method,
                    dataset_name=args.dataset,
                    missing_location=missing_location,
                    train_missing_prop=train_missing_prop,
                    seed=seed,
                    radio_aggregation_method=args.radio_aggregation_method,
                )
                best_epoch_warmup = min(5, int(args.epochs))
                print(
                    "Running seed="
                    f"{seed}, missing_location={missing_location}, "
                    f"train_missing_prop={train_missing_prop}"
                )
                print(f"Missing pattern seed: {int(args.missing_pattern_seed)}")
                print(f"Best-epoch warmup: {best_epoch_warmup}")
                print(f"Retrain outer train: {bool(args.retrain_outer)}")
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
                    "retrain_outer": bool(args.retrain_outer),
                    "best_epoch_warmup": best_epoch_warmup,
                    "inner_splits": args.inner_splits,
                    "outer_splits": args.outer_splits,
                    "train_missing_prop": float(train_missing_prop),
                    "missing_location": str(missing_location).lower(),
                    "missing_pattern_seed": int(args.missing_pattern_seed),
                    "imputation_method": args.imputation_method,
                    "radio_aggregation_method": str(args.radio_aggregation_method).strip().lower(),
                    "test_eval_combinations": len(test_eval_setups),
                    "test_missing_props_grid": ",".join(
                        str(float(setup["missing_prop"])) for setup in test_eval_setups
                    ),
                    "eval_missing_locations_grid": ",".join(
                        str(setup["missing_location"]).lower() for setup in test_eval_setups
                    ),
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

                inner_df, outer_df, history_df, split_df, test_predictions_df = nested_cv(
                    dfs=dfs,
                    inst_df=inst_df,
                    label_col=label_col,
                    epochs=args.epochs,
                    seed=seed,
                    hp_configs=hp_configs,
                    retrain_outer=bool(args.retrain_outer),
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
                    radio_aggregation_method=str(args.radio_aggregation_method).strip().lower(),
                    radio_pooling_kwargs={
                        "hidden_dim": int(args.radio_attention_hidden_dim),
                        "dropout": float(args.radio_attention_dropout),
                        "epochs": int(args.radio_attention_epochs),
                        "lr": float(args.radio_attention_lr),
                        "weight_decay": float(args.radio_attention_weight_decay),
                    },
                    candidate_model_dir=os.path.join(odir, "models"),
                )

                _save_run_outputs(
                    odir=odir,
                    inner_df=inner_df,
                    outer_df=outer_df,
                    history_df=history_df,
                    split_df=split_df,
                    test_predictions_df=test_predictions_df,
                    seed=seed,
                    missing_location=missing_location,
                    train_missing_prop=float(train_missing_prop),
                )

                print(f"Run finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
    print("Finish!")
