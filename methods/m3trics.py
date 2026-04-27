import argparse
import gc
import os
import sys
import time
from collections import OrderedDict

import pandas as pd

from configs.hyperparams import get_model_config, list_available_model_configs
from dataset.preprocess_dataset import (
    impute_modality_df,
    load_configured_modality_frames,
    load_endpoint_df,
    save_processed_outputs,
    summarize_duplicate_patient_rows,
    summarize_missing_values,
    validate_imputation_requirements,
)


# ------------------------ GENERIC ARG HELPERS ------------------------

def _parse_key_value_arg(raw_value, arg_name):
    if "=" not in raw_value:
        raise ValueError(f"{arg_name} expects NAME=VALUE, got '{raw_value}'.")
    key, value = raw_value.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"{arg_name} expects a non-empty NAME in '{raw_value}'.")
    return key, value


def _parse_keyed_str_map(values, arg_name):
    mapping = OrderedDict()
    for raw_value in values or []:
        key, value = _parse_key_value_arg(raw_value, arg_name)
        mapping[key] = value
    return mapping


def _parse_keyed_list_map(values, arg_name):
    mapping = OrderedDict()
    for raw_value in values or []:
        key, value = _parse_key_value_arg(raw_value, arg_name)
        if value == "":
            mapping[key] = []
        else:
            mapping[key] = [item.strip() for item in value.split(",") if item.strip()]
    return mapping


def _parse_csv_list(raw_value):
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


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


def _parse_modality_pooling(raw_value):
    mapping = OrderedDict()
    raw_text = str(raw_value).strip()
    if raw_text == "":
        return mapping

    for item in raw_text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid modality pooling item '{item}'. Use NAME=METHOD."
            )
        modality_name, method = item.split("=", 1)
        modality_name = modality_name.strip().lower()
        method = method.strip().lower()
        if not modality_name:
            raise argparse.ArgumentTypeError(
                f"Invalid modality pooling item '{item}': empty modality name."
            )
        if method not in {"mean", "attention"}:
            raise argparse.ArgumentTypeError(
                f"Invalid pooling method '{method}' for modality '{modality_name}'. "
                "Valid methods: mean, attention."
            )
        mapping[modality_name] = method
    return mapping


def _normalize_model_name(model_name):
    from utils import normalize_model_name

    return normalize_model_name(model_name)


def _parse_training_value_or_list(raw_value, dtype, to_lower=None):
    from utils import parse_value_or_list

    return parse_value_or_list(raw_value, dtype, to_lower=to_lower)


# ------------------------ PREPROCESS REPORTS -------------------------

def _print_missingness_report(summary_rows, modality_configs):
    print("\n=== Missing Values Report ===")
    for row in summary_rows:
        modality_name = row["modality"]
        modality_config = modality_configs[modality_name]
        print(
            f"[{row['modality']}] total_missing_cells={row['total_missing_cells']} | "
            f"columns_with_missing={row['columns_with_missing']} | "
            f"numeric_imputation={modality_config['numeric_imputation'] or 'none'} | "
            f"categorical_imputation={modality_config['categorical_imputation'] or 'none'} | "
            f"knn_neighbors={modality_config['knn_neighbors'] if modality_config['knn_neighbors'] is not None else 'none'}"
        )
        for col_name, miss_count in row["missing_by_column"].items():
            print(f"  - {col_name}: {miss_count}")


def _print_duplicate_report(summary_rows):
    print("\n=== Duplicate Patient-ID Report ===")
    for row in summary_rows:
        aggregation_method = row["aggregation_method"] if row["aggregation_method"] is not None else "none"
        print(
            f"[{row['modality']}] duplicated_patient_count={row['duplicated_patient_count']} | "
            f"aggregation_method={aggregation_method} | "
            f"max_rows_per_patient={row['max_rows_per_patient']}"
        )
        if row["example_patient_ids"]:
            preview = ", ".join(str(x) for x in row["example_patient_ids"])
            print(f"  - example duplicated IDs: {preview}")


def _build_training_modality_pooling(modality_configs):
    pairs = []
    for modality_name, config in modality_configs.items():
        aggregation_method = config.get("aggregation_method")
        if aggregation_method is None:
            continue
        pairs.append(f"{modality_name}={aggregation_method}")
    return ",".join(pairs)


def _cleanup_after_training_run():
    gc.collect()

    try:
        import wandb

        if getattr(wandb, "run", None) is not None:
            wandb.finish()
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


# ------------------------ TRAINING CLI ------------------------------

def build_training_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--odir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name suffix")
    parser.add_argument("--endpoint", type=str, required=True, help="Endpoint base name")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["MLP", "pAM", "PAMDiPAM", "MLPDiPAM", "HealNet", "SMILe"],
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help=(
            "Directory containing either a generic processed_data bundle produced by m3trics "
            "(or its parent results root), or a legacy raw dataset directory."
        ),
    )
    parser.add_argument("--inner_splits", type=int, default=5)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=str, default="16")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--retrain_outer",
        type=_parse_bool_flag,
        required=True,
        help=(
            "If true, refit one final model on the full outer-train split using the robustly selected HPs. "
            "If false, keep the previous behavior: ensemble the retained inner-fold models on outer-test."
        ),
    )
    parser.add_argument("--learning_rate", type=str, default="5e-5")
    parser.add_argument("--weight_decay", type=str, default="1e-4")
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="123")
    parser.add_argument(
        "--missing_pattern_seed",
        type=int,
        default=0,
        help="Deterministic seed used only for missing-modality mask simulation.",
    )
    parser.add_argument("--fusion_hidden_dim", type=str, default="32")
    parser.add_argument("--fusion_hidden_layers", type=str, default="1")
    parser.add_argument(
        "--fusion_batchnorm",
        type=str,
        default="false",
        help="Whether to use BatchNorm in the shared fusion block of the MLP. Supports scalar or comma-separated list.",
    )
    parser.add_argument("--modality_hidden_layers", type=str, default="1")
    parser.add_argument("--dropout", type=str, default="0.2")
    parser.add_argument("--pam_dropout", type=str, default="0.4")
    parser.add_argument("--pam_temperature", type=str, default="2.0")
    parser.add_argument(
        "--distill_alpha",
        type=str,
        default="1.0",
        help="Weight a for representation distillation loss in PAMDiPAM / MLPDiPAM. Supports scalar or comma-separated list.",
    )
    parser.add_argument(
        "--distill_beta",
        type=str,
        default="0.3",
        help="Weight b for feature/logit distillation loss in PAMDiPAM / MLPDiPAM. Supports scalar or comma-separated list.",
    )
    parser.add_argument("--smil_e_latent_dim", type=str, default="64")
    parser.add_argument("--smil_e_num_priors", type=str, default="64")
    parser.add_argument("--smil_e_num_heads", type=str, default="4")
    parser.add_argument("--smil_e_dropout", type=str, default="0.1")
    parser.add_argument("--smil_e_alpha", type=str, default="1e-2")
    parser.add_argument("--smil_e_beta", type=str, default="1e-2")
    parser.add_argument("--meta_inner_lr", type=str, default="1e-3")
    parser.add_argument("--meta_val_fraction", type=str, default="0.25")
    parser.add_argument("--classifier_hidden_dim", type=str, default="64")
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
    parser.add_argument(
        "--paired_hp_groups",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
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
    parser.add_argument("--vae_imputer_latent_dim", type=int, default=16)
    parser.add_argument("--vae_imputer_hidden_dim", type=int, default=128)
    parser.add_argument("--vae_imputer_epochs", type=int, default=30)
    parser.add_argument("--vae_imputer_batch_size", type=int, default=64)
    parser.add_argument("--vae_imputer_lr", type=float, default=1e-3)
    parser.add_argument("--vae_imputer_beta", type=float, default=1e-3)
    parser.add_argument("--radio_attention_hidden_dim", type=int, default=128)
    parser.add_argument("--radio_attention_dropout", type=float, default=0.1)
    parser.add_argument("--radio_attention_epochs", type=int, default=25)
    parser.add_argument("--radio_attention_lr", type=float, default=1e-3)
    parser.add_argument("--radio_attention_weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--modality_pooling",
        type=str,
        default="",
        help=(
            "Per-modality patient-level pooling configuration as comma-separated NAME=METHOD pairs. "
            "Supported methods: mean, attention. Example: 'radio=attention'. "
            "If omitted, all modalities use mean pooling."
        ),
    )
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
    return parser


def parse_training_args(argv=None):
    return build_training_arg_parser().parse_args(argv)


def _build_output_dir(
    base_odir,
    model_name,
    imputation_method,
    dataset_name,
    missing_location,
    train_missing_prop,
    seed,
):
    mapping = {
        "global": "GLOBAL",
        "radio": "RADIO",
        "path": "PATH",
        "clin": "CLIN",
        "radio_report": "RADIO_REPORT",
        "blood": "BLOOD",
    }
    model_name_norm = _normalize_model_name(model_name)
    if model_name_norm == "mlp":
        model_label = (
            f"{str(imputation_method).strip().upper()}_"
            f"{model_name_norm.upper().replace('_', '-')}"
        )
    elif model_name_norm == "pam":
        model_label = "PAM"
    elif model_name_norm == "pam_dipam":
        model_label = "PAMDiPAM"
    elif model_name_norm == "mlp_dipam":
        model_label = "MLPDiPAM"
    elif model_name_norm == "smil_e":
        model_label = "SMILE"
    else:
        model_label = model_name_norm.upper().replace("_", "-")
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

    metric_cols = [c for c in outer_df.columns if c.startswith("outer_test_")]
    if metric_cols:
        summary = {}
        for col in metric_cols:
            summary[f"{col}_mean"] = float(outer_df[col].mean())
            summary[f"{col}_std"] = float(outer_df[col].std())
        pd.DataFrame([summary]).to_csv(os.path.join(odir, "outer_test_summary.csv"), index=False)



def _build_test_eval_setups_for_run(
    missing_location,
    train_missing_prop,
    test_missing_props,
    num_modalities,
    modality_names,
):
    from dataset import MissingModalitySimulator

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


def run_training_from_args(args):
    if bool(args.wandb) and str(args.wandb_console).strip().lower() == "off":
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_QUIET", "true")
        os.environ.setdefault("WANDB_CONSOLE", "off")

    start_time = time.time()
    from dataset import MissingModalitySimulator, load_or_preprocess_dataset
    from train_ncv import nested_cv
    from utils import build_hyperparameter_grid

    model_name_norm = _normalize_model_name(args.model)
    print("Running")

    if str(args.imputation_method).strip().lower() != "zero" and model_name_norm != "mlp":
        raise ValueError(
            "imputation_method='knn' or 'vae' is currently supported only with model='MLP'."
        )

    inst_df, dfs, label_col, patient_id_col = load_or_preprocess_dataset(args)
    modality_names = list(dfs.keys())
    num_modalities = len(modality_names)
    print(f"Dataframes read. Starting {args.model} training.")

    seeds_list = _parse_training_value_or_list(args.seeds, int)
    missing_locations = _parse_training_value_or_list(args.missing_location, str, to_lower=True)
    train_missing_props = _parse_training_value_or_list(args.train_missing_prop, float)
    test_missing_props = _parse_training_value_or_list(args.test_missing_prop, float)

    invalid_train_locations = [
        loc for loc in missing_locations if loc != "global" and loc not in modality_names
    ]
    invalid_test_props = [p for p in test_missing_props if p < 0.0 or p > 1.0]
    invalid_train_props = [p for p in train_missing_props if p < 0.0 or p > 1.0]
    modality_pooling = _parse_modality_pooling(args.modality_pooling)
    if invalid_train_locations:
        valid = ", ".join(["global"] + sorted(modality_names))
        raise ValueError(
            f"Invalid --missing_location values: {', '.join(sorted(set(invalid_train_locations)))}. "
            f"Valid values: {valid}"
        )
    if invalid_train_props:
        raise ValueError(
            f"Invalid --train_missing_prop values: {invalid_train_props}. All values must be in [0, 1]."
        )
    if invalid_test_props:
        raise ValueError(
            f"Invalid --test_missing_prop values: {invalid_test_props}. All values must be in [0, 1]."
        )
    invalid_pooling_modalities = [name for name in modality_pooling.keys() if name not in modality_names]
    if invalid_pooling_modalities:
        raise ValueError(
            f"Invalid --modality_pooling modalities: {', '.join(sorted(invalid_pooling_modalities))}. "
            f"Available modalities: {', '.join(sorted(modality_names))}"
        )

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

    wandb_project_name = f"{args.wandb_project}_{args.dataset}"
    wandb_enabled_flag = bool(args.wandb and args.wandb_mode != "disabled")

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
                    "model": str(args.model),
                    "model_canonical": model_name_norm,
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
                    "modality_pooling": dict(modality_pooling),
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
                    early_stopping_patience=int(args.early_stopping_patience),
                    lr_scheduler_patience=int(args.lr_patience),
                    radio_pooling_kwargs={
                        "hidden_dim": int(args.radio_attention_hidden_dim),
                        "dropout": float(args.radio_attention_dropout),
                        "epochs": int(args.radio_attention_epochs),
                        "lr": float(args.radio_attention_lr),
                        "weight_decay": float(args.radio_attention_weight_decay),
                    },
                    modality_pooling=modality_pooling,
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


def training_cli_main(argv=None):
    args = parse_training_args(argv)
    run_training_from_args(args)


# ------------------------ M3TRICS PREPROCESS CLI ---------------------

def _build_training_args_from_model_config(shared_args, model_config, modality_pooling):
    fixed_args = dict(model_config.get("fixed_args", {}))
    output_root = os.path.join(
        shared_args.results_root,
        "training_runs",
        f"{model_config['display_name']}_retrain{str(bool(shared_args.retrain_outer)).lower()}_k{int(shared_args.outer_splits)}",
    )

    arg_list = [
        "--model",
        str(model_config["model"]),
        "--dataset",
        str(shared_args.dataset),
        "--odir",
        output_root,
        "--dataset_dir",
        str(shared_args.results_root),
        "--endpoint",
        str(shared_args.endpoint_col),
        "--inner_splits",
        str(int(shared_args.inner_splits)),
        "--outer_splits",
        str(int(shared_args.outer_splits)),
        "--epochs",
        str(int(fixed_args.get("epochs", 80))),
        "--retrain_outer",
        str(bool(shared_args.retrain_outer)).lower(),
        "--seeds",
        str(shared_args.seeds),
        "--missing_pattern_seed",
        str(int(shared_args.missing_pattern_seed)),
        "--missing_location",
        str(shared_args.missing_location),
        "--train_missing_prop",
        str(shared_args.train_missing_prop),
        "--test_missing_prop",
        str(shared_args.test_missing_prop),
    ]
    for arg_name, arg_value in fixed_args.items():
        if arg_name == "epochs":
            continue
        arg_list.extend([f"--{arg_name}", str(arg_value)])
    resolved_modality_pooling = str(modality_pooling).strip()
    if resolved_modality_pooling:
        arg_list.extend(["--modality_pooling", resolved_modality_pooling])
    if bool(shared_args.wandb):
        arg_list.extend(
            [
                "--wandb",
                "--wandb_project",
                str(model_config["display_name"]),
                "--wandb_mode",
                str(shared_args.wandb_mode),
            ]
        )

    for arg_name, arg_value in model_config.get("hp_grid_args", {}).items():
        arg_list.extend([f"--{arg_name}", str(arg_value)])

    paired_hp_groups = model_config.get("paired_hp_grid_args", [])
    if paired_hp_groups:
        serialized_groups = ";".join(
            ",".join(str(name) for name in group)
            for group in paired_hp_groups
            if group
        )
        if serialized_groups:
            arg_list.extend(["--paired_hp_groups", serialized_groups])

    return parse_training_args(arg_list)


def _run_selected_models(args, modality_configs):
    selected_models = _parse_csv_list(args.run_models)
    if not selected_models:
        print("No training models selected. Stopping after preprocessing.")
        return

    training_modality_pooling = _build_training_modality_pooling(modality_configs)
    print("\n=== Training Launch ===")
    print(f"Selected models: {', '.join(selected_models)}")
    print(f"Seeds: {args.seeds}")
    print(f"Missing pattern seed: {args.missing_pattern_seed}")
    print(f"Inner splits: {args.inner_splits}")
    print(f"Outer splits: {args.outer_splits}")
    print(f"Retrain outer: {bool(args.retrain_outer)}")
    print(f"Train missing prop: {args.train_missing_prop}")
    print(f"Test missing prop: {args.test_missing_prop}")
    print(f"Missing location: {args.missing_location}")
    if training_modality_pooling:
        print(f"Duplicate-row aggregation for training: {training_modality_pooling}")

    for raw_model_name in selected_models:
        model_config = get_model_config(raw_model_name)
        training_args = _build_training_args_from_model_config(
            shared_args=args,
            model_config=model_config,
            modality_pooling=training_modality_pooling,
        )
        print(f"\nLaunching {model_config['display_name']}...")
        try:
            run_training_from_args(training_args)
        finally:
            del training_args
            _cleanup_after_training_run()


def build_preprocessing_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--results_root", required=True, type=str)
    parser.add_argument("--endpoint_csv", required=True, type=str)
    parser.add_argument("--patient_id_col", required=True, type=str)
    parser.add_argument("--endpoint_col", required=True, type=str)
    parser.add_argument("--modality_csv", action="append", default=[])
    parser.add_argument("--categorical_cols", action="append", default=[])
    parser.add_argument("--drop_cols", action="append", default=[])
    parser.add_argument("--aggregation_method", action="append", default=[])
    parser.add_argument("--numeric_imputation", action="append", default=[])
    parser.add_argument("--categorical_imputation", action="append", default=[])
    parser.add_argument("--knn_neighbors", action="append", default=[])
    parser.add_argument(
        "--run_models",
        type=str,
        default="",
        help=(
            "Comma-separated list of model configs to launch after preprocessing. "
            f"Available: {', '.join(list_available_model_configs())}"
        ),
    )
    parser.add_argument("--inner_splits", required=True, type=int)
    parser.add_argument("--outer_splits", required=True, type=int)
    parser.add_argument("--retrain_outer", required=True, type=_parse_bool_flag)
    parser.add_argument("--missing_location", required=True, type=str)
    parser.add_argument("--train_missing_prop", required=True, type=str)
    parser.add_argument("--test_missing_prop", required=True, type=str)
    parser.add_argument("--seeds", required=True, type=str)
    parser.add_argument("--missing_pattern_seed", required=True, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser


def parse_preprocessing_args(argv=None):
    return build_preprocessing_arg_parser().parse_args(argv)


def main(argv=None):
    args = parse_preprocessing_args(argv)
    output_dir = os.path.join(args.results_root, "processed_data")

    endpoint_df = load_endpoint_df(
        path=args.endpoint_csv,
        patient_id_col=args.patient_id_col,
        endpoint_col=args.endpoint_col,
    )
    modality_frames, modality_configs = load_configured_modality_frames(
        modality_paths=_parse_keyed_str_map(args.modality_csv, "--modality_csv"),
        patient_id_col=args.patient_id_col,
        endpoint_df=endpoint_df,
        drop_cols_map=_parse_keyed_list_map(args.drop_cols, "--drop_cols"),
        categorical_cols_map=_parse_keyed_list_map(args.categorical_cols, "--categorical_cols"),
        aggregation_map=_parse_keyed_str_map(args.aggregation_method, "--aggregation_method"),
        numeric_imputation_map=_parse_keyed_str_map(args.numeric_imputation, "--numeric_imputation"),
        categorical_imputation_map=_parse_keyed_str_map(args.categorical_imputation, "--categorical_imputation"),
        knn_neighbors_map=_parse_keyed_str_map(args.knn_neighbors, "--knn_neighbors"),
    )

    missing_summary, total_missing = summarize_missing_values(
        modality_frames,
        modality_configs,
        id_col=args.patient_id_col,
    )
    _print_missingness_report(missing_summary, modality_configs)
    if total_missing > 0:
        validate_imputation_requirements(
            modality_frames=modality_frames,
            modality_configs=modality_configs,
            id_col=args.patient_id_col,
        )
        print("Missing values were found. Applying the configured imputation plan.")
        imputed_frames = OrderedDict()
        modalities_with_missing = {
            row["modality"]
            for row in missing_summary
            if int(row["total_missing_cells"]) > 0
        }
        for modality_name, df in modality_frames.items():
            if modality_name not in modalities_with_missing:
                imputed_frames[modality_name] = df
                continue
            imputed_frames[modality_name] = impute_modality_df(
                df=df,
                id_col=args.patient_id_col,
                categorical_cols=modality_configs[modality_name]["categorical_cols"],
                numeric_imputation=modality_configs[modality_name]["numeric_imputation"],
                categorical_imputation=modality_configs[modality_name]["categorical_imputation"],
                knn_neighbors=modality_configs[modality_name]["knn_neighbors"],
                modality_name=modality_name,
            )
        modality_frames = imputed_frames
        print("Missing-value imputation completed.")
    else:
        print("No missing values found in modality dataframes.")

    duplicate_summary = summarize_duplicate_patient_rows(
        modality_frames,
        modality_configs,
        id_col=args.patient_id_col,
    )
    _print_duplicate_report(duplicate_summary)
    duplicated_modalities = [row for row in duplicate_summary if int(row["duplicated_patient_count"]) > 0]
    if duplicated_modalities:
        missing_aggregation_modalities = [
            row["modality"]
            for row in duplicated_modalities
            if row["aggregation_method"] is None
        ]
        if missing_aggregation_modalities:
            raise ValueError(
                "Duplicated patient ids were found in modalities without an aggregation method: "
                f"{', '.join(missing_aggregation_modalities)}. "
                "Define --aggregation_method for those modalities before continuing."
            )
        print(
            "Duplicated patient ids were found. The configured aggregation methods will be kept "
            "for downstream training."
        )
        print("Duplicate rows were preserved for downstream fold-wise aggregation.")
    else:
        print("No duplicated patient ids found in modality dataframes.")

    final_shapes = {
        modality_name: {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        }
        for modality_name, df in modality_frames.items()
    }
    summary_payload = {
        "dataset": args.dataset,
        "results_root": args.results_root,
        "endpoint_csv": args.endpoint_csv,
        "endpoint_col": args.endpoint_col,
        "patient_id_col": args.patient_id_col,
        "modality_imputation_config": {
            modality_name: {
                "numeric_imputation": config["numeric_imputation"],
                "categorical_imputation": config["categorical_imputation"],
                "knn_neighbors": config["knn_neighbors"],
            }
            for modality_name, config in modality_configs.items()
        },
        "modality_aggregation_config": {
            modality_name: config["aggregation_method"]
            for modality_name, config in modality_configs.items()
        },
        "missing_summary": missing_summary,
        "duplicate_summary": duplicate_summary,
        "final_shapes": final_shapes,
    }
    save_processed_outputs(
        output_dir=output_dir,
        endpoint_df=endpoint_df,
        patient_id_col=args.patient_id_col,
        endpoint_col=args.endpoint_col,
        modality_frames=modality_frames,
        summary_payload=summary_payload,
    )

    print("\n=== Preprocessing Complete ===")
    print(f"Preprocessing results saved to: {args.results_root}")
    _run_selected_models(args, modality_configs)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[m3trics] ERROR: {exc}", file=sys.stderr)
        raise
