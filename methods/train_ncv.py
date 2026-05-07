import copy
import gc
import os
import pickle
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from imputation_methods import build_imputer
from dataset import (
    MultimodalBaseDataset,
    MultimodalDatasetWithMissing,
    multimodal_collate,
    build_loaders,
)
from utils import (
    build_model,
    safe_binary_metrics,
    select_device,
    fit_and_transform_modalities,
    set_global_seed,
    normalize_model_name,
)
from dataset.preprocess_dataset import collapse_patient_rows, filter_by_patients
from dataset.radiology_attention_pooling import RadiologyAttentionPooler
from model_training import (
    get_model_init_kwargs,
    train_model_on_full_dataset,
    train_model_with_validation,
)
try:
    import wandb
except ImportError:
    wandb = None

WANDB_INIT_TIMEOUT_SEC = int(os.getenv("WANDB_INIT_TIMEOUT_SEC", "180"))

# ---------------------------- HELPER FUNCTIONS -----------------------------

# Function to transform outer test with each inner train scaler
def _transform_modalities_with_fitted_scalers(dfs_raw, scalers, patient_id_col="patient"):
    """Apply pre-fitted per-modality scalers to raw modality dataframes."""
    dfs_scaled = {}
    for name, df_raw in dfs_raw.items():
        if name not in scalers:
            raise ValueError(f"Missing scaler for modality '{name}'.")

        df_scaled = df_raw.copy()
        feats = [c for c in df_scaled.columns if c != patient_id_col]
        if len(df_scaled) > 0 and feats:
            values = df_scaled[feats].to_numpy(dtype=np.float32, copy=True)
            transformed = scalers[name].transform(values).astype(np.float32)
            feat_df = pd.DataFrame(transformed, columns=feats, index=df_scaled.index)
            base_cols = [c for c in df_scaled.columns if c not in feats]
            df_scaled = pd.concat([df_scaled[base_cols], feat_df], axis=1)

        dfs_scaled[name] = df_scaled

    return dfs_scaled


def _prepare_patient_level_modalities(
    dfs_raw,
    patient_id_col="patient",
    modality_pooling=None,
    fit_attention_poolers=False,
    fitted_poolers=None,
    radio_pooling_kwargs=None,
    labels_df=None,
    label_col=None,
):
    """Collapse raw split dataframes to one row per patient, optionally with learned attention pooling."""
    modality_pooling = dict(modality_pooling or {})
    prepared = {}
    active_poolers = dict(fitted_poolers or {})

    for modality_name, df_raw in dfs_raw.items():
        pooling_method = str(modality_pooling.get(modality_name, "mean")).strip().lower()
        if pooling_method not in {"mean", "attention"}:
            raise ValueError(
                f"Unsupported pooling method '{pooling_method}' for modality '{modality_name}'. "
                "Valid methods: mean, attention."
            )

        if pooling_method == "attention":
            if fit_attention_poolers:
                if labels_df is None or label_col is None:
                    raise ValueError(
                        "labels_df and label_col are required when fitting an attention pooler."
                    )
                feature_cols = [c for c in df_raw.columns if c != patient_id_col]
                if not feature_cols:
                    raise ValueError(
                        f"Modality '{modality_name}' has no feature columns to attention-pool."
                    )
                active_poolers[modality_name] = RadiologyAttentionPooler(
                    input_dim=len(feature_cols),
                    **dict(radio_pooling_kwargs or {}),
                )
                active_poolers[modality_name].fit(
                    df_train=df_raw,
                    labels_df=labels_df,
                    id_col=patient_id_col,
                    label_col=label_col,
                )
            if modality_name not in active_poolers or active_poolers[modality_name] is None:
                raise RuntimeError(
                    f"Attention pooling was requested for modality '{modality_name}' but no fitted pooler is available."
                )
            df_patient = active_poolers[modality_name].transform(df_raw)
        else:
            df_patient = collapse_patient_rows(df_raw, id_col=patient_id_col, strategy="mean")

        prepared[modality_name] = df_patient.set_index(patient_id_col, drop=False)

    return prepared, active_poolers


def _build_model_kwargs_from_hp_cfg(model_name_l, hp_cfg):
    if model_name_l in {"mlp"}:
        return {
            "modality_hidden_layers": hp_cfg["modality_hidden_layers"],
            "fusion_hidden_dim": hp_cfg["fusion_hidden_dim"],
            "fusion_hidden_layers": hp_cfg["fusion_hidden_layers"],
            "dropout_p": hp_cfg["dropout"],
            "fusion_batchnorm": bool(hp_cfg["fusion_batchnorm"]),
        }
    if model_name_l in {"pam"}:
        return {
            "dropout_p": hp_cfg["pam_dropout"],
            "temperature": hp_cfg["pam_temperature"],
        }
    if model_name_l in {"dipam"}:
        return {
            "dropout_p": hp_cfg["pam_dropout"],
            "temperature": hp_cfg["pam_temperature"],
            "distill_alpha": hp_cfg["distill_alpha"],
            "distill_beta": hp_cfg["distill_beta"],
        }
    if model_name_l in {"di_mmlp"}:
        return {
            "modality_hidden_layers": hp_cfg["modality_hidden_layers"],
            "fusion_hidden_dim": hp_cfg["fusion_hidden_dim"],
            "fusion_hidden_layers": hp_cfg["fusion_hidden_layers"],
            "dropout_p": hp_cfg["dropout"],
            "fusion_batchnorm": bool(hp_cfg["fusion_batchnorm"]),
            "distill_alpha": hp_cfg["distill_alpha"],
            "distill_beta": hp_cfg["distill_beta"],
        }
    if model_name_l in {"smil_e"}:
        return {
            "latent_dim": hp_cfg["smil_e_latent_dim"],
            "num_priors": hp_cfg["smil_e_num_priors"],
            "num_heads": hp_cfg["smil_e_num_heads"],
            "dropout": hp_cfg["smil_e_dropout"],
            "classifier_hidden_dim": hp_cfg["classifier_hidden_dim"],
            "alpha": hp_cfg["smil_e_alpha"],
            "beta": hp_cfg["smil_e_beta"],
            "meta_inner_lr": hp_cfg["meta_inner_lr"],
            "meta_val_fraction": hp_cfg["meta_val_fraction"],
        }
    if model_name_l in {"healnet"}:
        return {
            "depth": hp_cfg["healnet_depth"],
            "num_freq_bands": hp_cfg["healnet_num_freq_bands"],
            "num_latents": hp_cfg["healnet_num_latents"],
            "latent_dim": hp_cfg["healnet_latent_dim"],
            "cross_heads": hp_cfg["healnet_cross_heads"],
            "latent_heads": hp_cfg["healnet_latent_heads"],
            "cross_dim_head": hp_cfg["healnet_cross_dim_head"],
            "latent_dim_head": hp_cfg["healnet_latent_dim_head"],
            "attn_dropout": hp_cfg["healnet_attn_dropout"],
            "ff_dropout": hp_cfg["healnet_ff_dropout"],
            "self_per_cross_attn": hp_cfg["healnet_self_per_cross_attn"],
        }
    raise ValueError(
        f"Unsupported model '{model_name_l}'. Supported: mlp, di_mmlp, pam, dipam, smile, healnet"
    )


def _candidate_bundle_path(candidate_model_dir, outer_fold_idx, inner_fold_idx, hp_name):
    safe_hp_name = str(hp_name).replace(os.sep, "_")
    fold_dir = os.path.join(
        candidate_model_dir,
        f"outer_fold_{int(outer_fold_idx)}",
        f"inner_fold_{int(inner_fold_idx)}",
    )
    os.makedirs(fold_dir, exist_ok=True)
    return os.path.join(fold_dir, f"{safe_hp_name}.pkl")


def _save_candidate_bundle(
    bundle_path,
    model,
    model_name,
    input_dims,
    model_kwargs,
    scalers,
    imputer,
    modality_poolers,
):
    bundle = {
        "model_name": normalize_model_name(model_name),
        "input_dims": [int(dim) for dim in input_dims],
        "model_kwargs": get_model_init_kwargs(model_name, model_kwargs),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "scalers": scalers,
        "imputer": imputer,
        "modality_poolers": dict(modality_poolers or {}),
    }
    with open(bundle_path, "wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_candidate_bundle(bundle_path, device):
    with open(bundle_path, "rb") as handle:
        bundle = pickle.load(handle)

    model = build_model(
        bundle["model_name"],
        bundle["input_dims"],
        get_model_init_kwargs(bundle["model_name"], bundle["model_kwargs"]),
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])

    modality_poolers = bundle.get("modality_poolers")
    if modality_poolers is None:
        legacy_radio_pooler = bundle.get("radio_pooler")
        modality_poolers = {} if legacy_radio_pooler is None else {"radio": legacy_radio_pooler}

    return {
        "model": model,
        "scalers": bundle["scalers"],
        "imputer": bundle["imputer"],
        "modality_poolers": modality_poolers,
    }


def _fit_split_imputer(
    split_dataset,
    split_missing_simulator,
    apply_split_missing,
    imputation_method,
    missing_pattern_seed,
    imputer_seed,
    imputer_kwargs=None,
):
    method_l = str(imputation_method).strip().lower()
    if method_l == "zero":
        return None

    split_reference_dataset = MultimodalDatasetWithMissing(
        base_dataset=split_dataset,
        simulator=split_missing_simulator,
        apply_missing=apply_split_missing,
        imputation_method="zero",
        missing_pattern_seed=missing_pattern_seed,
    )
    return build_imputer(
        imputation_method=method_l,
        reference_dataset=split_reference_dataset,
        knn_k=5,
        vae_kwargs=imputer_kwargs,
        imputer_seed=imputer_seed,
    )

# Function to predict on outer fold for each inner fold model
def _predict_model_probabilities(
    model,
    data_loader,
    device,
    bypass_mask=False,
    collect_pam_details=False,
    model_name=None,
):
    """Run one model on a loader and return y_true / logits / probabilities / pids."""
    model.eval()
    y_true = []
    y_logits = []
    y_prob = []
    pids = []
    pam_alpha = []
    pam_r_scores = []

    with torch.no_grad():
        for Xs, present_mask, y, pid_batch in data_loader:
            Xs = [x.to(device) for x in Xs]
            present_mask = present_mask.to(device)

            model_mask = None if bypass_mask else present_mask
            if collect_pam_details:
                model_out = model(Xs, model_mask, return_aux=True)
                logits = model_out[0].squeeze(1)
                if model_name not in {"pam", "dipam"}:
                    raise ValueError(
                        "collect_pam_details=True is only supported for model_name in "
                        "{'pam', 'dipam'}."
                    )
                pam_alpha.append(model_out[2].detach().cpu().numpy())
                pam_r_scores.append(model_out[3].detach().cpu().numpy())
            else:
                logits = model(Xs, model_mask).squeeze(1)
            logits_np = logits.detach().cpu().numpy().reshape(-1)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            y_logits.extend(logits_np.tolist())
            y_prob.extend(probs.tolist())
            y_true.extend(y.cpu().numpy().tolist())
            pids.extend(pid_batch)

    pam_details = None
    if collect_pam_details:
        pam_details = {
            "alpha": np.concatenate(pam_alpha, axis=0),
            "R": np.concatenate(pam_r_scores, axis=0),
        }

    return np.asarray(y_true), np.asarray(y_logits), np.asarray(y_prob), list(pids), pam_details

def _log_selected_inner_models_to_wandb(
    selected_candidates,
    seed,
    outer_fold_idx,
    train_missing_simulator,
    wandb_project,
    wandb_mode,
    wandb_base_config,
    model_name_l,
):
    missing_location = str((wandb_base_config or {}).get("missing_location", "na")).strip().lower()
    train_missing_prop = float((wandb_base_config or {}).get("train_missing_prop", 0.0))

    for candidate in selected_candidates:
        run_name = (
            f"loc{missing_location}_"
            f"trainprop{train_missing_prop:g}_"
            f"seed{seed}_"
            f"outer{outer_fold_idx}_"
            f"inner{int(candidate['inner_fold'])}"
        )
        run_config = dict(wandb_base_config or {})
        run_config.update(
            {
                "seed": seed,
                "outer_fold": outer_fold_idx,
                "inner_fold": int(candidate["inner_fold"]),
                "model_type": "inner",
                "phase": "selected_inner_model",
                "selected_hp_name": candidate["hp_name"],
            }
        )
        if model_name_l in {"healnet", "smil_e"}:
            run_config.update(dict(candidate.get("hp_cfg", {})))
        inner_run = wandb.init(
            project=wandb_project,
            group=f"outer_fold_{outer_fold_idx}",
            name=run_name,
            mode=wandb_mode,
            config=run_config,
            reinit="finish_previous",
            settings=wandb.Settings(init_timeout=WANDB_INIT_TIMEOUT_SEC),
        )

        for hrow in candidate["history"]:
            epoch_i = int(hrow["epoch"])
            log_payload = {
                "best_inner_model/train_loss": float(hrow["train_loss"]),
                "best_inner_model/val_loss": float(hrow["val_loss"]),
                "best_inner_model/val_auc": float(hrow["val_auc"]),
                "best_inner_model/val_aucpr": float(hrow["val_aucpr"]),
                "best_inner_model/val_acc": float(hrow["val_acc"]),
            }
            if model_name_l in {"dipam", "di_mmlp"}:
                log_payload.update(
                    {
                        "best_inner_model/teacher_loss": float(hrow["teacher_loss"]),
                        "best_inner_model/student_survival_loss": float(hrow["student_survival_loss"]),
                        "best_inner_model/student_repr_loss": float(hrow["student_repr_loss"]),
                        "best_inner_model/student_feature_loss": float(hrow["student_feature_loss"]),
                    }
                )
            elif model_name_l == "smil_e":
                log_payload.update(
                    {
                        "best_inner_model/smil_meta_train_loss": float(hrow["smil_meta_train_loss"]),
                        "best_inner_model/smil_meta_val_loss": float(hrow["smil_meta_val_loss"]),
                        "best_inner_model/smil_meta_val_ce": float(hrow["smil_meta_val_ce"]),
                        "best_inner_model/smil_align_fusion": float(hrow["smil_align_fusion"]),
                        "best_inner_model/smil_align_hidden": float(hrow["smil_align_hidden"]),
                    }
                )
            inner_run.log(log_payload, step=epoch_i)

        inner_run.finish()


def _evaluate_retained_inner_models_on_outer_test(
    selected_candidates,
    dfs_test_outer_raw,
    inst_df_test_outer,
    label_col,
    patient_id_col,
    modality_pooling,
    radio_pooling_kwargs,
    test_eval_setups,
    modality_names,
    outer_fold_idx,
    train_missing_simulator,
    predict_bypass_mask,
    model_name_l,
    outer_eval_batch_size,
    imputation_method,
    missing_pattern_seed,
    device,
    selected_inner_rows,
    best_hp_row,
    epsilon,
    best_mean_auc,
):
    outer_results = []
    test_prediction_rows = []

    for eval_setup in test_eval_setups:
        eval_simulator = eval_setup["simulator"]
        eval_missing_location = str(eval_setup["missing_location"]).lower()
        eval_missing_prop = float(eval_setup["missing_prop"])
        apply_missing_eval = eval_missing_prop > 0.0

        ref_y_true = None
        ref_pids = None
        model_logits = []
        model_probs = []
        model_details = []
        model_outer_metrics = []

        for candidate in selected_candidates:
            bundle_path = candidate.get("bundle_path")
            if not bundle_path:
                raise RuntimeError(
                    "Missing candidate bundle path for outer-test prediction."
                )

            loaded_bundle = _load_candidate_bundle(bundle_path, device=device)

            dfs_outer_eval_prepared, _ = _prepare_patient_level_modalities(
                dfs_test_outer_raw,
                patient_id_col=patient_id_col,
                modality_pooling=modality_pooling,
                fit_attention_poolers=False,
                fitted_poolers=loaded_bundle["modality_poolers"],
                radio_pooling_kwargs=radio_pooling_kwargs,
                labels_df=None,
                label_col=None,
            )
            dfs_outer_eval_scaled = _transform_modalities_with_fitted_scalers(
                dfs_outer_eval_prepared,
                loaded_bundle["scalers"],
                patient_id_col=patient_id_col,
            )
            outer_eval_base = MultimodalBaseDataset(
                dfs=dfs_outer_eval_scaled,
                label_df=inst_df_test_outer,
                label_col=label_col,
                id_col=patient_id_col,
            )
            outer_eval_ds = MultimodalDatasetWithMissing(
                base_dataset=outer_eval_base,
                simulator=eval_simulator,
                apply_missing=apply_missing_eval,
                imputation_method=imputation_method,
                missing_pattern_seed=missing_pattern_seed,
                prefit_imputer=loaded_bundle["imputer"],
                imputer_kwargs=None,
            )
            outer_eval_loader = DataLoader(
                outer_eval_ds,
                batch_size=outer_eval_batch_size,
                shuffle=False,
                collate_fn=multimodal_collate,
                drop_last=False,
            )

            y_true_outer, y_logits_outer, y_prob_outer, pids_outer, pam_details_outer = _predict_model_probabilities(
                model=loaded_bundle["model"],
                data_loader=outer_eval_loader,
                device=device,
                bypass_mask=predict_bypass_mask,
                collect_pam_details=model_name_l in {"pam", "dipam"},
                model_name=model_name_l,
            )

            if ref_y_true is None:
                ref_y_true = y_true_outer
                ref_pids = list(pids_outer)
            else:
                if not np.array_equal(ref_y_true, y_true_outer):
                    raise RuntimeError("Retained inner-model predictions are misaligned on y_true.")
                if ref_pids != list(pids_outer):
                    raise RuntimeError("Retained inner-model predictions are misaligned on patient IDs.")

            model_logits.append(y_logits_outer)
            model_probs.append(y_prob_outer)
            model_details.append(pam_details_outer)
            model_outer_metrics.append(safe_binary_metrics(y_true_outer, y_prob_outer))

            del loaded_bundle["model"]
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        per_patient_prediction_rows = []
        for patient_idx, pid in enumerate(ref_pids):
            row = {
                "outer_fold": outer_fold_idx,
                "outer_eval_target": "test_outer",
                "patient": pid,
                "train_missing_location": str(getattr(train_missing_simulator, "missing_location", "global")).lower(),
                "train_missing_prop": float(getattr(train_missing_simulator, "missing_prop", 0.0)),
                "test_missing_location": eval_missing_location,
                "test_missing_prop": eval_missing_prop,
                "y_true": int(ref_y_true[patient_idx]),
            }

            for model_idx, (logits_arr, probs_arr, details_arr) in enumerate(
                zip(model_logits, model_probs, model_details),
                1,
            ):
                row[f"inner_model_{model_idx}_logit"] = float(logits_arr[patient_idx])
                row[f"inner_model_{model_idx}_prob"] = float(probs_arr[patient_idx])
                row[f"inner_model_{model_idx}_pred_label"] = int(logits_arr[patient_idx] >= 0.0)
                if model_name_l in {"pam", "dipam"} and details_arr is not None:
                    for modality_idx, modality_name in enumerate(modality_names):
                        row[f"inner_model_{model_idx}_{modality_name}_alpha"] = float(
                            details_arr["alpha"][patient_idx, modality_idx]
                        )
                        row[f"inner_model_{model_idx}_{modality_name}_R"] = float(
                            details_arr["R"][patient_idx, modality_idx]
                        )

            per_patient_prediction_rows.append(row)

        test_prediction_rows.extend(per_patient_prediction_rows)

        metric_names = ["LOGLOSS", "AUC", "AUCPR", "ACC", "SEN", "SP", "MCC"]
        mean_outer_metrics = {
            name: float(np.mean([float(metrics[name]) for metrics in model_outer_metrics]))
            for name in metric_names
        }

        outer_results.append(
            {
                "outer_fold": outer_fold_idx,
                "outer_eval_target": "test_outer",
                "eval_missing_location": eval_missing_location,
                "eval_missing_prop": eval_missing_prop,
                "inner_models_count": int(len(selected_candidates)),
                "selected_inner_hp_names": str(best_hp_row["hp_name"]),
                "selected_inner_mean_AUC": float(np.mean([r["val_best_AUC"] for r in selected_inner_rows])),
                "selected_inner_mean_LOGLOSS": float(np.mean([r["val_best_LOGLOSS"] for r in selected_inner_rows])),
                "selected_inner_std_AUC": float(best_hp_row["std_auc"]),
                "hp_selection_epsilon": float(epsilon),
                "hp_selection_best_mean_AUC": float(best_mean_auc),
                "outer_test_metric_source": "mean_retained_inner_models",
                "outer_refit_epochs": np.nan,
                "outer_test_LOGLOSS": float(mean_outer_metrics["LOGLOSS"]),
                "outer_test_AUC": float(mean_outer_metrics["AUC"]),
                "outer_test_AUCPR": float(mean_outer_metrics["AUCPR"]),
                "outer_test_ACC": float(mean_outer_metrics["ACC"]),
                "outer_test_SEN": float(mean_outer_metrics["SEN"]),
                "outer_test_SP": float(mean_outer_metrics["SP"]),
                "outer_test_MCC": float(mean_outer_metrics["MCC"]),
            }
        )

    return outer_results, test_prediction_rows


# --------------------------- NESTED CV FUNCTION -----------------------------

# Main nested cross-validation function
def nested_cv(
    dfs,
    inst_df,
    label_col,
    epochs,
    seed,
    hp_configs,
    train_missing_simulator,
    model_name,
    imputation_method="zero",
    inner_splits=5,
    outer_splits=5,
    missing_pattern_seed=0,
    patient_id_col="patient",
    wandb_enabled=False,
    wandb_project=None,
    wandb_mode="online",
    wandb_base_config=None,
    test_eval_setups=None,
    imputer_kwargs=None,
    radio_pooling_kwargs=None,
    modality_pooling=None,
    candidate_model_dir=None,
    retrain_outer=True,
    save_inner=False,
    early_stopping_patience=20,
    min_lr=1e-6,
    hp_selection_epsilon=0.02,
):
    wandb_active = bool(wandb_enabled and wandb is not None)
    if wandb_enabled and wandb is None:
        print("wandb is not installed. Continuing without wandb logging.")

    # Train missingness is applied on both inner-train and inner-validation.
    apply_missing_train = float(getattr(train_missing_simulator, "missing_prop", 0.0)) > 0.0
    model_name_l = normalize_model_name(model_name)
    predict_bypass_mask = (
        model_name_l == "mlp"
        and str(imputation_method).strip().lower() in {"knn", "vae"}
    )
    set_global_seed(seed, deterministic=True)

    # Test-time evaluation setups:
    # by default evaluate once with the provided train simulator.
    if test_eval_setups is None:
        eval_missing_prop = float(getattr(train_missing_simulator, "missing_prop", 0.0))
        eval_missing_location = str(getattr(train_missing_simulator, "missing_location", "global")).lower()
        test_eval_setups = [
            {
                "missing_prop": eval_missing_prop,
                "missing_location": eval_missing_location,
                "simulator": train_missing_simulator,
            }
        ]
    elif len(test_eval_setups) == 0:
        raise ValueError("test_eval_setups cannot be empty when provided.")

    # Get patient IDs and labels from  inst_df
    patients = inst_df[patient_id_col].values
    y = inst_df[label_col].values

    # Set up outer and inner cross-validation splits
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    # Input dims for each modality (removing patient column) and select device
    input_dims = [df.shape[1] - 1 for df in dfs.values()]
    modality_names = [str(name) for name in dfs.keys()]
    device = select_device()
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Using device: cuda ({gpu_name})")
    elif device.type == "mps":
        print("Using device: mps (Apple Metal)")
    else:
        print("Using device: cpu")

    inner_eval_rows = []
    history_rows = []
    outer_results = []
    split_rows = []
    test_prediction_rows = []
    saved_inner_outer_results = []
    saved_inner_test_prediction_rows = []

    for outer_fold_idx, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(patients, y), 1):
        print(f"\nOuter fold {outer_fold_idx}")

        # Get and save patient IDs splits for each outer fold
        train_outer_ids = patients[train_outer_idx]
        test_outer_ids = patients[test_outer_idx]

        for pid in train_outer_ids:
            split_rows.append({"outer_fold": outer_fold_idx, "split": "train_outer", "patient": pid})
        for pid in test_outer_ids:
            split_rows.append({"outer_fold": outer_fold_idx, "split": "test_outer", "patient": pid})

        # Filter inst_df for outer-train fold and get patient IDs and ys
        inst_df_train_outer = filter_by_patients(inst_df, train_outer_ids, id_col=patient_id_col)
        patients_train_outer = inst_df_train_outer[patient_id_col].values
        y_train_outer = inst_df_train_outer[label_col].values

        # Split dfs into outer-train data.
        dfs_train_outer_raw = {
            name: filter_by_patients(df, train_outer_ids, id_col=patient_id_col)
            for name, df in dfs.items()
        }

        # Train all HPs across all inner folds, then select one HP for the whole outer fold.
        all_inner_candidates_by_hp = {}
        selected_inner_rows = []
        selected_inner_histories = []

        for inner_fold_idx, (train_inner_idx, val_inner_idx) in enumerate(
            inner_cv.split(patients_train_outer, y_train_outer), 1
        ):
            # Get patient IDs splits for each inner fold
            train_inner_ids = patients_train_outer[train_inner_idx]
            val_inner_ids = patients_train_outer[val_inner_idx]

            # Filter inst_df for inner-train and inner-val folds
            inst_df_train_inner = filter_by_patients(
                inst_df_train_outer, train_inner_ids, id_col=patient_id_col
            )
            inst_df_val_inner = filter_by_patients(
                inst_df_train_outer, val_inner_ids, id_col=patient_id_col
            )

            # Filter dfs for inner-train and inner-val folds, then scale features using only inner-train statistics
            dfs_train_inner_raw = {
                name: filter_by_patients(df, train_inner_ids, id_col=patient_id_col)
                for name, df in dfs_train_outer_raw.items()
            }
            dfs_val_inner_raw = {
                name: filter_by_patients(df, val_inner_ids, id_col=patient_id_col)
                for name, df in dfs_train_outer_raw.items()
            }
            dfs_train_inner_prepared, modality_poolers_inner = _prepare_patient_level_modalities(
                dfs_train_inner_raw,
                patient_id_col=patient_id_col,
                modality_pooling=modality_pooling,
                fit_attention_poolers=True,
                fitted_poolers=None,
                radio_pooling_kwargs=radio_pooling_kwargs,
                labels_df=inst_df_train_inner,
                label_col=label_col,
            )
            dfs_val_inner_prepared, _ = _prepare_patient_level_modalities(
                dfs_val_inner_raw,
                patient_id_col=patient_id_col,
                modality_pooling=modality_pooling,
                fit_attention_poolers=False,
                fitted_poolers=modality_poolers_inner,
                radio_pooling_kwargs=radio_pooling_kwargs,
                labels_df=None,
                label_col=None,
            )
            dfs_train_inner_scaled, dfs_val_inner_scaled, scalers_inner = fit_and_transform_modalities(
                dfs_train_inner_prepared,
                dfs_val_inner_prepared,
                id_col=patient_id_col,
            )

            train_split_dataset = MultimodalBaseDataset(
                dfs=dfs_train_inner_scaled,
                label_df=inst_df_train_inner,
                label_col=label_col,
                id_col=patient_id_col,
            )
            prefit_inner_imputer = _fit_split_imputer(
                split_dataset=train_split_dataset,
                split_missing_simulator=train_missing_simulator,
                apply_split_missing=apply_missing_train,
                imputation_method=imputation_method,
                missing_pattern_seed=missing_pattern_seed,
                imputer_seed=int(seed + outer_fold_idx * 10_000 + inner_fold_idx * 100),
                imputer_kwargs=imputer_kwargs,
            )

            # Iterate over each HP config, train a model, and evaluate on inner-val fold
            for hp_idx, hp_cfg in enumerate(hp_configs):
                hp_name = hp_cfg["name"]

                # Inner validation belongs to outer-train, so it follows train missingness.
                train_loader, val_loader, _ = build_loaders(
                    inst_df_train=inst_df_train_inner,
                    inst_df_eval=inst_df_val_inner,
                    dfs_train_scaled=dfs_train_inner_scaled,
                    dfs_eval_scaled=dfs_val_inner_scaled,
                    label_col=label_col,
                    missing_simulator=train_missing_simulator,
                    batch_size=hp_cfg["batch_size"],
                    train_missing=apply_missing_train,
                    val_missing=apply_missing_train,
                    imputation_method=imputation_method,
                    missing_pattern_seed=missing_pattern_seed,
                    model_name=model_name,
                    loader_seed=int(seed + outer_fold_idx * 10_000 + inner_fold_idx * 100 + hp_idx),
                    id_col=patient_id_col,
                    prefit_imputer=prefit_inner_imputer,
                    imputer_kwargs=imputer_kwargs,
                )

                model_kwargs = _build_model_kwargs_from_hp_cfg(model_name_l, hp_cfg)

                # Train the model and evaluate on inner-val fold
                model, history, best_metrics = train_model_with_validation(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    input_dims=input_dims,
                    epochs=epochs,
                    lr=hp_cfg["learning_rate"],
                    weight_decay=hp_cfg["weight_decay"],
                    early_stopping_patience=early_stopping_patience,
                    min_lr=min_lr,
                    model_name=model_name,
                    imputation_method=imputation_method,
                    model_kwargs=model_kwargs,
                    train_seed=int(seed + outer_fold_idx * 10_000 + inner_fold_idx * 100 + hp_idx),
                )

                # Save inner evaluation METRICS for this HP config and inner fold
                inner_eval_rows.append(
                    {
                        "outer_fold": outer_fold_idx,
                        "inner_fold": inner_fold_idx,
                        "hp_name": hp_name,
                        **hp_cfg,
                        "val_best_epoch": int(best_metrics["best_epoch"]),
                        "val_best_LOGLOSS": float(best_metrics["LOGLOSS"]),
                        "val_best_AUC": float(best_metrics["AUC"]),
                        "val_best_AUCPR": float(best_metrics["AUCPR"]),
                        "val_best_ACC": float(best_metrics["ACC"]),
                        "val_best_SEN": float(best_metrics["SEN"]),
                        "val_best_SP": float(best_metrics["SP"]),
                        "val_best_MCC": float(best_metrics["MCC"]),
                    }
                )

                # Save inner evaluation HISTORY (per epoch) for this HP config and inner fold
                for hrow in history:
                    history_rows.append(
                        {
                            "outer_fold": outer_fold_idx,
                            "inner_fold": inner_fold_idx,
                            "hp_name": hp_name,
                            **hp_cfg,
                            **hrow,
                        }
                    )

                bundle_path = None
                if (((not bool(retrain_outer)) or bool(save_inner)) and candidate_model_dir):
                    bundle_path = _candidate_bundle_path(
                        candidate_model_dir=candidate_model_dir,
                        outer_fold_idx=outer_fold_idx,
                        inner_fold_idx=inner_fold_idx,
                        hp_name=hp_name,
                    )
                    _save_candidate_bundle(
                        bundle_path=bundle_path,
                        model=model,
                        model_name=model_name,
                        input_dims=input_dims,
                        model_kwargs=model_kwargs,
                        scalers=scalers_inner,
                        imputer=prefit_inner_imputer,
                        modality_poolers=modality_poolers_inner,
                    )

                all_inner_candidates_by_hp.setdefault(hp_name, []).append(
                    {
                        "inner_fold": inner_fold_idx,
                        "hp_name": hp_name,
                        "hp_cfg": hp_cfg,
                        "metrics": best_metrics,
                        "history": history,
                        "model_kwargs": model_kwargs,
                        "bundle_path": bundle_path,
                    }
                )

                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if not all_inner_candidates_by_hp:
            raise RuntimeError(f"No inner-fold candidates were trained for outer fold {outer_fold_idx}.")

        hp_selection_rows = []
        expected_inner_folds = int(inner_splits)
        for hp_cfg in hp_configs:
            hp_name = hp_cfg["name"]
            candidates = sorted(
                all_inner_candidates_by_hp.get(hp_name, []),
                key=lambda item: int(item["inner_fold"]),
            )
            if len(candidates) != expected_inner_folds:
                raise RuntimeError(
                    f"HP '{hp_name}' is missing trained inner-fold models for outer fold {outer_fold_idx}. "
                    f"Expected {expected_inner_folds}, found {len(candidates)}."
                )

            aucs = np.asarray([float(c["metrics"]["AUC"]) for c in candidates], dtype=np.float32)
            loglosses = np.asarray([float(c["metrics"]["LOGLOSS"]) for c in candidates], dtype=np.float32)
            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs))
            mean_logloss = float(np.mean(loglosses))
            hp_selection_rows.append(
                {
                    "hp_name": hp_name,
                    "hp_cfg": hp_cfg,
                    "candidates": candidates,
                    "mean_auc": mean_auc,
                    "std_auc": std_auc,
                    "mean_logloss": mean_logloss,
                }
            )

        epsilon = max(0.0, float(hp_selection_epsilon))
        best_mean_auc = max(float(row["mean_auc"]) for row in hp_selection_rows)
        tied_hp_rows = [
            row
            for row in hp_selection_rows
            if float(row["mean_auc"]) >= (best_mean_auc - epsilon)
        ]
        best_hp_row = min(
            tied_hp_rows,
            key=lambda row: (
                float(row["std_auc"]),
                float(row["mean_logloss"]),
                str(row["hp_name"]),
            ),
        )
        selected_candidates = best_hp_row["candidates"]

        print(
            f"  Selected hp across inner folds: {best_hp_row['hp_name']} "
            f"(mean_AUC={best_hp_row['mean_auc']:.4f}, std_AUC={best_hp_row['std_auc']:.4f}, "
            f"mean_LOGLOSS={best_hp_row['mean_logloss']:.4f}, epsilon={epsilon:.4f}, "
            f"best_mean_AUC={best_mean_auc:.4f}, tied_configs={len(tied_hp_rows)})"
        )

        for candidate in selected_candidates:
            candidate_metrics = candidate["metrics"]
            print(
                f"    Inner fold {candidate['inner_fold']} retained model: "
                f"AUC={float(candidate_metrics['AUC']):.4f}, "
                f"LOGLOSS={float(candidate_metrics['LOGLOSS']):.4f}, "
                f"best_epoch={int(candidate_metrics['best_epoch'])}"
            )
            selected_inner_rows.append(
                {
                    "inner_fold": int(candidate["inner_fold"]),
                    "hp_name": candidate["hp_name"],
                    **candidate["hp_cfg"],
                    "val_best_AUC": float(candidate_metrics["AUC"]),
                    "val_best_LOGLOSS": float(candidate_metrics["LOGLOSS"]),
                    "selected_hp_mean_AUC": float(best_hp_row["mean_auc"]),
                    "selected_hp_std_AUC": float(best_hp_row["std_auc"]),
                    "hp_selection_epsilon": float(epsilon),
                    "hp_selection_best_mean_AUC": float(best_mean_auc),
                }
            )
            selected_inner_histories.append(candidate["history"])

        selected_hp_cfg = dict(best_hp_row["hp_cfg"])
        selected_model_kwargs = dict(selected_candidates[0]["model_kwargs"])
        inst_df_test_outer = filter_by_patients(inst_df, test_outer_ids, id_col=patient_id_col)
        dfs_test_outer_raw = {
            name: filter_by_patients(df, test_outer_ids, id_col=patient_id_col)
            for name, df in dfs.items()
        }
        outer_eval_batch_size = 1 if model_name_l == "healnet" else int(selected_hp_cfg["batch_size"])
        refit_epochs = np.nan

        if bool(retrain_outer):
            refit_epochs = max(
                1,
                int(round(np.median([int(candidate["metrics"]["best_epoch"]) for candidate in selected_candidates]))),
            )
            print(
                f"  Refit outer-train model with selected hp: "
                f"lr={float(selected_hp_cfg['learning_rate']):g}, "
                f"batch_size={int(selected_hp_cfg['batch_size'])}, "
                f"epochs={refit_epochs}"
            )

            dfs_train_outer_prepared, modality_poolers_outer = _prepare_patient_level_modalities(
                dfs_train_outer_raw,
                patient_id_col=patient_id_col,
                modality_pooling=modality_pooling,
                fit_attention_poolers=True,
                fitted_poolers=None,
                radio_pooling_kwargs=radio_pooling_kwargs,
                labels_df=inst_df_train_outer,
                label_col=label_col,
            )
            dfs_train_outer_scaled, _, scalers_outer = fit_and_transform_modalities(
                dfs_train_outer_prepared,
                dfs_train_outer_prepared,
                id_col=patient_id_col,
            )

            outer_train_split_dataset = MultimodalBaseDataset(
                dfs=dfs_train_outer_scaled,
                label_df=inst_df_train_outer,
                label_col=label_col,
                id_col=patient_id_col,
            )
            prefit_outer_imputer = _fit_split_imputer(
                split_dataset=outer_train_split_dataset,
                split_missing_simulator=train_missing_simulator,
                apply_split_missing=apply_missing_train,
                imputation_method=imputation_method,
                missing_pattern_seed=missing_pattern_seed,
                imputer_seed=int(seed + outer_fold_idx * 100_000),
                imputer_kwargs=imputer_kwargs,
            )

            outer_train_loader, _, _ = build_loaders(
                inst_df_train=inst_df_train_outer,
                inst_df_eval=inst_df_train_outer,
                dfs_train_scaled=dfs_train_outer_scaled,
                dfs_eval_scaled=dfs_train_outer_scaled,
                label_col=label_col,
                missing_simulator=train_missing_simulator,
                batch_size=selected_hp_cfg["batch_size"],
                train_missing=apply_missing_train,
                val_missing=False,
                imputation_method=imputation_method,
                missing_pattern_seed=missing_pattern_seed,
                model_name=model_name,
                loader_seed=int(seed + outer_fold_idx * 100_000 + 1),
                id_col=patient_id_col,
                prefit_imputer=prefit_outer_imputer,
                imputer_kwargs=imputer_kwargs,
            )

            outer_train_model, outer_refit_history = train_model_on_full_dataset(
                train_loader=outer_train_loader,
                device=device,
                input_dims=input_dims,
                epochs=refit_epochs,
                lr=selected_hp_cfg["learning_rate"],
                weight_decay=selected_hp_cfg["weight_decay"],
                min_lr=min_lr,
                model_name=model_name,
                imputation_method=imputation_method,
                model_kwargs=selected_model_kwargs,
                train_seed=int(seed + outer_fold_idx * 100_000 + 2),
            )

            for hrow in outer_refit_history:
                history_rows.append(
                    {
                        "outer_fold": outer_fold_idx,
                        "inner_fold": 0,
                        "hp_name": best_hp_row["hp_name"],
                        "phase": "outer_refit",
                        **selected_hp_cfg,
                        **hrow,
                    }
                )

            if wandb_active:
                missing_location = str((wandb_base_config or {}).get("missing_location", "na")).strip().lower()
                train_missing_prop = float((wandb_base_config or {}).get("train_missing_prop", 0.0))
                run_name = (
                    f"loc{missing_location}_"
                    f"trainprop{train_missing_prop:g}_"
                    f"seed{seed}_"
                    f"outer{outer_fold_idx}_"
                    f"outertrain"
                )
                run_config = dict(wandb_base_config or {})
                run_config.update(
                    {
                        "seed": seed,
                        "outer_fold": outer_fold_idx,
                        "model_type": "outer",
                        "phase": "outer_train_refit",
                        "selected_hp_name": best_hp_row["hp_name"],
                        "outer_refit_epochs": int(refit_epochs),
                    }
                )
                if model_name_l in {"healnet", "smil_e"}:
                    run_config.update(dict(selected_hp_cfg))
                outer_train_run = wandb.init(
                    project=wandb_project,
                    group=f"outer_fold_{outer_fold_idx}",
                    name=run_name,
                    mode=wandb_mode,
                    config=run_config,
                    reinit="finish_previous",
                    settings=wandb.Settings(init_timeout=WANDB_INIT_TIMEOUT_SEC),
                )

                for hrow in outer_refit_history:
                    epoch_i = int(hrow["epoch"])
                    log_payload = {
                        "outer_train_model/train_loss": float(hrow["train_loss"]),
                        "outer_train_model/train_auc": float(hrow["train_auc"]),
                        "outer_train_model/train_aucpr": float(hrow["train_aucpr"]),
                        "outer_train_model/train_acc": float(hrow["train_acc"]),
                    }

                    if model_name_l in {"dipam", "di_mmlp"}:
                        log_payload.update(
                            {
                                "outer_train_model/teacher_loss": float(hrow["teacher_loss"]),
                                "outer_train_model/student_survival_loss": float(hrow["student_survival_loss"]),
                                "outer_train_model/student_repr_loss": float(hrow["student_repr_loss"]),
                                "outer_train_model/student_feature_loss": float(hrow["student_feature_loss"]),
                            }
                        )
                    elif model_name_l == "smil_e":
                        log_payload.update(
                            {
                                "outer_train_model/smil_meta_train_loss": float(hrow["smil_meta_train_loss"]),
                                "outer_train_model/smil_meta_val_loss": float(hrow["smil_meta_val_loss"]),
                                "outer_train_model/smil_meta_val_ce": float(hrow["smil_meta_val_ce"]),
                                "outer_train_model/smil_align_fusion": float(hrow["smil_align_fusion"]),
                                "outer_train_model/smil_align_hidden": float(hrow["smil_align_hidden"]),
                            }
                        )

                    outer_train_run.log(log_payload, step=epoch_i)

                outer_train_run.finish()

            dfs_outer_eval_prepared, _ = _prepare_patient_level_modalities(
                dfs_test_outer_raw,
                patient_id_col=patient_id_col,
                modality_pooling=modality_pooling,
                fit_attention_poolers=False,
                fitted_poolers=modality_poolers_outer,
                radio_pooling_kwargs=radio_pooling_kwargs,
                labels_df=None,
                label_col=None,
            )
            dfs_outer_eval_scaled = _transform_modalities_with_fitted_scalers(
                dfs_outer_eval_prepared,
                scalers_outer,
                patient_id_col=patient_id_col,
            )
            outer_eval_base = MultimodalBaseDataset(
                dfs=dfs_outer_eval_scaled,
                label_df=inst_df_test_outer,
                label_col=label_col,
                id_col=patient_id_col,
            )

            for eval_setup in test_eval_setups:
                eval_simulator = eval_setup["simulator"]
                eval_missing_location = str(eval_setup["missing_location"]).lower()
                eval_missing_prop = float(eval_setup["missing_prop"])
                apply_missing_eval = eval_missing_prop > 0.0

                outer_eval_ds = MultimodalDatasetWithMissing(
                    base_dataset=outer_eval_base,
                    simulator=eval_simulator,
                    apply_missing=apply_missing_eval,
                    imputation_method=imputation_method,
                    missing_pattern_seed=missing_pattern_seed,
                    prefit_imputer=prefit_outer_imputer,
                    imputer_kwargs=imputer_kwargs,
                )
                outer_eval_loader = DataLoader(
                    outer_eval_ds,
                    batch_size=outer_eval_batch_size,
                    shuffle=False,
                    collate_fn=multimodal_collate,
                    drop_last=False,
                )

                y_true_outer, y_logits_outer, y_prob_outer, pids_outer, pam_details_outer = _predict_model_probabilities(
                    model=outer_train_model,
                    data_loader=outer_eval_loader,
                    device=device,
                    bypass_mask=predict_bypass_mask,
                    collect_pam_details=model_name_l in {"pam", "dipam"},
                    model_name=model_name_l,
                )
                outer_metrics = safe_binary_metrics(y_true_outer, y_prob_outer)

                per_patient_prediction_rows = []
                for patient_idx, pid in enumerate(pids_outer):
                    row = {
                        "outer_fold": outer_fold_idx,
                        "outer_eval_target": "test_outer",
                        "patient": pid,
                        "train_missing_location": str(getattr(train_missing_simulator, "missing_location", "global")).lower(),
                        "train_missing_prop": float(getattr(train_missing_simulator, "missing_prop", 0.0)),
                        "test_missing_location": eval_missing_location,
                        "test_missing_prop": eval_missing_prop,
                        "y_true": int(y_true_outer[patient_idx]),
                    }
                    logit_value = float(y_logits_outer[patient_idx])
                    prob_value = float(y_prob_outer[patient_idx])
                    pred_label = int(logit_value >= 0.0)
                    row["inner_model_1_logit"] = logit_value
                    row["inner_model_1_prob"] = prob_value
                    row["inner_model_1_pred_label"] = pred_label
                    if model_name_l in {"pam", "dipam"} and pam_details_outer is not None:
                        for modality_idx, modality_name in enumerate(modality_names):
                            row[f"inner_model_1_{modality_name}_alpha"] = float(
                                pam_details_outer["alpha"][patient_idx, modality_idx]
                            )
                            row[f"inner_model_1_{modality_name}_R"] = float(
                                pam_details_outer["R"][patient_idx, modality_idx]
                            )
                    per_patient_prediction_rows.append(row)

                test_prediction_rows.extend(per_patient_prediction_rows)

                outer_results.append(
                    {
                        "outer_fold": outer_fold_idx,
                        "outer_eval_target": "test_outer",
                        "eval_missing_location": eval_missing_location,
                        "eval_missing_prop": eval_missing_prop,
                        "inner_models_count": 1,
                        "selected_inner_hp_names": str(best_hp_row["hp_name"]),
                        "selected_inner_mean_AUC": float(np.mean([r["val_best_AUC"] for r in selected_inner_rows])),
                        "selected_inner_mean_LOGLOSS": float(np.mean([r["val_best_LOGLOSS"] for r in selected_inner_rows])),
                        "selected_inner_std_AUC": float(best_hp_row["std_auc"]),
                        "hp_selection_epsilon": float(epsilon),
                        "hp_selection_best_mean_AUC": float(best_mean_auc),
                        "outer_test_metric_source": "outer_refit_model",
                        "outer_refit_epochs": int(refit_epochs),
                        "outer_test_LOGLOSS": float(outer_metrics["LOGLOSS"]),
                        "outer_test_AUC": float(outer_metrics["AUC"]),
                        "outer_test_AUCPR": float(outer_metrics["AUCPR"]),
                        "outer_test_ACC": float(outer_metrics["ACC"]),
                        "outer_test_SEN": float(outer_metrics["SEN"]),
                        "outer_test_SP": float(outer_metrics["SP"]),
                        "outer_test_MCC": float(outer_metrics["MCC"]),
                    }
                )

            del outer_train_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if bool(save_inner):
                if wandb_active:
                    _log_selected_inner_models_to_wandb(
                        selected_candidates=selected_candidates,
                        seed=seed,
                        outer_fold_idx=outer_fold_idx,
                        train_missing_simulator=train_missing_simulator,
                        wandb_project=wandb_project,
                        wandb_mode=wandb_mode,
                        wandb_base_config=wandb_base_config,
                        model_name_l=model_name_l,
                    )
                inner_outer_results, inner_test_prediction_rows = _evaluate_retained_inner_models_on_outer_test(
                    selected_candidates=selected_candidates,
                    dfs_test_outer_raw=dfs_test_outer_raw,
                    inst_df_test_outer=inst_df_test_outer,
                    label_col=label_col,
                    patient_id_col=patient_id_col,
                    modality_pooling=modality_pooling,
                    radio_pooling_kwargs=radio_pooling_kwargs,
                    test_eval_setups=test_eval_setups,
                    modality_names=modality_names,
                    outer_fold_idx=outer_fold_idx,
                    train_missing_simulator=train_missing_simulator,
                    predict_bypass_mask=predict_bypass_mask,
                    model_name_l=model_name_l,
                    outer_eval_batch_size=outer_eval_batch_size,
                    imputation_method=imputation_method,
                    missing_pattern_seed=missing_pattern_seed,
                    device=device,
                    selected_inner_rows=selected_inner_rows,
                    best_hp_row=best_hp_row,
                    epsilon=epsilon,
                    best_mean_auc=best_mean_auc,
                )
                saved_inner_outer_results.extend(inner_outer_results)
                saved_inner_test_prediction_rows.extend(inner_test_prediction_rows)

            if candidate_model_dir and bool(save_inner):
                outer_fold_cache_dir = os.path.join(candidate_model_dir, f"outer_fold_{int(outer_fold_idx)}")
                shutil.rmtree(outer_fold_cache_dir, ignore_errors=True)
        else:
            print(
                f"  Outer test predictions will be saved for the retained inner-fold models "
                f"for hp='{best_hp_row['hp_name']}'."
            )

            if wandb_active:
                _log_selected_inner_models_to_wandb(
                    selected_candidates=selected_candidates,
                    seed=seed,
                    outer_fold_idx=outer_fold_idx,
                    train_missing_simulator=train_missing_simulator,
                    wandb_project=wandb_project,
                    wandb_mode=wandb_mode,
                    wandb_base_config=wandb_base_config,
                    model_name_l=model_name_l,
                )

            inner_outer_results, inner_test_prediction_rows = _evaluate_retained_inner_models_on_outer_test(
                selected_candidates=selected_candidates,
                dfs_test_outer_raw=dfs_test_outer_raw,
                inst_df_test_outer=inst_df_test_outer,
                label_col=label_col,
                patient_id_col=patient_id_col,
                modality_pooling=modality_pooling,
                radio_pooling_kwargs=radio_pooling_kwargs,
                test_eval_setups=test_eval_setups,
                modality_names=modality_names,
                outer_fold_idx=outer_fold_idx,
                train_missing_simulator=train_missing_simulator,
                predict_bypass_mask=predict_bypass_mask,
                model_name_l=model_name_l,
                outer_eval_batch_size=outer_eval_batch_size,
                imputation_method=imputation_method,
                missing_pattern_seed=missing_pattern_seed,
                device=device,
                selected_inner_rows=selected_inner_rows,
                best_hp_row=best_hp_row,
                epsilon=epsilon,
                best_mean_auc=best_mean_auc,
            )
            outer_results.extend(inner_outer_results)
            test_prediction_rows.extend(inner_test_prediction_rows)

            if candidate_model_dir:
                outer_fold_cache_dir = os.path.join(candidate_model_dir, f"outer_fold_{int(outer_fold_idx)}")
                shutil.rmtree(outer_fold_cache_dir, ignore_errors=True)

    auxiliary_outputs = None
    if bool(retrain_outer) and bool(save_inner):
        auxiliary_outputs = {
            "outer_df": pd.DataFrame(saved_inner_outer_results),
            "test_predictions_df": pd.DataFrame(saved_inner_test_prediction_rows),
        }

    return (
        pd.DataFrame(inner_eval_rows),
        pd.DataFrame(outer_results),
        pd.DataFrame(history_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(test_prediction_rows),
        auxiliary_outputs,
    )
