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
from dataset.imputation_methods import build_imputer
from dataset import (
    MissingModalitySimulator,
    MultimodalBaseDataset,
    MultimodalDatasetWithMissing,
    multimodal_collate,
    build_loaders,
)
from scripts.utils import (
    build_model,
    normalize_task_type,
    primary_loss_name,
    primary_metric_name,
    safe_binary_metrics,
    safe_task_metrics,
    select_device,
    concordance_index_censored,
    fit_and_transform_modalities,
    set_global_seed,
    normalize_model_name,
    survival_logits_to_outputs,
)
from dataset.preprocess_dataset import collapse_patient_rows, filter_by_patients
from dataset.attention_pooling import AttentionPooler
from scripts.model_training import (
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

def _add_binary_ensemble_prediction_columns(row):
    """Add probability-averaged ensemble prediction columns to a binary row."""
    prob_items = []
    for key, value in row.items():
        key_str = str(key)
        if not (key_str.startswith("inner_model_") and key_str.endswith("_prob")):
            continue
        try:
            model_idx = int(key_str.split("_")[2])
            prob_value = float(value)
        except (ValueError, TypeError):
            continue
        if np.isfinite(prob_value):
            prob_items.append((model_idx, prob_value))

    if not prob_items:
        return row

    prob_items.sort(key=lambda item: item[0])
    probs = np.asarray([value for _, value in prob_items], dtype=float)
    ensemble_prob = float(np.mean(probs))
    clipped_prob = float(np.clip(ensemble_prob, 1e-12, 1.0 - 1e-12))
    row["ensemble_prob"] = ensemble_prob
    row["ensemble_logit"] = float(np.log(clipped_prob / (1.0 - clipped_prob)))
    row["ensemble_pred_label"] = int(ensemble_prob >= 0.5)
    row["ensemble_n_models"] = int(probs.size)
    return row


class _ConstantBinaryClassifier:
    """Fallback binary classifier used when a fold contains only one class."""

    def __init__(self, positive_probability):
        self.positive_probability = float(np.clip(positive_probability, 1e-7, 1.0 - 1e-7))

    def predict_proba(self, X):
        n_rows = int(np.asarray(X).shape[0])
        pos = np.full(n_rows, self.positive_probability, dtype=np.float64)
        neg = 1.0 - pos
        return np.column_stack([neg, pos])


def _is_lr_model(model_name_l):
    return normalize_model_name(model_name_l) == "lr"


def _is_sklearn_classification_model(model_name_l):
    return normalize_model_name(model_name_l) in {"lr", "rf"}


def _is_sklearn_survival_model(model_name_l):
    return normalize_model_name(model_name_l) in {"coxnet", "rsf"}


def _is_sklearn_baseline_model(model_name_l):
    return normalize_model_name(model_name_l) in {"lr", "rf", "coxnet", "rsf"}


def _normalize_lr_class_weight(raw_value):
    value = str(raw_value).strip().lower()
    if value in {"none", "null", "", "false"}:
        return None
    if value == "balanced":
        return "balanced"
    raise ValueError("lr_class_weight must be one of: none, balanced")


def _build_lr_kwargs_from_hp_cfg(hp_cfg):
    return {
        "C": float(hp_cfg.get("lr_C", 1.0)),
        "penalty": str(hp_cfg.get("lr_penalty", "l2")).strip().lower(),
        "solver": str(hp_cfg.get("lr_solver", "lbfgs")).strip().lower(),
        "class_weight": _normalize_lr_class_weight(hp_cfg.get("lr_class_weight", "none")),
        "max_iter": int(hp_cfg.get("lr_max_iter", 1000)),
    }


def _normalize_rf_class_weight(raw_value):
    value = str(raw_value).strip().lower()
    if value in {"none", "null", "", "false"}:
        return None
    if value == "balanced":
        return "balanced"
    raise ValueError("rf_class_weight must be one of: none, balanced")


def _parse_optional_int(raw_value):
    value = str(raw_value).strip().lower()
    if value in {"none", "null", "", "false"}:
        return None
    return int(float(value))


def _parse_optional_float_or_str(raw_value):
    value = str(raw_value).strip().lower()
    if value in {"none", "null", "", "false"}:
        return None
    if value in {"sqrt", "log2", "auto"}:
        return value
    return float(value)


def _build_coxnet_kwargs_from_hp_cfg(hp_cfg):
    return {
        "alphas": np.asarray([float(hp_cfg.get("coxnet_alpha", 0.1))], dtype=float),
        "l1_ratio": float(hp_cfg.get("coxnet_l1_ratio", 0.5)),
        "max_iter": int(hp_cfg.get("coxnet_max_iter", 100000)),
        "tol": float(hp_cfg.get("coxnet_tol", 1e-7)),
    }


def _build_rsf_kwargs_from_hp_cfg(hp_cfg, seed):
    return {
        "n_estimators": int(hp_cfg.get("rsf_n_estimators", 100)),
        "max_depth": _parse_optional_int(hp_cfg.get("rsf_max_depth", "none")),
        "min_samples_split": int(hp_cfg.get("rsf_min_samples_split", 6)),
        "min_samples_leaf": int(hp_cfg.get("rsf_min_samples_leaf", 3)),
        "max_features": _parse_optional_float_or_str(hp_cfg.get("rsf_max_features", "sqrt")),
        "n_jobs": int(hp_cfg.get("rsf_n_jobs", -1)),
        "random_state": int(seed),
    }


def _build_rf_kwargs_from_hp_cfg(hp_cfg, seed):
    return {
        "n_estimators": int(hp_cfg.get("rf_n_estimators", 200)),
        "max_depth": _parse_optional_int(hp_cfg.get("rf_max_depth", "none")),
        "min_samples_split": int(hp_cfg.get("rf_min_samples_split", 2)),
        "min_samples_leaf": int(hp_cfg.get("rf_min_samples_leaf", 1)),
        "max_features": _parse_optional_float_or_str(hp_cfg.get("rf_max_features", "sqrt")),
        "class_weight": _normalize_rf_class_weight(hp_cfg.get("rf_class_weight", "none")),
        "n_jobs": int(hp_cfg.get("rf_n_jobs", -1)),
        "random_state": int(seed),
    }


def _dataset_to_binary_matrix(dataset):
    if len(dataset) == 0:
        raise ValueError("Cannot build sklearn classification design matrix from an empty dataset.")

    x_rows = []
    y_values = []
    patient_ids = []
    for idx in range(len(dataset)):
        Xs, _, y, pid = dataset[idx]
        if isinstance(y, dict):
            raise ValueError("LR and RF baselines only support binary classification.")
        x_rows.append(
            np.concatenate(
                [x.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1) for x in Xs],
                axis=0,
            )
        )
        y_values.append(int(float(y.detach().cpu().item())))
        patient_ids.append(pid)

    return (
        np.vstack(x_rows).astype(np.float32, copy=False),
        np.asarray(y_values, dtype=np.int64),
        list(patient_ids),
    )


def _tensor_scalar(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().item()
    return value


def _dataset_to_survival_matrix(dataset):
    if len(dataset) == 0:
        raise ValueError("Cannot build survival design matrix from an empty dataset.")

    x_rows = []
    event_times = []
    event_observed = []
    censorship = []
    y_disc = []
    patient_ids = []
    for idx in range(len(dataset)):
        Xs, _, y, pid = dataset[idx]
        if not isinstance(y, dict):
            raise ValueError("CoxNet and RSF baselines only support task_type=survival.")
        x_rows.append(
            np.concatenate(
                [x.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1) for x in Xs],
                axis=0,
            )
        )
        event_times.append(float(_tensor_scalar(y["event_time"])))
        event_observed.append(bool(float(_tensor_scalar(y["event"])) > 0.5))
        censorship.append(int(float(_tensor_scalar(y["censorship"]))))
        y_disc.append(int(_tensor_scalar(y["y_disc"])))
        patient_ids.append(pid)

    y_struct = np.asarray(
        list(zip(event_observed, event_times)),
        dtype=[("event", bool), ("time", np.float64)],
    )
    return (
        np.vstack(x_rows).astype(np.float32, copy=False),
        y_struct,
        np.asarray(event_times, dtype=np.float64),
        np.asarray(event_observed, dtype=bool),
        np.asarray(censorship, dtype=np.int64),
        np.asarray(y_disc, dtype=np.int64),
        list(patient_ids),
    )


def _fit_lr_classifier(dataset, hp_cfg, seed):
    from sklearn.linear_model import LogisticRegression

    X, y, _ = _dataset_to_binary_matrix(dataset)
    unique_classes = np.unique(y)
    if unique_classes.size < 2:
        return _ConstantBinaryClassifier(float(np.mean(y)))

    lr_kwargs = _build_lr_kwargs_from_hp_cfg(hp_cfg)
    model = LogisticRegression(
        C=lr_kwargs["C"],
        penalty=lr_kwargs["penalty"],
        solver=lr_kwargs["solver"],
        class_weight=lr_kwargs["class_weight"],
        max_iter=lr_kwargs["max_iter"],
        random_state=int(seed),
    )
    model.fit(X, y)
    return model


def _fit_rf_classifier(dataset, hp_cfg, seed):
    from sklearn.ensemble import RandomForestClassifier

    X, y, _ = _dataset_to_binary_matrix(dataset)
    unique_classes = np.unique(y)
    if unique_classes.size < 2:
        return _ConstantBinaryClassifier(float(np.mean(y)))

    model = RandomForestClassifier(**_build_rf_kwargs_from_hp_cfg(hp_cfg, seed=seed))
    model.fit(X, y)
    return model


def _fit_sklearn_classification_model(dataset, model_name_l, hp_cfg, seed):
    model_name_l = normalize_model_name(model_name_l)
    if model_name_l == "lr":
        return _fit_lr_classifier(dataset=dataset, hp_cfg=hp_cfg, seed=seed)
    if model_name_l == "rf":
        return _fit_rf_classifier(dataset=dataset, hp_cfg=hp_cfg, seed=seed)
    raise ValueError(f"Unsupported sklearn classification baseline: {model_name_l}")


class _ConstantRiskSurvivalModel:
    """Fallback survival model used when a split has no observed events."""

    def predict(self, X):
        return np.zeros(int(np.asarray(X).shape[0]), dtype=np.float64)


def _fit_survival_baseline_model(dataset, model_name_l, hp_cfg, seed):
    X, y_struct, _, event_observed, _, _, _ = _dataset_to_survival_matrix(dataset)
    if np.unique(event_observed).size < 2 or not np.any(event_observed):
        return _ConstantRiskSurvivalModel()

    model_name_l = normalize_model_name(model_name_l)
    try:
        if model_name_l == "coxnet":
            from sksurv.linear_model import CoxnetSurvivalAnalysis

            model = CoxnetSurvivalAnalysis(**_build_coxnet_kwargs_from_hp_cfg(hp_cfg))
        elif model_name_l == "rsf":
            from sksurv.ensemble import RandomSurvivalForest

            model = RandomSurvivalForest(**_build_rsf_kwargs_from_hp_cfg(hp_cfg, seed=seed))
        else:
            raise ValueError(f"Unsupported survival sklearn baseline: {model_name_l}")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ZI_CoxNet, KNN_CoxNet, ZI_RSF and KNN_RSF require scikit-survival. "
            "Install it with conda-forge package 'scikit-survival'."
        ) from exc

    try:
        model.fit(X, y_struct)
    except Exception:
        if model_name_l == "coxnet":
            from sksurv.linear_model import CoxPHSurvivalAnalysis

            # Rare small-fold fallback when the elastic-net path is numerically unstable.
            model = CoxPHSurvivalAnalysis(alpha=float(hp_cfg.get("coxnet_alpha", 0.1)))
            model.fit(X, y_struct)
        else:
            raise
    return model


def _predict_lr_outputs(model, dataset):
    X, y_true, pids = _dataset_to_binary_matrix(dataset)
    probs = model.predict_proba(X)[:, 1].astype(np.float32, copy=False)
    clipped = np.clip(probs.astype(np.float64), 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped)).astype(np.float32, copy=False)
    return {
        "y_true": y_true,
        "probs": probs,
        "logits": logits,
        "pids": pids,
        "pam_details": None,
    }


def _predict_survival_baseline_outputs(model, dataset):
    X, _, event_times, event_observed, censorship, y_disc, pids = _dataset_to_survival_matrix(dataset)
    risk = np.asarray(model.predict(X), dtype=np.float64)
    if risk.ndim > 1:
        risk = risk[:, -1]
    risk = risk.reshape(-1).astype(np.float32, copy=False)
    n = risk.shape[0]
    return {
        "event_times": event_times,
        "event_observed": event_observed.astype(np.int64, copy=False),
        "censorship": censorship,
        "y_disc": y_disc,
        "risk": risk,
        "logits": np.zeros((n, 0), dtype=np.float32),
        "hazards": np.zeros((n, 0), dtype=np.float32),
        "survival": np.zeros((n, 0), dtype=np.float32),
        "pids": pids,
        "pam_details": None,
    }


def _survival_risk_metrics_from_outputs(pred_out):
    return {
        "CINDEX": float(
            concordance_index_censored(
                event_observed=np.asarray(pred_out["event_observed"], dtype=bool),
                event_times=np.asarray(pred_out["event_times"], dtype=np.float64),
                risk_scores=np.asarray(pred_out["risk"], dtype=np.float64),
            )
        ),
        "LOSS": np.nan,
    }


def _metrics_from_prediction_output(task_config, pred_out, model_name_l):
    if _is_survival(task_config):
        if _is_sklearn_survival_model(model_name_l):
            return _survival_risk_metrics_from_outputs(pred_out)
        return safe_task_metrics(
            task_config,
            event_times=pred_out["event_times"],
            event_observed=pred_out["event_observed"],
            censorship=pred_out["censorship"],
            y_disc=pred_out["y_disc"],
            logits=pred_out["logits"],
        )
    return safe_task_metrics(
        task_config,
        y_true=pred_out["y_true"],
        y_prob=pred_out["probs"],
    )


def _lr_history_row(train_metrics, val_metrics=None):
    val_metrics = train_metrics if val_metrics is None else val_metrics
    return {
        "epoch": 1,
        "train_loss": float(train_metrics.get("LOGLOSS", np.nan)),
        "train_auc": float(train_metrics.get("AUC", 0.0)),
        "train_aucpr": float(train_metrics.get("AUCPR", 0.0)),
        "train_acc": float(train_metrics.get("ACC", 0.0)),
        "train_cindex": 0.0,
        "val_loss": float(val_metrics.get("LOGLOSS", np.nan)),
        "val_auc": float(val_metrics.get("AUC", 0.0)),
        "val_aucpr": float(val_metrics.get("AUCPR", 0.0)),
        "val_acc": float(val_metrics.get("ACC", 0.0)),
        "val_cindex": 0.0,
        "teacher_loss": 0.0,
        "student_survival_loss": 0.0,
        "student_repr_loss": 0.0,
        "student_feature_loss": 0.0,
        "smil_meta_train_loss": 0.0,
        "smil_meta_val_loss": 0.0,
        "smil_meta_val_ce": 0.0,
        "smil_align_fusion": 0.0,
        "smil_align_hidden": 0.0,
    }


def _survival_baseline_history_row(train_metrics, val_metrics=None):
    val_metrics = train_metrics if val_metrics is None else val_metrics
    return {
        "epoch": 1,
        "train_loss": float(train_metrics.get("LOSS", np.nan)),
        "train_auc": 0.0,
        "train_aucpr": 0.0,
        "train_acc": 0.0,
        "train_cindex": float(train_metrics.get("CINDEX", 0.0)),
        "val_loss": float(val_metrics.get("LOSS", np.nan)),
        "val_auc": 0.0,
        "val_aucpr": 0.0,
        "val_acc": 0.0,
        "val_cindex": float(val_metrics.get("CINDEX", 0.0)),
        "teacher_loss": 0.0,
        "student_survival_loss": 0.0,
        "student_repr_loss": 0.0,
        "student_feature_loss": 0.0,
        "smil_meta_train_loss": 0.0,
        "smil_meta_val_loss": 0.0,
        "smil_meta_val_ce": 0.0,
        "smil_align_fusion": 0.0,
        "smil_align_hidden": 0.0,
    }

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
    attention_pooling_kwargs=None,
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
                active_poolers[modality_name] = AttentionPooler(
                    input_dim=len(feature_cols),
                    **dict(attention_pooling_kwargs or {}),
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


def _add_distillation_kwargs(model_kwargs, hp_cfg):
    resolved = dict(model_kwargs)
    if bool(hp_cfg.get("knowledge_distillation", False)):
        resolved["knowledge_distillation"] = True
        resolved["distill_alpha"] = hp_cfg["distill_alpha"]
        resolved["distill_beta"] = hp_cfg["distill_beta"]
    return resolved


def _build_model_kwargs_from_hp_cfg(model_name_l, hp_cfg):
    if model_name_l in {"lr"}:
        return _build_lr_kwargs_from_hp_cfg(hp_cfg)
    if model_name_l in {"rf"}:
        return {
            key: value
            for key, value in hp_cfg.items()
            if str(key).startswith("rf_")
        }
    if model_name_l in {"coxnet"}:
        return _build_coxnet_kwargs_from_hp_cfg(hp_cfg)
    if model_name_l in {"rsf"}:
        return {
            key: value
            for key, value in hp_cfg.items()
            if str(key).startswith("rsf_")
        }
    if model_name_l in {"mlp"}:
        return _add_distillation_kwargs({
            "modality_hidden_layers": hp_cfg["modality_hidden_layers"],
            "fusion_hidden_dim": hp_cfg["fusion_hidden_dim"],
            "fusion_hidden_layers": hp_cfg["fusion_hidden_layers"],
            "dropout_p": hp_cfg["dropout"],
            "fusion_batchnorm": bool(hp_cfg["fusion_batchnorm"]),
        }, hp_cfg)
    if model_name_l in {"pam"}:
        return _add_distillation_kwargs({
            "dropout_p": hp_cfg["pam_dropout"],
            "temperature": hp_cfg["pam_temperature"],
        }, hp_cfg)
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
        return _add_distillation_kwargs({
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
        }, hp_cfg)
    raise ValueError(
        f"Unsupported model '{model_name_l}'. Supported: lr, rf, coxnet, rsf, mlp, pam, smile, healnet"
    )


def _is_survival(task_config):
    return normalize_task_type((task_config or {}).get("task_type", "binary_classification")) == "survival"


def _with_task_output_dim(model_kwargs, task_config):
    resolved = dict(model_kwargs or {})
    if _is_survival(task_config):
        resolved["output_dim"] = int((task_config or {}).get("survival_n_bins", 4))
    return resolved


def _prepare_base_dataset_kwargs(task_config):
    if not _is_survival(task_config):
        return {}
    return {
        "task_type": "survival",
        "survival_time_col": (task_config or {}).get("survival_time_col"),
        "survival_event_col": (task_config or {}).get("survival_event_col"),
        "survival_censorship_col": (task_config or {}).get("survival_censorship_col"),
        "survival_y_disc_col": (task_config or {}).get("survival_y_disc_col"),
    }


def _empty_eval_store(task_config):
    return {
        "y_true": [],
        "probs": [],
        "logits": [],
        "event_times": [],
        "event_observed": [],
        "censorship": [],
        "y_disc": [],
    }


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
    model_name_l = normalize_model_name(model_name)
    bundle = {
        "model_name": model_name_l,
        "input_dims": [int(dim) for dim in input_dims],
        "model_kwargs": get_model_init_kwargs(model_name, model_kwargs),
        "scalers": scalers,
        "imputer": imputer,
        "modality_poolers": dict(modality_poolers or {}),
    }
    if _is_sklearn_baseline_model(model_name_l):
        bundle["model_type"] = "sklearn_baseline"
        bundle["sklearn_model"] = model
    else:
        bundle["model_type"] = "torch"
        bundle["model_state_dict"] = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
    with open(bundle_path, "wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_candidate_bundle(bundle_path, device):
    with open(bundle_path, "rb") as handle:
        bundle = pickle.load(handle)

    if bundle.get("model_type") in {"sklearn_baseline", "sklearn_lr"} or _is_sklearn_baseline_model(bundle["model_name"]):
        model = bundle["sklearn_model"]
    else:
        model = build_model(
            bundle["model_name"],
            bundle["input_dims"],
            get_model_init_kwargs(bundle["model_name"], bundle["model_kwargs"]),
        ).to(device)
        model.load_state_dict(bundle["model_state_dict"])

    modality_poolers = bundle.get("modality_poolers")
    if modality_poolers is None:
        modality_poolers = {}

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


def _observed_modality_missing_prop(dfs, inst_df, patient_id_col):
    """Fraction of patient-modality slots absent in the observed aligned cohort."""
    if not dfs:
        return 0.0

    if patient_id_col in inst_df.columns:
        patient_ids = inst_df[patient_id_col].astype(str).tolist()
    else:
        patient_ids = inst_df.index.astype(str).tolist()

    if not patient_ids:
        return 0.0

    missing_slots = 0
    for df in dfs.values():
        if patient_id_col in df.columns:
            available_ids = set(df[patient_id_col].astype(str).tolist())
        else:
            available_ids = set(df.index.astype(str).tolist())
        missing_slots += sum(pid not in available_ids for pid in patient_ids)

    total_slots = len(patient_ids) * len(dfs)
    return float(missing_slots) / float(total_slots) if total_slots > 0 else 0.0


# Function to predict on outer fold for each inner fold model
def _predict_model_outputs(
    model,
    data_loader,
    device,
    bypass_mask=False,
    collect_pam_details=False,
    model_name=None,
    task_config=None,
):
    """Run one model on a loader and return task-specific outputs plus patient ids."""
    model.eval()
    outputs = _empty_eval_store(task_config)
    pids = []
    pam_alpha = []
    pam_r_scores = []

    with torch.no_grad():
        for Xs, present_mask, y, pid_batch in data_loader:
            Xs = [x.to(device) for x in Xs]
            present_mask = present_mask.to(device)
            if isinstance(y, dict):
                y = {k: v.to(device) for k, v in y.items()}

            model_mask = None if bypass_mask else present_mask
            if collect_pam_details:
                model_out = model(Xs, model_mask, return_aux=True)
                logits = model_out[0]
                if model_name not in {"pam"}:
                    raise ValueError(
                        "collect_pam_details=True is only supported for model_name='pam'."
                    )
                pam_alpha.append(model_out[2].detach().cpu().numpy())
                pam_r_scores.append(model_out[3].detach().cpu().numpy())
            else:
                logits = model(Xs, model_mask)
            if logits.ndim == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            if _is_survival(task_config):
                logits_np = logits.detach().cpu().numpy()
                hazards, survival, risk = survival_logits_to_outputs(logits)
                outputs["logits"].append(logits_np)
                outputs["event_times"].extend(y["event_time"].detach().cpu().numpy().tolist())
                outputs["event_observed"].extend(y["event"].detach().cpu().numpy().tolist())
                outputs["censorship"].extend(y["censorship"].detach().cpu().numpy().tolist())
                outputs["y_disc"].extend(y["y_disc"].detach().cpu().numpy().tolist())
                if "hazards" not in outputs:
                    outputs["hazards"] = []
                    outputs["survival"] = []
                    outputs["risk"] = []
                outputs["hazards"].append(hazards.detach().cpu().numpy())
                outputs["survival"].append(survival.detach().cpu().numpy())
                outputs["risk"].append(risk.detach().cpu().numpy())
            else:
                logits_np = logits.detach().cpu().numpy().reshape(-1)
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                outputs["logits"].append(logits_np)
                outputs["probs"].extend(probs.tolist())
                outputs["y_true"].extend(y.cpu().numpy().tolist())
            pids.extend(pid_batch)

    pam_details = None
    if collect_pam_details:
        pam_details = {
            "alpha": np.concatenate(pam_alpha, axis=0),
            "R": np.concatenate(pam_r_scores, axis=0),
        }

    outputs["pids"] = list(pids)
    outputs["pam_details"] = pam_details
    if outputs["logits"]:
        outputs["logits"] = np.concatenate(outputs["logits"], axis=0)
    else:
        outputs["logits"] = np.zeros((0, 0), dtype=np.float32) if _is_survival(task_config) else np.zeros((0,), dtype=np.float32)
    if _is_survival(task_config):
        outputs["hazards"] = np.concatenate(outputs.get("hazards", []), axis=0) if outputs.get("hazards") else np.zeros_like(outputs["logits"])
        outputs["survival"] = np.concatenate(outputs.get("survival", []), axis=0) if outputs.get("survival") else np.zeros_like(outputs["logits"])
        outputs["risk"] = np.concatenate(outputs.get("risk", []), axis=0) if outputs.get("risk") else np.zeros((outputs["logits"].shape[0],), dtype=np.float32)
        outputs["event_times"] = np.asarray(outputs["event_times"], dtype=np.float32)
        outputs["event_observed"] = np.asarray(outputs["event_observed"], dtype=np.int64)
        outputs["censorship"] = np.asarray(outputs["censorship"], dtype=np.float32)
        outputs["y_disc"] = np.asarray(outputs["y_disc"], dtype=np.int64)
    else:
        outputs["y_true"] = np.asarray(outputs["y_true"], dtype=np.int64)
        outputs["probs"] = np.asarray(outputs["probs"], dtype=np.float32)
    return outputs

def _log_selected_inner_models_to_wandb(
    selected_candidates,
    seed,
    outer_fold_idx,
    train_missing_simulator,
    wandb_project,
    wandb_mode,
    wandb_base_config,
    model_name_l,
    task_config,
):
    degrading_modality = str((wandb_base_config or {}).get("degrading_modality", "na")).strip().lower()
    train_missing_prop = float((wandb_base_config or {}).get("train_missing_prop", 0.0))

    for candidate in selected_candidates:
        run_name = (
            f"degmod{degrading_modality}_"
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
            }
            if _is_survival(task_config):
                log_payload["best_inner_model/val_cindex"] = float(hrow.get("val_cindex", 0.0))
            else:
                log_payload.update(
                    {
                        "best_inner_model/val_auc": float(hrow["val_auc"]),
                        "best_inner_model/val_aucpr": float(hrow["val_aucpr"]),
                        "best_inner_model/val_acc": float(hrow["val_acc"]),
                    }
                )
            if bool(candidate.get("model_kwargs", {}).get("knowledge_distillation", False)):
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
    attention_pooling_kwargs,
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
    best_mean_metric,
    task_config,
    use_ensemble=False,
):
    outer_results = []
    test_prediction_rows = []
    primary_metric_key = primary_metric_name(task_config)
    primary_loss_key = primary_loss_name(task_config)
    base_dataset_kwargs = _prepare_base_dataset_kwargs(task_config)

    for eval_setup in test_eval_setups:
        eval_simulator = eval_setup["simulator"]
        eval_degrading_modality = str(eval_setup["degrading_modality"]).lower()
        eval_missing_prop = float(eval_setup["missing_prop"])
        apply_missing_eval = eval_missing_prop > 0.0

        ref_targets = None
        ref_pids = None
        model_outputs = []
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
                attention_pooling_kwargs=attention_pooling_kwargs,
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
                **base_dataset_kwargs,
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

            if _is_sklearn_classification_model(model_name_l):
                pred_out = _predict_lr_outputs(
                    model=loaded_bundle["model"],
                    dataset=outer_eval_ds,
                )
            elif _is_sklearn_survival_model(model_name_l):
                pred_out = _predict_survival_baseline_outputs(
                    model=loaded_bundle["model"],
                    dataset=outer_eval_ds,
                )
            else:
                pred_out = _predict_model_outputs(
                    model=loaded_bundle["model"],
                    data_loader=outer_eval_loader,
                    device=device,
                    bypass_mask=predict_bypass_mask,
                    collect_pam_details=model_name_l in {"pam"},
                    model_name=model_name_l,
                    task_config=task_config,
                )

            if _is_survival(task_config):
                aligned_targets = (
                    pred_out["event_times"],
                    pred_out["event_observed"],
                    pred_out["censorship"],
                    pred_out["y_disc"],
                )
            else:
                aligned_targets = pred_out["y_true"]

            if ref_targets is None:
                ref_targets = aligned_targets
                ref_pids = list(pred_out["pids"])
            else:
                if _is_survival(task_config):
                    if not all(np.array_equal(a, b) for a, b in zip(ref_targets, aligned_targets)):
                        raise RuntimeError("Retained inner-model predictions are misaligned on survival targets.")
                else:
                    if not np.array_equal(ref_targets, aligned_targets):
                        raise RuntimeError("Retained inner-model predictions are misaligned on y_true.")
                if ref_pids != list(pred_out["pids"]):
                    raise RuntimeError("Retained inner-model predictions are misaligned on patient IDs.")

            model_outputs.append(pred_out)
            model_details.append(pred_out["pam_details"])
            model_outer_metrics.append(
                _metrics_from_prediction_output(task_config, pred_out, model_name_l)
            )

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
                "train_degrading_modality": str(getattr(train_missing_simulator, "degrading_modality", "global")).lower(),
                "train_missing_prop": float(getattr(train_missing_simulator, "missing_prop", 0.0)),
                "test_degrading_modality": eval_degrading_modality,
                "test_missing_prop": eval_missing_prop,
            }
            if _is_survival(task_config):
                row["event_time"] = float(ref_targets[0][patient_idx])
                row["event_observed"] = int(ref_targets[1][patient_idx])
                row["censorship"] = int(ref_targets[2][patient_idx])
                row["y_disc"] = int(ref_targets[3][patient_idx])
            else:
                row["y_true"] = int(ref_targets[patient_idx])

            for model_idx, (pred_out, details_arr) in enumerate(
                zip(model_outputs, model_details),
                1,
            ):
                if _is_survival(task_config):
                    row[f"inner_model_{model_idx}_risk"] = float(pred_out["risk"][patient_idx])
                    for bin_idx in range(pred_out["logits"].shape[1]):
                        row[f"inner_model_{model_idx}_logit_bin_{bin_idx}"] = float(
                            pred_out["logits"][patient_idx, bin_idx]
                        )
                        row[f"inner_model_{model_idx}_hazard_bin_{bin_idx}"] = float(
                            pred_out["hazards"][patient_idx, bin_idx]
                        )
                        row[f"inner_model_{model_idx}_survival_bin_{bin_idx}"] = float(
                            pred_out["survival"][patient_idx, bin_idx]
                        )
                else:
                    row[f"inner_model_{model_idx}_logit"] = float(pred_out["logits"][patient_idx])
                    row[f"inner_model_{model_idx}_prob"] = float(pred_out["probs"][patient_idx])
                    row[f"inner_model_{model_idx}_pred_label"] = int(pred_out["logits"][patient_idx] >= 0.0)
                if model_name_l in {"pam"} and details_arr is not None:
                    for modality_idx, modality_name in enumerate(modality_names):
                        row[f"inner_model_{model_idx}_{modality_name}_alpha"] = float(
                            details_arr["alpha"][patient_idx, modality_idx]
                        )
                        r_value = details_arr["R"][patient_idx, modality_idx]
                        if np.ndim(r_value) == 0:
                            row[f"inner_model_{model_idx}_{modality_name}_R"] = float(r_value)
                        else:
                            for bin_idx, bin_value in enumerate(np.asarray(r_value).reshape(-1)):
                                row[f"inner_model_{model_idx}_{modality_name}_R_bin_{bin_idx}"] = float(bin_value)

            if bool(use_ensemble) and not _is_survival(task_config):
                _add_binary_ensemble_prediction_columns(row)
            per_patient_prediction_rows.append(row)

        test_prediction_rows.extend(per_patient_prediction_rows)

        if _is_survival(task_config):
            outer_metric_source = "mean_retained_inner_models"
            loss_values = np.asarray([float(metrics.get("LOSS", np.nan)) for metrics in model_outer_metrics], dtype=float)
            finite_loss_values = loss_values[np.isfinite(loss_values)]
            mean_outer_metrics = {
                "LOSS": float(np.mean(finite_loss_values)) if finite_loss_values.size else np.nan,
                "CINDEX": float(np.mean([float(metrics["CINDEX"]) for metrics in model_outer_metrics])),
            }
        elif bool(use_ensemble):
            ensemble_probs = np.mean(
                [np.asarray(pred_out["probs"], dtype=float) for pred_out in model_outputs],
                axis=0,
            )
            mean_outer_metrics = safe_task_metrics(
                task_config,
                y_true=np.asarray(ref_targets, dtype=int),
                y_prob=ensemble_probs,
            )
            outer_metric_source = "probability_averaged_ensemble"
        else:
            metric_names = ["LOGLOSS", "AUC", "AUCPR", "ACC", "SEN", "SP", "MCC"]
            mean_outer_metrics = {
                name: float(np.mean([float(metrics[name]) for metrics in model_outer_metrics]))
                for name in metric_names
            }
            outer_metric_source = "mean_retained_inner_models"

        result_row = {
            "outer_fold": outer_fold_idx,
            "outer_eval_target": "test_outer",
            "eval_degrading_modality": eval_degrading_modality,
            "eval_missing_prop": eval_missing_prop,
            "inner_models_count": int(len(selected_candidates)),
            "selected_inner_hp_names": str(best_hp_row["hp_name"]),
            "hp_selection_epsilon": float(epsilon),
            "outer_test_metric_source": outer_metric_source,
            "outer_refit_epochs": np.nan,
        }
        result_row[f"selected_inner_mean_{primary_metric_key}"] = float(
            np.mean([r[f"val_best_{primary_metric_key}"] for r in selected_inner_rows])
        )
        result_row[f"selected_inner_mean_{primary_loss_key}"] = float(
            np.mean([r[f"val_best_{primary_loss_key}"] for r in selected_inner_rows])
        )
        result_row[f"selected_inner_std_{primary_metric_key}"] = float(best_hp_row["std_primary"])
        result_row[f"hp_selection_best_mean_{primary_metric_key}"] = float(best_mean_metric)
        for metric_name, metric_value in mean_outer_metrics.items():
            result_row[f"outer_test_{metric_name}"] = float(metric_value)

        outer_results.append(result_row)

    return outer_results, test_prediction_rows


# --------------------------- NESTED CV FUNCTION -----------------------------

# Main nested cross-validation function
def nested_cv(
    dfs,
    inst_df,
    label_col,
    task_config,
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
    attention_pooling_kwargs=None,
    modality_pooling=None,
    candidate_model_dir=None,
    retrain_outer=True,
    save_inner=False,
    early_stopping_patience=20,
    min_lr=1e-6,
    scheduler_type="cosine_annealing",
    lr_patience=5,
    hp_selection_epsilon=0.02,
    missingness_study=True,
    use_ensemble=False,
):
    wandb_active = bool(wandb_enabled and wandb is not None)
    if wandb_enabled and wandb is None:
        print("wandb is not installed. Continuing without wandb logging.")

    # Train missingness is applied on both inner-train and inner-validation.
    apply_missing_train = float(getattr(train_missing_simulator, "missing_prop", 0.0)) > 0.0
    model_name_l = normalize_model_name(model_name)
    if _is_sklearn_classification_model(model_name_l) and _is_survival(task_config):
        raise ValueError("ZI_LR, KNN_LR, ZI_RF and KNN_RF are classification-only baselines and do not support task_type=survival.")
    if _is_sklearn_survival_model(model_name_l) and not _is_survival(task_config):
        raise ValueError("ZI_CoxNet, KNN_CoxNet, ZI_RSF and KNN_RSF are survival-only baselines.")
    predict_bypass_mask = (
        model_name_l == "mlp"
        and str(imputation_method).strip().lower() in {"knn", "vae"}
    )
    set_global_seed(seed, deterministic=True)
    primary_metric_key = primary_metric_name(task_config)
    primary_loss_key = primary_loss_name(task_config)
    base_dataset_kwargs = _prepare_base_dataset_kwargs(task_config)

    # Test-time evaluation setups:
    # by default evaluate once with the provided train simulator.
    if test_eval_setups is None:
        eval_missing_prop = float(getattr(train_missing_simulator, "missing_prop", 0.0))
        eval_degrading_modality = str(getattr(train_missing_simulator, "degrading_modality", "global")).lower()
        test_eval_setups = [
            {
                "missing_prop": eval_missing_prop,
                "degrading_modality": eval_degrading_modality,
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
    student_train_missing_simulator = train_missing_simulator
    apply_student_missing_train = apply_missing_train
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
                attention_pooling_kwargs=attention_pooling_kwargs,
                labels_df=inst_df_train_inner,
                label_col=label_col,
            )
            dfs_val_inner_prepared, _ = _prepare_patient_level_modalities(
                dfs_val_inner_raw,
                patient_id_col=patient_id_col,
                modality_pooling=modality_pooling,
                fit_attention_poolers=False,
                fitted_poolers=modality_poolers_inner,
                attention_pooling_kwargs=attention_pooling_kwargs,
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
                **base_dataset_kwargs,
            )
            prefit_inner_imputer = _fit_split_imputer(
                split_dataset=train_split_dataset,
                split_missing_simulator=student_train_missing_simulator,
                apply_split_missing=apply_student_missing_train,
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
                    missing_simulator=student_train_missing_simulator,
                    batch_size=hp_cfg["batch_size"],
                    train_missing=apply_student_missing_train,
                    val_missing=apply_student_missing_train,
                    imputation_method=imputation_method,
                    missing_pattern_seed=missing_pattern_seed,
                    model_name=model_name,
                    loader_seed=int(seed + outer_fold_idx * 10_000 + inner_fold_idx * 100 + hp_idx),
                    id_col=patient_id_col,
                    prefit_imputer=prefit_inner_imputer,
                    imputer_kwargs=imputer_kwargs,
                    task_type=normalize_task_type((task_config or {}).get("task_type", "binary_classification")),
                    survival_time_col=(task_config or {}).get("survival_time_col"),
                    survival_event_col=(task_config or {}).get("survival_event_col"),
                    survival_censorship_col=(task_config or {}).get("survival_censorship_col"),
                    survival_y_disc_col=(task_config or {}).get("survival_y_disc_col"),
                )

                model_kwargs = _with_task_output_dim(
                    _build_model_kwargs_from_hp_cfg(model_name_l, hp_cfg),
                    task_config,
                )

                # Train the model and evaluate on inner-val fold
                if _is_sklearn_classification_model(model_name_l):
                    model = _fit_sklearn_classification_model(
                        dataset=train_loader.dataset,
                        model_name_l=model_name_l,
                        hp_cfg=hp_cfg,
                        seed=int(seed + outer_fold_idx * 10_000 + inner_fold_idx * 100 + hp_idx),
                    )
                    train_pred_out = _predict_lr_outputs(model, train_loader.dataset)
                    val_pred_out = _predict_lr_outputs(model, val_loader.dataset)
                    train_metrics = safe_task_metrics(
                        task_config,
                        y_true=train_pred_out["y_true"],
                        y_prob=train_pred_out["probs"],
                    )
                    best_metrics = safe_task_metrics(
                        task_config,
                        y_true=val_pred_out["y_true"],
                        y_prob=val_pred_out["probs"],
                    )
                    best_metrics["best_epoch"] = 1
                    history = [_lr_history_row(train_metrics=train_metrics, val_metrics=best_metrics)]
                elif _is_sklearn_survival_model(model_name_l):
                    model = _fit_survival_baseline_model(
                        dataset=train_loader.dataset,
                        model_name_l=model_name_l,
                        hp_cfg=hp_cfg,
                        seed=int(seed + outer_fold_idx * 10_000 + inner_fold_idx * 100 + hp_idx),
                    )
                    train_pred_out = _predict_survival_baseline_outputs(model, train_loader.dataset)
                    val_pred_out = _predict_survival_baseline_outputs(model, val_loader.dataset)
                    train_metrics = _metrics_from_prediction_output(task_config, train_pred_out, model_name_l)
                    best_metrics = _metrics_from_prediction_output(task_config, val_pred_out, model_name_l)
                    best_metrics["best_epoch"] = 1
                    history = [_survival_baseline_history_row(train_metrics=train_metrics, val_metrics=best_metrics)]
                else:
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
                        scheduler_type=scheduler_type,
                        lr_patience=lr_patience,
                        model_name=model_name,
                        imputation_method=imputation_method,
                        model_kwargs=model_kwargs,
                        train_seed=int(seed + outer_fold_idx * 10_000 + inner_fold_idx * 100 + hp_idx),
                        task_config=task_config,
                    )

                # Save inner evaluation METRICS for this HP config and inner fold
                inner_eval_rows.append(
                    {
                        "outer_fold": outer_fold_idx,
                        "inner_fold": inner_fold_idx,
                        "hp_name": hp_name,
                        **hp_cfg,
                        "val_best_epoch": int(best_metrics["best_epoch"]),
                        "val_best_LOGLOSS": float(best_metrics.get("LOGLOSS", np.nan)),
                        "val_best_AUC": float(best_metrics.get("AUC", np.nan)),
                        "val_best_AUCPR": float(best_metrics.get("AUCPR", np.nan)),
                        "val_best_ACC": float(best_metrics.get("ACC", np.nan)),
                        "val_best_SEN": float(best_metrics.get("SEN", np.nan)),
                        "val_best_SP": float(best_metrics.get("SP", np.nan)),
                        "val_best_MCC": float(best_metrics.get("MCC", np.nan)),
                        "val_best_LOSS": float(best_metrics.get("LOSS", np.nan)),
                        "val_best_CINDEX": float(best_metrics.get("CINDEX", np.nan)),
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

            primary_scores = np.asarray(
                [float(c["metrics"][primary_metric_key]) for c in candidates],
                dtype=np.float32,
            )
            loss_scores = np.asarray(
                [float(c["metrics"][primary_loss_key]) for c in candidates],
                dtype=np.float32,
            )
            mean_primary = float(np.mean(primary_scores))
            std_primary = float(np.std(primary_scores))
            finite_loss_scores = loss_scores[np.isfinite(loss_scores)]
            mean_loss = float(np.mean(finite_loss_scores)) if finite_loss_scores.size else float("inf")
            hp_selection_rows.append(
                {
                    "hp_name": hp_name,
                    "hp_cfg": hp_cfg,
                    "candidates": candidates,
                    "mean_primary": mean_primary,
                    "std_primary": std_primary,
                    "mean_loss": mean_loss,
                }
            )

        epsilon = max(0.0, float(hp_selection_epsilon))
        best_mean_metric = max(float(row["mean_primary"]) for row in hp_selection_rows)
        tied_hp_rows = [
            row
            for row in hp_selection_rows
            if float(row["mean_primary"]) >= (best_mean_metric - epsilon)
        ]
        best_hp_row = min(
            tied_hp_rows,
            key=lambda row: (
                float(row["std_primary"]),
                float(row["mean_loss"]),
                str(row["hp_name"]),
            ),
        )
        selected_candidates = best_hp_row["candidates"]

        print(
            f"  Selected hp across inner folds: {best_hp_row['hp_name']} "
            f"(mean_{primary_metric_key}={best_hp_row['mean_primary']:.4f}, "
            f"std_{primary_metric_key}={best_hp_row['std_primary']:.4f}, "
            f"mean_{primary_loss_key}={best_hp_row['mean_loss']:.4f}, epsilon={epsilon:.4f}, "
            f"best_mean_{primary_metric_key}={best_mean_metric:.4f}, tied_configs={len(tied_hp_rows)})"
        )

        for candidate in selected_candidates:
            candidate_metrics = candidate["metrics"]
            print(
                f"    Inner fold {candidate['inner_fold']} retained model: "
                f"{primary_metric_key}={float(candidate_metrics[primary_metric_key]):.4f}, "
                f"{primary_loss_key}={float(candidate_metrics[primary_loss_key]):.4f}, "
                f"best_epoch={int(candidate_metrics['best_epoch'])}"
            )
            selected_inner_rows.append(
                {
                    "inner_fold": int(candidate["inner_fold"]),
                    "hp_name": candidate["hp_name"],
                    **candidate["hp_cfg"],
                    "val_best_AUC": float(candidate_metrics.get("AUC", np.nan)),
                    "val_best_LOGLOSS": float(candidate_metrics.get("LOGLOSS", np.nan)),
                    "val_best_CINDEX": float(candidate_metrics.get("CINDEX", np.nan)),
                    "val_best_LOSS": float(candidate_metrics.get("LOSS", np.nan)),
                    f"selected_hp_mean_{primary_metric_key}": float(best_hp_row["mean_primary"]),
                    f"selected_hp_std_{primary_metric_key}": float(best_hp_row["std_primary"]),
                    "hp_selection_epsilon": float(epsilon),
                    f"hp_selection_best_mean_{primary_metric_key}": float(best_mean_metric),
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
                attention_pooling_kwargs=attention_pooling_kwargs,
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
                **base_dataset_kwargs,
            )
            prefit_outer_imputer = _fit_split_imputer(
                split_dataset=outer_train_split_dataset,
                split_missing_simulator=student_train_missing_simulator,
                apply_split_missing=apply_student_missing_train,
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
                missing_simulator=student_train_missing_simulator,
                batch_size=selected_hp_cfg["batch_size"],
                train_missing=apply_student_missing_train,
                val_missing=False,
                imputation_method=imputation_method,
                missing_pattern_seed=missing_pattern_seed,
                model_name=model_name,
                loader_seed=int(seed + outer_fold_idx * 100_000 + 1),
                id_col=patient_id_col,
                prefit_imputer=prefit_outer_imputer,
                imputer_kwargs=imputer_kwargs,
                task_type=normalize_task_type((task_config or {}).get("task_type", "binary_classification")),
                survival_time_col=(task_config or {}).get("survival_time_col"),
                survival_event_col=(task_config or {}).get("survival_event_col"),
                survival_censorship_col=(task_config or {}).get("survival_censorship_col"),
                survival_y_disc_col=(task_config or {}).get("survival_y_disc_col"),
            )

            if _is_sklearn_classification_model(model_name_l):
                outer_train_model = _fit_sklearn_classification_model(
                    dataset=outer_train_loader.dataset,
                    model_name_l=model_name_l,
                    hp_cfg=selected_hp_cfg,
                    seed=int(seed + outer_fold_idx * 100_000 + 2),
                )
                outer_train_pred_out = _predict_lr_outputs(outer_train_model, outer_train_loader.dataset)
                outer_train_metrics = safe_task_metrics(
                    task_config,
                    y_true=outer_train_pred_out["y_true"],
                    y_prob=outer_train_pred_out["probs"],
                )
                outer_refit_history = [_lr_history_row(train_metrics=outer_train_metrics)]
            elif _is_sklearn_survival_model(model_name_l):
                outer_train_model = _fit_survival_baseline_model(
                    dataset=outer_train_loader.dataset,
                    model_name_l=model_name_l,
                    hp_cfg=selected_hp_cfg,
                    seed=int(seed + outer_fold_idx * 100_000 + 2),
                )
                outer_train_pred_out = _predict_survival_baseline_outputs(
                    outer_train_model,
                    outer_train_loader.dataset,
                )
                outer_train_metrics = _metrics_from_prediction_output(
                    task_config,
                    outer_train_pred_out,
                    model_name_l,
                )
                outer_refit_history = [_survival_baseline_history_row(train_metrics=outer_train_metrics)]
            else:
                outer_train_model, outer_refit_history = train_model_on_full_dataset(
                    train_loader=outer_train_loader,
                    device=device,
                    input_dims=input_dims,
                    epochs=refit_epochs,
                    lr=selected_hp_cfg["learning_rate"],
                    weight_decay=selected_hp_cfg["weight_decay"],
                    min_lr=min_lr,
                    scheduler_type=scheduler_type,
                    lr_patience=lr_patience,
                    model_name=model_name,
                    imputation_method=imputation_method,
                    model_kwargs=selected_model_kwargs,
                    train_seed=int(seed + outer_fold_idx * 100_000 + 2),
                    task_config=task_config,
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
                degrading_modality = str((wandb_base_config or {}).get("degrading_modality", "na")).strip().lower()
                train_missing_prop = float((wandb_base_config or {}).get("train_missing_prop", 0.0))
                run_name = (
                    f"degmod{degrading_modality}_"
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
                    log_payload = {"outer_train_model/train_loss": float(hrow["train_loss"])}
                    if _is_survival(task_config):
                        log_payload["outer_train_model/train_cindex"] = float(hrow.get("train_cindex", 0.0))
                    else:
                        log_payload.update(
                            {
                                "outer_train_model/train_auc": float(hrow["train_auc"]),
                                "outer_train_model/train_aucpr": float(hrow["train_aucpr"]),
                                "outer_train_model/train_acc": float(hrow["train_acc"]),
                            }
                        )

                    if bool(selected_model_kwargs.get("knowledge_distillation", False)):
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
                attention_pooling_kwargs=attention_pooling_kwargs,
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
                **base_dataset_kwargs,
            )

            for eval_setup in test_eval_setups:
                eval_simulator = eval_setup["simulator"]
                eval_degrading_modality = str(eval_setup["degrading_modality"]).lower()
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

                if _is_sklearn_classification_model(model_name_l):
                    pred_out = _predict_lr_outputs(
                        model=outer_train_model,
                        dataset=outer_eval_ds,
                    )
                elif _is_sklearn_survival_model(model_name_l):
                    pred_out = _predict_survival_baseline_outputs(
                        model=outer_train_model,
                        dataset=outer_eval_ds,
                    )
                else:
                    pred_out = _predict_model_outputs(
                        model=outer_train_model,
                        data_loader=outer_eval_loader,
                        device=device,
                        bypass_mask=predict_bypass_mask,
                        collect_pam_details=model_name_l in {"pam"},
                        model_name=model_name_l,
                        task_config=task_config,
                    )
                outer_metrics = _metrics_from_prediction_output(task_config, pred_out, model_name_l)

                per_patient_prediction_rows = []
                for patient_idx, pid in enumerate(pred_out["pids"]):
                    row = {
                        "outer_fold": outer_fold_idx,
                        "outer_eval_target": "test_outer",
                        "patient": pid,
                        "train_degrading_modality": str(getattr(train_missing_simulator, "degrading_modality", "global")).lower(),
                        "train_missing_prop": float(getattr(train_missing_simulator, "missing_prop", 0.0)),
                        "test_degrading_modality": eval_degrading_modality,
                        "test_missing_prop": eval_missing_prop,
                    }
                    if _is_survival(task_config):
                        row["event_time"] = float(pred_out["event_times"][patient_idx])
                        row["event_observed"] = int(pred_out["event_observed"][patient_idx])
                        row["censorship"] = int(pred_out["censorship"][patient_idx])
                        row["y_disc"] = int(pred_out["y_disc"][patient_idx])
                        row["inner_model_1_risk"] = float(pred_out["risk"][patient_idx])
                        for bin_idx in range(pred_out["logits"].shape[1]):
                            row[f"inner_model_1_logit_bin_{bin_idx}"] = float(
                                pred_out["logits"][patient_idx, bin_idx]
                            )
                            row[f"inner_model_1_hazard_bin_{bin_idx}"] = float(
                                pred_out["hazards"][patient_idx, bin_idx]
                            )
                            row[f"inner_model_1_survival_bin_{bin_idx}"] = float(
                                pred_out["survival"][patient_idx, bin_idx]
                            )
                    else:
                        row["y_true"] = int(pred_out["y_true"][patient_idx])
                        logit_value = float(pred_out["logits"][patient_idx])
                        prob_value = float(pred_out["probs"][patient_idx])
                        pred_label = int(logit_value >= 0.0)
                        row["inner_model_1_logit"] = logit_value
                        row["inner_model_1_prob"] = prob_value
                        row["inner_model_1_pred_label"] = pred_label
                        if bool(use_ensemble):
                            _add_binary_ensemble_prediction_columns(row)
                    if model_name_l in {"pam"} and pred_out["pam_details"] is not None:
                        for modality_idx, modality_name in enumerate(modality_names):
                            row[f"inner_model_1_{modality_name}_alpha"] = float(
                                pred_out["pam_details"]["alpha"][patient_idx, modality_idx]
                            )
                            r_value = pred_out["pam_details"]["R"][patient_idx, modality_idx]
                            if np.ndim(r_value) == 0:
                                row[f"inner_model_1_{modality_name}_R"] = float(r_value)
                            else:
                                for bin_idx, bin_value in enumerate(np.asarray(r_value).reshape(-1)):
                                    row[f"inner_model_1_{modality_name}_R_bin_{bin_idx}"] = float(bin_value)
                    per_patient_prediction_rows.append(row)

                test_prediction_rows.extend(per_patient_prediction_rows)

                result_row = {
                    "outer_fold": outer_fold_idx,
                    "outer_eval_target": "test_outer",
                    "eval_degrading_modality": eval_degrading_modality,
                    "eval_missing_prop": eval_missing_prop,
                    "inner_models_count": 1,
                    "selected_inner_hp_names": str(best_hp_row["hp_name"]),
                    "hp_selection_epsilon": float(epsilon),
                    "outer_test_metric_source": "outer_refit_model",
                    "outer_refit_epochs": int(refit_epochs),
                    f"selected_inner_mean_{primary_metric_key}": float(
                        np.mean([r[f"val_best_{primary_metric_key}"] for r in selected_inner_rows])
                    ),
                    f"selected_inner_mean_{primary_loss_key}": float(
                        np.mean([r[f"val_best_{primary_loss_key}"] for r in selected_inner_rows])
                    ),
                    f"selected_inner_std_{primary_metric_key}": float(best_hp_row["std_primary"]),
                    f"hp_selection_best_mean_{primary_metric_key}": float(best_mean_metric),
                }
                for metric_name, metric_value in outer_metrics.items():
                    result_row[f"outer_test_{metric_name}"] = float(metric_value)
                outer_results.append(result_row)

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
                        task_config=task_config,
                    )
                inner_outer_results, inner_test_prediction_rows = _evaluate_retained_inner_models_on_outer_test(
                    selected_candidates=selected_candidates,
                    dfs_test_outer_raw=dfs_test_outer_raw,
                    inst_df_test_outer=inst_df_test_outer,
                    label_col=label_col,
                    patient_id_col=patient_id_col,
                    modality_pooling=modality_pooling,
                    attention_pooling_kwargs=attention_pooling_kwargs,
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
                    best_mean_metric=best_mean_metric,
                    task_config=task_config,
                    use_ensemble=bool(use_ensemble),
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
                    task_config=task_config,
                )

            inner_outer_results, inner_test_prediction_rows = _evaluate_retained_inner_models_on_outer_test(
                selected_candidates=selected_candidates,
                dfs_test_outer_raw=dfs_test_outer_raw,
                inst_df_test_outer=inst_df_test_outer,
                label_col=label_col,
                patient_id_col=patient_id_col,
                modality_pooling=modality_pooling,
                attention_pooling_kwargs=attention_pooling_kwargs,
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
                best_mean_metric=best_mean_metric,
                task_config=task_config,
                use_ensemble=bool(use_ensemble),
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
