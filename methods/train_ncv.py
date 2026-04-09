import copy
import gc
import os
import pickle
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from imputation_methods import build_imputer
from custom_learning.meta_learning import train_smil_e_with_meta_learning
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
    filter_by_patients,
    fit_and_transform_modalities,
    set_global_seed,
    normalize_model_name,
)
from dataset.preprocess_dataset import collapse_patient_rows
from dataset.radiology_attention_pooling import RadiologyAttentionPooler
try:
    import wandb
except ImportError:
    wandb = None

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
    radio_aggregation_method="mean",
    fit_radio_pooler=False,
    radio_pooler=None,
    radio_pooling_kwargs=None,
    labels_df=None,
    label_col=None,
):
    """Collapse raw split dataframes to one row per patient, optionally with learned radio pooling."""
    radio_aggregation_l = str(radio_aggregation_method).strip().lower()
    prepared = {}
    fitted_radio_pooler = radio_pooler

    for modality_name, df_raw in dfs_raw.items():
        if modality_name == "radio" and radio_aggregation_l == "attention":
            if fit_radio_pooler:
                if labels_df is None or label_col is None:
                    raise ValueError(
                        "labels_df and label_col are required when fitting the radiology attention pooler."
                    )
                feature_cols = [c for c in df_raw.columns if c != patient_id_col]
                if not feature_cols:
                    raise ValueError("Radiology dataframe has no feature columns to pool.")
                fitted_radio_pooler = RadiologyAttentionPooler(
                    input_dim=len(feature_cols),
                    **dict(radio_pooling_kwargs or {}),
                )
                fitted_radio_pooler.fit(
                    df_train=df_raw,
                    labels_df=labels_df,
                    id_col=patient_id_col,
                    label_col=label_col,
                )
            if fitted_radio_pooler is None:
                raise RuntimeError(
                    "Radiology attention pooling was requested but no fitted pooler is available."
                )
            df_patient = fitted_radio_pooler.transform(df_raw)
        else:
            df_patient = collapse_patient_rows(df_raw, id_col=patient_id_col, strategy="mean")

        prepared[modality_name] = df_patient.set_index(patient_id_col, drop=False)

    return prepared, fitted_radio_pooler


def _build_model_kwargs_from_hp_cfg(model_name_l, hp_cfg):
    if model_name_l in {"mlp"}:
        return {
            "modality_hidden_layers": hp_cfg["modality_hidden_layers"],
            "fusion_hidden_dim": hp_cfg["fusion_hidden_dim"],
            "fusion_hidden_layers": hp_cfg["fusion_hidden_layers"],
            "dropout_p": hp_cfg["dropout"],
            "fusion_batchnorm": bool(hp_cfg["fusion_batchnorm"]),
        }
    if model_name_l in {"dyam"}:
        return {
            "dropout_p": hp_cfg["dyam_dropout"],
            "temperature": hp_cfg["dyam_temperature"],
        }
    if model_name_l in {"distill_dyam"}:
        return {
            "dropout_p": hp_cfg["dyam_dropout"],
            "temperature": hp_cfg["dyam_temperature"],
            "concat_masks_input": True,
            "distill_alpha": hp_cfg["distill_alpha"],
            "distill_beta": hp_cfg["distill_beta"],
        }
    if model_name_l in {"smil_e"}:
        return {
            "latent_dim": hp_cfg["smil_e_latent_dim"],
            "num_priors": hp_cfg["smil_e_num_priors"],
            "num_heads": hp_cfg["smil_e_num_heads"],
            "dropout": hp_cfg["smil_e_dropout"],
            "alpha": hp_cfg["smil_e_alpha"],
            "beta": hp_cfg["smil_e_beta"],
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
        f"Unsupported model '{model_name_l}'. Supported: mlp, dyam, distill_dyam, smil_e, healnet"
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
    radio_pooler,
):
    bundle = {
        "model_name": normalize_model_name(model_name),
        "input_dims": [int(dim) for dim in input_dims],
        "model_kwargs": dict(model_kwargs or {}),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "scalers": scalers,
        "imputer": imputer,
        "radio_pooler": radio_pooler,
    }
    with open(bundle_path, "wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_candidate_bundle(bundle_path, device):
    with open(bundle_path, "rb") as handle:
        bundle = pickle.load(handle)

    model = build_model(
        bundle["model_name"],
        bundle["input_dims"],
        bundle["model_kwargs"],
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])

    return {
        "model": model,
        "scalers": bundle["scalers"],
        "imputer": bundle["imputer"],
        "radio_pooler": bundle["radio_pooler"],
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
    collect_dyam_details=False,
    model_name=None,
):
    """Run one model on a loader and return y_true / logits / probabilities / pids."""
    model.eval()
    y_true = []
    y_logits = []
    y_prob = []
    pids = []
    dyam_alpha = []
    dyam_r_scores = []

    with torch.no_grad():
        for Xs, present_mask, y, pid_batch in data_loader:
            Xs = [x.to(device) for x in Xs]
            present_mask = present_mask.to(device)

            model_mask = None if bypass_mask else present_mask
            if collect_dyam_details:
                model_out = model(Xs, model_mask, return_aux=True)
                logits = model_out[0].squeeze(1)
                if model_name not in {"dyam", "distill_dyam"}:
                    raise ValueError(
                        "collect_dyam_details=True is only supported for model_name in "
                        "{'dyam', 'distill_dyam'}."
                    )
                dyam_alpha.append(model_out[2].detach().cpu().numpy())
                dyam_r_scores.append(model_out[3].detach().cpu().numpy())
            else:
                logits = model(Xs, model_mask).squeeze(1)
            logits_np = logits.detach().cpu().numpy().reshape(-1)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            y_logits.extend(logits_np.tolist())
            y_prob.extend(probs.tolist())
            y_true.extend(y.cpu().numpy().tolist())
            pids.extend(pid_batch)

    dyam_details = None
    if collect_dyam_details:
        dyam_details = {
            "alpha": np.concatenate(dyam_alpha, axis=0),
            "R": np.concatenate(dyam_r_scores, axis=0),
        }

    return np.asarray(y_true), np.asarray(y_logits), np.asarray(y_prob), list(pids), dyam_details

# Function to build a complete-modality batch for teacher input in distillation
def _build_full_batch_from_patient_ids(base_dataset, pid_batch, device):
    """Build a complete-modality batch (teacher input) from patient IDs."""
    xs_rows = []
    ys = []
    for pid in pid_batch:
        xs_i, y_i, _ = base_dataset.get_by_patient_id(pid)
        xs_rows.append(xs_i)
        ys.append(y_i)

    n_modalities = len(xs_rows[0])
    xs_full = []
    for m_idx in range(n_modalities):
        xs_full.append(torch.stack([row[m_idx] for row in xs_rows], dim=0).to(device))
    y_full = torch.stack(ys, dim=0).to(dtype=torch.float32, device=device)
    full_mask = torch.ones((len(pid_batch), n_modalities), dtype=torch.bool, device=device)
    return xs_full, full_mask, y_full

# Functions to compute losses for supervised and distillation cases
def _compute_supervised_bce_loss(logits, targets, bce_criterion):
    return bce_criterion(logits, targets)

# For distillation, we combine the student's supervised loss with weighted representation and feature matching losses against the teacher
def _compute_distill_student_loss(
    student_logits,
    student_repr,
    teacher_logits,
    teacher_repr,
    targets,
    bce_criterion,
    repr_criterion,
    feat_criterion,
    alpha_repr,
    beta_feat,
):
    loss_survival = bce_criterion(student_logits, targets)
    loss_repr = repr_criterion(student_repr, teacher_repr)
    loss_feature = feat_criterion(student_logits, teacher_logits)
    total = loss_survival + (alpha_repr * loss_repr) + (beta_feat * loss_feature)
    return total, loss_survival, loss_repr, loss_feature


# ---------------------------- TRAIN FUNCTION -------------------------------

# Train function with validation and early stopping
def train_model_with_validation(
    train_loader,
    val_loader,
    device,
    input_dims,
    epochs,
    lr,
    model_name,
    imputation_method="zero",
    model_kwargs=None,
    train_seed=0,
):
    min_best_epoch = min(5, int(epochs))
    model_kwargs = model_kwargs or {}
    model_name_l = normalize_model_name(model_name)

    if model_name_l in {"smil_e"}:
        return train_smil_e_with_meta_learning(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            input_dims=input_dims,
            epochs=epochs,
            lr=lr,
            model_kwargs=model_kwargs,
            train_seed=train_seed,
        )

    # For MLP with learned/external imputation, do not re-mask imputed modalities.
    bypass_mask = (
        model_name_l == "mlp"
        and str(imputation_method).strip().lower() in {"knn", "vae"}
    )

    criterion = nn.BCEWithLogitsLoss()

    if model_name_l in {"distill_dyam"}:
        dyam_kwargs = {k: v for k, v in model_kwargs.items() if k not in {"distill_alpha", "distill_beta"}}
        distill_alpha = float(model_kwargs.get("distill_alpha", 1.0))
        distill_beta = float(model_kwargs.get("distill_beta", 0.3))

        teacher_model = build_model("distill_dyam", input_dims, dyam_kwargs).to(device)
        student_model = build_model("distill_dyam", input_dims, dyam_kwargs).to(device)
        teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=lr, weight_decay=1e-4)
        student_optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            student_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        repr_criterion = nn.MSELoss()
        feat_criterion = nn.MSELoss()
        teacher_base_dataset = train_loader.dataset.base_dataset
    else:
        model = build_model(model_name, input_dims, model_kwargs).to(device)
        weight_decay = 1e-4
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler_patience = 5
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=scheduler_patience,
        )

    # Align epoch selection with inner-fold HP selection:
    # maximize AUC, break ties with lower validation loss.
    best_epoch_score = (-np.inf, -np.inf)  # (AUC, -VAL_LOSS)
    best_epoch = 1
    best_model_state = None
    best_val_targets = None
    best_val_probs = None

    early_stop = 0
    patience = 20

    history = []

    for epoch in range(1, epochs + 1):
        if model_name_l in {"distill_dyam"}:
            teacher_model.train()
            student_model.train()
        else:
            model.train()
        train_loss = 0.0
        train_steps = 0
        train_teacher_loss = 0.0
        train_student_survival = 0.0
        train_student_repr = 0.0
        train_student_feature = 0.0

        for Xs, present_mask, y, pids in train_loader:
            Xs = [x.to(device) for x in Xs]
            present_mask = present_mask.to(device)
            y = y.to(device)

            if model_name_l in {"distill_dyam"}:
                Xs_teacher, teacher_mask, y_teacher = _build_full_batch_from_patient_ids(
                    teacher_base_dataset,
                    pids,
                    device=device,
                )

                teacher_optimizer.zero_grad()
                teacher_out = teacher_model(Xs_teacher, teacher_mask, return_aux=True)
                teacher_logits = teacher_out[0].squeeze(1)
                teacher_repr = teacher_out[4]
                teacher_loss = _compute_supervised_bce_loss(teacher_logits, y_teacher, criterion)
                teacher_loss.backward()
                teacher_optimizer.step()

                student_optimizer.zero_grad()
                student_out = student_model(Xs, present_mask, return_aux=True)
                student_logits = student_out[0].squeeze(1)
                student_repr = student_out[4]
                student_loss, student_survival, student_repr_loss, student_feature_loss = _compute_distill_student_loss(
                    student_logits=student_logits,
                    student_repr=student_repr,
                    teacher_logits=teacher_logits.detach(),
                    teacher_repr=teacher_repr.detach(),
                    targets=y,
                    bce_criterion=criterion,
                    repr_criterion=repr_criterion,
                    feat_criterion=feat_criterion,
                    alpha_repr=distill_alpha,
                    beta_feat=distill_beta,
                )
                student_loss.backward()
                student_optimizer.step()

                train_loss += student_loss.item()
                train_teacher_loss += teacher_loss.item()
                train_student_survival += student_survival.item()
                train_student_repr += student_repr_loss.item()
                train_student_feature += student_feature_loss.item()
                train_steps += 1
            else:
                model_mask = None if bypass_mask else present_mask
                optimizer.zero_grad()
                logits_out = model(Xs, model_mask)
                if logits_out is None:
                    continue
                logits = logits_out.squeeze(1)
                loss = _compute_supervised_bce_loss(logits, y, criterion)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)
        avg_teacher_loss = train_teacher_loss / max(train_steps, 1)
        avg_student_survival = train_student_survival / max(train_steps, 1)
        avg_student_repr = train_student_repr / max(train_steps, 1)
        avg_student_feature = train_student_feature / max(train_steps, 1)

        if model_name_l in {"distill_dyam"}:
            student_model.eval()
        else:
            model.eval()
        val_loss = 0.0
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for Xs, present_mask, y, _ in val_loader:
                Xs = [x.to(device) for x in Xs]
                present_mask = present_mask.to(device)
                y = y.to(device)

                if model_name_l in {"distill_dyam"}:
                    logits = student_model(Xs, present_mask).squeeze(1)
                    loss = _compute_supervised_bce_loss(logits, y, criterion)
                else:
                    model_mask = None if bypass_mask else present_mask
                    logits = model(Xs, model_mask).squeeze(1)
                    loss = _compute_supervised_bce_loss(logits, y, criterion)

                val_loss += loss.item()
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                val_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        scheduler.step(avg_val_loss)

        val_metrics_epoch = safe_binary_metrics(val_targets, val_probs)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "val_auc": float(val_metrics_epoch["AUC"]),
                "val_aucpr": float(val_metrics_epoch["AUCPR"]),
                "val_acc": float(val_metrics_epoch["ACC"]),
                "teacher_loss": float(avg_teacher_loss),
                "student_survival_loss": float(avg_student_survival),
                "student_repr_loss": float(avg_student_repr),
                "student_feature_loss": float(avg_student_feature),
                "smil_meta_train_loss": 0.0,
                "smil_meta_val_loss": 0.0,
                "smil_meta_val_ce": 0.0,
                "smil_align_fusion": 0.0,
                "smil_align_hidden": 0.0,
            }
        )

        epoch_score = (float(val_metrics_epoch["AUC"]), -float(avg_val_loss))
        if epoch >= min_best_epoch and epoch_score > best_epoch_score:
            best_epoch_score = epoch_score
            best_epoch = epoch
            if model_name_l in {"distill_dyam"}:
                best_model_state = copy.deepcopy(student_model.state_dict())
            else:
                best_model_state = copy.deepcopy(model.state_dict())
            best_val_targets = np.asarray(val_targets)
            best_val_probs = np.asarray(val_probs)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= patience:
                break

    if best_model_state is None:
        raise RuntimeError(
            f"No best epoch was selected. Check epochs={epochs} and min_best_epoch={min_best_epoch}."
        )

    if model_name_l in {"distill_dyam"}:
        student_model.load_state_dict(best_model_state)
    else:
        model.load_state_dict(best_model_state)

    best_metrics = safe_binary_metrics(best_val_targets, best_val_probs)
    best_metrics["best_epoch"] = int(best_epoch)

    if model_name_l in {"distill_dyam"}:
        return student_model, history, best_metrics
    return model, history, best_metrics


def train_model_on_full_dataset(
    train_loader,
    device,
    input_dims,
    epochs,
    lr,
    model_name,
    imputation_method="zero",
    model_kwargs=None,
    train_seed=0,
):
    """Train a final model on the full outer-train split for a fixed number of epochs."""
    model_kwargs = model_kwargs or {}
    model_name_l = normalize_model_name(model_name)
    set_global_seed(train_seed, deterministic=True)

    bypass_mask = (
        model_name_l == "mlp"
        and str(imputation_method).strip().lower() in {"knn", "vae"}
    )

    criterion = nn.BCEWithLogitsLoss()

    if model_name_l in {"distill_dyam"}:
        dyam_kwargs = {k: v for k, v in model_kwargs.items() if k not in {"distill_alpha", "distill_beta"}}
        distill_alpha = float(model_kwargs.get("distill_alpha", 1.0))
        distill_beta = float(model_kwargs.get("distill_beta", 0.3))

        teacher_model = build_model("distill_dyam", input_dims, dyam_kwargs).to(device)
        student_model = build_model("distill_dyam", input_dims, dyam_kwargs).to(device)
        teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=lr, weight_decay=1e-4)
        student_optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            student_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        repr_criterion = nn.MSELoss()
        feat_criterion = nn.MSELoss()
        teacher_base_dataset = train_loader.dataset.base_dataset
    else:
        model = build_model(model_name, input_dims, model_kwargs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

    history = []

    for epoch in range(1, int(epochs) + 1):
        if model_name_l in {"distill_dyam"}:
            teacher_model.train()
            student_model.train()
        else:
            model.train()

        train_loss = 0.0
        train_steps = 0
        train_targets = []
        train_probs = []
        train_teacher_loss = 0.0
        train_student_survival = 0.0
        train_student_repr = 0.0
        train_student_feature = 0.0

        for Xs, present_mask, y, pids in train_loader:
            Xs = [x.to(device) for x in Xs]
            present_mask = present_mask.to(device)
            y = y.to(device)

            if model_name_l in {"distill_dyam"}:
                Xs_teacher, teacher_mask, y_teacher = _build_full_batch_from_patient_ids(
                    teacher_base_dataset,
                    pids,
                    device=device,
                )

                teacher_optimizer.zero_grad()
                teacher_out = teacher_model(Xs_teacher, teacher_mask, return_aux=True)
                teacher_logits = teacher_out[0].squeeze(1)
                teacher_repr = teacher_out[4]
                teacher_loss = _compute_supervised_bce_loss(teacher_logits, y_teacher, criterion)
                teacher_loss.backward()
                teacher_optimizer.step()

                student_optimizer.zero_grad()
                student_out = student_model(Xs, present_mask, return_aux=True)
                student_logits = student_out[0].squeeze(1)
                student_repr = student_out[4]
                student_loss, student_survival, student_repr_loss, student_feature_loss = _compute_distill_student_loss(
                    student_logits=student_logits,
                    student_repr=student_repr,
                    teacher_logits=teacher_logits.detach(),
                    teacher_repr=teacher_repr.detach(),
                    targets=y,
                    bce_criterion=criterion,
                    repr_criterion=repr_criterion,
                    feat_criterion=feat_criterion,
                    alpha_repr=distill_alpha,
                    beta_feat=distill_beta,
                )
                student_loss.backward()
                student_optimizer.step()

                probs = torch.sigmoid(student_logits).detach().cpu().numpy().reshape(-1)
                train_targets.extend(y.detach().cpu().numpy().tolist())
                train_probs.extend(probs.tolist())
                train_loss += student_loss.item()
                train_teacher_loss += teacher_loss.item()
                train_student_survival += student_survival.item()
                train_student_repr += student_repr_loss.item()
                train_student_feature += student_feature_loss.item()
                train_steps += 1
            else:
                model_mask = None if bypass_mask else present_mask
                optimizer.zero_grad()
                logits_out = model(Xs, model_mask)
                if logits_out is None:
                    continue
                logits = logits_out.squeeze(1)
                loss = _compute_supervised_bce_loss(logits, y, criterion)
                loss.backward()
                optimizer.step()

                probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                train_targets.extend(y.detach().cpu().numpy().tolist())
                train_probs.extend(probs.tolist())
                train_loss += loss.item()
                train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)
        scheduler.step(avg_train_loss)
        train_metrics_epoch = safe_binary_metrics(train_targets, train_probs)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "train_auc": float(train_metrics_epoch["AUC"]),
                "train_aucpr": float(train_metrics_epoch["AUCPR"]),
                "train_acc": float(train_metrics_epoch["ACC"]),
                "teacher_loss": float(train_teacher_loss / max(train_steps, 1)),
                "student_survival_loss": float(train_student_survival / max(train_steps, 1)),
                "student_repr_loss": float(train_student_repr / max(train_steps, 1)),
                "student_feature_loss": float(train_student_feature / max(train_steps, 1)),
            }
        )

    if model_name_l in {"distill_dyam"}:
        return student_model, history
    return model, history


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
                "phase": "selected_inner_model",
                "selected_hp_name": candidate["hp_name"],
            }
        )
        inner_run = wandb.init(
            project=wandb_project,
            group=f"outer_fold_{outer_fold_idx}",
            name=run_name,
            mode=wandb_mode,
            config=run_config,
            reinit="finish_previous",
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
            if model_name_l == "distill_dyam":
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
    radio_aggregation_method="mean",
    radio_pooling_kwargs=None,
    candidate_model_dir=None,
    retrain_outer=True,
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

        # Train all HPs across all inner folds, then select one robust HP for the whole outer fold.
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
            dfs_train_inner_prepared, radio_pooler_inner = _prepare_patient_level_modalities(
                dfs_train_inner_raw,
                patient_id_col=patient_id_col,
                radio_aggregation_method=radio_aggregation_method,
                fit_radio_pooler=True,
                radio_pooler=None,
                radio_pooling_kwargs=radio_pooling_kwargs,
                labels_df=inst_df_train_inner,
                label_col=label_col,
            )
            dfs_val_inner_prepared, _ = _prepare_patient_level_modalities(
                dfs_val_inner_raw,
                patient_id_col=patient_id_col,
                radio_aggregation_method=radio_aggregation_method,
                fit_radio_pooler=False,
                radio_pooler=radio_pooler_inner,
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
                if (not bool(retrain_outer)) and candidate_model_dir:
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
                        radio_pooler=radio_pooler_inner,
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

        robust_hp_rows = []
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
            robust_score = float(mean_auc - (0.5 * std_auc))

            robust_hp_rows.append(
                {
                    "hp_name": hp_name,
                    "hp_cfg": hp_cfg,
                    "candidates": candidates,
                    "mean_auc": mean_auc,
                    "std_auc": std_auc,
                    "mean_logloss": mean_logloss,
                    "robust_score": robust_score,
                }
            )

        best_hp_row = max(
            robust_hp_rows,
            key=lambda row: (float(row["robust_score"]), -float(row["mean_logloss"])),
        )
        selected_candidates = best_hp_row["candidates"]

        print(
            f"  Selected robust hp across inner folds: {best_hp_row['hp_name']} "
            f"(score={best_hp_row['robust_score']:.4f}, mean_AUC={best_hp_row['mean_auc']:.4f}, "
            f"std_AUC={best_hp_row['std_auc']:.4f}, mean_LOGLOSS={best_hp_row['mean_logloss']:.4f})"
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
                    "robust_selected_mean_AUC": float(best_hp_row["mean_auc"]),
                    "robust_selected_std_AUC": float(best_hp_row["std_auc"]),
                    "robust_selected_score": float(best_hp_row["robust_score"]),
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

            dfs_train_outer_prepared, radio_pooler_outer = _prepare_patient_level_modalities(
                dfs_train_outer_raw,
                patient_id_col=patient_id_col,
                radio_aggregation_method=radio_aggregation_method,
                fit_radio_pooler=True,
                radio_pooler=None,
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
                        "phase": "outer_train_refit",
                        "selected_hp_name": best_hp_row["hp_name"],
                        "outer_refit_epochs": int(refit_epochs),
                    }
                )
                outer_train_run = wandb.init(
                    project=wandb_project,
                    group=f"outer_fold_{outer_fold_idx}",
                    name=run_name,
                    mode=wandb_mode,
                    config=run_config,
                    reinit="finish_previous",
                )

                for hrow in outer_refit_history:
                    epoch_i = int(hrow["epoch"])
                    log_payload = {
                        "outer_train_model/train_loss": float(hrow["train_loss"]),
                        "outer_train_model/train_auc": float(hrow["train_auc"]),
                        "outer_train_model/train_aucpr": float(hrow["train_aucpr"]),
                        "outer_train_model/train_acc": float(hrow["train_acc"]),
                    }

                    if model_name_l == "distill_dyam":
                        log_payload.update(
                            {
                                "outer_train_model/teacher_loss": float(hrow["teacher_loss"]),
                                "outer_train_model/student_survival_loss": float(hrow["student_survival_loss"]),
                                "outer_train_model/student_repr_loss": float(hrow["student_repr_loss"]),
                                "outer_train_model/student_feature_loss": float(hrow["student_feature_loss"]),
                            }
                        )

                    outer_train_run.log(log_payload, step=epoch_i)

                outer_train_run.finish()

            dfs_outer_eval_prepared, _ = _prepare_patient_level_modalities(
                dfs_test_outer_raw,
                patient_id_col=patient_id_col,
                radio_aggregation_method=radio_aggregation_method,
                fit_radio_pooler=False,
                radio_pooler=radio_pooler_outer,
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

                y_true_outer, y_logits_outer, y_prob_outer, pids_outer, dyam_details_outer = _predict_model_probabilities(
                    model=outer_train_model,
                    data_loader=outer_eval_loader,
                    device=device,
                    bypass_mask=predict_bypass_mask,
                    collect_dyam_details=model_name_l in {"dyam", "distill_dyam"},
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
                        "ensemble_prob": float(y_prob_outer[patient_idx]),
                        "ensemble_pred_label": int(y_prob_outer[patient_idx] >= 0.5),
                    }
                    logit_value = float(y_logits_outer[patient_idx])
                    prob_value = float(y_prob_outer[patient_idx])
                    pred_label = int(logit_value >= 0.0)
                    row["inner_model_1_logit"] = logit_value
                    row["inner_model_1_prob"] = prob_value
                    row["inner_model_1_pred_label"] = pred_label
                    if model_name_l in {"dyam", "distill_dyam"} and dyam_details_outer is not None:
                        for modality_idx, modality_name in enumerate(modality_names):
                            row[f"inner_model_1_{modality_name}_alpha"] = float(
                                dyam_details_outer["alpha"][patient_idx, modality_idx]
                            )
                            row[f"inner_model_1_{modality_name}_R"] = float(
                                dyam_details_outer["R"][patient_idx, modality_idx]
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
                        "selected_inner_robust_score": float(best_hp_row["robust_score"]),
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
        else:
            print(
                f"  Outer test prediction will use the retained inner-fold ensemble "
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

                for candidate in selected_candidates:
                    bundle_path = candidate.get("bundle_path")
                    if not bundle_path:
                        raise RuntimeError(
                            "Missing candidate bundle path for outer-test ensemble prediction."
                        )

                    loaded_bundle = _load_candidate_bundle(bundle_path, device=device)

                    dfs_outer_eval_prepared, _ = _prepare_patient_level_modalities(
                        dfs_test_outer_raw,
                        patient_id_col=patient_id_col,
                        radio_aggregation_method=radio_aggregation_method,
                        fit_radio_pooler=False,
                        radio_pooler=loaded_bundle["radio_pooler"],
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
                        imputer_kwargs=imputer_kwargs,
                    )
                    outer_eval_loader = DataLoader(
                        outer_eval_ds,
                        batch_size=outer_eval_batch_size,
                        shuffle=False,
                        collate_fn=multimodal_collate,
                        drop_last=False,
                    )

                    y_true_outer, y_logits_outer, y_prob_outer, pids_outer, dyam_details_outer = _predict_model_probabilities(
                        model=loaded_bundle["model"],
                        data_loader=outer_eval_loader,
                        device=device,
                        bypass_mask=predict_bypass_mask,
                        collect_dyam_details=model_name_l in {"dyam", "distill_dyam"},
                        model_name=model_name_l,
                    )

                    if ref_y_true is None:
                        ref_y_true = y_true_outer
                        ref_pids = list(pids_outer)
                    else:
                        if not np.array_equal(ref_y_true, y_true_outer):
                            raise RuntimeError("Ensemble member predictions are misaligned on y_true.")
                        if ref_pids != list(pids_outer):
                            raise RuntimeError("Ensemble member predictions are misaligned on patient IDs.")

                    model_logits.append(y_logits_outer)
                    model_probs.append(y_prob_outer)
                    model_details.append(dyam_details_outer)

                    del loaded_bundle["model"]
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                stacked_probs = np.stack(model_probs, axis=0)
                ensemble_prob = np.mean(stacked_probs, axis=0)
                ensemble_pred_label = (ensemble_prob >= 0.5).astype(np.int32)
                outer_metrics = safe_binary_metrics(ref_y_true, ensemble_prob)

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
                        "ensemble_prob": float(ensemble_prob[patient_idx]),
                        "ensemble_pred_label": int(ensemble_pred_label[patient_idx]),
                    }

                    for model_idx, (logits_arr, probs_arr, details_arr) in enumerate(
                        zip(model_logits, model_probs, model_details),
                        1,
                    ):
                        row[f"inner_model_{model_idx}_logit"] = float(logits_arr[patient_idx])
                        row[f"inner_model_{model_idx}_prob"] = float(probs_arr[patient_idx])
                        row[f"inner_model_{model_idx}_pred_label"] = int(logits_arr[patient_idx] >= 0.0)
                        if model_name_l in {"dyam", "distill_dyam"} and details_arr is not None:
                            for modality_idx, modality_name in enumerate(modality_names):
                                row[f"inner_model_{model_idx}_{modality_name}_alpha"] = float(
                                    details_arr["alpha"][patient_idx, modality_idx]
                                )
                                row[f"inner_model_{model_idx}_{modality_name}_R"] = float(
                                    details_arr["R"][patient_idx, modality_idx]
                                )

                    per_patient_prediction_rows.append(row)

                test_prediction_rows.extend(per_patient_prediction_rows)

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
                        "selected_inner_robust_score": float(best_hp_row["robust_score"]),
                        "outer_refit_epochs": np.nan,
                        "outer_test_LOGLOSS": float(outer_metrics["LOGLOSS"]),
                        "outer_test_AUC": float(outer_metrics["AUC"]),
                        "outer_test_AUCPR": float(outer_metrics["AUCPR"]),
                        "outer_test_ACC": float(outer_metrics["ACC"]),
                        "outer_test_SEN": float(outer_metrics["SEN"]),
                        "outer_test_SP": float(outer_metrics["SP"]),
                        "outer_test_MCC": float(outer_metrics["MCC"]),
                    }
                )

            if candidate_model_dir:
                outer_fold_cache_dir = os.path.join(candidate_model_dir, f"outer_fold_{int(outer_fold_idx)}")
                shutil.rmtree(outer_fold_cache_dir, ignore_errors=True)

    return (
        pd.DataFrame(inner_eval_rows),
        pd.DataFrame(outer_results),
        pd.DataFrame(history_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(test_prediction_rows),
    )
