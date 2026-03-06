import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
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

# Function to build KNN reference pool dataset in the current scaler space
def _build_knn_reference_base_dataset(
    dfs_raw_full,
    scalers,
    inst_df_full,
    label_col,
    patient_id_col="patient",
):
    """Build full-dataset KNN reference pool in the current scaler space."""
    dfs_knn_ref_scaled = _transform_modalities_with_fitted_scalers(
        dfs_raw_full,
        scalers,
        patient_id_col=patient_id_col,
    )
    return MultimodalBaseDataset(
        dfs=dfs_knn_ref_scaled,
        label_df=inst_df_full,
        label_col=label_col,
        id_col=patient_id_col,
    )

# Function to predict on outer fold for each inner fold model
def _predict_model_probabilities(
    model,
    data_loader,
    device,
    model_name,
    imputation_method="zero",
):
    """Run one model on a loader and return y_true / probabilities."""
    model.eval()
    y_true = []
    y_prob = []
    pids = []
    model_name_l = normalize_model_name(model_name)
    bypass_mask = (
        model_name_l == "mlp"
        and str(imputation_method).strip().lower() == "knn"
    )

    with torch.no_grad():
        for Xs, present_mask, y, pid_batch in data_loader:
            Xs = [x.to(device) for x in Xs]
            present_mask = present_mask.to(device)

            model_mask = None if bypass_mask else present_mask
            logits = model(Xs, model_mask).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            y_prob.extend(probs.tolist())
            y_true.extend(y.cpu().numpy().tolist())
            pids.extend(pid_batch)

    return np.asarray(y_true), np.asarray(y_prob), list(pids)

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
):
    model_kwargs = model_kwargs or {}
    model_name_l = normalize_model_name(model_name)

    # For MLP with KNN imputation, we do not use masking since KNN already imputes missing modalities
    bypass_mask = (
        model_name_l == "mlp"
        and str(imputation_method).strip().lower() == "knn"
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
        weight_decay = 1e-3 if model_name_l == "healnet" else 1e-4
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler_patience = 3 if model_name_l == "healnet" else 5
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
    patience = 8 if model_name_l == "healnet" else 20

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
            }
        )

        epoch_score = (float(val_metrics_epoch["AUC"]), -float(avg_val_loss))
        if epoch_score > best_epoch_score:
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

    if best_model_state is not None:
        if model_name_l in {"distill_dyam"}:
            student_model.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)

    best_metrics = safe_binary_metrics(best_val_targets, best_val_probs)
    best_metrics["best_epoch"] = int(best_epoch)

    if model_name_l in {"distill_dyam"}:
        return student_model, history, best_metrics
    return model, history, best_metrics


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
    gpu_memory_fraction=1.0,
    missing_pattern_seed=0,
    patient_id_col="patient",
    wandb_enabled=False,
    wandb_project=None,
    wandb_mode="online",
    wandb_base_config=None,
    test_eval_setups=None,
):
    wandb_active = bool(wandb_enabled and wandb is not None)
    if wandb_enabled and wandb is None:
        print("wandb is not installed. Continuing without wandb logging.")

    # Train missingness is applied on both inner-train and inner-validation.
    apply_missing_train = float(getattr(train_missing_simulator, "missing_prob", 0.0)) > 0.0
    model_name_l = normalize_model_name(model_name)
    set_global_seed(seed, deterministic=True)

    # Test-time evaluation setups:
    # by default evaluate once with the provided train simulator.
    if test_eval_setups is None:
        eval_missing_prob = float(getattr(train_missing_simulator, "missing_prob", 0.0))
        eval_missing_location = str(getattr(train_missing_simulator, "missing_location", "global")).lower()
        test_eval_setups = [
            {
                "missing_prob": eval_missing_prob,
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
    device = select_device()
    if device.type == "cuda":
        # Optionally cap per-process CUDA memory usage.
        fraction = float(gpu_memory_fraction)
        if not (0.0 < fraction <= 1.0):
            raise ValueError("gpu_memory_fraction must be in (0, 1].")
        gpu_idx = device.index if device.index is not None else 0
        torch.cuda.set_per_process_memory_fraction(fraction, gpu_idx)
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Using device: cuda ({gpu_name}) | memory_fraction={fraction}")
    elif device.type == "mps":
        print("Using device: mps (Apple Metal)")
    else:
        print("Using device: cpu")

    inner_eval_rows = []
    history_rows = []
    outer_results = []
    split_rows = []

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

        # Select one best HP config per inner fold and keep the 5 trained models.
        selected_inner_members = []
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
            dfs_train_inner_scaled, dfs_val_inner_scaled, scalers_inner = fit_and_transform_modalities(
                dfs_train_inner_raw,
                dfs_val_inner_raw,
                id_col=patient_id_col,
            )

            # For KNN imputation, use a reference pool built from the full dataset
            # transformed with the current inner-fold scalers (split-independent pool).
            knn_reference_base_inner = None
            if str(imputation_method).strip().lower() == "knn":
                knn_reference_base_inner = _build_knn_reference_base_dataset(
                    dfs_raw_full=dfs,
                    scalers=scalers_inner,
                    inst_df_full=inst_df,
                    label_col=label_col,
                    patient_id_col=patient_id_col,
                )

            # Keep track of the best model and HP config for this inner fold
            best_inner_model = None
            best_inner_hp = None
            best_inner_metrics = None
            best_inner_scalers = None
            best_inner_history = None
            best_inner_score = (-np.inf, -np.inf)  # (AUC, -LOGLOSS)

            # Iterate over each HP config, train a model, and evaluate on inner-val fold
            for hp_idx, hp_cfg in enumerate(hp_configs):
                hp_name = hp_cfg["name"]

                # Inner validation belongs to outer-train, so it follows train missingness.
                train_loader, val_loader = build_loaders(
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
                    knn_reference_base_dataset=knn_reference_base_inner,
                )

                # Get model kwargs from this HP config
                if model_name_l in {"mlp"}:
                    model_kwargs = {
                        "modality_hidden_layers": hp_cfg["modality_hidden_layers"],
                        "fusion_hidden_dim": hp_cfg["fusion_hidden_dim"],
                        "fusion_hidden_layers": hp_cfg["fusion_hidden_layers"],
                        "dropout_p": hp_cfg["dropout"],
                    }
                elif model_name_l in {"dyam"}:
                    model_kwargs = {
                        "dropout_p": hp_cfg["dyam_dropout"],
                        "temperature": hp_cfg["dyam_temperature"],
                    }
                elif model_name_l in {"distill_dyam"}:
                    model_kwargs = {
                        "dropout_p": hp_cfg["dyam_dropout"],
                        "temperature": hp_cfg["dyam_temperature"],
                        "concat_masks_input": True,
                        "distill_alpha": hp_cfg["distill_alpha"],
                        "distill_beta": hp_cfg["distill_beta"],
                    }
                elif model_name_l in {"healnet"}:
                    model_kwargs = {
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
                else:
                    raise ValueError(
                        "Unsupported model "
                        f"'{model_name}'. Supported: mlp, dyam, distill_dyam, healnet"
                    )

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

                score = (float(best_metrics["AUC"]), -float(best_metrics["LOGLOSS"]))
                if score > best_inner_score:
                    best_inner_score = score
                    best_inner_model = model
                    best_inner_hp = hp_cfg
                    best_inner_metrics = best_metrics
                    best_inner_scalers = scalers_inner
                    best_inner_history = history

            if best_inner_model is None:
                raise RuntimeError(
                    f"No model selected for outer fold {outer_fold_idx}, inner fold {inner_fold_idx}."
                )
            print(
                f"  Inner fold {inner_fold_idx} best hp: {best_inner_hp['name']} "
                f"(AUC={best_inner_metrics['AUC']:.4f}, LOGLOSS={best_inner_metrics['LOGLOSS']:.4f}, "
                f"best_epoch={int(best_inner_metrics['best_epoch'])})"
            )

            # After iterating over all HP configs, save the best model and its scalers for this inner fold, along with its metrics and history.
            selected_inner_members.append(
                {
                    "model": best_inner_model,
                    "scalers": best_inner_scalers,
                }
            )
            selected_inner_rows.append(
                {
                    "inner_fold": inner_fold_idx,
                    "hp_name": best_inner_hp["name"],
                    **best_inner_hp,
                    "val_best_AUC": float(best_inner_metrics["AUC"]),
                    "val_best_LOGLOSS": float(best_inner_metrics["LOGLOSS"]),
                }
            )
            selected_inner_histories.append(best_inner_history)

        # Log averaged learning curve of the selected best inner-fold models for this outer fold
        inner_curve_run = None
        if wandb_active and selected_inner_histories:
            selected_hp_names = [row["hp_name"] for row in selected_inner_rows]
            train_missing_location = str((wandb_base_config or {}).get("train_missing_location", "na")).strip().lower()
            train_missing_prob = float((wandb_base_config or {}).get("train_missing_prob", 0.0))
            run_name = (
                f"trainloc{train_missing_location}_"
                f"trainprob{train_missing_prob:g}_"
                f"seed{seed}_"
                f"outer{outer_fold_idx}"
            )
            run_config = dict(wandb_base_config or {})
            run_config.update(
                {
                    "seed": seed,
                    "outer_fold": outer_fold_idx,
                    "phase": "avg_inner_curve_best_per_fold",
                    "selected_inner_hp_names": selected_hp_names,
                }
            )
            inner_curve_run = wandb.init(
                project=wandb_project,
                group=f"outer_fold_{outer_fold_idx}",
                name=run_name,
                mode=wandb_mode,
                config=run_config,
                reinit="finish_previous",
            )

            max_epoch = max(len(h) for h in selected_inner_histories)
            for epoch_i in range(1, max_epoch + 1):
                epoch_rows = [h[epoch_i - 1] for h in selected_inner_histories if len(h) >= epoch_i]
                if not epoch_rows:
                    continue

                train_losses = [float(r["train_loss"]) for r in epoch_rows]
                val_losses = [float(r["val_loss"]) for r in epoch_rows]
                val_aucs = [float(r["val_auc"]) for r in epoch_rows]

                inner_curve_run.log(
                    {
                        "avg_inner_best_models/train_loss_mean": float(np.mean(train_losses)),
                        "avg_inner_best_models/val_loss_mean": float(np.mean(val_losses)),
                        "avg_inner_best_models/val_auc_mean": float(np.mean(val_aucs)),
                    },
                    step=epoch_i,
                )

            inner_curve_run.finish()

        # Outer prediction = average predictions of selected inner models on outer-test data (20%).
        # Each model must evaluate on data transformed with its own inner-train scalers.
        inst_df_test_outer = filter_by_patients(inst_df, test_outer_ids, id_col=patient_id_col)
        dfs_test_outer_raw = {
            name: filter_by_patients(df, test_outer_ids, id_col=patient_id_col)
            for name, df in dfs.items()
        }

        outer_eval_batch_size = 1 if model_name_l == "healnet" else max(int(h["batch_size"]) for h in hp_configs)

        # Pre-build per-inner-model datasets in each model/scaler space once.
        outer_member_contexts = []
        for member in selected_inner_members:
            model = member["model"]
            scalers = member["scalers"]

            dfs_outer_eval_scaled = _transform_modalities_with_fitted_scalers(
                dfs_test_outer_raw,
                scalers,
                patient_id_col=patient_id_col,
            )
            outer_eval_base = MultimodalBaseDataset(
                dfs=dfs_outer_eval_scaled,
                label_df=inst_df_test_outer,
                label_col=label_col,
                id_col=patient_id_col,
            )

            knn_reference_base_outer = None
            if str(imputation_method).strip().lower() == "knn":
                knn_reference_base_outer = _build_knn_reference_base_dataset(
                    dfs_raw_full=dfs,
                    scalers=scalers,
                    inst_df_full=inst_df,
                    label_col=label_col,
                    patient_id_col=patient_id_col,
                )

            outer_member_contexts.append(
                {
                    "model": model,
                    "outer_eval_base": outer_eval_base,
                    "knn_reference_base": knn_reference_base_outer,
                }
            )

        for eval_setup in test_eval_setups:
            eval_simulator = eval_setup["simulator"]
            eval_missing_location = str(eval_setup["missing_location"]).lower()
            eval_missing_prob = float(eval_setup["missing_prob"])
            apply_missing_eval = eval_missing_prob > 0.0

            model_probs = []
            y_true_outer = None
            pids_outer = None

            for member_idx, member_ctx in enumerate(outer_member_contexts):
                model = member_ctx["model"]
                ensemble_eval_ds = MultimodalDatasetWithMissing(
                    base_dataset=member_ctx["outer_eval_base"],
                    simulator=eval_simulator,
                    apply_missing=apply_missing_eval,
                    imputation_method=imputation_method,
                    missing_pattern_seed=missing_pattern_seed,
                    knn_reference_base_dataset=member_ctx["knn_reference_base"],
                )
                ensemble_eval_loader = DataLoader(
                    ensemble_eval_ds,
                    batch_size=outer_eval_batch_size,
                    shuffle=False,
                    collate_fn=multimodal_collate,
                    drop_last=False,
                )

                y_true_i, y_prob_i, pids_i = _predict_model_probabilities(
                    model=model,
                    data_loader=ensemble_eval_loader,
                    device=device,
                    model_name=model_name,
                    imputation_method=imputation_method,
                )
                if y_true_outer is None:
                    y_true_outer = y_true_i
                    pids_outer = pids_i
                else:
                    if pids_i != pids_outer:
                        raise ValueError(
                            f"Outer-test patient ID order mismatch across inner models "
                            f"(outer_fold={outer_fold_idx}, eval_missing_location={eval_missing_location}, "
                            f"eval_missing_prob={eval_missing_prob}, inner_model_idx={member_idx + 1})."
                        )
                    if not np.array_equal(y_true_outer, y_true_i):
                        raise ValueError(
                            f"Outer-test y_true mismatch across inner models "
                            f"(outer_fold={outer_fold_idx}, eval_missing_location={eval_missing_location}, "
                            f"eval_missing_prob={eval_missing_prob}, inner_model_idx={member_idx + 1})."
                        )
                model_probs.append(y_prob_i)

            # Average probabilities across inner models to get ensemble prediction
            ensemble_prob = np.mean(np.stack(model_probs, axis=0), axis=0)
            outer_metrics = safe_binary_metrics(y_true_outer, ensemble_prob)

            outer_results.append(
                {
                    "outer_fold": outer_fold_idx,
                    "outer_eval_target": "test_outer",
                    "eval_missing_location": eval_missing_location,
                    "eval_missing_prob": eval_missing_prob,
                    "inner_models_count": len(selected_inner_members),
                    "selected_inner_hp_names": "|".join(r["hp_name"] for r in selected_inner_rows),
                    "selected_inner_mean_AUC": float(np.mean([r["val_best_AUC"] for r in selected_inner_rows])),
                    "selected_inner_mean_LOGLOSS": float(np.mean([r["val_best_LOGLOSS"] for r in selected_inner_rows])),
                    "outer_test_LOGLOSS": float(outer_metrics["LOGLOSS"]),
                    "outer_test_AUC": float(outer_metrics["AUC"]),
                    "outer_test_AUCPR": float(outer_metrics["AUCPR"]),
                    "outer_test_ACC": float(outer_metrics["ACC"]),
                    "outer_test_SEN": float(outer_metrics["SEN"]),
                    "outer_test_SP": float(outer_metrics["SP"]),
                    "outer_test_MCC": float(outer_metrics["MCC"]),
                }
            )

    return (
        pd.DataFrame(inner_eval_rows),
        pd.DataFrame(outer_results),
        pd.DataFrame(history_rows),
        pd.DataFrame(split_rows),
    )
