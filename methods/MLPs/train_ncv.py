import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from dataset import MLPDataset, MultimodalDatasetWithMissing, multimodal_collate
from models import MultimodalMLP
from utils import safe_binary_metrics, select_device, filter_by_patients, fit_and_transform_modalities
try:
    import wandb
except ImportError:
    wandb = None

# Function to build data loaders with missing data for training and evaluation
def _build_loaders(
    dfs_train_scaled,
    inst_df_train,
    dfs_eval_scaled,
    inst_df_eval,
    label_col,
    missing_simulator,
    batch_size,
    train_missing=True,
    eval_missing=False,
):
    train_base = MLPDataset(dfs=dfs_train_scaled, label_df=inst_df_train, label_col=label_col)
    eval_base = MLPDataset(dfs=dfs_eval_scaled, label_df=inst_df_eval, label_col=label_col)

    train_ds = MultimodalDatasetWithMissing(
        base_dataset=train_base,
        simulator=missing_simulator,
        apply_missing=train_missing,
    )
    eval_ds = MultimodalDatasetWithMissing(
        base_dataset=eval_base,
        simulator=missing_simulator,
        apply_missing=eval_missing,
    )

    n_train = len(train_ds)
    if n_train < 2:
        raise ValueError(
            f"Inner-train split has only {n_train} sample(s). "
            "At least 2 are required with BatchNorm."
        )
    # Drop only when the last batch would have size 1 (BatchNorm issue).
    drop_last_train = (n_train % batch_size) == 1

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multimodal_collate,
        drop_last=drop_last_train,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_collate,
        drop_last=False,
    )

    return train_loader, eval_loader

# Function to transform outer train with each inner train scaler
def _transform_modalities_with_fitted_scalers(dfs_raw, scalers):
    """Apply pre-fitted per-modality scalers to raw modality dataframes."""
    dfs_scaled = {}
    for name, df_raw in dfs_raw.items():
        if name not in scalers:
            raise ValueError(f"Missing scaler for modality '{name}'.")

        df_scaled = df_raw.copy()
        feats = [c for c in df_scaled.columns if c != "patient"]
        if len(df_scaled) > 0 and feats:
            values = df_scaled[feats].to_numpy(dtype=np.float32, copy=True)
            df_scaled.loc[:, feats] = scalers[name].transform(values).astype(np.float32)

        dfs_scaled[name] = df_scaled

    return dfs_scaled

# Train function with validation and early stopping
def train_model_with_validation(train_loader, val_loader, device, input_dims, epochs, lr, model_kwargs=None):
    model_kwargs = model_kwargs or {}

    model = MultimodalMLP(input_dims, **model_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_epoch = 1
    best_model_state = None
    best_val_targets = None
    best_val_probs = None

    early_stop = 0
    patience = 10

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for Xs, present_mask, y, _ in train_loader:
            Xs = [x.to(device) for x in Xs]
            present_mask = present_mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(Xs, present_mask).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for Xs, present_mask, y, _ in val_loader:
                Xs = [x.to(device) for x in Xs]
                present_mask = present_mask.to(device)
                y = y.to(device)

                logits = model(Xs, present_mask).squeeze(1)
                loss = criterion(logits, y)

                val_loss += loss.item()
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
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
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = float(avg_val_loss)
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_targets = np.asarray(val_targets)
            best_val_probs = np.asarray(val_probs)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    best_metrics = safe_binary_metrics(best_val_targets, best_val_probs)
    best_metrics["best_epoch"] = int(best_epoch)

    return model, history, best_metrics

# Function to predict on outer fold for each inner fold model
def _predict_model_probabilities(model, data_loader, device):
    """Run one model on a loader and return y_true / probabilities."""
    model.eval()
    y_true = []
    y_prob = []

    with torch.no_grad():
        for Xs, present_mask, y, _ in data_loader:
            Xs = [x.to(device) for x in Xs]
            present_mask = present_mask.to(device)
            logits = model(Xs, present_mask).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            y_prob.extend(probs.tolist())
            y_true.extend(y.cpu().numpy().tolist())

    return np.asarray(y_true), np.asarray(y_prob)

# Main nested cross-validation function
def nested_cv(
    dfs,
    inst_df,
    label_col,
    epochs,
    seed,
    hp_configs,
    missing_simulator,
    missing_scope="none",
    inner_splits=5,
    outer_splits=5,
    wandb_enabled=False,
    wandb_project=None,
    wandb_mode="online",
    wandb_base_config=None,
):
    wandb_active = bool(wandb_enabled and wandb is not None)
    if wandb_enabled and wandb is None:
        print("wandb is not installed. Continuing without wandb logging.")
    
    # Create boolean flags for missing modalities in train and/or test
    missing_scope = str(missing_scope).lower()
    if missing_scope not in {"train", "test", "both", "none"}:
        raise ValueError("missing_scope must be one of: train, test, both, none")
    apply_missing_train = missing_scope in {"train", "both"}
    apply_missing_eval = missing_scope in {"test", "both"}

    # Get patient IDs and labels from  inst_df
    patients = inst_df["patient"].values
    y = inst_df[label_col].values

    # Set up outer and inner cross-validation splits
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    # Input dims for each modality (removing patient column) and select device
    input_dims = [df.shape[1] - 1 for df in dfs.values()]
    device = select_device()

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
            split_rows.append({"outer_fold": outer_fold_idx, "split": "train_outer", "patient": int(pid)})
        for pid in test_outer_ids:
            split_rows.append({"outer_fold": outer_fold_idx, "split": "test_outer", "patient": int(pid)})

        # Filter inst_df for outer-train fold and get patient IDs and ys
        inst_df_train_outer = filter_by_patients(inst_df, train_outer_ids)
        patients_train_outer = inst_df_train_outer["patient"].values
        y_train_outer = inst_df_train_outer[label_col].values

        # Split dfs into outer-train data.
        dfs_train_outer_raw = {name: filter_by_patients(df, train_outer_ids) for name, df in dfs.items()}

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
            inst_df_train_inner = filter_by_patients(inst_df_train_outer, train_inner_ids)
            inst_df_val_inner = filter_by_patients(inst_df_train_outer, val_inner_ids)

            # Filter dfs for inner-train and inner-val folds, then scale features using only inner-train statistics
            dfs_train_inner_raw = {
                name: filter_by_patients(df, train_inner_ids) for name, df in dfs_train_outer_raw.items()
            }
            dfs_val_inner_raw = {
                name: filter_by_patients(df, val_inner_ids) for name, df in dfs_train_outer_raw.items()
            }
            dfs_train_inner_scaled, dfs_val_inner_scaled, scalers_inner = fit_and_transform_modalities(
                dfs_train_inner_raw, dfs_val_inner_raw
            )

            # Keep track of the best model and HP config for this inner fold
            best_inner_model = None
            best_inner_hp = None
            best_inner_metrics = None
            best_inner_scalers = None
            best_inner_history = None
            best_inner_score = (-np.inf, -np.inf)  # (AUC, -LOGLOSS)

            # Iterate over each HP config, train a model, and evaluate on inner-val fold
            for hp_cfg in hp_configs:
                hp_name = hp_cfg["name"]
                print(f"  Inner fold {inner_fold_idx} - tuning hp: {hp_name}")

                # Get train and val loaders according to missing scope, simulator and batch size
                train_loader, val_loader = _build_loaders(
                    inst_df_train=inst_df_train_inner,
                    inst_df_eval=inst_df_val_inner,
                    dfs_train_scaled=dfs_train_inner_scaled,
                    dfs_eval_scaled=dfs_val_inner_scaled,
                    label_col=label_col,
                    missing_simulator=missing_simulator,
                    batch_size=hp_cfg["batch_size"],
                    train_missing=apply_missing_train,
                    eval_missing=apply_missing_eval,
                )

                # Get model kwargs from this HP config
                model_kwargs = {
                    "modality_hidden_layers": hp_cfg["modality_hidden_layers"],
                    "fusion_hidden_dim": hp_cfg["fusion_hidden_dim"],
                    "fusion_hidden_layers": hp_cfg["fusion_hidden_layers"],
                    "dropout_p": hp_cfg["dropout"],
                }

                # Train the model and evaluate on inner-val fold
                model, history, best_metrics = train_model_with_validation(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    input_dims=input_dims,
                    epochs=epochs,
                    lr=hp_cfg["learning_rate"],
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
        if wandb_active and selected_inner_histories:
            selected_hp_names = [row["hp_name"] for row in selected_inner_rows]
            run_name = (
                f"avg_inner_best_seed{seed}_outer{outer_fold_idx}_"
                + "__".join(selected_hp_names)
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
            run = wandb.init(
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
                val_accs = [float(r["val_acc"]) for r in epoch_rows]

                run.log(
                    {
                        "avg_inner_best_models/train_loss_mean": float(np.mean(train_losses)),
                        "avg_inner_best_models/train_loss_std": float(np.std(train_losses)),
                        "avg_inner_best_models/val_loss_mean": float(np.mean(val_losses)),
                        "avg_inner_best_models/val_loss_std": float(np.std(val_losses)),
                        "avg_inner_best_models/val_auc_mean": float(np.mean(val_aucs)),
                        "avg_inner_best_models/val_auc_std": float(np.std(val_aucs)),
                        "avg_inner_best_models/val_acc_mean": float(np.mean(val_accs)),
                        "avg_inner_best_models/val_acc_std": float(np.std(val_accs)),
                        "cv/outer_fold": outer_fold_idx,
                        "cv/inner_folds_count": len(epoch_rows),
                    },
                    step=epoch_i,
                )

            run.summary["avg_inner_best_models/selected_inner_count"] = len(selected_inner_histories)
            run.summary["avg_inner_best_models/selected_inner_hp_names"] = "|".join(selected_hp_names)
            run.finish()

        # Outer prediction = average predictions of selected inner models on outer-train data (80%).
        # Each model must evaluate on data transformed with its own inner-train scalers.
        outer_eval_batch_size = max(int(h["batch_size"]) for h in hp_configs)
        model_probs = []
        y_true_outer = None
        
        for member in selected_inner_members:
            model = member["model"]
            scalers = member["scalers"]

            dfs_outer_eval_scaled = _transform_modalities_with_fitted_scalers(
                dfs_train_outer_raw,
                scalers,
            )
            ensemble_eval_base = MLPDataset(
                dfs=dfs_outer_eval_scaled,
                label_df=inst_df_train_outer,
                label_col=label_col,
            )
            ensemble_eval_ds = MultimodalDatasetWithMissing(
                base_dataset=ensemble_eval_base,
                simulator=missing_simulator,
                apply_missing=apply_missing_eval,
            )
            ensemble_eval_loader = DataLoader(
                ensemble_eval_ds,
                batch_size=outer_eval_batch_size,
                shuffle=False,
                collate_fn=multimodal_collate,
                drop_last=False,
            )

            y_true_i, y_prob_i = _predict_model_probabilities(model, ensemble_eval_loader, device)
            if y_true_outer is None:
                y_true_outer = y_true_i
            model_probs.append(y_prob_i)

        
        # Average probabilities across inner models to get ensemble prediction
        ensemble_prob = np.mean(np.stack(model_probs, axis=0), axis=0)
        outer_metrics = safe_binary_metrics(y_true_outer, ensemble_prob)
        outer_results.append(
            {
                "outer_fold": outer_fold_idx,
                "outer_eval_target": "train_outer",
                "inner_models_count": len(selected_inner_members),
                "selected_inner_hp_names": "|".join(r["hp_name"] for r in selected_inner_rows),
                "selected_inner_mean_AUC": float(np.mean([r["val_best_AUC"] for r in selected_inner_rows])),
                "selected_inner_mean_LOGLOSS": float(np.mean([r["val_best_LOGLOSS"] for r in selected_inner_rows])),
                "outer_train_LOGLOSS": float(outer_metrics["LOGLOSS"]),
                "outer_train_AUC": float(outer_metrics["AUC"]),
                "outer_train_AUCPR": float(outer_metrics["AUCPR"]),
                "outer_train_ACC": float(outer_metrics["ACC"]),
                "outer_train_SEN": float(outer_metrics["SEN"]),
                "outer_train_SP": float(outer_metrics["SP"]),
                "outer_train_MCC": float(outer_metrics["MCC"]),
            }
        )

    return (
        pd.DataFrame(inner_eval_rows),
        pd.DataFrame(outer_results),
        pd.DataFrame(history_rows),
        pd.DataFrame(split_rows),
    )
