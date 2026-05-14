import copy
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from dataset import MultimodalDatasetWithMissing, multimodal_collate
from models import learn_priors as learn_smil_priors
from models import meta_train_step as smil_meta_train_step
from utils import (
    build_model,
    compute_survival_loss_from_logits,
    normalize_model_name,
    normalize_task_type,
    primary_metric_name,
    safe_binary_metrics,
    safe_task_metrics,
    set_global_seed,
    survival_logits_to_outputs,
)


def get_model_init_kwargs(model_name, model_kwargs=None):
    """Keep only constructor kwargs needed to instantiate the model itself."""
    model_kwargs = dict(model_kwargs or {})
    model_name_l = normalize_model_name(model_name)
    if model_name_l in {"dipam", "di_mmlp"}:
        return {
            key: value
            for key, value in model_kwargs.items()
            if key not in {"distill_alpha", "distill_beta"}
        }
    if model_name_l in {"smil_e"}:
        return {
            key: value
            for key, value in model_kwargs.items()
            if key not in {"meta_inner_lr", "meta_val_fraction", "meta_inner_steps"}
        }
    return model_kwargs


def _loader_batch_size(loader, fallback=16):
    if getattr(loader, "batch_size", None) is not None:
        return int(loader.batch_size)
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None and getattr(batch_sampler, "batch_size", None) is not None:
        return int(batch_sampler.batch_size)
    return int(fallback)


def _split_meta_indices(base_dataset, seed, meta_val_fraction=0.2):
    n = len(base_dataset)
    if n < 2:
        return np.arange(n), np.arange(n)

    labels = (
        base_dataset.label_df.loc[base_dataset.patient_ids, base_dataset.label_col]
        .to_numpy()
        .astype(np.int64, copy=False)
    )

    meta_val_size = max(1, int(round(n * float(meta_val_fraction))))
    if meta_val_size >= n:
        meta_val_size = n - 1

    unique, counts = np.unique(labels, return_counts=True)
    can_stratify = unique.size > 1 and np.all(counts >= 2)

    indices = np.arange(n)
    if can_stratify:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=meta_val_size,
            random_state=int(seed),
        )
        train_idx, val_idx = next(splitter.split(indices, labels))
        return train_idx, val_idx

    rng = np.random.default_rng(int(seed))
    shuffled = rng.permutation(indices)
    val_idx = np.sort(shuffled[:meta_val_size])
    train_idx = np.sort(shuffled[meta_val_size:])
    return train_idx, val_idx


def _move_batch_to_device(batch, device):
    Xs, present_mask, y, _ = batch
    Xs = [x.to(device) for x in Xs]
    present_mask = present_mask.to(device)
    if isinstance(y, dict):
        y = {
            "event_time": y["event_time"].to(device),
            "event": y["event"].to(device),
            "censorship": y["censorship"].to(device),
            "y_disc": y["y_disc"].to(device),
        }
    else:
        y = y.to(device)
    return Xs, present_mask, y


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
    for modality_idx in range(n_modalities):
        xs_full.append(torch.stack([row[modality_idx] for row in xs_rows], dim=0).to(device))
    if isinstance(ys[0], dict):
        y_full = {
            "event_time": torch.stack([item["event_time"] for item in ys], dim=0).to(device=device),
            "event": torch.stack([item["event"] for item in ys], dim=0).to(device=device),
            "censorship": torch.stack([item["censorship"] for item in ys], dim=0).to(device=device),
            "y_disc": torch.stack([item["y_disc"] for item in ys], dim=0).to(device=device),
        }
    else:
        y_full = torch.stack(ys, dim=0).to(dtype=torch.float32, device=device)
    full_mask = torch.ones((len(pid_batch), n_modalities), dtype=torch.bool, device=device)
    return xs_full, full_mask, y_full


def _task_type(task_config):
    return normalize_task_type((task_config or {}).get("task_type", "binary_classification"))


def _prepare_logits_for_task(logits):
    if logits.ndim == 2 and logits.size(1) == 1:
        return logits.squeeze(1)
    return logits


def _compute_supervised_task_loss(logits, targets, task_config, bce_criterion):
    if _task_type(task_config) == "survival":
        return compute_survival_loss_from_logits(
            logits=_prepare_logits_for_task(logits),
            y_disc=targets["y_disc"],
            censorship=targets["censorship"],
            loss_name=str((task_config or {}).get("survival_loss", "nll")).strip().lower(),
        )
    return bce_criterion(_prepare_logits_for_task(logits), targets)


def _accumulate_eval_batch(task_config, logits, targets, probs_or_logits_store):
    task_type = _task_type(task_config)
    logits_prepared = _prepare_logits_for_task(logits)
    if task_type == "survival":
        probs_or_logits_store["logits"].append(logits_prepared.detach().cpu().numpy())
        probs_or_logits_store["event_times"].extend(targets["event_time"].detach().cpu().numpy().tolist())
        probs_or_logits_store["event_observed"].extend(targets["event"].detach().cpu().numpy().tolist())
        probs_or_logits_store["censorship"].extend(targets["censorship"].detach().cpu().numpy().tolist())
        probs_or_logits_store["y_disc"].extend(targets["y_disc"].detach().cpu().numpy().tolist())
        return
    probs = torch.sigmoid(logits_prepared).detach().cpu().numpy().reshape(-1)
    probs_or_logits_store["probs"].extend(probs.tolist())
    probs_or_logits_store["y_true"].extend(targets.detach().cpu().numpy().tolist())


def _finalize_task_metrics(task_config, store):
    task_type = _task_type(task_config)
    if task_type == "survival":
        logits = np.concatenate(store["logits"], axis=0) if store["logits"] else np.zeros((0, 0), dtype=np.float32)
        return safe_task_metrics(
            task_config,
            event_times=np.asarray(store["event_times"], dtype=np.float32),
            event_observed=np.asarray(store["event_observed"], dtype=np.int64),
            censorship=np.asarray(store["censorship"], dtype=np.float32),
            y_disc=np.asarray(store["y_disc"], dtype=np.int64),
            logits=logits,
        )
    return safe_task_metrics(
        task_config,
        y_true=np.asarray(store["y_true"], dtype=np.int64),
        y_prob=np.asarray(store["probs"], dtype=np.float32),
    )


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
    task_config,
):
    loss_survival = _compute_supervised_task_loss(student_logits, targets, task_config, bce_criterion)
    loss_repr = repr_criterion(student_repr, teacher_repr)
    loss_feature = feat_criterion(student_logits, teacher_logits)
    total = loss_survival + (alpha_repr * loss_repr) + (beta_feat * loss_feature)
    return total, loss_survival, loss_repr, loss_feature


def _extract_distill_outputs(model_name, model_out):
    model_name_l = normalize_model_name(model_name)
    if model_name_l in {"dipam"}:
        logits = _prepare_logits_for_task(model_out[0])
        attn_weights = model_out[1]
        risk_tensor = model_out[3]
        repr_vec = (attn_weights.unsqueeze(-1) * risk_tensor).reshape(risk_tensor.shape[0], -1)
        return logits, repr_vec
    if model_name_l in {"mlp", "di_mmlp"}:
        logits = _prepare_logits_for_task(model_out[0])
        repr_vec = model_out[1]
        return logits, repr_vec
    raise ValueError(f"Unsupported distillation model '{model_name_l}'.")


def _normalize_scheduler_type(scheduler_type):
    value = str(scheduler_type or "cosine_annealing").strip().lower()
    if value not in {"cosine_annealing", "reduce_lr_on_plateau"}:
        raise ValueError(
            f"Unsupported scheduler_type '{scheduler_type}'. Supported: "
            "cosine_annealing, reduce_lr_on_plateau."
        )
    return value


def _build_scheduler(optimizer, epochs, min_lr, scheduler_type="cosine_annealing", lr_patience=5):
    scheduler_type = _normalize_scheduler_type(scheduler_type)
    if scheduler_type == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=float(min_lr),
        )
    if scheduler_type == "reduce_lr_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(int(lr_patience), 0),
            min_lr=float(min_lr),
        )
    raise ValueError(f"Unsupported scheduler_type '{scheduler_type}'.")


def _step_scheduler(scheduler, scheduler_type="cosine_annealing", metric=None):
    scheduler_type = _normalize_scheduler_type(scheduler_type)
    if scheduler_type == "cosine_annealing":
        scheduler.step()
        return
    if metric is None:
        raise ValueError("ReduceLROnPlateau requires a metric value for scheduler.step(metric).")
    scheduler.step(float(metric))


def train_smil_e_with_meta_learning(
    train_loader,
    val_loader,
    device,
    input_dims,
    epochs,
    lr,
    model_kwargs=None,
    train_seed=0,
    weight_decay=1e-4,
    early_stopping_patience=8,
    min_lr=1e-6,
    scheduler_type="cosine_annealing",
    lr_patience=5,
    task_config=None,
):
    """Train SMIL-E with a SMIL-style meta loop fully contained in inner-train."""
    min_best_epoch = min(5, int(epochs))
    model_kwargs = dict(model_kwargs or {})
    inner_steps = int(model_kwargs.pop("meta_inner_steps", 1))
    inner_lr = float(model_kwargs.pop("meta_inner_lr", 1e-2))
    meta_val_fraction = float(model_kwargs.pop("meta_val_fraction", 0.2))

    model = build_model("smil_e", input_dims, model_kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    scheduler = _build_scheduler(
        optimizer,
        epochs=epochs,
        min_lr=min_lr,
        scheduler_type=scheduler_type,
        lr_patience=lr_patience,
    )

    train_dataset = train_loader.dataset
    base_dataset = train_dataset.base_dataset
    meta_train_idx, meta_val_idx = _split_meta_indices(
        base_dataset,
        seed=int(train_seed),
        meta_val_fraction=meta_val_fraction,
    )

    meta_train_incomplete_ds = Subset(train_dataset, meta_train_idx.tolist())
    meta_val_incomplete_ds = Subset(train_dataset, meta_val_idx.tolist())

    meta_complete_base_ds = MultimodalDatasetWithMissing(
        base_dataset=base_dataset,
        simulator=train_dataset.simulator,
        apply_missing=False,
        imputation_method="zero",
        missing_pattern_seed=train_dataset.missing_pattern_seed,
    )
    meta_val_complete_ds = Subset(meta_complete_base_ds, meta_val_idx.tolist())

    meta_batch_size = _loader_batch_size(train_loader, fallback=16)
    meta_train_loader = DataLoader(
        meta_train_incomplete_ds,
        batch_size=min(meta_batch_size, max(len(meta_train_incomplete_ds), 1)),
        shuffle=True,
        collate_fn=multimodal_collate,
        drop_last=False,
    )
    meta_val_incomplete_loader = DataLoader(
        meta_val_incomplete_ds,
        batch_size=min(meta_batch_size, max(len(meta_val_incomplete_ds), 1)),
        shuffle=False,
        collate_fn=multimodal_collate,
        drop_last=False,
    )
    meta_val_complete_loader = DataLoader(
        meta_val_complete_ds,
        batch_size=min(meta_batch_size, max(len(meta_val_complete_ds), 1)),
        shuffle=False,
        collate_fn=multimodal_collate,
        drop_last=False,
    )

    priors = learn_smil_priors(
        base_dataset=base_dataset,
        encoders=model.encoders,
        num_modalities=model.num_modalities,
        num_priors=model.num_priors,
        device=device,
    ).to(device)
    model.set_priors(priors)

    best_metric_key = primary_metric_name(task_config)
    best_val_score = -np.inf
    best_stop_loss = np.inf
    best_epoch = 1
    best_model_state = None
    best_val_targets = None
    best_val_probs = None
    early_stop = 0
    patience = int(early_stopping_patience)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        meta_train_losses = []
        meta_val_losses = []
        meta_align_fusion = []
        meta_align_hidden = []
        meta_val_ce = []

        incomplete_val_cycle = cycle(meta_val_incomplete_loader)
        complete_val_cycle = cycle(meta_val_complete_loader)

        for incomplete_train_batch in meta_train_loader:
            incomplete_val_batch = next(incomplete_val_cycle)
            complete_val_batch = next(complete_val_cycle)

            stats = smil_meta_train_step(
                model=model,
                optimizer=optimizer,
                incomplete_train_batch=_move_batch_to_device(incomplete_train_batch, device),
                incomplete_val_batch=_move_batch_to_device(incomplete_val_batch, device),
                complete_val_batch=_move_batch_to_device(complete_val_batch, device),
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                alpha=model.alpha,
                beta=model.beta,
                task_config=task_config,
            )
            meta_train_losses.append(float(stats["meta_train_loss"]))
            meta_val_losses.append(float(stats["meta_val_loss"]))
            meta_align_fusion.append(float(stats["align_fusion"]))
            meta_align_hidden.append(float(stats["align_hidden"]))
            meta_val_ce.append(float(stats["ce_noise"]))

        avg_meta_train_loss = float(np.mean(meta_train_losses)) if meta_train_losses else 0.0
        avg_meta_val_loss = float(np.mean(meta_val_losses)) if meta_val_losses else 0.0
        avg_align_fusion = float(np.mean(meta_align_fusion)) if meta_align_fusion else 0.0
        avg_align_hidden = float(np.mean(meta_align_hidden)) if meta_align_hidden else 0.0
        avg_meta_val_ce = float(np.mean(meta_val_ce)) if meta_val_ce else 0.0

        model.eval()
        val_loss = 0.0
        val_store = {
            "probs": [],
            "y_true": [],
            "logits": [],
            "event_times": [],
            "event_observed": [],
            "censorship": [],
            "y_disc": [],
        }

        with torch.no_grad():
            for Xs, present_mask, y, _ in val_loader:
                Xs = [x.to(device) for x in Xs]
                present_mask = present_mask.to(device)
                if isinstance(y, dict):
                    y = {k: v.to(device) for k, v in y.items()}
                else:
                    y = y.to(device)

                logits_out = model(
                    Xs,
                    present_mask,
                    mode="incomplete",
                    meta_train=False,
                )
                logits = _prepare_logits_for_task(logits_out)
                loss = _compute_supervised_task_loss(logits, y, task_config, criterion)

                val_loss += float(loss.item())
                _accumulate_eval_batch(task_config, logits, y, val_store)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        _step_scheduler(scheduler, scheduler_type=scheduler_type, metric=avg_val_loss)

        val_metrics_epoch = _finalize_task_metrics(task_config, val_store)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_meta_train_loss),
                "val_loss": float(avg_val_loss),
                "val_auc": float(val_metrics_epoch.get("AUC", 0.0)),
                "val_aucpr": float(val_metrics_epoch.get("AUCPR", 0.0)),
                "val_acc": float(val_metrics_epoch.get("ACC", 0.0)),
                "val_cindex": float(val_metrics_epoch.get("CINDEX", 0.0)),
                "teacher_loss": 0.0,
                "student_survival_loss": 0.0,
                "student_repr_loss": 0.0,
                "student_feature_loss": 0.0,
                "smil_meta_train_loss": float(avg_meta_train_loss),
                "smil_meta_val_loss": float(avg_meta_val_loss),
                "smil_meta_val_ce": float(avg_meta_val_ce),
                "smil_align_fusion": float(avg_align_fusion),
                "smil_align_hidden": float(avg_align_hidden),
            }
        )

        if epoch >= min_best_epoch:
            current_score = float(val_metrics_epoch[best_metric_key])
            if current_score > best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                best_val_targets = copy.deepcopy(val_store)

            if float(avg_val_loss) < best_stop_loss:
                best_stop_loss = float(avg_val_loss)
                early_stop = 0
            else:
                early_stop += 1
                if early_stop >= patience:
                    break

    if best_model_state is None:
        raise RuntimeError(
            f"No best epoch was selected. Check epochs={epochs} and min_best_epoch={min_best_epoch}."
        )

    model.load_state_dict(best_model_state)
    best_metrics = _finalize_task_metrics(task_config, best_val_targets)
    best_metrics["best_epoch"] = int(best_epoch)
    return model, history, best_metrics


def train_smil_e_on_full_dataset_with_meta_learning(
    train_loader,
    device,
    input_dims,
    epochs,
    lr,
    model_kwargs=None,
    train_seed=0,
    weight_decay=1e-4,
    min_lr=1e-6,
    scheduler_type="cosine_annealing",
    lr_patience=5,
    task_config=None,
):
    """Train SMILe on the full outer-train split with an internal meta split."""
    model_kwargs = dict(model_kwargs or {})
    inner_steps = int(model_kwargs.pop("meta_inner_steps", 1))
    inner_lr = float(model_kwargs.pop("meta_inner_lr", 1e-3))
    meta_val_fraction = float(model_kwargs.pop("meta_val_fraction", 0.25))
    min_best_epoch = min(5, int(epochs))

    model = build_model("smil_e", input_dims, model_kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    scheduler = _build_scheduler(
        optimizer,
        epochs=epochs,
        min_lr=min_lr,
        scheduler_type=scheduler_type,
        lr_patience=lr_patience,
    )

    train_dataset = train_loader.dataset
    base_dataset = train_dataset.base_dataset
    meta_train_idx, meta_val_idx = _split_meta_indices(
        base_dataset,
        seed=int(train_seed),
        meta_val_fraction=meta_val_fraction,
    )

    meta_train_incomplete_ds = Subset(train_dataset, meta_train_idx.tolist())
    meta_val_incomplete_ds = Subset(train_dataset, meta_val_idx.tolist())
    meta_complete_base_ds = MultimodalDatasetWithMissing(
        base_dataset=base_dataset,
        simulator=train_dataset.simulator,
        apply_missing=False,
        imputation_method="zero",
        missing_pattern_seed=train_dataset.missing_pattern_seed,
    )
    meta_val_complete_ds = Subset(meta_complete_base_ds, meta_val_idx.tolist())

    meta_batch_size = _loader_batch_size(train_loader, fallback=16)
    meta_train_loader = DataLoader(
        meta_train_incomplete_ds,
        batch_size=min(meta_batch_size, max(len(meta_train_incomplete_ds), 1)),
        shuffle=True,
        collate_fn=multimodal_collate,
        drop_last=False,
    )
    meta_val_incomplete_loader = DataLoader(
        meta_val_incomplete_ds,
        batch_size=min(meta_batch_size, max(len(meta_val_incomplete_ds), 1)),
        shuffle=False,
        collate_fn=multimodal_collate,
        drop_last=False,
    )
    meta_val_complete_loader = DataLoader(
        meta_val_complete_ds,
        batch_size=min(meta_batch_size, max(len(meta_val_complete_ds), 1)),
        shuffle=False,
        collate_fn=multimodal_collate,
        drop_last=False,
    )

    priors = learn_smil_priors(
        base_dataset=base_dataset,
        encoders=model.encoders,
        num_modalities=model.num_modalities,
        num_priors=model.num_priors,
        device=device,
    ).to(device)
    model.set_priors(priors)

    best_meta_val_loss = np.inf
    best_model_state = None
    history = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        meta_train_losses = []
        meta_val_losses = []
        meta_align_fusion = []
        meta_align_hidden = []
        meta_val_ce = []

        incomplete_val_cycle = cycle(meta_val_incomplete_loader)
        complete_val_cycle = cycle(meta_val_complete_loader)

        for incomplete_train_batch in meta_train_loader:
            incomplete_val_batch = next(incomplete_val_cycle)
            complete_val_batch = next(complete_val_cycle)

            stats = smil_meta_train_step(
                model=model,
                optimizer=optimizer,
                incomplete_train_batch=_move_batch_to_device(incomplete_train_batch, device),
                incomplete_val_batch=_move_batch_to_device(incomplete_val_batch, device),
                complete_val_batch=_move_batch_to_device(complete_val_batch, device),
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                alpha=model.alpha,
                beta=model.beta,
                task_config=task_config,
            )
            meta_train_losses.append(float(stats["meta_train_loss"]))
            meta_val_losses.append(float(stats["meta_val_loss"]))
            meta_align_fusion.append(float(stats["align_fusion"]))
            meta_align_hidden.append(float(stats["align_hidden"]))
            meta_val_ce.append(float(stats["ce_noise"]))

        avg_meta_train_loss = float(np.mean(meta_train_losses)) if meta_train_losses else 0.0
        avg_meta_val_loss = float(np.mean(meta_val_losses)) if meta_val_losses else 0.0
        avg_align_fusion = float(np.mean(meta_align_fusion)) if meta_align_fusion else 0.0
        avg_align_hidden = float(np.mean(meta_align_hidden)) if meta_align_hidden else 0.0
        avg_meta_val_ce = float(np.mean(meta_val_ce)) if meta_val_ce else 0.0

        model.eval()
        train_loss = 0.0
        train_store = {
            "probs": [],
            "y_true": [],
            "logits": [],
            "event_times": [],
            "event_observed": [],
            "censorship": [],
            "y_disc": [],
        }
        with torch.no_grad():
            for Xs, present_mask, y, _ in train_loader:
                Xs = [x.to(device) for x in Xs]
                present_mask = present_mask.to(device)
                if isinstance(y, dict):
                    y = {k: v.to(device) for k, v in y.items()}
                else:
                    y = y.to(device)
                logits_out = model(Xs, present_mask, mode="incomplete", meta_train=False)
                logits = _prepare_logits_for_task(logits_out)
                loss = _compute_supervised_task_loss(logits, y, task_config, criterion)
                train_loss += float(loss.item())
                _accumulate_eval_batch(task_config, logits, y, train_store)

        avg_train_loss = train_loss / max(len(train_loader), 1)
        _step_scheduler(scheduler, scheduler_type=scheduler_type, metric=avg_meta_val_loss)
        train_metrics_epoch = _finalize_task_metrics(task_config, train_store)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "train_auc": float(train_metrics_epoch.get("AUC", 0.0)),
                "train_aucpr": float(train_metrics_epoch.get("AUCPR", 0.0)),
                "train_acc": float(train_metrics_epoch.get("ACC", 0.0)),
                "train_cindex": float(train_metrics_epoch.get("CINDEX", 0.0)),
                "teacher_loss": 0.0,
                "student_survival_loss": 0.0,
                "student_repr_loss": 0.0,
                "student_feature_loss": 0.0,
                "smil_meta_train_loss": float(avg_meta_train_loss),
                "smil_meta_val_loss": float(avg_meta_val_loss),
                "smil_meta_val_ce": float(avg_meta_val_ce),
                "smil_align_fusion": float(avg_align_fusion),
                "smil_align_hidden": float(avg_align_hidden),
            }
        )

        if epoch >= min_best_epoch and avg_meta_val_loss < best_meta_val_loss:
            best_meta_val_loss = avg_meta_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    return model, history


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
    weight_decay=1e-4,
    early_stopping_patience=20,
    min_lr=1e-6,
    scheduler_type="cosine_annealing",
    lr_patience=5,
    task_config=None,
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
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            min_lr=min_lr,
            scheduler_type=scheduler_type,
            lr_patience=lr_patience,
            task_config=task_config,
        )

    bypass_mask = (
        model_name_l == "mlp"
        and str(imputation_method).strip().lower() in {"knn", "vae"}
    )

    criterion = nn.BCEWithLogitsLoss()
    best_metric_key = primary_metric_name(task_config)

    if model_name_l in {"dipam", "di_mmlp"}:
        student_kwargs = get_model_init_kwargs(model_name_l, model_kwargs)
        distill_alpha = float(model_kwargs.get("distill_alpha", 1.0))
        distill_beta = float(model_kwargs.get("distill_beta", 0.3))
        if model_name_l == "di_mmlp":
            teacher_model = build_model("mlp", input_dims, student_kwargs).to(device)
        else:
            teacher_kwargs = {
                "dropout_p": float(model_kwargs.get("dropout_p", 0.4)),
                "temperature": float(model_kwargs.get("temperature", 2.0)),
            }
            teacher_model = build_model("pam", input_dims, teacher_kwargs).to(device)
        student_model = build_model(model_name_l, input_dims, student_kwargs).to(device)
        teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=lr, weight_decay=float(weight_decay))
        student_optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=float(weight_decay))
        scheduler = _build_scheduler(
            student_optimizer,
            epochs=epochs,
            min_lr=min_lr,
            scheduler_type=scheduler_type,
            lr_patience=lr_patience,
        )
        repr_criterion = nn.MSELoss()
        feat_criterion = nn.MSELoss()
        teacher_base_dataset = train_loader.dataset.base_dataset
    else:
        model = build_model(model_name, input_dims, model_kwargs).to(device)
        if model_name_l in {"smil_e"}:
            base_dataset = train_loader.dataset.base_dataset
            priors = learn_smil_priors(
                base_dataset=base_dataset,
                encoders=model.encoders,
                num_modalities=model.num_modalities,
                num_priors=model.num_priors,
                device=device,
            ).to(device)
            model.set_priors(priors)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))
        scheduler = _build_scheduler(
            optimizer,
            epochs=epochs,
            min_lr=min_lr,
            scheduler_type=scheduler_type,
            lr_patience=lr_patience,
        )

    best_epoch_score = (-np.inf, -np.inf)
    best_epoch = 1
    best_model_state = None
    best_val_targets = None
    best_val_probs = None
    early_stop = 0
    patience = int(early_stopping_patience)
    history = []

    for epoch in range(1, epochs + 1):
        if model_name_l in {"dipam", "di_mmlp"}:
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
            if isinstance(y, dict):
                y = {k: v.to(device) for k, v in y.items()}
            else:
                y = y.to(device)

            if model_name_l in {"dipam", "di_mmlp"}:
                Xs_teacher, teacher_mask, y_teacher = _build_full_batch_from_patient_ids(
                    teacher_base_dataset,
                    pids,
                    device=device,
                )

                teacher_optimizer.zero_grad()
                teacher_out = teacher_model(Xs_teacher, teacher_mask, return_aux=True)
                teacher_logits, teacher_repr = _extract_distill_outputs(
                    "di_mmlp" if model_name_l == "di_mmlp" else "dipam",
                    teacher_out,
                )
                teacher_loss = _compute_supervised_task_loss(teacher_logits, y_teacher, task_config, criterion)
                teacher_loss.backward()
                teacher_optimizer.step()

                student_optimizer.zero_grad()
                student_out = student_model(Xs, present_mask, return_aux=True)
                student_logits, student_repr = _extract_distill_outputs(model_name_l, student_out)
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
                    task_config=task_config,
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
                logits = _prepare_logits_for_task(logits_out)
                loss = _compute_supervised_task_loss(logits, y, task_config, criterion)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)
        avg_teacher_loss = train_teacher_loss / max(train_steps, 1)
        avg_student_survival = train_student_survival / max(train_steps, 1)
        avg_student_repr = train_student_repr / max(train_steps, 1)
        avg_student_feature = train_student_feature / max(train_steps, 1)

        if model_name_l in {"dipam", "di_mmlp"}:
            student_model.eval()
        else:
            model.eval()
        val_loss = 0.0
        val_store = {
            "probs": [],
            "y_true": [],
            "logits": [],
            "event_times": [],
            "event_observed": [],
            "censorship": [],
            "y_disc": [],
        }

        with torch.no_grad():
            for Xs, present_mask, y, _ in val_loader:
                Xs = [x.to(device) for x in Xs]
                present_mask = present_mask.to(device)
                if isinstance(y, dict):
                    y = {k: v.to(device) for k, v in y.items()}
                else:
                    y = y.to(device)

                if model_name_l in {"dipam", "di_mmlp"}:
                    logits = _prepare_logits_for_task(student_model(Xs, present_mask))
                    loss = _compute_supervised_task_loss(logits, y, task_config, criterion)
                else:
                    model_mask = None if bypass_mask else present_mask
                    logits = _prepare_logits_for_task(model(Xs, model_mask))
                    loss = _compute_supervised_task_loss(logits, y, task_config, criterion)

                val_loss += loss.item()
                _accumulate_eval_batch(task_config, logits, y, val_store)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        _step_scheduler(scheduler, scheduler_type=scheduler_type, metric=avg_val_loss)

        val_metrics_epoch = _finalize_task_metrics(task_config, val_store)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "val_auc": float(val_metrics_epoch.get("AUC", 0.0)),
                "val_aucpr": float(val_metrics_epoch.get("AUCPR", 0.0)),
                "val_acc": float(val_metrics_epoch.get("ACC", 0.0)),
                "val_cindex": float(val_metrics_epoch.get("CINDEX", 0.0)),
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

        epoch_score = (float(val_metrics_epoch[best_metric_key]), -float(avg_val_loss))
        if epoch >= min_best_epoch and epoch_score > best_epoch_score:
            best_epoch_score = epoch_score
            best_epoch = epoch
            if model_name_l in {"dipam", "di_mmlp"}:
                best_model_state = copy.deepcopy(student_model.state_dict())
            else:
                best_model_state = copy.deepcopy(model.state_dict())
            best_val_targets = copy.deepcopy(val_store)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= patience:
                break

    if best_model_state is None:
        raise RuntimeError(
            f"No best epoch was selected. Check epochs={epochs} and min_best_epoch={min_best_epoch}."
        )

    if model_name_l in {"dipam", "di_mmlp"}:
        student_model.load_state_dict(best_model_state)
    else:
        model.load_state_dict(best_model_state)

    best_metrics = _finalize_task_metrics(task_config, best_val_targets)
    best_metrics["best_epoch"] = int(best_epoch)

    if model_name_l in {"dipam", "di_mmlp"}:
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
    weight_decay=1e-4,
    min_lr=1e-6,
    scheduler_type="cosine_annealing",
    lr_patience=5,
    task_config=None,
):
    """Train a final model on the full outer-train split for a fixed number of epochs."""
    model_kwargs = model_kwargs or {}
    model_name_l = normalize_model_name(model_name)
    set_global_seed(train_seed, deterministic=True)

    if model_name_l in {"smil_e"}:
        return train_smil_e_on_full_dataset_with_meta_learning(
            train_loader=train_loader,
            device=device,
            input_dims=input_dims,
            epochs=epochs,
            lr=lr,
            model_kwargs=model_kwargs,
            train_seed=train_seed,
            weight_decay=weight_decay,
            min_lr=min_lr,
            scheduler_type=scheduler_type,
            lr_patience=lr_patience,
            task_config=task_config,
        )

    bypass_mask = (
        model_name_l == "mlp"
        and str(imputation_method).strip().lower() in {"knn", "vae"}
    )

    criterion = nn.BCEWithLogitsLoss()

    if model_name_l in {"dipam", "di_mmlp"}:
        student_kwargs = get_model_init_kwargs(model_name_l, model_kwargs)
        distill_alpha = float(model_kwargs.get("distill_alpha", 1.0))
        distill_beta = float(model_kwargs.get("distill_beta", 0.3))
        if model_name_l == "di_mmlp":
            teacher_model = build_model("mlp", input_dims, student_kwargs).to(device)
        else:
            teacher_kwargs = {
                "dropout_p": float(model_kwargs.get("dropout_p", 0.4)),
                "temperature": float(model_kwargs.get("temperature", 2.0)),
            }
            teacher_model = build_model("pam", input_dims, teacher_kwargs).to(device)
        student_model = build_model(model_name_l, input_dims, student_kwargs).to(device)
        teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=lr, weight_decay=float(weight_decay))
        student_optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=float(weight_decay))
        scheduler = _build_scheduler(
            student_optimizer,
            epochs=epochs,
            min_lr=min_lr,
            scheduler_type=scheduler_type,
            lr_patience=lr_patience,
        )
        repr_criterion = nn.MSELoss()
        feat_criterion = nn.MSELoss()
        teacher_base_dataset = train_loader.dataset.base_dataset
    else:
        model = build_model(model_name, input_dims, model_kwargs).to(device)
        if model_name_l in {"smil_e"}:
            base_dataset = train_loader.dataset.base_dataset
            priors = learn_smil_priors(
                base_dataset=base_dataset,
                encoders=model.encoders,
                num_modalities=model.num_modalities,
                num_priors=model.num_priors,
                device=device,
            ).to(device)
            model.set_priors(priors)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))
        scheduler = _build_scheduler(
            optimizer,
            epochs=epochs,
            min_lr=min_lr,
            scheduler_type=scheduler_type,
            lr_patience=lr_patience,
        )

    history = []

    for epoch in range(1, int(epochs) + 1):
        if model_name_l in {"dipam", "di_mmlp"}:
            teacher_model.train()
            student_model.train()
        else:
            model.train()

        train_loss = 0.0
        train_steps = 0
        train_store = {
            "probs": [],
            "y_true": [],
            "logits": [],
            "event_times": [],
            "event_observed": [],
            "censorship": [],
            "y_disc": [],
        }
        train_teacher_loss = 0.0
        train_student_survival = 0.0
        train_student_repr = 0.0
        train_student_feature = 0.0

        for Xs, present_mask, y, pids in train_loader:
            Xs = [x.to(device) for x in Xs]
            present_mask = present_mask.to(device)
            if isinstance(y, dict):
                y = {k: v.to(device) for k, v in y.items()}
            else:
                y = y.to(device)

            if model_name_l in {"dipam", "di_mmlp"}:
                Xs_teacher, teacher_mask, y_teacher = _build_full_batch_from_patient_ids(
                    teacher_base_dataset,
                    pids,
                    device=device,
                )

                teacher_optimizer.zero_grad()
                teacher_out = teacher_model(Xs_teacher, teacher_mask, return_aux=True)
                teacher_logits, teacher_repr = _extract_distill_outputs(
                    "di_mmlp" if model_name_l == "di_mmlp" else "dipam",
                    teacher_out,
                )
                teacher_loss = _compute_supervised_task_loss(teacher_logits, y_teacher, task_config, criterion)
                teacher_loss.backward()
                teacher_optimizer.step()

                student_optimizer.zero_grad()
                student_out = student_model(Xs, present_mask, return_aux=True)
                student_logits, student_repr = _extract_distill_outputs(model_name_l, student_out)
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
                    task_config=task_config,
                )
                student_loss.backward()
                student_optimizer.step()

                _accumulate_eval_batch(task_config, student_logits, y, train_store)
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
                logits = _prepare_logits_for_task(logits_out)
                loss = _compute_supervised_task_loss(logits, y, task_config, criterion)
                loss.backward()
                optimizer.step()

                _accumulate_eval_batch(task_config, logits, y, train_store)
                train_loss += loss.item()
                train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)
        _step_scheduler(scheduler, scheduler_type=scheduler_type, metric=avg_train_loss)
        train_metrics_epoch = _finalize_task_metrics(task_config, train_store)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "train_auc": float(train_metrics_epoch.get("AUC", 0.0)),
                "train_aucpr": float(train_metrics_epoch.get("AUCPR", 0.0)),
                "train_acc": float(train_metrics_epoch.get("ACC", 0.0)),
                "train_cindex": float(train_metrics_epoch.get("CINDEX", 0.0)),
                "teacher_loss": float(train_teacher_loss / max(train_steps, 1)),
                "student_survival_loss": float(train_student_survival / max(train_steps, 1)),
                "student_repr_loss": float(train_student_repr / max(train_steps, 1)),
                "student_feature_loss": float(train_student_feature / max(train_steps, 1)),
            }
        )

    if model_name_l in {"dipam", "di_mmlp"}:
        return student_model, history
    return model, history
