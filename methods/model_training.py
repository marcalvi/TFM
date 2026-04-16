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
from utils import build_model, normalize_model_name, safe_binary_metrics, set_global_seed


def get_model_init_kwargs(model_name, model_kwargs=None):
    """Keep only constructor kwargs needed to instantiate the model itself."""
    model_kwargs = dict(model_kwargs or {})
    model_name_l = normalize_model_name(model_name)
    if model_name_l == "di_pam":
        return {
            key: value
            for key, value in model_kwargs.items()
            if key not in {"distill_alpha", "distill_beta"}
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
    y_full = torch.stack(ys, dim=0).to(dtype=torch.float32, device=device)
    full_mask = torch.ones((len(pid_batch), n_modalities), dtype=torch.bool, device=device)
    return xs_full, full_mask, y_full


def _compute_supervised_bce_loss(logits, targets, bce_criterion):
    return bce_criterion(logits, targets)


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


def train_smil_e_with_meta_learning(
    train_loader,
    val_loader,
    device,
    input_dims,
    epochs,
    lr,
    model_kwargs=None,
    train_seed=0,
):
    """Train SMIL-E with a SMIL-style meta loop fully contained in inner-train."""
    min_best_epoch = min(5, int(epochs))
    model_kwargs = dict(model_kwargs or {})
    inner_steps = int(model_kwargs.pop("meta_inner_steps", 1))
    inner_lr = float(model_kwargs.pop("meta_inner_lr", 1e-2))
    meta_val_fraction = float(model_kwargs.pop("meta_val_fraction", 0.2))

    model = build_model("smil_e", input_dims, model_kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
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

    best_epoch_score = (-np.inf, -np.inf)
    best_epoch = 1
    best_model_state = None
    best_val_targets = None
    best_val_probs = None
    early_stop = 0
    patience = 20
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
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for Xs, present_mask, y, _ in val_loader:
                Xs = [x.to(device) for x in Xs]
                present_mask = present_mask.to(device)
                y = y.to(device)

                logits_out = model(
                    Xs,
                    present_mask,
                    mode="incomplete",
                    meta_train=False,
                )
                logits = logits_out.squeeze(1)
                loss = criterion(logits, y)

                val_loss += float(loss.item())
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                val_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        scheduler.step(avg_val_loss)

        val_metrics_epoch = safe_binary_metrics(val_targets, val_probs)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_meta_train_loss),
                "val_loss": float(avg_val_loss),
                "val_auc": float(val_metrics_epoch["AUC"]),
                "val_aucpr": float(val_metrics_epoch["AUCPR"]),
                "val_acc": float(val_metrics_epoch["ACC"]),
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

        epoch_score = (float(val_metrics_epoch["AUC"]), -float(avg_val_loss))
        if epoch >= min_best_epoch and epoch_score > best_epoch_score:
            best_epoch_score = epoch_score
            best_epoch = epoch
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

    model.load_state_dict(best_model_state)
    best_metrics = safe_binary_metrics(best_val_targets, best_val_probs)
    best_metrics["best_epoch"] = int(best_epoch)
    return model, history, best_metrics


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

    bypass_mask = (
        model_name_l == "mlp"
        and str(imputation_method).strip().lower() in {"knn", "vae"}
    )

    criterion = nn.BCEWithLogitsLoss()

    if model_name_l in {"di_pam"}:
        pam_kwargs = get_model_init_kwargs(model_name_l, model_kwargs)
        distill_alpha = float(model_kwargs.get("distill_alpha", 1.0))
        distill_beta = float(model_kwargs.get("distill_beta", 0.3))

        teacher_model = build_model("di_pam", input_dims, pam_kwargs).to(device)
        student_model = build_model("di_pam", input_dims, pam_kwargs).to(device)
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

    best_epoch_score = (-np.inf, -np.inf)
    best_epoch = 1
    best_model_state = None
    best_val_targets = None
    best_val_probs = None
    early_stop = 0
    patience = 20
    history = []

    for epoch in range(1, epochs + 1):
        if model_name_l in {"di_pam"}:
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

            if model_name_l in {"di_pam"}:
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

        if model_name_l in {"di_pam"}:
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

                if model_name_l in {"di_pam"}:
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
            if model_name_l in {"di_pam"}:
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

    if model_name_l in {"di_pam"}:
        student_model.load_state_dict(best_model_state)
    else:
        model.load_state_dict(best_model_state)

    best_metrics = safe_binary_metrics(best_val_targets, best_val_probs)
    best_metrics["best_epoch"] = int(best_epoch)

    if model_name_l in {"di_pam"}:
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

    if model_name_l in {"di_pam"}:
        pam_kwargs = get_model_init_kwargs(model_name_l, model_kwargs)
        distill_alpha = float(model_kwargs.get("distill_alpha", 1.0))
        distill_beta = float(model_kwargs.get("distill_beta", 0.3))

        teacher_model = build_model("di_pam", input_dims, pam_kwargs).to(device)
        student_model = build_model("di_pam", input_dims, pam_kwargs).to(device)
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
        if model_name_l in {"di_pam"}:
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

            if model_name_l in {"di_pam"}:
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

    if model_name_l in {"di_pam"}:
        return student_model, history
    return model, history
