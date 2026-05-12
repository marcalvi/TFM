import numpy as np
import pandas as pd
import torch


SURVIVAL_Y_DISC_COL = "y_disc"
SURVIVAL_CENSORSHIP_COL = "censorship"


def normalize_task_type(task_type):
    task_type_l = str(task_type).strip().lower()
    if task_type_l == "classification":
        return "binary_classification"
    if task_type_l in {"binary_classification", "survival"}:
        return task_type_l
    return task_type_l


def _normalize_binary_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isna().all():
        return numeric

    mapping = {
        "1": 1.0,
        "true": 1.0,
        "yes": 1.0,
        "y": 1.0,
        "dead": 1.0,
        "deceased": 1.0,
        "event": 1.0,
        "0": 0.0,
        "false": 0.0,
        "no": 0.0,
        "n": 0.0,
        "alive": 0.0,
        "living": 0.0,
        "censored": 0.0,
        "censor": 0.0,
    }
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.map(mapping)


def add_survival_target_columns(
    endpoint_df,
    patient_id_col,
    time_col,
    event_col,
    n_bins=4,
    y_disc_col=SURVIVAL_Y_DISC_COL,
    censorship_col=SURVIVAL_CENSORSHIP_COL,
):
    work_df = endpoint_df.copy()
    required_cols = [patient_id_col, time_col, event_col]
    for required_col in required_cols:
        if required_col not in work_df.columns:
            raise ValueError(
                f"Required survival column '{required_col}' not found in endpoint CSV."
            )

    event_time = pd.to_numeric(work_df[time_col], errors="coerce")
    event_observed = _normalize_binary_series(work_df[event_col])
    invalid_mask = event_time.isna() | event_observed.isna()
    if invalid_mask.any():
        invalid_examples = (
            work_df.loc[invalid_mask, required_cols]
            .head(10)
            .to_dict("records")
        )
        n_invalid = int(invalid_mask.sum())
        print(
            f"[survival] Dropping {n_invalid} rows with invalid time/event values. "
            f"Examples: {invalid_examples}"
        )
        work_df = work_df.loc[~invalid_mask].copy()
        event_time = event_time.loc[~invalid_mask].copy()
        event_observed = event_observed.loc[~invalid_mask].copy()

    if work_df.empty:
        raise ValueError("No valid endpoint rows left after survival target filtering.")

    event_observed = event_observed.astype(np.int64)
    invalid_binary_mask = ~event_observed.isin([0, 1])
    if invalid_binary_mask.any():
        invalid_examples = (
            work_df.loc[invalid_binary_mask, [patient_id_col, event_col]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"Survival event column '{event_col}' must contain binary values. "
            f"Examples of invalid rows: {invalid_examples}"
        )

    work_df[time_col] = event_time.astype(np.float32)
    work_df[event_col] = event_observed.astype(np.int64)
    work_df[censorship_col] = (1 - work_df[event_col].astype(np.int64)).astype(np.int64)

    times = work_df[time_col].astype(np.float64)
    if times.nunique() <= 1:
        work_df[y_disc_col] = 0
        bin_edges = np.array([times.min() - 1e-6, times.max() + 1e-6], dtype=np.float64)
        return work_df, bin_edges

    requested_bins = max(int(n_bins), 2)
    try:
        y_disc, bin_edges = pd.qcut(
            times,
            q=requested_bins,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
    except ValueError:
        min_t = float(times.min()) - 1e-6
        max_t = float(times.max()) + 1e-6
        bin_edges = np.linspace(min_t, max_t, num=requested_bins + 1, dtype=np.float64)
        y_disc = pd.cut(
            times,
            bins=bin_edges,
            labels=False,
            include_lowest=True,
            right=False,
        )

    y_disc = pd.Series(y_disc, index=work_df.index).fillna(0).astype(np.int64)
    work_df[y_disc_col] = y_disc
    return work_df, np.asarray(bin_edges, dtype=np.float64)


def survival_logits_to_outputs(logits):
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1)
    return hazards, survival, risk


def nll_survival_loss(hazards, survival, y_disc, censorship, eps=1e-7):
    batch_size = len(y_disc)
    y_disc = y_disc.view(batch_size, 1).long()
    censorship = censorship.view(batch_size, 1).float()
    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1)
    survival_padded = torch.cat([torch.ones_like(censorship), survival], dim=1)

    uncensored = -(
        (1 - censorship)
        * (
            torch.log(torch.gather(survival_padded, 1, y_disc).clamp(min=eps))
            + torch.log(torch.gather(hazards, 1, y_disc).clamp(min=eps))
        )
    )
    censored = -(
        censorship
        * torch.log(torch.gather(survival_padded, 1, y_disc + 1).clamp(min=eps))
    )
    return (uncensored + censored).mean()


def ce_survival_loss(hazards, survival, y_disc, censorship, alpha=0.4, eps=1e-7):
    batch_size = len(y_disc)
    y_disc = y_disc.view(batch_size, 1).long()
    censorship = censorship.view(batch_size, 1).float()
    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1)
    survival_padded = torch.cat([torch.ones_like(censorship), survival], dim=1)
    reg = -(
        (1 - censorship)
        * (
            torch.log(torch.gather(survival_padded, 1, y_disc).clamp(min=eps))
            + torch.log(torch.gather(hazards, 1, y_disc).clamp(min=eps))
        )
    )
    ce_term = -(
        censorship * torch.log(torch.gather(survival, 1, y_disc).clamp(min=eps))
        + (1 - censorship) * torch.log(
            1 - torch.gather(survival, 1, y_disc).clamp(min=eps)
        )
    )
    return ((1 - float(alpha)) * ce_term + float(alpha) * reg).mean()


def cox_survival_loss(hazards, survival, censorship):
    current_batch_len = len(survival)
    risk_mat = np.zeros([current_batch_len, current_batch_len], dtype=np.float32)
    survival_np = survival.detach().cpu().numpy().reshape(-1)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            risk_mat[i, j] = float(survival_np[j] >= survival_np[i])

    risk_mat = torch.FloatTensor(risk_mat).to(hazards.device)
    theta = hazards.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * risk_mat, dim=1)))
        * (1 - censorship.float())
    )
    return loss_cox


def compute_survival_loss_from_logits(logits, y_disc, censorship, loss_name="nll"):
    hazards, survival, _ = survival_logits_to_outputs(logits)
    loss_name_l = str(loss_name).strip().lower()
    if loss_name_l == "nll":
        return nll_survival_loss(hazards, survival, y_disc, censorship)
    if loss_name_l == "ce_survival":
        return ce_survival_loss(hazards, survival, y_disc, censorship)
    if loss_name_l == "cox":
        return cox_survival_loss(hazards, survival, censorship)
    raise ValueError(
        f"Unsupported survival loss '{loss_name}'. Valid values: nll, ce_survival, cox."
    )


def concordance_index_censored(event_observed, event_times, risk_scores):
    event_observed = np.asarray(event_observed, dtype=bool).reshape(-1)
    event_times = np.asarray(event_times, dtype=np.float64).reshape(-1)
    risk_scores = np.asarray(risk_scores, dtype=np.float64).reshape(-1)
    if not (len(event_observed) == len(event_times) == len(risk_scores)):
        raise ValueError("concordance_index_censored expects equal-length arrays.")

    concordant = 0.0
    permissible = 0.0

    n = len(event_times)
    for i in range(n):
        for j in range(i + 1, n):
            comparable = False
            sign = 0.0
            if event_observed[i] and event_times[i] < event_times[j]:
                comparable = True
                sign = np.sign(risk_scores[i] - risk_scores[j])
            elif event_observed[j] and event_times[j] < event_times[i]:
                comparable = True
                sign = np.sign(risk_scores[j] - risk_scores[i])

            if not comparable:
                continue

            permissible += 1.0
            if sign > 0:
                concordant += 1.0
            elif sign == 0:
                concordant += 0.5

    if permissible == 0.0:
        return 0.5
    return float(concordant / permissible)


def safe_survival_metrics(
    event_times,
    event_observed,
    censorship,
    y_disc,
    logits,
    loss_name="nll",
    include_raw=False,
):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(
            f"safe_survival_metrics expects logits with shape [N, n_bins], got {tuple(logits.shape)}."
        )

    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    y_disc_t = torch.as_tensor(np.asarray(y_disc, dtype=np.int64), dtype=torch.long)
    censorship_t = torch.as_tensor(np.asarray(censorship, dtype=np.float32), dtype=torch.float32)
    hazards_t, survival_t, risk_t = survival_logits_to_outputs(logits_t)
    loss_t = compute_survival_loss_from_logits(
        logits_t,
        y_disc=y_disc_t,
        censorship=censorship_t,
        loss_name=loss_name,
    )
    cindex = concordance_index_censored(
        event_observed=np.asarray(event_observed, dtype=bool),
        event_times=np.asarray(event_times, dtype=np.float64),
        risk_scores=risk_t.detach().cpu().numpy(),
    )
    metrics = {
        "CINDEX": float(cindex),
        "LOSS": float(loss_t.detach().cpu().item()),
    }
    if include_raw:
        metrics["Hazards"] = hazards_t.detach().cpu().numpy().tolist()
        metrics["Survival"] = survival_t.detach().cpu().numpy().tolist()
        metrics["Risk"] = risk_t.detach().cpu().numpy().tolist()
        metrics["Logits"] = logits_t.detach().cpu().numpy().tolist()
    return metrics
