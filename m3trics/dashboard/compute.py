"""
M3TRICS Dashboard Compute Module
=================================
Computes all dashboard statistics directly from results/ directory,
replacing the analysis notebooks as the data source.

Entry points
------------
compute_progressive_dataset(results_dir, ds_key, modality, retrain, distillation_models)
    Full progressive-missingness pipeline for one dataset/modality/retrain combination.
    Returns a dict matching the structure expected by generate_dashboard.py.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional deps (graceful degradation if missing)
# ---------------------------------------------------------------------------
try:
    from scipy import stats as _scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRIC_COL     = 'outer_test_AUC'          # binary classification
ENSEMBLE_SRC   = 'probability_averaged_ensemble'


# ===========================================================================
# I/O helpers
# ===========================================================================

def _read_csv(path: Path) -> list[dict]:
    try:
        with open(path, newline='') as f:
            return list(csv.DictReader(f))
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return []


def _f(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _binary_auc(y_true: list[float], scores: list[float]) -> float | None:
    """Rank-based binary AUC with average ranks for tied scores."""
    pairs = [(float(y), float(s)) for y, s in zip(y_true, scores) if y is not None and s is not None]
    if not pairs:
        return None
    n_pos = sum(1 for y, _ in pairs if y == 1)
    n_neg = sum(1 for y, _ in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None

    ordered = sorted(enumerate(pairs), key=lambda x: x[1][1])
    ranks = [0.0] * len(ordered)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1][1] == ordered[i][1][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    rank_by_original = [0.0] * len(ordered)
    for rank, (orig_idx, _) in zip(ranks, ordered):
        rank_by_original[orig_idx] = rank
    sum_pos_ranks = sum(rank_by_original[i] for i, (y, _) in enumerate(pairs) if y == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _concordance_index(times: list[float], events: list[float], risks: list[float]) -> float | None:
    """Simple Harrell C-index. Higher risk should correspond to shorter time."""
    vals = [
        (float(t), int(float(e)), float(r))
        for t, e, r in zip(times, events, risks)
        if t is not None and e is not None and r is not None
    ]
    permissible = concordant = ties = 0.0
    for i in range(len(vals)):
        ti, ei, ri = vals[i]
        for j in range(i + 1, len(vals)):
            tj, ej, rj = vals[j]
            if ti == tj:
                continue
            if ti < tj and ei == 1:
                permissible += 1
                if ri > rj:
                    concordant += 1
                elif ri == rj:
                    ties += 1
            elif tj < ti and ej == 1:
                permissible += 1
                if rj > ri:
                    concordant += 1
                elif ri == rj:
                    ties += 1
    if permissible == 0:
        return None
    return (concordant + 0.5 * ties) / permissible


def _inner_prediction_metric(rows: list[dict], inner_cols: list[str], metric_col: str) -> float | None:
    scores_by_inner = []
    if metric_col == 'outer_test_CINDEX':
        times = [_f(r.get('event_time')) for r in rows]
        events = [_f(r.get('event_observed')) for r in rows]
        for col in inner_cols:
            risks = [_f(r.get(col)) for r in rows]
            metric = _concordance_index(times, events, risks)
            if metric is not None:
                scores_by_inner.append(metric)
    else:
        y_true = [_f(r.get('y_true')) for r in rows]
        for col in inner_cols:
            probs = [_f(r.get(col)) for r in rows]
            metric = _binary_auc(y_true, probs)
            if metric is not None:
                scores_by_inner.append(metric)
    if not scores_by_inner:
        return None
    return sum(scores_by_inner) / len(scores_by_inner)


def load_retained_inner_replicates_from_predictions(
    results_dir: Path,
    ds_key: str,
    modality: str = 'global',
    retrain: str = 'false',
    metric_col: str = METRIC_COL,
) -> list[dict]:
    """
    Compute retained-inner-model performance directly from test_predictions.csv.

    Some historical runs stored only probability_averaged_ensemble rows in
    outer_test_metrics.csv even though inner_model_* predictions were present.
    This loader avoids losing those methods in the dashboard.
    """
    training_runs = results_dir / ds_key / 'training_runs'
    if not training_runs.exists():
        return []

    suffix = f'_retrain{retrain}_'
    mod_upper = modality.upper()
    records: list[dict] = []

    for model_dir in sorted(training_runs.iterdir()):
        if not model_dir.is_dir() or suffix not in model_dir.name:
            continue
        model_name = model_dir.name[: model_dir.name.index(suffix)]
        miss_root = model_dir / 'TRAIN_MISSING' / mod_upper
        if not miss_root.exists():
            continue

        for train_pct_dir in sorted(miss_root.iterdir()):
            if not train_pct_dir.is_dir():
                continue
            for seed_dir in sorted(train_pct_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
                    continue
                try:
                    seed = int(seed_dir.name.split('_', 1)[1])
                except ValueError:
                    continue

                rows = _read_csv(seed_dir / 'test_predictions.csv')
                if not rows:
                    continue
                if metric_col == 'outer_test_CINDEX':
                    inner_cols = sorted(c for c in rows[0] if c.startswith('inner_model_') and c.endswith('_risk'))
                else:
                    inner_cols = sorted(c for c in rows[0] if c.startswith('inner_model_') and c.endswith('_prob'))
                if not inner_cols:
                    continue

                grouped: dict[tuple[int, float, float], list[dict]] = defaultdict(list)
                for row in rows:
                    outer_fold = row.get('outer_fold')
                    try:
                        outer_fold = int(outer_fold)
                    except (TypeError, ValueError):
                        continue
                    train_prop = round(_f(row.get('train_missing_prop'), _f(train_pct_dir.name, 0.0)), 4)
                    test_prop = round(_f(row.get('test_missing_prop'), 0.0), 4)
                    grouped[(outer_fold, train_prop, test_prop)].append(row)

                for (outer_fold, train_prop, test_prop), group_rows in grouped.items():
                    metric = _inner_prediction_metric(group_rows, inner_cols, metric_col)
                    if metric is None:
                        continue
                    records.append({
                        'model_name':        model_name,
                        'train_missing_prop': train_prop,
                        'test_missing_prop':  test_prop,
                        'seed':              seed,
                        'outer_fold':        outer_fold,
                        'auc':               metric,
                    })

    return records


# ===========================================================================
# Step 1 — collect per-replicate AUC records from results/
# ===========================================================================

def load_replicates(
    results_dir: Path,
    ds_key: str,
    modality: str = 'global',
    retrain: str = 'false',
    metric_col: str = METRIC_COL,
    source_filter: str | None = ENSEMBLE_SRC,
) -> list[dict]:
    """
    Walk results/<ds_key>/training_runs/ and return one record per
    (model, train_missing_prop, test_missing_prop, seed, outer_fold).

    Parameters
    ----------
    results_dir   Root of the results/ directory.
    ds_key        Dataset key, e.g. 'mmColorectal_OS_21_label'.
    modality      Degrading-modality folder name (case-insensitive), e.g. 'global'.
    retrain       'false' or 'true' — selects the _retrain<X>_k* subfolder.
    metric_col    Column to use as the performance metric.
    source_filter If set, only rows whose outer_test_metric_source matches this value
                  are kept (use None to keep all).
    """
    if source_filter == 'mean_retained_inner_models':
        prediction_records = load_retained_inner_replicates_from_predictions(
            results_dir, ds_key, modality, retrain, metric_col
        )
        if prediction_records:
            return prediction_records

    training_runs = results_dir / ds_key / 'training_runs'
    if not training_runs.exists():
        return []

    suffix     = f'_retrain{retrain}_'
    mod_upper  = modality.upper()
    records    = []

    for model_dir in sorted(training_runs.iterdir()):
        if not model_dir.is_dir() or suffix not in model_dir.name:
            continue
        model_name = model_dir.name[: model_dir.name.index(suffix)]

        miss_root = model_dir / 'TRAIN_MISSING' / mod_upper
        if not miss_root.exists():
            continue

        for train_pct_dir in sorted(miss_root.iterdir()):
            if not train_pct_dir.is_dir():
                continue

            for seed_dir in sorted(train_pct_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
                    continue
                try:
                    seed = int(seed_dir.name.split('_', 1)[1])
                except ValueError:
                    continue

                for row in _read_csv(seed_dir / 'outer_test_metrics.csv'):
                    if source_filter and row.get('outer_test_metric_source', '') != source_filter:
                        continue
                    auc = _f(row.get(metric_col))
                    if auc is None:
                        continue
                    records.append({
                        'model_name':        model_name,
                        'train_missing_prop': round(_f(row.get('train_missing_prop', 0), 0), 4),
                        'test_missing_prop':  round(_f(row.get('eval_missing_prop',    0), 0), 4),
                        'seed':              seed,
                        'outer_fold':        int(row.get('outer_fold', 1)),
                        'auc':               auc,
                    })

    return records


# ===========================================================================
# Step 1b — hyperparameter selection summary
# ===========================================================================

def _fmt_hp_value(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == '' or s.lower() in {'nan', 'none', 'null'}:
        return None
    try:
        f = float(s)
        if math.isnan(f):
            return None
        if f.is_integer():
            return str(int(f))
        return f'{f:.6g}'
    except ValueError:
        return s


def _split_selected_hp_names(value: str | None) -> list[str]:
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() in {'nan', 'none', 'null'}:
        return []
    for sep in (';', '|'):
        if sep in s:
            return [x.strip() for x in s.split(sep) if x.strip()]
    return [s]


def _normalise_hp_display_name(hp_name: str) -> str:
    """Remove scenario suffixes from historical run names for display."""
    return re.sub(r'trmiss[^_]*_degmod[^_]*$', '', str(hp_name)).rstrip('_')


def _base_model_name(model_name: str) -> str:
    """Normalise dashboard method names to their underlying model family."""
    name = str(model_name or '').strip()
    if name.endswith('_KD'):
        name = name[:-3]
    name_l = name.lower().replace('-', '_')
    for prefix in ('zi_', 'knn_'):
        if name_l.startswith(prefix):
            name_l = name_l[len(prefix):]
            break
    aliases = {
        'lr': 'lr',
        'logistic_regression': 'lr',
        'rf': 'rf',
        'random_forest': 'rf',
        'coxnet': 'coxnet',
        'rsf': 'rsf',
        'mlp': 'mlp',
        'mmlp': 'mlp',
        'vae_mlp': 'mlp',
        'pam': 'pam',
        'healnet': 'healnet',
        'smile': 'smile',
        'smile_e': 'smile',
    }
    return aliases.get(name_l, name_l)


def _reportable_hp_columns(model_name: str) -> set[str] | None:
    """
    Return HP columns that are meaningful for the selected method family.

    Sklearn baselines carry generic training columns in inner_hp_eval.csv because
    the runner uses a common interface. Those values are fixed runtime
    placeholders, not searched hyperparameters, so they should not define HP
    signatures or dashboard columns for those methods.
    """
    family = _base_model_name(model_name)
    common_neural = {
        'batch_size', 'learning_rate', 'weight_decay',
        'distill_alpha', 'distill_beta', 'knowledge_distillation',
    }
    if family == 'lr':
        return {'lr_C', 'lr_penalty', 'lr_solver', 'lr_class_weight', 'lr_max_iter'}
    if family == 'rf':
        return {
            'rf_n_estimators', 'rf_max_depth', 'rf_min_samples_split',
            'rf_min_samples_leaf', 'rf_max_features', 'rf_class_weight', 'rf_n_jobs',
        }
    if family == 'coxnet':
        return {'coxnet_alpha', 'coxnet_l1_ratio', 'coxnet_max_iter', 'coxnet_tol'}
    if family == 'rsf':
        return {
            'rsf_n_estimators', 'rsf_max_depth', 'rsf_min_samples_split',
            'rsf_min_samples_leaf', 'rsf_max_features', 'rsf_n_jobs',
        }
    if family == 'pam':
        return common_neural | {'pam_dropout', 'pam_temperature'}
    if family == 'healnet':
        return common_neural | {
            'healnet_depth', 'healnet_num_freq_bands', 'healnet_num_latents',
            'healnet_latent_dim', 'healnet_cross_heads', 'healnet_latent_heads',
            'healnet_cross_dim_head', 'healnet_latent_dim_head',
            'healnet_attn_dropout', 'healnet_ff_dropout', 'healnet_self_per_cross_attn',
        }
    if family == 'smile':
        return common_neural | {
            'smil_e_latent_dim', 'smil_e_num_priors', 'smil_e_num_heads',
            'smil_e_dropout', 'classifier_hidden_dim', 'smil_e_alpha', 'smil_e_beta',
            'meta_inner_lr', 'meta_val_fraction',
        }
    if family == 'mlp':
        return common_neural | {
            'modality_hidden_layers', 'fusion_hidden_dim', 'fusion_hidden_layers',
            'fusion_batchnorm', 'dropout',
        }
    return None


def compute_hp_selection_summary(
    results_dir: Path,
    ds_key: str,
    modality: str = 'global',
    retrain: str = 'false',
) -> list[dict]:
    """
    Count how often each selected hyperparameter combination is used.

    Selection is counted once per (model, seed, outer fold, hp_name). Rows
    repeated across test-missing or train-missing proportions are intentionally
    deduplicated so the denominator is the nested-CV evaluation count
    (n_seeds x n_outer_folds), not the number of missingness cells.
    """
    training_runs = results_dir / ds_key / 'training_runs'
    if not training_runs.exists():
        return []

    suffix = f'_retrain{retrain}_'
    mod_upper = modality.upper()
    excluded_cols = {
        'outer_fold', 'inner_fold', 'hp_name', 'name',
        'seed', 'train_missing_prop',
        'degrading_modality', 'missing_location',
        'train_degrading_modality', 'test_degrading_modality', 'eval_degrading_modality',
        'train_missing_location', 'test_missing_location', 'eval_missing_location',
    }
    excluded_prefixes = ('val_best_',)

    hp_params: dict[tuple[str, str], dict[str, str]] = {}
    selected_units: set[tuple[str, int, int, float, str]] = set()
    method_events: set[tuple[str, int, int, float]] = set()

    for model_dir in sorted(training_runs.iterdir()):
        if not model_dir.is_dir() or suffix not in model_dir.name:
            continue
        model_name = model_dir.name[: model_dir.name.index(suffix)]
        miss_root = model_dir / 'TRAIN_MISSING' / mod_upper
        if not miss_root.exists():
            continue

        for train_pct_dir in sorted(miss_root.iterdir()):
            if not train_pct_dir.is_dir():
                continue
            try:
                train_prop = round(float(train_pct_dir.name) / 100.0, 6)
            except ValueError:
                continue
            for seed_dir in sorted(train_pct_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
                    continue
                try:
                    seed = int(seed_dir.name.split('_', 1)[1])
                except ValueError:
                    continue

                # Map normalised hp_name -> parameter values from the full evaluated grid.
                allowed_cols = _reportable_hp_columns(model_name)
                for row in _read_csv(seed_dir / 'inner_hp_eval.csv'):
                    hp_name = str(row.get('hp_name') or row.get('name') or '').strip()
                    if not hp_name:
                        continue
                    norm_name = _normalise_hp_display_name(hp_name)
                    key = (model_name, norm_name)
                    if key not in hp_params:
                        params = {}
                        for col, val in row.items():
                            if col in excluded_cols or any(col.startswith(p) for p in excluded_prefixes):
                                continue
                            if allowed_cols is not None and col not in allowed_cols:
                                continue
                            fmt = _fmt_hp_value(val)
                            if fmt is not None:
                                params[col] = fmt
                        hp_params[key] = params

                # Count selected HPs once per (seed, outer_fold, train_prop).
                for row in _read_csv(seed_dir / 'outer_test_metrics.csv'):
                    try:
                        outer_fold = int(row.get('outer_fold', 1))
                    except (TypeError, ValueError):
                        continue
                    hp_names = _split_selected_hp_names(row.get('selected_inner_hp_names'))
                    if not hp_names:
                        continue
                    method_events.add((model_name, seed, outer_fold, train_prop))
                    for hp_name in hp_names:
                        norm_name = _normalise_hp_display_name(hp_name)
                        selected_units.add((model_name, seed, outer_fold, train_prop, norm_name))

    if not selected_units:
        return []

    # Count per (model, signature, train_prop)
    counts: dict[tuple[str, tuple, float], int] = defaultdict(int)
    display_names: dict[tuple[str, tuple], str] = {}
    signature_params: dict[tuple[str, tuple], dict[str, str]] = {}
    for model_name, _seed, _outer_fold, train_prop, norm_name in selected_units:
        params = hp_params.get((model_name, norm_name), {})
        signature = tuple(sorted(params.items())) if params else (('hp_name', norm_name),)
        key = (model_name, signature)
        counts[(model_name, signature, train_prop)] += 1
        signature_params.setdefault(key, dict(params))
        display_names.setdefault(key, norm_name)

    # Total events per (model, train_prop) = n_seeds × n_outer_folds for that prop
    totals: dict[tuple[str, float], int] = defaultdict(int)
    for model_name, _seed, _outer_fold, train_prop in method_events:
        totals[(model_name, train_prop)] += 1

    all_param_cols = sorted({k for params in signature_params.values() for k in params})
    rows = []
    for (model_name, signature, train_prop), selected_count in sorted(
        counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][2])
    ):
        total = totals.get((model_name, train_prop), selected_count)
        row = {
            'model_name': model_name,
            'hp_combination': display_names.get((model_name, signature), 'HP combination'),
            'train_prop': train_prop,
            'selected_count': selected_count,
            'selection_events': total,
            'selected_pct': 100.0 * selected_count / total if total else None,
        }
        params = signature_params.get((model_name, signature), {})
        for col in all_param_cols:
            row[col] = params.get(col)
        rows.append(row)
    return rows


# ===========================================================================
# Step 2 — mean AUC summary  (method_condition_mean_auc_summary equivalent)
# ===========================================================================

def compute_mean_auc_summary(replicates: list[dict]) -> list[dict]:
    """
    Aggregate per-replicate AUCs to mean ± std per
    (model_name, train_missing_prop, test_missing_prop).
    """
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in replicates:
        key = (r['model_name'], r['train_missing_prop'], r['test_missing_prop'])
        groups[key].append(r['auc'])

    result = []
    for (model, train_p, test_p), aucs in sorted(groups.items()):
        n    = len(aucs)
        mean = sum(aucs) / n
        var  = sum((a - mean) ** 2 for a in aucs) / max(n - 1, 1) if n > 1 else 0.0
        result.append({
            'model_name':        model,
            'train_missing_prop': train_p,
            'test_missing_prop':  test_p,
            'mean_auc':          mean,
            'std_auc':           math.sqrt(var),
            'n_replicates':      n,
        })
    return result


# ===========================================================================
# Step 3 — AUPMC & method-level metrics  (method_level_metrics equivalent)
# ===========================================================================

def _trapz_normalized(xs: list[float], ys: list[float]) -> float:
    """Trapezoidal integral normalized to the x range [min, max]."""
    pairs = sorted(zip(xs, ys))
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    if len(xs) < 2:
        return ys[0] if ys else 0.0
    area = sum(
        (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2
        for i in range(len(xs) - 1)
    )
    x_range = xs[-1] - xs[0]
    return area / x_range if x_range > 0 else ys[0]


def compute_method_metrics(
    mean_auc_summary: list[dict],
    distillation_models: set[str] | None = None,
) -> list[dict]:
    """
    Compute per-method metrics:
      baseline_auc, test_aupmc, train_aupmc, bft_aupmc,
      bft_train_prop, and positive degradation coefficients.

    Distillation methods are excluded from train-time metrics.
    """
    distillation_models = distillation_models or set()

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in mean_auc_summary:
        by_model[r['model_name']].append(r)

    all_test_props  = sorted({r['test_missing_prop']  for r in mean_auc_summary})
    all_train_props = sorted({r['train_missing_prop'] for r in mean_auc_summary})

    result = []
    for model, rows in sorted(by_model.items()):

        def _get(train_p: float, test_p: float) -> float | None:
            for r in rows:
                if abs(r['train_missing_prop'] - train_p) < 1e-4 \
                        and abs(r['test_missing_prop'] - test_p) < 1e-4:
                    return r['mean_auc']
            return None

        baseline = _get(0.0, 0.0)
        if baseline is None:
            continue

        # Test-time AUPMC: fix train_prop=0, integrate over test_props
        test_curve = [(tp, _get(0.0, tp)) for tp in all_test_props]
        test_curve = [(x, y) for x, y in test_curve if y is not None]
        test_aupmc = _trapz_normalized([x for x, _ in test_curve],
                                       [y for _, y in test_curve]) if test_curve else None

        # Train-time AUPMC: fix test_prop=0, integrate over train_props
        train_aupmc = None
        if model not in distillation_models:
            train_curve = [(tp, _get(tp, 0.0)) for tp in all_train_props]
            train_curve = [(x, y) for x, y in train_curve if y is not None]
            if train_curve:
                train_aupmc = _trapz_normalized([x for x, _ in train_curve],
                                                [y for _, y in train_curve])

        # BFT AUPMC: best fixed train prop
        bft_aupmc, bft_train_prop = None, None
        for tp in all_train_props:
            curve = [(tst, _get(tp, tst)) for tst in all_test_props]
            curve = [(x, y) for x, y in curve if y is not None]
            if curve:
                a = _trapz_normalized([x for x, _ in curve], [y for _, y in curve])
                if bft_aupmc is None or a > bft_aupmc:
                    bft_aupmc, bft_train_prop = a, tp

        # Positive degradation coefficients: area where baseline/performance > 1.
        train_deg = None
        if baseline and baseline > 0 and model not in distillation_models:
            train_curve_for_deg = [(tp, _get(tp, 0.0)) for tp in all_train_props]
            train_curve_for_deg = [(x, y) for x, y in train_curve_for_deg if y is not None]
            if train_curve_for_deg:
                ratios = [max(baseline / y - 1, 0) if y else 0 for _, y in train_curve_for_deg]
                train_deg = _trapz_normalized([x for x, _ in train_curve_for_deg], ratios)

        test_deg = None
        if baseline and baseline > 0 and test_curve:
            ratios = [max(baseline / y - 1, 0) if y else 0 for _, y in test_curve]
            test_deg = _trapz_normalized([x for x, _ in test_curve], ratios)

        bft_deg = None
        if bft_train_prop is not None:
            bft_baseline = _get(bft_train_prop, 0.0)
            bft_curve_for_deg = [(tp, _get(bft_train_prop, tp)) for tp in all_test_props]
            bft_curve_for_deg = [(x, y) for x, y in bft_curve_for_deg if y is not None]
            if bft_baseline and bft_baseline > 0 and bft_curve_for_deg:
                ratios = [max(bft_baseline / y - 1, 0) if y else 0 for _, y in bft_curve_for_deg]
                bft_deg = _trapz_normalized([x for x, _ in bft_curve_for_deg], ratios)

        result.append({
            'model_name':           model,
            'baseline_auc':         baseline,
            'test_aupmc':           test_aupmc,
            'train_aupmc':          train_aupmc,
            'bft_aupmc':            bft_aupmc,
            'bft_train_prop':       bft_train_prop,
            'train_degradation_coef': train_deg,
            'test_degradation_coef': test_deg,
            'bft_degradation_coef':   bft_deg,
        })

    return result


# ===========================================================================
# Step 4 — Degradation curves  (degradation_curve_summary equivalent)
# ===========================================================================

def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1) if len(values) > 1 else 0.0
    return mean, math.sqrt(var)


def _ci95(values: list[float]) -> tuple[float | None, float | None, float | None]:
    mean, std = _mean_std(values)
    if mean is None:
        return None, None, None
    half = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return half, mean - half, mean + half


def _condition_lookup(rows: list[dict]) -> dict[tuple[float, float], dict]:
    return {
        (round(r['train_missing_prop'], 4), round(r['test_missing_prop'], 4)): r
        for r in rows
    }


def _paired_ratio_ci(
    replicate_lookup: dict[tuple[float, float], dict[tuple[int, int], float]],
    baseline_cond: tuple[float, float],
    target_cond: tuple[float, float],
) -> tuple[float | None, float | None, float | None]:
    base = replicate_lookup.get(baseline_cond, {})
    targ = replicate_lookup.get(target_cond, {})
    ratios = [
        base[k] / targ[k]
        for k in sorted(set(base) & set(targ))
        if targ[k] not in (None, 0)
    ]
    return _ci95(ratios)


def compute_degradation_curves(
    mean_auc_summary: list[dict],
    replicates: list[dict] | None = None,
    distillation_models: set[str] | None = None,
) -> list[dict]:
    """
    Degradation curves per model.

    Baselines are trajectory-specific:
      - train-time:       performance(train=0, test=0)
      - test-time:        performance(train=0, test=0)
      - best fixed-train: performance(train=m_train*, test=0)

    The plotted ratio is baseline_mean / condition_mean. Confidence intervals
    are computed from paired replicate ratios when replicate-level results are
    available; this makes the baseline point exactly 1 with zero-width CI.
    """
    distillation_models = distillation_models or set()
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in mean_auc_summary:
        by_model[r['model_name']].append(r)

    rep_by_model: dict[str, dict[tuple[float, float], dict[tuple[int, int], float]]] = defaultdict(lambda: defaultdict(dict))
    for r in replicates or []:
        cond = (round(r['train_missing_prop'], 4), round(r['test_missing_prop'], 4))
        rep_key = (int(r['seed']), int(r['outer_fold']))
        rep_by_model[r['model_name']][cond][rep_key] = r['auc']

    all_test_props = sorted({r['test_missing_prop'] for r in mean_auc_summary})
    result = []
    for model, rows in sorted(by_model.items()):
        lookup = _condition_lookup(rows)
        global_base_row = lookup.get((0.0, 0.0))
        if not global_base_row or not global_base_row.get('mean_auc') or global_base_row['mean_auc'] <= 0:
            continue
        global_baseline = global_base_row['mean_auc']
        reps_for_model = rep_by_model.get(model, {})

        def _add_curve(scenario: str, baseline_cond: tuple[float, float], curve_rows: list[dict]) -> None:
            base_row = lookup.get(baseline_cond)
            if not base_row or not base_row.get('mean_auc') or base_row['mean_auc'] <= 0:
                return
            baseline = base_row['mean_auc']
            for r in sorted(curve_rows, key=lambda x: (x['train_missing_prop'], x['test_missing_prop'])):
                perf = r.get('mean_auc')
                if not perf:
                    continue
                target_cond = (round(r['train_missing_prop'], 4), round(r['test_missing_prop'], 4))
                ci, lo, hi = _paired_ratio_ci(reps_for_model, baseline_cond, target_cond)
                result.append({
                    'model_name':                    model,
                    'scenario':                      scenario,
                    'train_prop':                    r['train_missing_prop'],
                    'test_prop':                     r['test_missing_prop'],
                    'missing_prop':                  r['train_missing_prop'] if scenario == 'train' else r['test_missing_prop'],
                    'mean_auc':                      perf,
                    'std_auc':                       r.get('std_auc'),
                    'n_replicates':                  r.get('n_replicates'),
                    'degradation_ratio':             baseline / perf,
                    'degradation_ratio_ci95':        ci,
                    'degradation_ratio_ci95_lower':  lo,
                    'degradation_ratio_ci95_upper':  hi,
                    'baseline_auc':                  baseline,
                })

        train_rows = [r for r in rows if abs(r['test_missing_prop']) < 1e-4]
        if model not in distillation_models:
            _add_curve('train', (0.0, 0.0), train_rows)

        test_rows = [r for r in rows if abs(r['train_missing_prop']) < 1e-4]
        _add_curve('test', (0.0, 0.0), test_rows)

        best_train_prop, best_aupmc = None, None
        for train_prop in sorted({r['train_missing_prop'] for r in rows}):
            curve = [(tp, lookup.get((round(train_prop, 4), round(tp, 4)), {}).get('mean_auc')) for tp in all_test_props]
            curve = [(x, y) for x, y in curve if y is not None]
            if curve:
                aupmc = _trapz_normalized([x for x, _ in curve], [y for _, y in curve])
                if best_aupmc is None or aupmc > best_aupmc:
                    best_aupmc, best_train_prop = aupmc, train_prop
        if best_train_prop is not None:
            bft_rows = [r for r in rows if abs(r['train_missing_prop'] - best_train_prop) < 1e-4]
            _add_curve('envelope', (round(best_train_prop, 4), 0.0), bft_rows)

    return result


# ===========================================================================
# Step 5 — Best-fixed-train curve  (best_fixed_train_curve equivalent)
# ===========================================================================

def compute_bft_curve(mean_auc_summary: list[dict]) -> list[dict]:
    """
    For each model: find the train_prop that maximises test-time AUPMC
    and return the full test curve for that train_prop.
    """
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in mean_auc_summary:
        by_model[r['model_name']].append(r)

    all_test_props = sorted({r['test_missing_prop'] for r in mean_auc_summary})
    result = []

    for model, rows in sorted(by_model.items()):
        train_props = sorted({r['train_missing_prop'] for r in rows})
        best_aupmc, best_train_prop = None, None

        for tp in train_props:
            curve = sorted(
                [(r['test_missing_prop'], r['mean_auc'])
                 for r in rows if abs(r['train_missing_prop'] - tp) < 1e-4],
                key=lambda x: x[0],
            )
            if len(curve) >= 2:
                a = _trapz_normalized([x for x, _ in curve], [y for _, y in curve])
                if best_aupmc is None or a > best_aupmc:
                    best_aupmc, best_train_prop = a, tp

        if best_train_prop is None:
            continue
        for r in sorted(rows, key=lambda x: x['test_missing_prop']):
            if abs(r['train_missing_prop'] - best_train_prop) < 1e-4:
                result.append({
                    'model_name':       model,
                    'best_train_prop':  best_train_prop,
                    'test_missing_prop': r['test_missing_prop'],
                    'mean_auc':         r['mean_auc'],
                })
    return result


# ===========================================================================
# Step 6 — Global Friedman test  (level1_global_friedman equivalent)
# ===========================================================================

def compute_friedman(replicates: list[dict]) -> list[dict]:
    """
    Global Friedman chi-square test across all models using per-replicate AUCs.
    A 'replicate' is one (seed, outer_fold, train_prop, test_prop) combination.
    """
    if not _SCIPY:
        return []

    models = sorted({r['model_name'] for r in replicates})
    if len(models) < 2:
        return []

    by_rep: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in replicates:
        key = (r['seed'], r['outer_fold'], r['train_missing_prop'], r['test_missing_prop'])
        by_rep[key][r['model_name']] = r['auc']

    # Keep only replicates that have every model
    complete = {k: v for k, v in by_rep.items() if all(m in v for m in models)}
    if len(complete) < 2:
        return []

    rep_keys = sorted(complete)
    matrix   = [[complete[k][m] for k in rep_keys] for m in models]

    try:
        stat, p = _scipy_stats.friedmanchisquare(*matrix)
        return [{
            'statistic':    stat,
            'p_value':      p,
            'significant_p0_05': bool(p < 0.05),
            'n_methods':    len(models),
            'n_replicates': len(complete),
        }]
    except Exception:
        return []


# ===========================================================================
# Step 7 — Pairwise Wilcoxon + BH FDR  (wilcoxon_significant equivalent)
# ===========================================================================

def _bh_adjust(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvalues)
    if n == 0:
        return []
    order   = sorted(range(n), key=lambda i: pvalues[i])
    adj     = [1.0] * n
    running = 1.0
    for rank, idx in enumerate(reversed(order), 1):
        adj[idx] = min(running, pvalues[idx] * n / (n - rank + 1))
        running  = adj[idx]
    return adj


def compute_wilcoxon(replicates: list[dict], alpha: float = 0.05) -> list[dict]:
    """
    Pairwise Wilcoxon signed-rank tests per (train_prop, test_prop) condition
    with Benjamini-Hochberg FDR correction.
    Returns only significantly different pairs (adj_p < alpha).
    """
    if not _SCIPY:
        return []

    by_cond: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in replicates:
        cond = (r['train_missing_prop'], r['test_missing_prop'])
        by_cond[cond][r['model_name']].append(r['auc'])

    result = []
    for cond, model_aucs in by_cond.items():
        models = sorted(model_aucs)
        pairs, raw_ps, deltas = [], [], []

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                a1, a2 = model_aucs[m1], model_aucs[m2]
                n = min(len(a1), len(a2))
                if n < 2:
                    continue
                try:
                    _, p = _scipy_stats.wilcoxon(a1[:n], a2[:n], alternative='two-sided')
                except Exception:
                    p = 1.0
                pairs.append((m1, m2))
                raw_ps.append(p)
                deltas.append(sum(a1[:n]) / n - sum(a2[:n]) / n)

        if not pairs:
            continue

        adj_ps = _bh_adjust(raw_ps)
        for (m1, m2), raw_p, adj_p, delta in zip(pairs, raw_ps, adj_ps, deltas):
            if adj_p < alpha:
                result.append({
                    'train_missing_prop': cond[0],
                    'test_missing_prop':  cond[1],
                    'model_a':            m1,
                    'model_b':            m2,
                    'delta_mean_auc':     round(delta, 6),
                    'p_value':            raw_p,
                    'p_adj':              adj_p,
                    'winner':             m1 if delta > 0 else m2,
                })
    return result


# ===========================================================================
# Main entry point
# ===========================================================================

def compute_progressive_dataset(
    results_dir: Path,
    ds_key: str,
    modality: str = 'global',
    retrain: str = 'false',
    distillation_models: set[str] | None = None,
    metric_col: str = METRIC_COL,
    source_filter: str | None = ENSEMBLE_SRC,
) -> dict:
    """
    Full progressive-missingness pipeline for one dataset/modality combination.

    Returns a dict with keys:
        mean_auc, metrics, degradation, bft, friedman, wilcoxon

    Each value is a list of dicts matching the CSV format produced by the
    analysis notebooks, so generate_dashboard.py can use this as a drop-in
    replacement for reading pre-computed CSV files.
    """
    replicates = load_replicates(
        results_dir, ds_key, modality, retrain, metric_col, source_filter
    )
    if not replicates:
        return {}

    mean_auc = compute_mean_auc_summary(replicates)

    return {
        'mean_auc':   mean_auc,
        'metrics':    compute_method_metrics(mean_auc, distillation_models),
        'degradation': compute_degradation_curves(mean_auc, replicates, distillation_models or set()),
        'bft':        compute_bft_curve(mean_auc),
        'friedman':   compute_friedman(replicates),
        'wilcoxon':   compute_wilcoxon(replicates),
        'hp_selection': compute_hp_selection_summary(results_dir, ds_key, modality, retrain),
    }
