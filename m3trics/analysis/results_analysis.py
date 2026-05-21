
"""Backend logic for analysis/statistical_tests.ipynb.

This module contains the dense data-loading, aggregation, statistical-testing,
and plotting helpers so the notebook can stay focused on configuration,
tables, and final figures.
"""

from pathlib import Path

BASE_LINE_FIGSIZE = (15.8, 10.0)
LINE_LEGEND_EXTRA_WIDTH_PER_MODEL = 0.55
BASE_HEATMAP_PANEL_WIDTH = 16.0 / 3.0
BASE_HEATMAP_PANEL_HEIGHT = 16.0 / 3.0
DEFAULT_HEATMAP_FIGSIZE = (16.0, 10.0)

import itertools
import math
import re
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import roc_auc_score

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch, Rectangle
    HAVE_MPL = True
except ImportError:
    plt = None
    ListedColormap = None
    LinearSegmentedColormap = None
    Normalize = None
    Patch = None
    Rectangle = None
    HAVE_MPL = False

DISPLAY_NAME_MAP = {
    'ZERO_MLP': 'ZI_MLP',
    'KNN_MLP': 'KNN_MLP',
    'VAE_MLP': 'VAE_MLP',
    'PAM': 'pAM',
    'DIMMLP': 'Di-MMLP',
    'DI_MMLP': 'Di-MMLP',
    'DIPAM': 'Di-PAM',
    'DI_PAM': 'Di-PAM',
    'HEALNET': 'HealNet',
    'SMILE': 'SMILe',
}
INNER_MODEL_PROB_RE = re.compile(r'^inner_model_(\d+)_prob$')

MANAGUA_HEX = [
    '#ffcf67', '#e3a358', '#c77c4b', '#aa5b41', '#863c39', '#632a3e',
    '#502f59', '#4c4a85', '#556eac', '#6292c9', '#71bae4', '#81e7ff',
]


def _normalize_model_label(name: str) -> str:
    raw = str(name).strip()
    if '_retrain' in raw:
        raw = raw.split('_retrain', 1)[0]
    compact = raw.replace('-', '_').upper()
    return DISPLAY_NAME_MAP.get(compact, raw)


def _stable_seed(*parts, base_seed=0):
    text = '|'.join(str(part) for part in parts)
    return (zlib.crc32(text.encode('utf-8')) + int(base_seed)) % (2**32 - 1)


def _parse_retrain_flag_from_run_name(run_name: str):
    raw = str(run_name).strip().lower()
    marker = '_retrain'
    if marker not in raw:
        return None
    suffix = raw.split(marker, 1)[1]
    if suffix.startswith('true'):
        return True
    if suffix.startswith('false'):
        return False
    return None


def list_model_sources(results_root: Path, dataset_name: str, train_missing_location: str = 'GLOBAL', model_names=None, retrain_outer=None, results_mode: str = 'decay'):
    sources = []
    requested = None if model_names is None else set(model_names)
    results_mode = str(results_mode).strip().lower()
    if results_mode not in {'decay', 'fixed_dataset'}:
        raise ValueError(f"Unsupported results_mode='{results_mode}'. Expected 'decay' or 'fixed_dataset'.")

    for run_dir in sorted(p for p in results_root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        run_retrain_flag = _parse_retrain_flag_from_run_name(run_dir.name)
        if retrain_outer is not None:
            if run_retrain_flag is None:
                continue
            if bool(run_retrain_flag) != bool(retrain_outer):
                continue
        model_label = _normalize_model_label(run_dir.name)
        if requested is not None and model_label not in requested:
            continue

        if results_mode == 'fixed_dataset':
            run_data_dir = run_dir / 'FIXED'
            if not run_data_dir.exists():
                continue
        else:
            run_data_dir = run_dir / 'TRAIN_MISSING' / train_missing_location.upper()
            if not run_data_dir.exists():
                continue

        sources.append({
            'model_name': model_label,
            'run_dir': run_dir,
            'run_data_dir': run_data_dir,
            'results_mode': results_mode,
        })
    return sources


def resolve_requested_model_names(results_root: Path, dataset_name: str, train_missing_location: str = 'GLOBAL', retrain_outer=None, results_mode: str = 'decay'):
    return sorted({
        source['model_name']
        for source in list_model_sources(
            results_root,
            dataset_name,
            train_missing_location,
            model_names=None,
            retrain_outer=retrain_outer,
            results_mode=results_mode,
        )
    })


def _parse_seed_from_path(path: Path):
    for parent in path.parents:
        if parent.name.startswith('seed_'):
            try:
                return int(parent.name.replace('seed_', ''))
            except ValueError:
                return parent.name.replace('seed_', '')
    return np.nan


def _list_inner_model_prob_cols(df: pd.DataFrame):
    cols = []
    for col in df.columns:
        match = INNER_MODEL_PROB_RE.match(str(col))
        if match:
            cols.append((int(match.group(1)), col))
    cols.sort(key=lambda item: item[0])
    return cols


def normalize_prediction_df(pred_df: pd.DataFrame):
    out = pred_df.copy()
    required_cols = [
        'model_name',
        'patient',
        'train_missing_prop',
        'test_missing_prop',
        'y_true',
    ]
    missing_cols = [col for col in required_cols if col not in out.columns]
    if missing_cols:
        raise ValueError(f'Missing required prediction columns: {missing_cols}')

    inner_prob_cols = _list_inner_model_prob_cols(out)
    if not inner_prob_cols:
        raise ValueError('No inner_model_*_prob columns found in prediction dataframe.')

    if 'outer_eval_target' in out.columns:
        out = out.loc[out['outer_eval_target'].astype(str) == 'test_outer'].copy()

    out['model_name'] = out['model_name'].astype(str)
    out['patient'] = out['patient'].astype(str)
    out['train_missing_prop'] = out['train_missing_prop'].astype(float)
    out['test_missing_prop'] = out['test_missing_prop'].astype(float)
    out['y_true'] = out['y_true'].astype(int)
    if 'seed' not in out.columns:
        out['seed'] = np.nan
    if 'outer_fold' not in out.columns:
        out['outer_fold'] = np.nan
    return out.reset_index(drop=True)


def load_all_test_predictions(results_root: Path, dataset_name: str, train_missing_location: str = 'GLOBAL', model_names=None, retrain_outer=None, results_mode: str = 'decay'):
    frames = []
    missing_prediction_files = []
    for source in list_model_sources(
        results_root,
        dataset_name,
        train_missing_location,
        model_names=model_names,
        retrain_outer=retrain_outer,
        results_mode=results_mode,
    ):
        found_any = False
        for path in sorted(source['run_data_dir'].rglob('test_predictions.csv')):
            found_any = True
            df = pd.read_csv(path)
            if df.empty:
                continue
            if 'seed' not in df.columns:
                df['seed'] = _parse_seed_from_path(path)
            df['model_name'] = source['model_name']
            df['source_file'] = str(path)
            frames.append(df)
        if not found_any:
            missing_prediction_files.append(str(source['run_data_dir']))

    if not frames:
        return pd.DataFrame(), missing_prediction_files
    return normalize_prediction_df(pd.concat(frames, ignore_index=True)), missing_prediction_files


def expand_inner_model_predictions(pred_df: pd.DataFrame):
    if pred_df.empty:
        return pd.DataFrame()

    prob_cols = _list_inner_model_prob_cols(pred_df)
    rows = []
    base_cols = [
        'model_name', 'patient', 'train_missing_prop', 'test_missing_prop', 'y_true',
        'seed', 'outer_fold'
    ]
    for member_idx, prob_col in prob_cols:
        sub_df = pred_df[base_cols + [prob_col]].copy()
        sub_df = sub_df.rename(columns={prob_col: 'member_prob'})
        sub_df['inner_model_idx'] = int(member_idx)
        sub_df['replicate_id'] = (
            sub_df['seed'].astype(str)
            + '|outer_' + sub_df['outer_fold'].astype(str)
            + '|inner_' + sub_df['inner_model_idx'].astype(str)
        )
        rows.append(sub_df)

    long_df = pd.concat(rows, ignore_index=True)
    long_df['member_prob'] = long_df['member_prob'].astype(float)
    long_df = long_df.dropna(subset=['member_prob']).reset_index(drop=True)
    return long_df


def aggregate_member_patient_predictions(member_pred_df: pd.DataFrame):
    if member_pred_df.empty:
        return pd.DataFrame()

    group_cols = [
        'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold',
        'inner_model_idx', 'replicate_id', 'patient', 'y_true'
    ]
    counts_df = (
        member_pred_df
        .groupby(group_cols, as_index=False)
        .size()
        .rename(columns={'size': 'n_prediction_rows'})
    )
    duplicated_df = counts_df.loc[counts_df['n_prediction_rows'] > 1].copy()
    if not duplicated_df.empty:
        sample_rows = duplicated_df.head(10).to_dict(orient='records')
        raise ValueError(
            'Found duplicated predictions for the same seed x outer_fold x inner_model x patient x missingness cell. '
            'This notebook does not collapse them into a single predictor. Sample duplicated groups: '
            f'{sample_rows}'
        )

    out_df = (
        member_pred_df[group_cols + ['member_prob']]
        .copy()
        .sort_values([
            'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold',
            'inner_model_idx', 'patient'
        ])
        .reset_index(drop=True)
    )
    out_df['n_prediction_rows'] = 1
    return out_df


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def bootstrap_mean_ci(values, n_bootstrap=2000, confidence=0.95, random_seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            'mean': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'ci_half_width': np.nan,
            'n_bootstrap_valid': 0,
        }
    observed_mean = float(np.mean(values))
    if values.size == 1:
        return {
            'mean': observed_mean,
            'ci_lower': observed_mean,
            'ci_upper': observed_mean,
            'ci_half_width': 0.0,
            'n_bootstrap_valid': 1,
        }

    rng = np.random.default_rng(int(random_seed))
    alpha = 1.0 - float(confidence)
    samples = []
    n = values.size
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        samples.append(float(np.mean(values[idx])))
    arr = np.asarray(samples, dtype=float)
    ci_lower, ci_upper = np.quantile(arr, [alpha / 2.0, 1.0 - (alpha / 2.0)])
    return {
        'mean': observed_mean,
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_half_width': float((ci_upper - ci_lower) / 2.0),
        'n_bootstrap_valid': int(arr.size),
    }


def build_replicate_auc_table(member_patient_df: pd.DataFrame):
    if member_patient_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold',
        'inner_model_idx', 'replicate_id'
    ]
    for key, group_df in member_patient_df.groupby(group_cols, sort=True):
        model_name, train_prop, test_prop, seed, outer_fold, inner_idx, replicate_id = key
        auc_val = safe_auc(group_df['y_true'].to_numpy(), group_df['member_prob'].to_numpy())
        rows.append({
            'model_name': model_name,
            'train_missing_prop': float(train_prop),
            'test_missing_prop': float(test_prop),
            'seed': seed,
            'outer_fold': outer_fold,
            'inner_model_idx': int(inner_idx),
            'replicate_id': str(replicate_id),
            'n_patients': int(group_df['patient'].nunique()),
            'auc': auc_val,
        })
    return pd.DataFrame(rows).sort_values([
        'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold', 'inner_model_idx'
    ]).reset_index(drop=True)


def build_level1_summary(replicate_auc_df: pd.DataFrame):
    rows = []
    for (model_name, train_prop, test_prop), group_df in replicate_auc_df.groupby(['model_name', 'train_missing_prop', 'test_missing_prop'], sort=True):
        auc_values = group_df['auc'].to_numpy(dtype=float)
        auc_values = auc_values[np.isfinite(auc_values)]
        rows.append({
            'model_name': model_name,
            'scenario': 'both',
            'train_prop': float(train_prop),
            'test_prop': float(test_prop),
            'train_missing_prop': float(train_prop),
            'test_missing_prop': float(test_prop),
            'mean_auc': float(np.mean(auc_values)) if auc_values.size else np.nan,
            'std_auc': float(np.std(auc_values, ddof=1)) if auc_values.size > 1 else 0.0,
            'n_replicates': int(auc_values.size),
        })
    return pd.DataFrame(rows).sort_values(['model_name', 'train_prop', 'test_prop']).reset_index(drop=True)


def build_fixed_dataset_method_summary(replicate_auc_df: pd.DataFrame):
    if replicate_auc_df.empty:
        return pd.DataFrame(columns=['model_name', 'mean_auc', 'std_auc', 'n_replicates', 'auc_ci95', 'auc_ci95_lower', 'auc_ci95_upper'])
    rows = []
    for model_name, group_df in replicate_auc_df.groupby('model_name', sort=True):
        auc_values = group_df['auc'].to_numpy(dtype=float)
        auc_values = auc_values[np.isfinite(auc_values)]
        stats = bootstrap_mean_ci(auc_values, n_bootstrap=2000, confidence=0.95, random_seed=_stable_seed(model_name, 'fixed_dataset'))
        rows.append({
            'model_name': str(model_name),
            'mean_auc': float(stats['mean']),
            'std_auc': float(np.std(auc_values, ddof=1)) if auc_values.size > 1 else 0.0,
            'n_replicates': int(auc_values.size),
            'auc_ci95': float(stats['ci_half_width']),
            'auc_ci95_lower': float(stats['ci_lower']),
            'auc_ci95_upper': float(stats['ci_upper']),
        })
    return pd.DataFrame(rows).sort_values(['mean_auc', 'model_name'], ascending=[False, True]).reset_index(drop=True)


def plot_fixed_dataset_auc_violins(
    replicate_auc_df: pd.DataFrame,
    method_summary_df: pd.DataFrame,
    title: str,
    figures_dir: Path,
    file_name: str,
    figsize=None,
):
    if not HAVE_MPL:
        print('Matplotlib is not available in this environment. Skipping fixed-dataset AUC violin plot.')
        return
    if replicate_auc_df.empty:
        print('No replicate AUCs available for fixed-dataset violin plot.')
        return

    ordered_models = method_summary_df['model_name'].astype(str).tolist() if not method_summary_df.empty else sorted(replicate_auc_df['model_name'].astype(str).unique().tolist())
    data = []
    labels = []
    for model_name in ordered_models:
        values = replicate_auc_df.loc[replicate_auc_df['model_name'].astype(str) == model_name, 'auc'].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            data.append(values)
            labels.append(model_name)
    if not data:
        print('No finite AUC values available for fixed-dataset violin plot.')
        return

    resolved_figsize = tuple(figsize) if figsize is not None else (max(12.0, 1.25 * len(labels) + 4.0), 8.0)
    fig, ax = plt.subplots(figsize=resolved_figsize)
    positions = np.arange(1, len(labels) + 1)
    violin_parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    palette = plt.cm.Set2(np.linspace(0.0, 1.0, max(len(labels), 3)))
    for idx, body in enumerate(violin_parts['bodies']):
        body.set_facecolor(palette[idx])
        body.set_edgecolor('#263238')
        body.set_alpha(0.55)
        body.set_linewidth(1.0)

    rng = np.random.default_rng(42)
    for pos, values in zip(positions, data):
        jitter = rng.normal(0.0, 0.075, size=len(values))
        ax.scatter(
            np.full_like(values, float(pos), dtype=float) + jitter,
            values,
            s=26,
            color='#111111',
            edgecolor='none',
            alpha=0.72,
            zorder=3,
        )

    ax.set_title(title, fontsize=18, pad=16)
    ax.set_ylabel('Replicate AUC', fontsize=15)
    ax.set_xlabel('Method', fontsize=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', color='#D9DEE3', linewidth=0.9, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#A7B0BA')
    ax.spines['bottom'].set_color('#A7B0BA')

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / file_name
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f'Saved figure to: {figure_path}')
    plt.show()


def build_cell_model_mean_matrix(level1_df: pd.DataFrame):
    both_df = level1_df.loc[level1_df['scenario'] == 'both'].copy()
    if both_df.empty:
        return pd.DataFrame()
    matrix = (
        both_df
        .pivot(index=['train_missing_prop', 'test_missing_prop'], columns='model_name', values='mean_auc')
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return matrix




def build_post_adaptation_envelope(level1_df: pd.DataFrame):
    both_df = level1_df.loc[level1_df['scenario'] == 'both'].copy()
    if both_df.empty:
        return pd.DataFrame(columns=['model_name', 'test_missing_prop', 'envelope_train_missing_prop', 'envelope_mean_auc'])

    idx = both_df.groupby(['model_name', 'test_missing_prop'])['mean_auc'].idxmax()
    envelope_df = (
        both_df.loc[idx, ['model_name', 'test_missing_prop', 'train_missing_prop', 'mean_auc']]
        .rename(columns={
            'train_missing_prop': 'envelope_train_missing_prop',
            'mean_auc': 'envelope_mean_auc',
        })
        .sort_values(['model_name', 'test_missing_prop'])
        .reset_index(drop=True)
    )
    return envelope_df


def _normalized_trapezoid_auc(curve_df: pd.DataFrame, x_col: str, y_col: str):
    """Area under a performance-vs-missingness curve, normalized by x-range."""
    if curve_df.empty:
        return np.nan
    sub_df = curve_df[[x_col, y_col]].dropna().copy()
    if sub_df.empty:
        return np.nan
    sub_df[x_col] = sub_df[x_col].astype(float)
    sub_df[y_col] = sub_df[y_col].astype(float)
    sub_df = (
        sub_df
        .groupby(x_col, as_index=False)[y_col]
        .mean()
        .sort_values(x_col)
    )
    x = sub_df[x_col].to_numpy(dtype=float)
    y = sub_df[y_col].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if y.size == 0:
        return np.nan
    if y.size == 1 or np.isclose(float(np.max(x) - np.min(x)), 0.0):
        return float(y[0])
    trapezoid = getattr(np, 'trapezoid', np.trapz)
    return float(trapezoid(y, x) / (float(np.max(x) - np.min(x))))


def _safe_ratio(numerator, denominator):
    numerator = float(numerator) if np.isfinite(numerator) else np.nan
    denominator = float(denominator) if np.isfinite(denominator) else np.nan
    if not np.isfinite(numerator) or not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return np.nan
    return float(numerator / denominator)


def build_method_level_metrics(
    replicate_auc_df: pd.DataFrame,
    level1_df: pd.DataFrame,
    distillation_model_names=None,
):
    both_df = level1_df.loc[level1_df['scenario'] == 'both'].copy()
    if both_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    distillation_model_set = {
        _normalize_model_label(model_name)
        for model_name in (distillation_model_names or [])
    }

    baseline_df = both_df.loc[
        np.isclose(both_df['train_missing_prop'], 0.0) & np.isclose(both_df['test_missing_prop'], 0.0),
        ['model_name', 'mean_auc']
    ].copy()
    baseline_lookup = baseline_df.set_index('model_name')['mean_auc'].to_dict()

    envelope_df = build_post_adaptation_envelope(both_df)

    rows = []
    model_names = sorted(both_df['model_name'].astype(str).unique().tolist())
    for model_name in model_names:
        model_df = both_df.loc[both_df['model_name'].astype(str) == model_name].copy()
        train_curve_df = model_df.loc[np.isclose(model_df['test_missing_prop'], 0.0)].copy()
        test_curve_df = model_df.loc[np.isclose(model_df['train_missing_prop'], 0.0)].copy()
        envelope_curve_df = envelope_df.loc[envelope_df['model_name'].astype(str) == model_name].copy()

        baseline_auc = float(baseline_lookup.get(model_name, np.nan))
        is_distillation_method = _normalize_model_label(model_name) in distillation_model_set
        training_intuition_aupmc = _normalized_trapezoid_auc(
            train_curve_df,
            x_col='train_missing_prop',
            y_col='mean_auc',
        )
        if is_distillation_method:
            training_intuition_aupmc = np.nan
        inference_resilience_aupmc = _normalized_trapezoid_auc(
            test_curve_df,
            x_col='test_missing_prop',
            y_col='mean_auc',
        )
        adapted_resilience_aupmc = _normalized_trapezoid_auc(
            envelope_curve_df,
            x_col='test_missing_prop',
            y_col='envelope_mean_auc',
        )

        rows.append({
            'model_name': model_name,
            'is_distillation_method': bool(is_distillation_method),
            'baseline_performance_auc': baseline_auc,
            'training_intuition_aupmc': float(training_intuition_aupmc),
            'inference_resilience_aupmc': float(inference_resilience_aupmc),
            'adapted_resilience_aupmc': float(adapted_resilience_aupmc),
            'train_degradation_coefficient': _safe_ratio(training_intuition_aupmc, baseline_auc),
            'test_degradation_coefficient': _safe_ratio(inference_resilience_aupmc, baseline_auc),
            'minimum_degradation_coefficient': _safe_ratio(adapted_resilience_aupmc, baseline_auc),
        })

    metrics_df = pd.DataFrame(rows).sort_values(
        ['baseline_performance_auc', 'model_name'],
        ascending=[False, True],
    ).reset_index(drop=True)
    return metrics_df, envelope_df



def build_method_plot_summary(replicate_auc_df: pd.DataFrame, n_bootstrap=2000, confidence=0.95, random_seed=42):
    rows = []

    def _append_rows(group_df: pd.DataFrame, scenario_name: str, train_prop_fn, test_prop_fn, missing_prop_fn):
        for key, sub_df in group_df.groupby(['model_name', 'train_missing_prop', 'test_missing_prop'], sort=True):
            model_name, train_prop, test_prop = key
            stats = bootstrap_mean_ci(
                sub_df['auc'].to_numpy(dtype=float),
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                random_seed=_stable_seed(model_name, scenario_name, train_prop, test_prop, base_seed=random_seed),
            )
            rows.append({
                'model_name': model_name,
                'scenario': scenario_name,
                'train_prop': float(train_prop_fn(train_prop, test_prop)),
                'test_prop': float(test_prop_fn(train_prop, test_prop)),
                'missing_prop': float(missing_prop_fn(train_prop, test_prop)),
                'mean_auc': stats['mean'],
                'std_auc': float(sub_df['auc'].std(ddof=1)) if sub_df['auc'].notna().sum() > 1 else 0.0,
                'n_replicates': int(sub_df['auc'].notna().sum()),
                'auc_ci95': stats['ci_half_width'],
                'auc_ci95_lower': stats['ci_lower'],
                'auc_ci95_upper': stats['ci_upper'],
            })

    both_df = replicate_auc_df.copy()
    _append_rows(
        both_df,
        'both',
        train_prop_fn=lambda train_prop, test_prop: train_prop,
        test_prop_fn=lambda train_prop, test_prop: test_prop,
        missing_prop_fn=lambda train_prop, test_prop: np.nan,
    )

    train_df = replicate_auc_df.loc[np.isclose(replicate_auc_df['test_missing_prop'], 0.0)].copy()
    _append_rows(
        train_df,
        'train',
        train_prop_fn=lambda train_prop, test_prop: train_prop,
        test_prop_fn=lambda train_prop, test_prop: 0.0,
        missing_prop_fn=lambda train_prop, test_prop: float(train_prop),
    )

    test_df = replicate_auc_df.loc[np.isclose(replicate_auc_df['train_missing_prop'], 0.0)].copy()
    _append_rows(
        test_df,
        'test',
        train_prop_fn=lambda train_prop, test_prop: 0.0,
        test_prop_fn=lambda train_prop, test_prop: test_prop,
        missing_prop_fn=lambda train_prop, test_prop: float(test_prop),
    )

    summary_df = pd.DataFrame(rows).sort_values(['scenario', 'model_name', 'train_prop', 'test_prop']).reset_index(drop=True)
    if summary_df.empty:
        return summary_df

    both_summary_df = summary_df.loc[summary_df['scenario'] == 'both'].copy()
    envelope_idx = both_summary_df.groupby(['model_name', 'test_prop'])['mean_auc'].idxmax()
    envelope_df = both_summary_df.loc[envelope_idx].copy()
    envelope_df['scenario'] = 'envelope'
    envelope_df['missing_prop'] = envelope_df['test_prop'].astype(float)
    summary_df = pd.concat([summary_df, envelope_df], ignore_index=True)
    summary_df = summary_df.sort_values(['scenario', 'model_name', 'train_prop', 'test_prop']).reset_index(drop=True)

    return summary_df


def build_metric_ordering_table(method_level_metrics_df: pd.DataFrame, distillation_model_names=None):
    if method_level_metrics_df.empty:
        return pd.DataFrame()
    distillation_model_set = {
        _normalize_model_label(model_name)
        for model_name in (distillation_model_names or [])
    }
    metric_specs = [
        ('baseline_performance_auc', 'Baseline performance'),
        ('training_intuition_aupmc', 'Training intuition'),
        ('inference_resilience_aupmc', 'Inference resilience'),
        ('adapted_resilience_aupmc', 'Adapted resilience'),
        ('train_degradation_coefficient', 'Train degradation coefficient'),
        ('test_degradation_coefficient', 'Test degradation coefficient'),
        ('minimum_degradation_coefficient', 'Minimum degradation coefficient'),
    ]
    max_len = int(method_level_metrics_df['model_name'].nunique())
    out = {}
    for metric_col, label in metric_specs:
        if metric_col not in method_level_metrics_df.columns:
            continue
        metric_df = method_level_metrics_df[['model_name', metric_col]].copy()
        if metric_col in {'training_intuition_aupmc', 'train_degradation_coefficient'} and distillation_model_set:
            metric_df = metric_df.loc[
                ~metric_df['model_name'].astype(str).map(_normalize_model_label).isin(distillation_model_set)
            ].copy()
        ordered = (
            metric_df
            .sort_values([metric_col, 'model_name'], ascending=[False, True], na_position='last')
            ['model_name']
            .tolist()
        )
        if len(ordered) < max_len:
            ordered.extend([''] * (max_len - len(ordered)))
        out[label] = ordered
    return pd.DataFrame(out)


def plot_method_line_triplet(summary_df: pd.DataFrame, metric_col: str, title: str, ylabel: str, figures_dir: Path, file_name: str, figsize=None):
    if not HAVE_MPL:
        print(f'Matplotlib is not available. Skipping {metric_col} line plot.')
        return
    if summary_df.empty:
        print(f'No data available for {metric_col} line plot.')
        return

    color_map = _build_color_map(summary_df)
    ci_col = 'auc_ci95'
    n_models = summary_df['model_name'].nunique() if not summary_df.empty else 1
    resolved_figsize = tuple(figsize) if figsize is not None else _resolve_plot_figsize('line', n_models)
    fig, axes = plt.subplots(1, 3, figsize=resolved_figsize, sharey=True)

    panels = [
        ('train', 'Missing at Train', 'train_prop'),
        ('test', 'Missing at Test', 'test_prop'),
        ('envelope', 'Adapted Resilience Envelope', 'test_prop'),
    ]

    for ax, (scenario, panel_title, x_col) in zip(axes, panels):
        scenario_df = summary_df.loc[summary_df['scenario'] == scenario].copy()
        ax.set_title(panel_title, fontsize=14)
        ax.set_xlabel('Missing prop', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(alpha=0.25)

        if scenario_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        for model_name in sorted(scenario_df['model_name'].unique().tolist()):
            model_df = scenario_df.loc[scenario_df['model_name'] == model_name].sort_values(x_col)
            x = model_df[x_col].to_numpy(dtype=float)
            y = model_df[metric_col].to_numpy(dtype=float)
            ci = model_df[ci_col].fillna(0.0).to_numpy(dtype=float)
            lower = y - ci
            upper = y + ci
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)

            ax.plot(x, y, marker='o', linewidth=2, label=model_name, color=color_map[model_name])
            ax.fill_between(x, lower, upper, color=color_map[model_name], alpha=0.16)
        ax.set_xticks(sorted(scenario_df[x_col].dropna().unique().tolist()))
        ax.tick_params(axis='both', labelsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.92, 0.5))

    fig.suptitle(title, fontsize=15, y=0.955)
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.17, top=0.85, wspace=0.18)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / file_name
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f'Saved figure to: {figure_path}')
    plt.show()


def compute_level1_global_friedman(level1_df: pd.DataFrame):
    matrix = build_cell_model_mean_matrix(level1_df)
    if matrix.empty:
        return pd.DataFrame(columns=['n_cells', 'n_models', 'friedman_statistic', 'p_value', 'significant_p0_05'])

    complete_matrix = matrix.dropna(axis=0, how='any').copy()
    n_cells, n_models = complete_matrix.shape
    if n_models < 3 or n_cells < 2:
        return pd.DataFrame([{
            'n_cells': int(n_cells),
            'n_models': int(n_models),
            'friedman_statistic': np.nan,
            'p_value': np.nan,
            'significant_p0_05': False,
        }])

    stat, p_value = friedmanchisquare(*[complete_matrix[col].to_numpy(dtype=float) for col in complete_matrix.columns])
    return pd.DataFrame([{
        'n_cells': int(n_cells),
        'n_models': int(n_models),
        'friedman_statistic': float(stat),
        'p_value': float(p_value),
        'significant_p0_05': bool(float(p_value) < 0.05),
    }])


def compute_fixed_dataset_global_friedman(replicate_auc_df: pd.DataFrame):
    if replicate_auc_df.empty:
        return pd.DataFrame(columns=['n_replicates', 'n_models', 'friedman_statistic', 'p_value', 'significant_p0_05'])

    complete_matrix = (
        replicate_auc_df[['replicate_id', 'model_name', 'auc']]
        .pivot(index='replicate_id', columns='model_name', values='auc')
        .dropna(axis=0, how='any')
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    n_replicates, n_models = complete_matrix.shape
    if n_models < 3 or n_replicates < 2:
        return pd.DataFrame([{
            'n_replicates': int(n_replicates),
            'n_models': int(n_models),
            'friedman_statistic': np.nan,
            'p_value': np.nan,
            'significant_p0_05': False,
        }])

    stat, p_value = friedmanchisquare(*[complete_matrix[col].to_numpy(dtype=float) for col in complete_matrix.columns])
    return pd.DataFrame([{
        'n_replicates': int(n_replicates),
        'n_models': int(n_models),
        'friedman_statistic': float(stat),
        'p_value': float(p_value),
        'significant_p0_05': bool(float(p_value) < 0.05),
    }])


def _build_color_map(summary_df: pd.DataFrame):
    model_names = sorted(summary_df['model_name'].unique().tolist()) if not summary_df.empty else []
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(model_names), 1)))
    return {model_name: colors[idx] for idx, model_name in enumerate(model_names)}


def _resolve_plot_figsize(kind: str, n_models: int):
    n_models = max(int(n_models), 1)
    if kind == 'line':
        width = float(BASE_LINE_FIGSIZE[0]) + max(0, n_models - 3) * float(LINE_LEGEND_EXTRA_WIDTH_PER_MODEL)
        return (width, float(BASE_LINE_FIGSIZE[1]))
    if kind == 'heatmap':
        nrows, ncols = _resolve_heatmap_grid(n_models)
        width = float(BASE_HEATMAP_PANEL_WIDTH) * int(ncols)
        height = float(BASE_HEATMAP_PANEL_HEIGHT) * int(nrows)
        return (width, height)
    raise ValueError(f'Unknown plot kind: {kind}')


def _resolve_heatmap_grid(n_models: int):
    n_models = max(int(n_models), 1)
    if n_models <= 4:
        return 1, n_models
    if n_models <= 8:
        return 2, math.ceil(n_models / 2)
    raise ValueError(f'Heatmap layout supports at most 8 models with max 2 rows and max 4 columns, got {n_models}.')


def _resolve_heatmap_layout(nrows: int):
    if int(nrows) <= 1:
        return {
            'top': 0.81,
            'suptitle_y': 0.985,
            'hspace': 0.42,
            'bottom': 0.16,
        }
    return {
        'top': 0.87,
        'suptitle_y': 0.975,
        'hspace': 0.30,
        'bottom': 0.12,
    }


def plot_level1_auc_heatmaps(summary_df: pd.DataFrame, friedman_global_df: pd.DataFrame, title: str, figures_dir: Path, file_name: str, figsize=None):
    if not HAVE_MPL:
        print('Matplotlib is not available in this environment. Skipping level 1 heatmap plot.')
        return

    both_df = summary_df.loc[summary_df['scenario'] == 'both'].copy()
    if both_df.empty:
        print('No data available for level 1 AUC heatmaps.')
        return

    global_significant = False
    global_p_value = np.nan
    if friedman_global_df is not None and not friedman_global_df.empty:
        global_significant = bool(friedman_global_df.iloc[0].get('significant_p0_05', False))
        global_p_value = float(friedman_global_df.iloc[0].get('p_value', np.nan))

    model_names = sorted(both_df['model_name'].unique().tolist())
    n_models = len(model_names)
    nrows, ncols = _resolve_heatmap_grid(n_models)
    resolved_figsize = tuple(figsize) if figsize is not None else _resolve_plot_figsize('heatmap', n_models)
    fig, axes = plt.subplots(nrows, ncols, figsize=resolved_figsize, squeeze=False)
    axes_flat = axes.ravel()

    values = both_df['mean_auc'].to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        print('No finite values available for level 1 AUC heatmaps.')
        return

    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6

    level1_cmap = ListedColormap(plt.cm.viridis(np.linspace(0.00, 0.95, 256)))
    image = None
    used_axes = []
    for idx, model_name in enumerate(model_names):
        ax = axes_flat[idx]
        used_axes.append(ax)
        model_df = both_df.loc[both_df['model_name'] == model_name].copy()
        pivot = (
            model_df
            .pivot(index='train_prop', columns='test_prop', values='mean_auc')
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        pivot_ci = (
            model_df
            .pivot(index='train_prop', columns='test_prop', values='auc_ci95')
            .reindex(index=pivot.index, columns=pivot.columns)
        )

        x_edges = np.arange(len(pivot.columns) + 1, dtype=float)
        y_edges = np.arange(len(pivot.index) + 1, dtype=float)
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            pivot.to_numpy(dtype=float),
            cmap=level1_cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors='white',
            linewidth=0.45,
            shading='flat',
        )
        ax.set_xlim(0, len(pivot.columns))
        ax.set_ylim(len(pivot.index), 0)
        ax.set_aspect('equal')
        ax.set_title(model_name, fontsize=14)
        ax.set_xlabel('Test missing prop', fontsize=13)
        ax.set_ylabel('Train missing prop', fontsize=13)
        ax.set_xticks(np.arange(len(pivot.columns), dtype=float) + 0.5)
        ax.set_xticklabels([f'{value:g}' for value in pivot.columns], rotation=45, ha='right', fontsize=12)
        ax.set_yticks(np.arange(len(pivot.index), dtype=float) + 0.5)
        ax.set_yticklabels([f'{value:g}' for value in pivot.index], fontsize=12)
        ax.tick_params(axis='both', which='major', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for row_idx, train_prop in enumerate(pivot.index):
            for col_idx, test_prop in enumerate(pivot.columns):
                val = pivot.loc[train_prop, test_prop]
                ci = pivot_ci.loc[train_prop, test_prop]
                x_center = float(col_idx) + 0.5
                y_center = float(row_idx) + 0.5

                text_color = 'white'
                if pd.notna(val):
                    mean_text = f'{val:.3f}'
                    ax.text(
                        x_center,
                        y_center - 0.17,
                        mean_text,
                        ha='center',
                        va='center',
                        color=text_color,
                        fontsize=10,
                        fontweight='bold',
                    )
                    if pd.notna(ci):
                        ax.text(
                            x_center,
                            y_center + 0.19,
                            f'±{ci:.3f}',
                            ha='center',
                            va='center',
                            color=text_color,
                            fontsize=9,
                        )

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    layout = _resolve_heatmap_layout(nrows)
    fig.subplots_adjust(
        left=0.06,
        right=0.90,
        bottom=layout['bottom'],
        top=layout['top'],
        wspace=0.34,
        hspace=layout['hspace'],
    )
    footnote = '* global Friedman test significant (p < 0.05)'
    if np.isfinite(global_p_value):
        footnote = f'* global Friedman test significant (p = {global_p_value:.4g})' if global_significant else f'global Friedman test not significant (p = {global_p_value:.4g})'
    fig.text(0.5, -0.02, footnote, ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round,pad=1.10,rounding_size=0.40', facecolor='white', edgecolor='#7A8088', linewidth=1.0))

    fig.suptitle(str(title), fontsize=15, y=layout['suptitle_y'])
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / file_name
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f'Saved figure to: {figure_path}')
    plt.show()

def fdr_bh_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    out = np.full_like(p_values, np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)
    if not finite_mask.any():
        return out

    p = p_values[finite_mask]
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        running = min(running, val)
        adjusted[i] = running
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    out[finite_mask] = restored
    return out


def build_paired_replicate_auc_df(replicate_auc_df: pd.DataFrame, model_a: str, model_b: str, train_prop: float, test_prop: float):
    sub_df = replicate_auc_df.loc[
        np.isclose(replicate_auc_df['train_missing_prop'], float(train_prop))
        & np.isclose(replicate_auc_df['test_missing_prop'], float(test_prop))
        & replicate_auc_df['model_name'].isin([model_a, model_b])
    ].copy()
    if sub_df.empty:
        return pd.DataFrame(columns=['replicate_id', 'auc_a', 'auc_b'])

    wide_df = (
        sub_df
        .pivot(index='replicate_id', columns='model_name', values='auc')
        .dropna(how='any')
        .reset_index()
    )
    if model_a not in wide_df.columns or model_b not in wide_df.columns:
        return pd.DataFrame(columns=['replicate_id', 'auc_a', 'auc_b'])

    return wide_df[['replicate_id', model_a, model_b]].rename(columns={model_a: 'auc_a', model_b: 'auc_b'})


def compute_level2_pairwise_tests(level1_df: pd.DataFrame, replicate_auc_df: pd.DataFrame):
    rows = []
    both_df = level1_df.loc[level1_df['scenario'] == 'both'].copy()
    for (train_prop, test_prop), cond_df in both_df.groupby(['train_prop', 'test_prop'], sort=True):
        cond_df = cond_df.dropna(subset=['mean_auc']).sort_values(['mean_auc', 'model_name'], ascending=[False, True]).reset_index(drop=True)
        ranked_models = cond_df['model_name'].tolist()
        if len(ranked_models) < 2:
            continue

        mean_auc_lookup = cond_df.set_index('model_name')['mean_auc'].to_dict()
        cell_rows = []
        for winner_rank, winner_model in enumerate(ranked_models, start=1):
            for loser_rank, loser_model in enumerate(ranked_models[winner_rank:], start=winner_rank + 1):
                paired_df = build_paired_replicate_auc_df(
                    replicate_auc_df=replicate_auc_df,
                    model_a=winner_model,
                    model_b=loser_model,
                    train_prop=train_prop,
                    test_prop=test_prop,
                )
                winner_values = paired_df['auc_a'].to_numpy(dtype=float)
                loser_values = paired_df['auc_b'].to_numpy(dtype=float)
                delta_mean_auc = float(mean_auc_lookup[winner_model] - mean_auc_lookup[loser_model])
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        w_stat, p_value = wilcoxon(
                            winner_values,
                            loser_values,
                            zero_method='wilcox',
                            alternative='two-sided',
                            method='auto',
                        )
                except ValueError:
                    w_stat, p_value = np.nan, 1.0

                cell_rows.append({
                    'train_missing_prop': float(train_prop),
                    'test_missing_prop': float(test_prop),
                    'winner_rank': int(winner_rank),
                    'loser_rank': int(loser_rank),
                    'winner_model': str(winner_model),
                    'loser_model': str(loser_model),
                    'winner_mean_auc': float(mean_auc_lookup[winner_model]),
                    'loser_mean_auc': float(mean_auc_lookup[loser_model]),
                    'delta_mean_auc': delta_mean_auc,
                    'wilcoxon_statistic': float(w_stat) if np.isfinite(w_stat) else np.nan,
                    'p_value': float(p_value),
                    'n_replicates': int(len(paired_df)),
                })

        cell_df = pd.DataFrame(cell_rows)
        if not cell_df.empty:
            cell_df['p_value_fdr'] = fdr_bh_adjust(cell_df['p_value'].to_numpy(dtype=float))
            cell_df['significant_fdr_0p05'] = cell_df['p_value_fdr'] < 0.05
            cell_df['n_pairs_tested_in_cell'] = int(len(cell_df))
            rows.append(cell_df)

    if not rows:
        return pd.DataFrame(columns=[
            'train_missing_prop', 'test_missing_prop', 'winner_rank', 'loser_rank',
            'winner_model', 'loser_model', 'winner_mean_auc', 'loser_mean_auc',
            'delta_mean_auc', 'wilcoxon_statistic', 'p_value', 'p_value_fdr',
            'significant_fdr_0p05', 'n_replicates', 'n_pairs_tested_in_cell'
        ])

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values([
        'train_missing_prop', 'test_missing_prop', 'winner_rank', 'loser_rank'
    ]).reset_index(drop=True)


def build_fixed_dataset_method_significance_summary(method_summary_df: pd.DataFrame, pairwise_df: pd.DataFrame):
    if method_summary_df.empty:
        return pd.DataFrame(columns=['model_name', 'mean_auc', 'std_auc', 'n_replicates', 'auc_ci95', 'significant_wins', 'significant_losses'])

    summary_df = method_summary_df.copy()
    if pairwise_df.empty:
        summary_df['significant_wins'] = 0
        summary_df['significant_losses'] = 0
        return summary_df

    sig_df = pairwise_df.loc[pairwise_df['significant_fdr_0p05']].copy()
    wins = sig_df.groupby('winner_model').size().to_dict()
    losses = sig_df.groupby('loser_model').size().to_dict()
    summary_df['significant_wins'] = summary_df['model_name'].map(lambda name: int(wins.get(name, 0)))
    summary_df['significant_losses'] = summary_df['model_name'].map(lambda name: int(losses.get(name, 0)))
    return summary_df


def select_level2_plot_pairs(level2_pairwise_df: pd.DataFrame):
    if level2_pairwise_df.empty:
        return pd.DataFrame(columns=[
            'train_missing_prop', 'test_missing_prop', 'winner_rank', 'loser_rank',
            'winner_model', 'loser_model', 'winner_mean_auc', 'loser_mean_auc',
            'delta_mean_auc', 'wilcoxon_statistic', 'p_value', 'p_value_fdr',
            'significant_fdr_0p05', 'n_replicates', 'n_pairs_tested_in_cell',
            'winner_group_models', 'winner_group_ranks', 'winner_group_size', 'winner_group_text'
        ])

    rows = []
    for _, cell_df in level2_pairwise_df.groupby(['train_missing_prop', 'test_missing_prop'], sort=True):
        ordered_df = cell_df.sort_values(['winner_rank', 'loser_rank']).reset_index(drop=True)
        top_model_df = ordered_df.loc[ordered_df['winner_rank'] == 1].copy()
        sig_df = top_model_df.loc[top_model_df['significant_fdr_0p05']].sort_values('loser_rank').reset_index(drop=True)
        if sig_df.empty:
            continue

        target_row = sig_df.iloc[0]
        target_loser_model = str(target_row['loser_model'])
        target_loser_rank = int(target_row['loser_rank'])

        same_loser_sig_df = ordered_df.loc[
            (ordered_df['loser_model'].astype(str) == target_loser_model)
            & (ordered_df['loser_rank'].astype(int) == target_loser_rank)
            & (ordered_df['winner_rank'].astype(int) < target_loser_rank)
            & (ordered_df['significant_fdr_0p05'])
        ].sort_values('winner_rank').reset_index(drop=True)
        if same_loser_sig_df.empty:
            continue

        winner_group_models = same_loser_sig_df['winner_model'].astype(str).tolist()
        winner_group_ranks = same_loser_sig_df['winner_rank'].astype(int).tolist()

        plot_row = target_row.to_dict()
        plot_row['winner_group_models'] = '|'.join(winner_group_models)
        plot_row['winner_group_ranks'] = '|'.join(str(rank) for rank in winner_group_ranks)
        plot_row['winner_group_size'] = int(len(winner_group_models))
        plot_row['winner_group_text'] = ', '.join(winner_group_models)
        rows.append(plot_row)

    if not rows:
        return pd.DataFrame(columns=list(level2_pairwise_df.columns) + [
            'winner_group_models', 'winner_group_ranks', 'winner_group_size', 'winner_group_text'
        ])
    return pd.DataFrame(rows).sort_values(['train_missing_prop', 'test_missing_prop']).reset_index(drop=True)


def _format_winner_group_text(models_text: str, max_names_per_line=2):
    names = [part.strip() for part in str(models_text).split(',') if part.strip()]
    if not names:
        return ''
    lines = []
    for idx in range(0, len(names), max_names_per_line):
        lines.append(', '.join(names[idx:idx + max_names_per_line]))
    return '\n'.join(lines)


def plot_level2_significant_pairs_heatmap(level2_plot_df: pd.DataFrame, title: str, figures_dir: Path, file_name: str):
    if not HAVE_MPL:
        print('Matplotlib is not available in this environment. Skipping condition-level summary heatmap.')
        return
    if level2_plot_df.empty:
        print('No statistically significant condition-level summary pairs available for plotting after FDR correction.')
        return

    winner_model_names = sorted(level2_plot_df['winner_model'].astype(str).unique().tolist())
    color_lookup = {name: idx for idx, name in enumerate(winner_model_names)}
    level2_hex = ['#012a4a', '#013a63', '#01497c', '#014f86', '#2a6f97', '#2c7da0', '#468faf', '#61a5c2', '#89c2d9', '#a9d6e5']
    base_cmap = LinearSegmentedColormap.from_list('level2_custom', level2_hex)
    cmap = ListedColormap(base_cmap(np.linspace(0.0, 1.0, max(len(winner_model_names), 1))))
    cmap.set_bad('#F2F2F2')

    train_props = sorted(level2_plot_df['train_missing_prop'].unique().tolist())
    test_props = sorted(level2_plot_df['test_missing_prop'].unique().tolist())
    matrix = np.full((len(train_props), len(test_props)), np.nan, dtype=float)
    for _, row in level2_plot_df.iterrows():
        i = train_props.index(float(row['train_missing_prop']))
        j = test_props.index(float(row['test_missing_prop']))
        matrix[i, j] = float(color_lookup[row['winner_model']])

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    x_edges = np.arange(len(test_props) + 1, dtype=float)
    y_edges = np.arange(len(train_props) + 1, dtype=float)
    ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(winner_model_names) - 0.5,
        edgecolors='white',
        linewidth=0.55,
        shading='flat',
    )
    ax.set_xlim(0, len(test_props))
    ax.set_ylim(len(train_props), 0)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13, pad=22, loc='center')
    ax.set_xlabel('Test missing prop', fontsize=12)
    ax.set_ylabel('Train missing prop', fontsize=12)
    ax.set_xticks(np.arange(len(test_props), dtype=float) + 0.5)
    ax.set_xticklabels([f'{value:g}' for value in test_props], rotation=45, ha='right', fontsize=11)
    ax.set_yticks(np.arange(len(train_props), dtype=float) + 0.5)
    ax.set_yticklabels([f'{value:g}' for value in train_props], fontsize=11)
    ax.tick_params(axis='both', which='major', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for _, row in level2_plot_df.iterrows():
        i = train_props.index(float(row['train_missing_prop']))
        j = test_props.index(float(row['test_missing_prop']))
        x_center = float(j) + 0.5
        y_center = float(i) + 0.5
        winner_group_text = _format_winner_group_text(row.get('winner_group_text', row['winner_model']), max_names_per_line=1)
        full_text = winner_group_text + f"\n> {row['loser_model']} ({int(row['loser_rank'])})"
        ax.text(
            x_center,
            y_center,
            full_text,
            ha='center',
            va='center',
            color='white',
            fontsize=7.5,
            fontweight='bold',
            linespacing=1.12,
        )

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / file_name
    plt.savefig(figure_path, dpi=1000, bbox_inches='tight')
    print(f'Saved figure to: {figure_path}')
    plt.show()


def plot_level3_pairwise_condition_matrices(level2_pairwise_df: pd.DataFrame, title: str, figures_dir: Path, file_name: str):
    if not HAVE_MPL:
        print('Matplotlib is not available in this environment. Skipping condition-level pairwise ΔAUC matrices.')
        return
    if level2_pairwise_df.empty:
        print('No pairwise condition-level results available for plotting.')
        return

    sig_df = level2_pairwise_df.loc[level2_pairwise_df['significant_fdr_0p05']].copy()
    if sig_df.empty:
        print('No statistically significant pairwise comparisons available for the condition-level matrices.')
        return

    train_props = sorted(level2_pairwise_df['train_missing_prop'].unique().tolist())
    test_props = sorted(level2_pairwise_df['test_missing_prop'].unique().tolist())

    max_abs_delta = float(sig_df['delta_mean_auc'].abs().max()) if not sig_df.empty else 0.0
    if not np.isfinite(max_abs_delta) or max_abs_delta <= 0.0:
        max_abs_delta = 1e-6

    panel_model_counts = []
    for train_prop in train_props:
        for test_prop in test_props:
            cond_df = level2_pairwise_df.loc[
                np.isclose(level2_pairwise_df['train_missing_prop'], float(train_prop))
                & np.isclose(level2_pairwise_df['test_missing_prop'], float(test_prop))
            ].copy()
            mean_auc_lookup = {}
            for row in cond_df.itertuples(index=False):
                mean_auc_lookup[str(row.winner_model)] = float(row.winner_mean_auc)
                mean_auc_lookup[str(row.loser_model)] = float(row.loser_mean_auc)
            panel_model_counts.append(len(mean_auc_lookup))
    n_models_max = max(panel_model_counts) if panel_model_counts else 0

    single_panel = len(train_props) == 1 and len(test_props) == 1
    fig_width = 15.0 if single_panel else max(22.0, 4.1 * len(test_props) + 4.2)
    fig_height = 13.0 if single_panel else max(22.0, 4.1 * len(train_props) + 4.0)
    fig, axes = plt.subplots(len(train_props), len(test_props), figsize=(fig_width, fig_height), squeeze=False)

    cmap = plt.cm.Blues.copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    norm = Normalize(vmin=0.0, vmax=max_abs_delta)
    image = None
    title_fontsize = 18 if single_panel else 12
    tick_fontsize = 15 if single_panel else 8.4
    xtick_fontsize = 14 if single_panel else 7.8
    cell_fontsize = 13 if single_panel else 6.0
    panel_title_pad = 0 if single_panel else 11
    cell_linewidth = 0.55 if single_panel else 0.30

    for row_idx, train_prop in enumerate(train_props):
        for col_idx, test_prop in enumerate(test_props):
            ax = axes[row_idx, col_idx]
            cond_df = level2_pairwise_df.loc[
                np.isclose(level2_pairwise_df['train_missing_prop'], float(train_prop))
                & np.isclose(level2_pairwise_df['test_missing_prop'], float(test_prop))
            ].copy()
            condition_df = cond_df.loc[cond_df['significant_fdr_0p05']].copy()

            mean_auc_lookup = {}
            for row in cond_df.itertuples(index=False):
                mean_auc_lookup[str(row.winner_model)] = float(row.winner_mean_auc)
                mean_auc_lookup[str(row.loser_model)] = float(row.loser_mean_auc)
            ordered_models = [
                name for name, _ in sorted(mean_auc_lookup.items(), key=lambda item: (-item[1], item[0]))
            ]
            n_models = len(ordered_models)
            display_model_names_x = [f'{name} ({idx + 1})' for idx, name in enumerate(ordered_models)]
            display_model_names_y = [f'{name} ({idx + 1})' for idx, name in enumerate(ordered_models)]
            model_to_idx = {name: idx for idx, name in enumerate(ordered_models)}

            matrix = np.full((n_models, n_models), np.nan, dtype=float)
            for _, pair_row in condition_df.iterrows():
                winner_idx = model_to_idx[pair_row['winner_model']]
                loser_idx = model_to_idx[pair_row['loser_model']]
                delta = float(pair_row['delta_mean_auc'])
                matrix[winner_idx, loser_idx] = delta

            matrix[np.diag_indices(n_models)] = np.nan

            x_edges = np.arange(n_models + 1, dtype=float)
            y_edges = np.arange(n_models + 1, dtype=float)
            ax.set_facecolor('white')
            for mat_row in range(n_models):
                for mat_col in range(mat_row + 1, n_models):
                    ax.add_patch(Rectangle(
                        (float(mat_col), float(mat_row)),
                        1.0,
                        1.0,
                        facecolor='#E6E6E6',
                        edgecolor='white',
                        linewidth=cell_linewidth,
                        zorder=0,
                    ))
            image = ax.pcolormesh(
                x_edges,
                y_edges,
                matrix,
                cmap=cmap,
                norm=norm,
                edgecolors='white',
                linewidth=cell_linewidth,
                shading='flat',
                zorder=1,
            )
            ax.set_xlim(0, n_models)
            ax.set_ylim(n_models, 0)
            ax.set_aspect('equal')
            if not single_panel:
                ax.set_title(f'train={train_prop:g} | test={test_prop:g}', fontsize=title_fontsize, pad=panel_title_pad)
            ax.tick_params(axis='both', which='major', length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_xticks(np.arange(n_models, dtype=float) + 0.5)
            ax.set_yticks(np.arange(n_models, dtype=float) + 0.5)
            ax.set_xticklabels(display_model_names_x, rotation=90, ha='center', va='top', fontsize=xtick_fontsize)
            ax.set_yticklabels(display_model_names_y, fontsize=tick_fontsize)

            finite_coords = np.argwhere(np.isfinite(matrix))
            for mat_row, mat_col in finite_coords:
                value = float(matrix[mat_row, mat_col])
                text_color = 'white' if abs(value) >= (0.42 * max_abs_delta) else 'black'
                ax.text(
                    float(mat_col) + 0.5,
                    float(mat_row) + 0.5,
                    f'{value:+.03f}',
                    ha='center',
                    va='center',
                    fontsize=cell_fontsize,
                    fontweight='bold' if single_panel else 'normal',
                    color=text_color,
                )

    if single_panel:
        fig.subplots_adjust(left=0.18, right=0.96, bottom=0.30, top=0.86)
        fig.suptitle(title, fontsize=22, y=0.94)
    else:
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.14, top=0.94, wspace=0.50, hspace=0.34)
        fig.suptitle(title, fontsize=19, y=0.985)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / file_name
    target_long_side_px = 3840
    export_dpi = max(300, int(round(target_long_side_px / max(fig_width, fig_height))))
    plt.savefig(figure_path, dpi=export_dpi, bbox_inches='tight')
    print(f'Saved figure to: {figure_path}')
    plt.show()
