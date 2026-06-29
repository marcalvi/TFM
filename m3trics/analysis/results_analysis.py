
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

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    Image = None
    HAVE_PIL = False

MAX_FIGURE_LONG_SIDE_PX = 2200

DISPLAY_NAME_MAP = {
    'ZERO_MLP': 'ZI_MLP',
    'KNN_MLP': 'KNN_MLP',
    'ZI_LR': 'ZI_LR',
    'ZILR': 'ZI_LR',
    'KNN_LR': 'KNN_LR',
    'KNNLR': 'KNN_LR',
    'ZI_RF': 'ZI_RF',
    'ZIRF': 'ZI_RF',
    'KNN_RF': 'KNN_RF',
    'KNNRF': 'KNN_RF',
    'ZI_COXNET': 'ZI_CoxNet',
    'ZICOXNET': 'ZI_CoxNet',
    'KNN_COXNET': 'KNN_CoxNet',
    'KNNCOXNET': 'KNN_CoxNet',
    'ZI_RSF': 'ZI_RSF',
    'ZIRSF': 'ZI_RSF',
    'KNN_RSF': 'KNN_RSF',
    'KNNRSF': 'KNN_RSF',
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
INNER_MODEL_RISK_RE = re.compile(r'^inner_model_(\d+)_risk$')

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


def _distillation_model_set(distillation_model_names=None):
    return {
        _normalize_model_label(model_name)
        for model_name in (distillation_model_names or [])
    }


def _is_distillation_method(model_name, distillation_model_names=None):
    raw = str(model_name).strip()
    normalized = _normalize_model_label(raw)
    configured = _distillation_model_set(distillation_model_names)
    return (
        normalized in configured
        or raw.upper().endswith('_KD')
        or normalized in {'Di-PAM', 'Di-MMLP'}  # legacy outputs only
    )


def _stable_seed(*parts, base_seed=0):
    text = '|'.join(str(part) for part in parts)
    return (zlib.crc32(text.encode('utf-8')) + int(base_seed)) % (2**32 - 1)


def _save_current_figure(figures_dir: Path, file_name: str, dpi: int = 220, max_long_side_px: int = MAX_FIGURE_LONG_SIDE_PX):
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / file_name

    fig = plt.gcf()
    width_in, height_in = fig.get_size_inches()
    max_inches = max(float(width_in), float(height_in), 1.0)
    dpi_cap = max(72, int(math.floor(max_long_side_px / max_inches)))
    export_dpi = min(int(dpi), dpi_cap)

    plt.savefig(figure_path, dpi=export_dpi, bbox_inches='tight')

    if HAVE_PIL and figure_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
        with Image.open(figure_path) as image:
            width_px, height_px = image.size
            long_side_px = max(width_px, height_px)
            if long_side_px > max_long_side_px:
                scale = max_long_side_px / float(long_side_px)
                new_size = (
                    max(1, int(round(width_px * scale))),
                    max(1, int(round(height_px * scale))),
                )
                resized = image.resize(new_size, Image.LANCZOS)
                save_kwargs = {'optimize': True}
                if figure_path.suffix.lower() in {'.jpg', '.jpeg'}:
                    save_kwargs['quality'] = 90
                resized.save(figure_path, **save_kwargs)

    print(f'Saved figure to: {figure_path}')
    return figure_path


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


def list_model_sources(results_root: Path, dataset_name: str, train_degrading_modality: str = 'GLOBAL', model_names=None, retrain_outer=None, results_mode: str = 'decay'):
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
            run_data_dir = run_dir / 'TRAIN_MISSING' / train_degrading_modality.upper()
            if not run_data_dir.exists():
                continue

        sources.append({
            'model_name': model_label,
            'run_dir': run_dir,
            'run_data_dir': run_data_dir,
            'results_mode': results_mode,
        })
    return sources


def resolve_requested_model_names(results_root: Path, dataset_name: str, train_degrading_modality: str = 'GLOBAL', retrain_outer=None, results_mode: str = 'decay'):
    return sorted({
        source['model_name']
        for source in list_model_sources(
            results_root,
            dataset_name,
            train_degrading_modality,
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


def _has_git_conflict_markers(path: Path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
            for line in handle:
                if line.startswith(('<<<<<<<', '=======', '>>>>>>>')):
                    return True
    except OSError:
        return False
    return False


def _list_inner_model_prob_cols(df: pd.DataFrame):
    cols = []
    for col in df.columns:
        match = INNER_MODEL_PROB_RE.match(str(col))
        if match:
            cols.append((int(match.group(1)), col))
    cols.sort(key=lambda item: item[0])
    return cols


def _list_inner_model_risk_cols(df: pd.DataFrame):
    cols = []
    for col in df.columns:
        match = INNER_MODEL_RISK_RE.match(str(col))
        if match:
            cols.append((int(match.group(1)), col))
    cols.sort(key=lambda item: item[0])
    return cols


def add_ensemble_prediction_columns(pred_df: pd.DataFrame):
    """Return a copy with probability-averaged ensemble prediction columns."""
    out = pred_df.copy()
    prob_cols = [col for _, col in _list_inner_model_prob_cols(out)]
    if not prob_cols:
        raise ValueError('Cannot compute ensemble predictions because no inner_model_*_prob columns were found.')

    prob_df = out[prob_cols].apply(pd.to_numeric, errors='coerce')
    ensemble_prob = prob_df.mean(axis=1, skipna=True)
    ensemble_n_models = prob_df.notna().sum(axis=1)

    out['ensemble_prob'] = ensemble_prob
    out['ensemble_n_models'] = ensemble_n_models.astype(int)
    clipped_prob = np.clip(ensemble_prob.astype(float), 1e-12, 1.0 - 1e-12)
    out['ensemble_logit'] = np.log(clipped_prob / (1.0 - clipped_prob))
    out.loc[ensemble_n_models.eq(0), ['ensemble_prob', 'ensemble_logit']] = np.nan
    out['ensemble_pred_label'] = (out['ensemble_prob'].astype(float) >= 0.5).astype('Int64')
    out.loc[out['ensemble_prob'].isna(), 'ensemble_pred_label'] = pd.NA
    return out


def normalize_prediction_df(pred_df: pd.DataFrame, use_ensemble: bool = False):
    out = pred_df.copy()
    legacy_column_map = {
        'train_missing_location': 'train_degrading_modality',
        'test_missing_location': 'test_degrading_modality',
        'eval_missing_location': 'eval_degrading_modality',
        'missing_location': 'degrading_modality',
    }
    for old_col, new_col in legacy_column_map.items():
        if new_col not in out.columns and old_col in out.columns:
            out[new_col] = out[old_col]
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
    if bool(use_ensemble):
        if 'ensemble_prob' not in out.columns:
            out = add_ensemble_prediction_columns(out)
    elif not inner_prob_cols:
        raise ValueError('No inner_model_*_prob columns found in prediction dataframe.')

    if 'outer_eval_target' in out.columns:
        out = out.loc[out['outer_eval_target'].astype(str) == 'test_outer'].copy()

    out['model_name'] = out['model_name'].astype(str)
    out['patient'] = out['patient'].astype(str)
    out['train_missing_prop'] = out['train_missing_prop'].astype(float)
    out['test_missing_prop'] = out['test_missing_prop'].astype(float)
    out['y_true'] = out['y_true'].astype(int)
    if bool(use_ensemble):
        out['ensemble_prob'] = out['ensemble_prob'].astype(float)
        if 'ensemble_logit' in out.columns:
            out['ensemble_logit'] = out['ensemble_logit'].astype(float)
        if 'ensemble_n_models' in out.columns:
            out['ensemble_n_models'] = pd.to_numeric(
                out['ensemble_n_models'],
                errors='coerce',
            ).fillna(0).astype(int)
    if 'seed' not in out.columns:
        out['seed'] = np.nan
    if 'outer_fold' not in out.columns:
        out['outer_fold'] = np.nan
    return out.reset_index(drop=True)


def load_all_test_predictions(results_root: Path, dataset_name: str, train_degrading_modality: str = 'GLOBAL', model_names=None, retrain_outer=None, results_mode: str = 'decay', use_ensemble: bool = False):
    frames = []
    missing_prediction_files = []
    for source in list_model_sources(
        results_root,
        dataset_name,
        train_degrading_modality,
        model_names=model_names,
        retrain_outer=retrain_outer,
        results_mode=results_mode,
    ):
        found_any = False
        for path in sorted(source['run_data_dir'].rglob('test_predictions.csv')):
            found_any = True
            if _has_git_conflict_markers(path):
                missing_prediction_files.append(
                    f'Skipped corrupted prediction file with unresolved git conflict markers: {path}'
                )
                continue
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
    return normalize_prediction_df(pd.concat(frames, ignore_index=True), use_ensemble=use_ensemble), missing_prediction_files


def add_survival_ensemble_prediction_columns(pred_df: pd.DataFrame):
    """Return a copy with risk-averaged ensemble prediction columns."""
    out = pred_df.copy()
    risk_cols = [col for _, col in _list_inner_model_risk_cols(out)]
    if not risk_cols:
        raise ValueError('Cannot compute survival ensemble predictions because no inner_model_*_risk columns were found.')

    risk_df = out[risk_cols].apply(pd.to_numeric, errors='coerce')
    out['ensemble_risk'] = risk_df.mean(axis=1, skipna=True)
    out['ensemble_n_models'] = risk_df.notna().sum(axis=1).astype(int)
    out.loc[out['ensemble_n_models'].eq(0), 'ensemble_risk'] = np.nan
    return out


def normalize_survival_prediction_df(pred_df: pd.DataFrame, use_ensemble: bool = False):
    out = pred_df.copy()
    legacy_column_map = {
        'train_missing_location': 'train_degrading_modality',
        'test_missing_location': 'test_degrading_modality',
        'eval_missing_location': 'eval_degrading_modality',
        'missing_location': 'degrading_modality',
    }
    for old_col, new_col in legacy_column_map.items():
        if new_col not in out.columns and old_col in out.columns:
            out[new_col] = out[old_col]
    required_cols = [
        'model_name',
        'patient',
        'train_missing_prop',
        'test_missing_prop',
        'event_time',
        'event_observed',
    ]
    missing_cols = [col for col in required_cols if col not in out.columns]
    if missing_cols:
        raise ValueError(f'Missing required survival prediction columns: {missing_cols}')

    inner_risk_cols = _list_inner_model_risk_cols(out)
    if bool(use_ensemble):
        if 'ensemble_risk' not in out.columns:
            out = add_survival_ensemble_prediction_columns(out)
    elif not inner_risk_cols:
        raise ValueError('No inner_model_*_risk columns found in survival prediction dataframe.')

    if 'outer_eval_target' in out.columns:
        out = out.loc[out['outer_eval_target'].astype(str) == 'test_outer'].copy()

    out['model_name'] = out['model_name'].astype(str)
    out['patient'] = out['patient'].astype(str)
    out['train_missing_prop'] = out['train_missing_prop'].astype(float)
    out['test_missing_prop'] = out['test_missing_prop'].astype(float)
    out['event_time'] = pd.to_numeric(out['event_time'], errors='coerce').astype(float)
    out['event_observed'] = pd.to_numeric(out['event_observed'], errors='coerce').fillna(0).astype(int)
    if 'censorship' in out.columns:
        out['censorship'] = pd.to_numeric(out['censorship'], errors='coerce')
    if 'y_disc' in out.columns:
        out['y_disc'] = pd.to_numeric(out['y_disc'], errors='coerce')
    if bool(use_ensemble):
        out['ensemble_risk'] = pd.to_numeric(out['ensemble_risk'], errors='coerce')
        if 'ensemble_n_models' in out.columns:
            out['ensemble_n_models'] = pd.to_numeric(
                out['ensemble_n_models'],
                errors='coerce',
            ).fillna(0).astype(int)
    if 'seed' not in out.columns:
        out['seed'] = np.nan
    if 'outer_fold' not in out.columns:
        out['outer_fold'] = np.nan
    return out.reset_index(drop=True)


def load_all_survival_test_predictions(results_root: Path, dataset_name: str, train_degrading_modality: str = 'GLOBAL', model_names=None, retrain_outer=None, results_mode: str = 'decay', use_ensemble: bool = False):
    frames = []
    missing_prediction_files = []
    for source in list_model_sources(
        results_root,
        dataset_name,
        train_degrading_modality,
        model_names=model_names,
        retrain_outer=retrain_outer,
        results_mode=results_mode,
    ):
        found_any = False
        for path in sorted(source['run_data_dir'].rglob('test_predictions.csv')):
            found_any = True
            if _has_git_conflict_markers(path):
                missing_prediction_files.append(
                    f'Skipped corrupted prediction file with unresolved git conflict markers: {path}'
                )
                continue
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
    return normalize_survival_prediction_df(pd.concat(frames, ignore_index=True), use_ensemble=use_ensemble), missing_prediction_files


def expand_inner_model_survival_predictions(pred_df: pd.DataFrame, use_ensemble: bool = False):
    if pred_df.empty:
        return pd.DataFrame()

    base_cols = [
        'model_name', 'patient', 'train_missing_prop', 'test_missing_prop',
        'event_time', 'event_observed', 'seed', 'outer_fold'
    ]
    optional_cols = [col for col in ['censorship', 'y_disc'] if col in pred_df.columns]

    if bool(use_ensemble):
        if 'ensemble_risk' not in pred_df.columns:
            pred_df = add_survival_ensemble_prediction_columns(pred_df)
        long_df = pred_df[base_cols + optional_cols + ['ensemble_risk']].copy()
        long_df = long_df.rename(columns={'ensemble_risk': 'member_risk'})
        long_df['inner_model_idx'] = 0
        long_df['prediction_source'] = 'ensemble'
        long_df['replicate_id'] = (
            long_df['seed'].astype(str)
            + '|outer_' + long_df['outer_fold'].astype(str)
            + '|ensemble'
        )
        long_df['member_risk'] = long_df['member_risk'].astype(float)
        return long_df.dropna(subset=['member_risk', 'event_time', 'event_observed']).reset_index(drop=True)

    risk_cols = _list_inner_model_risk_cols(pred_df)
    rows = []
    for member_idx, risk_col in risk_cols:
        sub_df = pred_df[base_cols + optional_cols + [risk_col]].copy()
        sub_df = sub_df.rename(columns={risk_col: 'member_risk'})
        sub_df['inner_model_idx'] = int(member_idx)
        sub_df['prediction_source'] = 'inner_model'
        sub_df['replicate_id'] = (
            sub_df['seed'].astype(str)
            + '|outer_' + sub_df['outer_fold'].astype(str)
            + '|inner_' + sub_df['inner_model_idx'].astype(str)
        )
        rows.append(sub_df)

    if not rows:
        return pd.DataFrame()
    long_df = pd.concat(rows, ignore_index=True)
    long_df['member_risk'] = long_df['member_risk'].astype(float)
    return long_df.dropna(subset=['member_risk', 'event_time', 'event_observed']).reset_index(drop=True)


def aggregate_member_patient_survival_predictions(member_pred_df: pd.DataFrame):
    if member_pred_df.empty:
        return pd.DataFrame()

    if 'prediction_source' not in member_pred_df.columns:
        member_pred_df = member_pred_df.copy()
        member_pred_df['prediction_source'] = 'inner_model'

    group_cols = [
        'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold',
        'prediction_source', 'inner_model_idx', 'replicate_id', 'patient',
        'event_time', 'event_observed'
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
            'Found duplicated survival predictions for the same seed x outer_fold x replicate x patient x missingness cell. '
            'This notebook does not collapse them into a single predictor. Sample duplicated groups: '
            f'{sample_rows}'
        )

    out_df = (
        member_pred_df[group_cols + ['member_risk']]
        .copy()
        .sort_values([
            'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold',
            'prediction_source', 'inner_model_idx', 'patient'
        ])
        .reset_index(drop=True)
    )
    out_df['n_prediction_rows'] = 1
    return out_df


def safe_cindex(event_observed, event_time, risk_score):
    event_observed = np.asarray(event_observed, dtype=bool).reshape(-1)
    event_time = np.asarray(event_time, dtype=np.float64).reshape(-1)
    risk_score = np.asarray(risk_score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(event_time) & np.isfinite(risk_score)
    event_observed = event_observed[finite]
    event_time = event_time[finite]
    risk_score = risk_score[finite]
    if not (len(event_observed) == len(event_time) == len(risk_score)):
        raise ValueError('safe_cindex expects equal-length arrays.')
    if len(event_time) < 2:
        return np.nan

    concordant = 0.0
    permissible = 0.0
    for i in range(len(event_time)):
        for j in range(i + 1, len(event_time)):
            comparable = False
            sign = 0.0
            if event_observed[i] and event_time[i] < event_time[j]:
                comparable = True
                sign = np.sign(risk_score[i] - risk_score[j])
            elif event_observed[j] and event_time[j] < event_time[i]:
                comparable = True
                sign = np.sign(risk_score[j] - risk_score[i])
            if not comparable:
                continue
            permissible += 1.0
            if sign > 0:
                concordant += 1.0
            elif sign == 0:
                concordant += 0.5
    if permissible == 0.0:
        return np.nan
    return float(concordant / permissible)


def build_replicate_cindex_table(member_patient_df: pd.DataFrame):
    if member_patient_df.empty:
        return pd.DataFrame()
    if 'prediction_source' not in member_patient_df.columns:
        member_patient_df = member_patient_df.copy()
        member_patient_df['prediction_source'] = 'inner_model'
    rows = []
    group_cols = [
        'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold',
        'prediction_source', 'inner_model_idx', 'replicate_id'
    ]
    for key, group_df in member_patient_df.groupby(group_cols, sort=True):
        model_name, train_prop, test_prop, seed, outer_fold, prediction_source, inner_idx, replicate_id = key
        cindex_val = safe_cindex(
            group_df['event_observed'].to_numpy(),
            group_df['event_time'].to_numpy(),
            group_df['member_risk'].to_numpy(),
        )
        rows.append({
            'model_name': model_name,
            'train_missing_prop': float(train_prop),
            'test_missing_prop': float(test_prop),
            'seed': seed,
            'outer_fold': outer_fold,
            'prediction_source': str(prediction_source),
            'inner_model_idx': int(inner_idx),
            'replicate_id': str(replicate_id),
            'n_patients': int(group_df['patient'].nunique()),
            'cindex': cindex_val,
        })
    return pd.DataFrame(rows).sort_values([
        'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold',
        'prediction_source', 'inner_model_idx'
    ]).reset_index(drop=True)


def cindex_replicates_to_auc_compatible(replicate_cindex_df: pd.DataFrame):
    """Rename C-index replicate values to the internal AUC-compatible column name."""
    if replicate_cindex_df.empty:
        return replicate_cindex_df.copy()
    if 'cindex' not in replicate_cindex_df.columns:
        raise ValueError("Expected a 'cindex' column.")
    out = replicate_cindex_df.copy()
    out['auc'] = out['cindex']
    return out.drop(columns=['cindex'])


_CINDEX_COLUMN_RENAME = {
    'auc': 'cindex',
    'mean_auc': 'mean_cindex',
    'std_auc': 'std_cindex',
    'auc_ci95': 'cindex_ci95',
    'auc_ci95_lower': 'cindex_ci95_lower',
    'auc_ci95_upper': 'cindex_ci95_upper',
    'baseline_auc': 'baseline_cindex',
    'train_time_aupmc': 'train_time_cindex_aupmc',
    'test_time_aupmc': 'test_time_cindex_aupmc',
    'best_fixed_train_aupmc': 'best_fixed_train_cindex_aupmc',
    'selected_train_aupmc': 'selected_train_cindex_aupmc',
    'envelope_mean_auc': 'envelope_mean_cindex',
    'winner_mean_auc': 'winner_mean_cindex',
    'loser_mean_auc': 'loser_mean_cindex',
    'delta_mean_auc': 'delta_mean_cindex',
}


def rename_auc_outputs_for_cindex(df: pd.DataFrame):
    """Return a copy with AUC-specific column names replaced by C-index names."""
    out = df.copy()
    rename_map = {}
    for col in out.columns:
        if col in _CINDEX_COLUMN_RENAME:
            rename_map[col] = _CINDEX_COLUMN_RENAME[col]
            continue
        new_col = col
        for old_fragment, new_fragment in [
            ('baseline_auc', 'baseline_cindex'),
            ('train_time_aupmc', 'train_time_cindex_aupmc'),
            ('test_time_aupmc', 'test_time_cindex_aupmc'),
            ('best_fixed_train_aupmc', 'best_fixed_train_cindex_aupmc'),
            ('selected_train_aupmc', 'selected_train_cindex_aupmc'),
            ('mean_auc', 'mean_cindex'),
            ('std_auc', 'std_cindex'),
            ('auc_ci95', 'cindex_ci95'),
        ]:
            new_col = new_col.replace(old_fragment, new_fragment)
        if new_col != col:
            rename_map[col] = new_col
    return out.rename(columns=rename_map)


def rename_general_results_summary_for_cindex(df: pd.DataFrame):
    out = df.copy()
    if out.empty:
        return out
    text_replacements = {
        'baseline_auc': 'baseline_cindex',
        'train_time_aupmc': 'train_time_cindex_aupmc',
        'test_time_aupmc': 'test_time_cindex_aupmc',
        'best_fixed_train_aupmc': 'best_fixed_train_cindex_aupmc',
        'Baseline AUC': 'Baseline C-index',
        'Train-time AUPMC': 'Train-time C-index AUPMC',
        'Test-time AUPMC': 'Test-time C-index AUPMC',
        'Best fixed-train AUPMC': 'Best fixed-train C-index AUPMC',
        'AUC': 'C-index',
    }
    for col in ['selection_metric', 'rank_basis', 'details']:
        if col not in out.columns:
            continue
        series = out[col].astype(str)
        for old, new in text_replacements.items():
            series = series.str.replace(old, new, regex=False)
        out[col] = series
    return out


def expand_inner_model_predictions(pred_df: pd.DataFrame, use_ensemble: bool = False):
    if pred_df.empty:
        return pd.DataFrame()

    base_cols = [
        'model_name', 'patient', 'train_missing_prop', 'test_missing_prop', 'y_true',
        'seed', 'outer_fold'
    ]

    if bool(use_ensemble):
        if 'ensemble_prob' not in pred_df.columns:
            pred_df = add_ensemble_prediction_columns(pred_df)
        long_df = pred_df[base_cols + ['ensemble_prob']].copy()
        long_df = long_df.rename(columns={'ensemble_prob': 'member_prob'})
        long_df['inner_model_idx'] = 0
        long_df['prediction_source'] = 'ensemble'
        long_df['replicate_id'] = (
            long_df['seed'].astype(str)
            + '|outer_' + long_df['outer_fold'].astype(str)
            + '|ensemble'
        )
        long_df['member_prob'] = long_df['member_prob'].astype(float)
        return long_df.dropna(subset=['member_prob']).reset_index(drop=True)

    prob_cols = _list_inner_model_prob_cols(pred_df)
    rows = []
    for member_idx, prob_col in prob_cols:
        sub_df = pred_df[base_cols + [prob_col]].copy()
        sub_df = sub_df.rename(columns={prob_col: 'member_prob'})
        sub_df['inner_model_idx'] = int(member_idx)
        sub_df['prediction_source'] = 'inner_model'
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

    if 'prediction_source' not in member_pred_df.columns:
        member_pred_df = member_pred_df.copy()
        member_pred_df['prediction_source'] = 'inner_model'

    group_cols = [
        'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold',
        'prediction_source', 'inner_model_idx', 'replicate_id', 'patient', 'y_true'
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
            'Found duplicated predictions for the same seed x outer_fold x replicate x patient x missingness cell. '
            'This notebook does not collapse them into a single predictor. Sample duplicated groups: '
            f'{sample_rows}'
        )

    out_df = (
        member_pred_df[group_cols + ['member_prob']]
        .copy()
        .sort_values([
            'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold',
            'prediction_source', 'inner_model_idx', 'patient'
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
    if 'prediction_source' not in member_patient_df.columns:
        member_patient_df = member_patient_df.copy()
        member_patient_df['prediction_source'] = 'inner_model'
    rows = []
    group_cols = [
        'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold',
        'prediction_source', 'inner_model_idx', 'replicate_id'
    ]
    for key, group_df in member_patient_df.groupby(group_cols, sort=True):
        model_name, train_prop, test_prop, seed, outer_fold, prediction_source, inner_idx, replicate_id = key
        auc_val = safe_auc(group_df['y_true'].to_numpy(), group_df['member_prob'].to_numpy())
        rows.append({
            'model_name': model_name,
            'train_missing_prop': float(train_prop),
            'test_missing_prop': float(test_prop),
            'seed': seed,
            'outer_fold': outer_fold,
            'prediction_source': str(prediction_source),
            'inner_model_idx': int(inner_idx),
            'replicate_id': str(replicate_id),
            'n_patients': int(group_df['patient'].nunique()),
            'auc': auc_val,
        })
    return pd.DataFrame(rows).sort_values([
        'model_name', 'train_missing_prop', 'test_missing_prop', 'seed', 'outer_fold',
        'prediction_source', 'inner_model_idx'
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

    _save_current_figure(figures_dir, file_name, dpi=220)
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




def build_best_fixed_train_curve(level1_df: pd.DataFrame):
    """Select one train-missingness setting per model and return its test-missingness curve.

    The selected setting is the train_missing_prop with the highest normalized AUPMC
    over test_missing_prop. This avoids an oracle envelope that changes the training
    missingness setting independently at each test missingness point.
    """
    both_df = level1_df.loc[level1_df['scenario'] == 'both'].copy()
    if both_df.empty:
        return pd.DataFrame(columns=[
            'model_name',
            'test_missing_prop',
            'envelope_train_missing_prop',
            'envelope_mean_auc',
            'selected_train_aupmc',
        ])

    selected_rows = []
    for model_name, model_df in both_df.groupby('model_name', sort=True):
        train_scores = []
        for train_prop, train_df in model_df.groupby('train_missing_prop', sort=True):
            train_scores.append({
                'model_name': model_name,
                'envelope_train_missing_prop': float(train_prop),
                'selected_train_aupmc': _normalized_trapezoid_auc(
                    train_df,
                    x_col='test_missing_prop',
                    y_col='mean_auc',
                ),
            })
        score_df = pd.DataFrame(train_scores).dropna(subset=['selected_train_aupmc'])
        if score_df.empty:
            continue
        best_score = (
            score_df
            .sort_values(
                ['selected_train_aupmc', 'envelope_train_missing_prop'],
                ascending=[False, True],
            )
            .iloc[0]
        )
        selected_train_prop = float(best_score['envelope_train_missing_prop'])
        selected_curve_df = model_df.loc[
            np.isclose(model_df['train_missing_prop'].astype(float), selected_train_prop)
        ].copy()
        for _, row in selected_curve_df.iterrows():
            selected_rows.append({
                'model_name': model_name,
                'test_missing_prop': float(row['test_missing_prop']),
                'envelope_train_missing_prop': selected_train_prop,
                'envelope_mean_auc': float(row['mean_auc']),
                'selected_train_aupmc': float(best_score['selected_train_aupmc']),
            })

    return (
        pd.DataFrame(selected_rows)
        .sort_values(['model_name', 'test_missing_prop'])
        .reset_index(drop=True)
    )


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


def _positive_degradation_auc(curve_df: pd.DataFrame, x_col: str, y_col: str, baseline_auc):
    """Normalized area of positive degradation: max(baseline/performance - 1, 0)."""
    baseline_auc = float(baseline_auc) if np.isfinite(baseline_auc) else np.nan
    if curve_df.empty or not np.isfinite(baseline_auc):
        return np.nan
    sub_df = curve_df[[x_col, y_col]].copy()
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
    finite = np.isfinite(x) & np.isfinite(y) & ~np.isclose(y, 0.0)
    x = x[finite]
    y = y[finite]
    if y.size == 0:
        return np.nan
    positive_degradation = np.maximum((baseline_auc / y) - 1.0, 0.0)
    if positive_degradation.size == 1 or np.isclose(float(np.max(x) - np.min(x)), 0.0):
        return float(positive_degradation[0])
    trapezoid = getattr(np, 'trapezoid', np.trapz)
    return float(trapezoid(positive_degradation, x) / (float(np.max(x) - np.min(x))))


def _normalized_trapezoid_auc_arrays(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if y.size == 0:
        return np.nan
    unique_x = np.unique(x)
    if unique_x.size != x.size:
        grouped_y = []
        for value in unique_x:
            grouped_y.append(float(np.nanmean(y[np.isclose(x, value)])))
        x = unique_x
        y = np.asarray(grouped_y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if y.size == 1 or np.isclose(float(np.max(x) - np.min(x)), 0.0):
        return float(y[0])
    trapezoid = getattr(np, 'trapezoid', np.trapz)
    return float(trapezoid(y, x) / (float(np.max(x) - np.min(x))))


def _positive_degradation_auc_arrays(x, y, baseline_auc):
    baseline_auc = float(baseline_auc) if np.isfinite(baseline_auc) else np.nan
    if not np.isfinite(baseline_auc):
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & ~np.isclose(y, 0.0)
    x = x[finite]
    y = y[finite]
    if y.size == 0:
        return np.nan
    unique_x = np.unique(x)
    if unique_x.size != x.size:
        grouped_y = []
        for value in unique_x:
            grouped_y.append(float(np.nanmean(y[np.isclose(x, value)])))
        x = unique_x
        y = np.asarray(grouped_y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    positive_degradation = np.maximum((baseline_auc / y) - 1.0, 0.0)
    if positive_degradation.size == 1 or np.isclose(float(np.max(x) - np.min(x)), 0.0):
        return float(positive_degradation[0])
    trapezoid = getattr(np, 'trapezoid', np.trapz)
    return float(trapezoid(positive_degradation, x) / (float(np.max(x) - np.min(x))))


def _method_metrics_from_arrays(train_props, test_props, mean_auc, is_distillation_method=False):
    train_props = np.asarray(train_props, dtype=float)
    test_props = np.asarray(test_props, dtype=float)
    mean_auc = np.asarray(mean_auc, dtype=float)
    finite = np.isfinite(train_props) & np.isfinite(test_props) & np.isfinite(mean_auc)
    train_props = train_props[finite]
    test_props = test_props[finite]
    mean_auc = mean_auc[finite]
    if mean_auc.size == 0:
        return {
            'baseline_auc': np.nan,
            'train_time_aupmc': np.nan,
            'test_time_aupmc': np.nan,
            'best_fixed_train_aupmc': np.nan,
            'best_fixed_train_missing_prop': np.nan,
            'train_degradation_coefficient': np.nan,
            'test_degradation_coefficient': np.nan,
            'minimum_degradation_coefficient': np.nan,
        }

    baseline_mask = np.isclose(train_props, 0.0) & np.isclose(test_props, 0.0)
    baseline_auc = float(np.nanmean(mean_auc[baseline_mask])) if baseline_mask.any() else np.nan

    train_mask = np.isclose(test_props, 0.0)
    train_time_aupmc = _normalized_trapezoid_auc_arrays(train_props[train_mask], mean_auc[train_mask])
    train_degradation_coefficient = _positive_degradation_auc_arrays(
        train_props[train_mask],
        mean_auc[train_mask],
        baseline_auc=baseline_auc,
    )
    if bool(is_distillation_method):
        train_time_aupmc = np.nan
        train_degradation_coefficient = np.nan

    test_mask = np.isclose(train_props, 0.0)
    test_time_aupmc = _normalized_trapezoid_auc_arrays(test_props[test_mask], mean_auc[test_mask])
    test_degradation_coefficient = _positive_degradation_auc_arrays(
        test_props[test_mask],
        mean_auc[test_mask],
        baseline_auc=baseline_auc,
    )

    best_fixed_train_missing_prop = np.nan
    best_fixed_train_aupmc = np.nan
    minimum_degradation_coefficient = np.nan
    train_scores = []
    for train_prop in np.unique(train_props):
        mask = np.isclose(train_props, train_prop)
        score = _normalized_trapezoid_auc_arrays(test_props[mask], mean_auc[mask])
        if np.isfinite(score):
            train_scores.append((float(train_prop), float(score)))
    if train_scores:
        train_scores = sorted(train_scores, key=lambda item: (-item[1], item[0]))
        best_fixed_train_missing_prop = float(train_scores[0][0])
        best_mask = np.isclose(train_props, best_fixed_train_missing_prop)
        best_baseline_mask = best_mask & np.isclose(test_props, 0.0)
        best_fixed_train_baseline_auc = (
            float(np.nanmean(mean_auc[best_baseline_mask]))
            if best_baseline_mask.any()
            else baseline_auc
        )
        best_fixed_train_aupmc = _normalized_trapezoid_auc_arrays(
            test_props[best_mask],
            mean_auc[best_mask],
        )
        minimum_degradation_coefficient = _positive_degradation_auc_arrays(
            test_props[best_mask],
            mean_auc[best_mask],
            baseline_auc=best_fixed_train_baseline_auc,
        )

    return {
        'baseline_auc': float(baseline_auc),
        'train_time_aupmc': float(train_time_aupmc),
        'test_time_aupmc': float(test_time_aupmc),
        'best_fixed_train_aupmc': float(best_fixed_train_aupmc),
        'best_fixed_train_missing_prop': best_fixed_train_missing_prop,
        'train_degradation_coefficient': float(train_degradation_coefficient),
        'test_degradation_coefficient': float(test_degradation_coefficient),
        'minimum_degradation_coefficient': float(minimum_degradation_coefficient),
    }


def _weighted_condition_mean_auc(model_replicate_df: pd.DataFrame, replicate_weights=None):
    """Build condition-level mean AUCs for one model, optionally with bootstrap weights."""
    required_cols = {'train_missing_prop', 'test_missing_prop', 'replicate_id', 'auc'}
    missing_cols = required_cols.difference(model_replicate_df.columns)
    if missing_cols:
        raise ValueError(f'Missing required columns for condition means: {sorted(missing_cols)}')

    sub_df = model_replicate_df[list(required_cols)].copy()
    sub_df['auc'] = pd.to_numeric(sub_df['auc'], errors='coerce')
    sub_df = sub_df.dropna(subset=['auc'])
    if sub_df.empty:
        return pd.DataFrame(columns=['train_missing_prop', 'test_missing_prop', 'mean_auc'])

    if replicate_weights is None:
        out_df = (
            sub_df
            .groupby(['train_missing_prop', 'test_missing_prop'], as_index=False)['auc']
            .mean()
            .rename(columns={'auc': 'mean_auc'})
        )
    else:
        weights = pd.Series(replicate_weights, name='bootstrap_weight')
        weights.index = weights.index.astype(str)
        sub_df['replicate_id'] = sub_df['replicate_id'].astype(str)
        sub_df = sub_df.merge(
            weights.reset_index().rename(columns={'index': 'replicate_id'}),
            on='replicate_id',
            how='inner',
        )
        if sub_df.empty:
            return pd.DataFrame(columns=['train_missing_prop', 'test_missing_prop', 'mean_auc'])
        sub_df['weighted_auc'] = sub_df['auc'].astype(float) * sub_df['bootstrap_weight'].astype(float)
        out_df = (
            sub_df
            .groupby(['train_missing_prop', 'test_missing_prop'], as_index=False)
            .agg(weighted_auc=('weighted_auc', 'sum'), bootstrap_weight=('bootstrap_weight', 'sum'))
        )
        out_df['mean_auc'] = out_df['weighted_auc'] / out_df['bootstrap_weight']
        out_df = out_df[['train_missing_prop', 'test_missing_prop', 'mean_auc']]

    out_df['train_missing_prop'] = out_df['train_missing_prop'].astype(float)
    out_df['test_missing_prop'] = out_df['test_missing_prop'].astype(float)
    out_df['mean_auc'] = out_df['mean_auc'].astype(float)
    return out_df.sort_values(['train_missing_prop', 'test_missing_prop']).reset_index(drop=True)


def _method_metrics_from_condition_means(condition_mean_df: pd.DataFrame, is_distillation_method=False):
    """Compute method-level scalar metrics from condition-level mean AUCs."""
    if condition_mean_df.empty:
        return {
            'baseline_auc': np.nan,
            'train_time_aupmc': np.nan,
            'test_time_aupmc': np.nan,
            'best_fixed_train_aupmc': np.nan,
            'best_fixed_train_missing_prop': np.nan,
            'train_degradation_coefficient': np.nan,
            'test_degradation_coefficient': np.nan,
            'minimum_degradation_coefficient': np.nan,
        }
    return _method_metrics_from_arrays(
        train_props=condition_mean_df['train_missing_prop'].to_numpy(dtype=float),
        test_props=condition_mean_df['test_missing_prop'].to_numpy(dtype=float),
        mean_auc=condition_mean_df['mean_auc'].to_numpy(dtype=float),
        is_distillation_method=is_distillation_method,
    )


def _summarize_bootstrap_distribution(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            'ci_half_width': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_bootstrap_valid': 0,
        }
    if values.size == 1:
        value = float(values[0])
        return {
            'ci_half_width': 0.0,
            'ci_lower': value,
            'ci_upper': value,
            'n_bootstrap_valid': 1,
        }
    alpha = 1.0 - float(confidence)
    ci_lower, ci_upper = np.quantile(values, [alpha / 2.0, 1.0 - (alpha / 2.0)])
    return {
        'ci_half_width': float((ci_upper - ci_lower) / 2.0),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_bootstrap_valid': int(values.size),
    }


def _bootstrap_method_metric_intervals(
    model_replicate_df: pd.DataFrame,
    is_distillation_method=False,
    n_bootstrap=2000,
    confidence=0.95,
    random_seed=42,
):
    """Bootstrap CIs for method-level AUPMC and degradation metrics.

    Bootstrap samples are drawn over replicate identifiers. Each sampled replicate
    contributes all available missingness cells, preserving the trajectory structure
    used by the integrated metrics.
    """
    metric_cols = [
        'train_time_aupmc',
        'test_time_aupmc',
        'best_fixed_train_aupmc',
        'train_degradation_coefficient',
        'test_degradation_coefficient',
        'minimum_degradation_coefficient',
    ]
    empty = {}
    for metric_col in metric_cols:
        empty[f'{metric_col}_ci95'] = np.nan
        empty[f'{metric_col}_ci95_lower'] = np.nan
        empty[f'{metric_col}_ci95_upper'] = np.nan
        empty[f'{metric_col}_n_bootstrap_valid'] = 0

    if model_replicate_df.empty:
        return empty

    replicate_ids = (
        model_replicate_df['replicate_id']
        .astype(str)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not replicate_ids:
        return empty

    wide_df = (
        model_replicate_df
        .assign(
            replicate_id=lambda df: df['replicate_id'].astype(str),
            train_missing_prop=lambda df: df['train_missing_prop'].astype(float),
            test_missing_prop=lambda df: df['test_missing_prop'].astype(float),
            auc=lambda df: pd.to_numeric(df['auc'], errors='coerce'),
        )
        .pivot_table(
            index='replicate_id',
            columns=['train_missing_prop', 'test_missing_prop'],
            values='auc',
            aggfunc='mean',
        )
        .reindex(index=replicate_ids)
        .sort_index(axis=1)
    )
    if wide_df.empty:
        return empty
    condition_pairs = list(wide_df.columns)
    train_props = np.asarray([float(pair[0]) for pair in condition_pairs], dtype=float)
    test_props = np.asarray([float(pair[1]) for pair in condition_pairs], dtype=float)
    auc_matrix = wide_df.to_numpy(dtype=float)

    rng = np.random.default_rng(int(random_seed))
    values_by_metric = {metric_col: [] for metric_col in metric_cols}
    n_replicates = len(replicate_ids)
    for _ in range(int(n_bootstrap)):
        sampled_idx = rng.integers(0, n_replicates, size=n_replicates)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            sampled_means = np.nanmean(auc_matrix[sampled_idx, :], axis=0)
        metrics = _method_metrics_from_arrays(
            train_props=train_props,
            test_props=test_props,
            mean_auc=sampled_means,
            is_distillation_method=is_distillation_method,
        )
        for metric_col in metric_cols:
            values_by_metric[metric_col].append(metrics.get(metric_col, np.nan))

    out = {}
    for metric_col in metric_cols:
        stats = _summarize_bootstrap_distribution(values_by_metric[metric_col], confidence=confidence)
        out[f'{metric_col}_ci95'] = stats['ci_half_width']
        out[f'{metric_col}_ci95_lower'] = stats['ci_lower']
        out[f'{metric_col}_ci95_upper'] = stats['ci_upper']
        out[f'{metric_col}_n_bootstrap_valid'] = stats['n_bootstrap_valid']
    return out


def build_method_level_metrics(
    replicate_auc_df: pd.DataFrame,
    level1_df: pd.DataFrame,
    distillation_model_names=None,
    n_bootstrap=2000,
    confidence=0.95,
    random_seed=42,
):
    both_df = level1_df.loc[level1_df['scenario'] == 'both'].copy()
    if both_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    baseline_df = both_df.loc[
        np.isclose(both_df['train_missing_prop'], 0.0) & np.isclose(both_df['test_missing_prop'], 0.0),
        ['model_name', 'mean_auc']
    ].copy()
    baseline_lookup = baseline_df.set_index('model_name')['mean_auc'].to_dict()

    envelope_df = build_best_fixed_train_curve(both_df)

    rows = []
    model_names = sorted(both_df['model_name'].astype(str).unique().tolist())
    for model_name in model_names:
        model_df = both_df.loc[both_df['model_name'].astype(str) == model_name].copy()
        is_distillation_method = _is_distillation_method(model_name, distillation_model_names)

        condition_mean_df = model_df[[
            'train_missing_prop',
            'test_missing_prop',
            'mean_auc',
        ]].copy()
        point_metrics = _method_metrics_from_condition_means(
            condition_mean_df,
            is_distillation_method=is_distillation_method,
        )
        # Preserve exact baseline lookup used elsewhere in the notebook.
        point_metrics['baseline_auc'] = float(baseline_lookup.get(model_name, point_metrics['baseline_auc']))

        model_replicate_df = replicate_auc_df.loc[
            replicate_auc_df['model_name'].astype(str) == model_name
        ].copy()
        interval_metrics = _bootstrap_method_metric_intervals(
            model_replicate_df,
            is_distillation_method=is_distillation_method,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            random_seed=_stable_seed(model_name, 'method_level_metrics', base_seed=random_seed),
        )

        row = {
            'model_name': model_name,
            'is_distillation_method': bool(is_distillation_method),
            'baseline_auc': point_metrics['baseline_auc'],
            'train_time_aupmc': point_metrics['train_time_aupmc'],
            'test_time_aupmc': point_metrics['test_time_aupmc'],
            'best_fixed_train_aupmc': point_metrics['best_fixed_train_aupmc'],
            'best_fixed_train_missing_prop': point_metrics['best_fixed_train_missing_prop'],
            'train_degradation_coefficient': point_metrics['train_degradation_coefficient'],
            'test_degradation_coefficient': point_metrics['test_degradation_coefficient'],
            'minimum_degradation_coefficient': point_metrics['minimum_degradation_coefficient'],
        }
        row.update(interval_metrics)
        rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values(
        ['baseline_auc', 'model_name'],
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
    envelope_rows = []
    for model_name, model_df in both_summary_df.groupby('model_name', sort=True):
        train_scores = []
        for train_prop, train_df in model_df.groupby('train_prop', sort=True):
            train_scores.append({
                'train_prop': float(train_prop),
                'aupmc': _normalized_trapezoid_auc(
                    train_df,
                    x_col='test_prop',
                    y_col='mean_auc',
                ),
            })
        score_df = pd.DataFrame(train_scores).dropna(subset=['aupmc'])
        if score_df.empty:
            continue
        selected_train_prop = float(
            score_df
            .sort_values(['aupmc', 'train_prop'], ascending=[False, True])
            .iloc[0]['train_prop']
        )
        selected_curve_df = model_df.loc[np.isclose(model_df['train_prop'].astype(float), selected_train_prop)].copy()
        selected_curve_df['scenario'] = 'envelope'
        selected_curve_df['missing_prop'] = selected_curve_df['test_prop'].astype(float)
        selected_curve_df['selected_train_aupmc'] = float(score_df.loc[
            np.isclose(score_df['train_prop'].astype(float), selected_train_prop),
            'aupmc',
        ].iloc[0])
        envelope_rows.append(selected_curve_df)
    if envelope_rows:
        envelope_df = pd.concat(envelope_rows, ignore_index=True)
        summary_df = pd.concat([summary_df, envelope_df], ignore_index=True)
    summary_df = summary_df.sort_values(['scenario', 'model_name', 'train_prop', 'test_prop']).reset_index(drop=True)

    return summary_df


def build_degradation_curve_summary(method_plot_summary_df: pd.DataFrame, distillation_model_names=None):
    """Build raw pointwise degradation-ratio curves.

    Train-time and test-time curves use the complete-data baseline
    (train=0, test=0). Best fixed-train/envelope curves use their own fixed-train
    baseline at (train=m_train*, test=0), because the trajectory evaluates
    degradation as test missingness increases after selecting one train regime.
    """
    if method_plot_summary_df.empty:
        return pd.DataFrame()

    required_cols = {'model_name', 'scenario', 'train_prop', 'test_prop', 'mean_auc', 'auc_ci95'}
    missing_cols = required_cols.difference(method_plot_summary_df.columns)
    if missing_cols:
        raise ValueError(f'Missing required columns for degradation curves: {sorted(missing_cols)}')

    baseline_df = method_plot_summary_df.loc[
        (method_plot_summary_df['scenario'] == 'both')
        & np.isclose(method_plot_summary_df['train_prop'].astype(float), 0.0)
        & np.isclose(method_plot_summary_df['test_prop'].astype(float), 0.0),
        ['model_name', 'mean_auc']
    ].copy()
    baseline_lookup = baseline_df.set_index('model_name')['mean_auc'].to_dict()

    out_df = method_plot_summary_df.loc[
        method_plot_summary_df['scenario'].isin(['train', 'test', 'envelope'])
    ].copy()
    out_df['baseline_auc'] = out_df['model_name'].map(baseline_lookup).astype(float)

    envelope_baseline_df = out_df.loc[
        (out_df['scenario'] == 'envelope')
        & np.isclose(out_df['test_prop'].astype(float), 0.0),
        ['model_name', 'mean_auc']
    ].copy()
    envelope_baseline_lookup = envelope_baseline_df.set_index('model_name')['mean_auc'].to_dict()
    envelope_mask = out_df['scenario'] == 'envelope'
    out_df.loc[envelope_mask, 'baseline_auc'] = (
        out_df.loc[envelope_mask, 'model_name']
        .map(envelope_baseline_lookup)
        .fillna(out_df.loc[envelope_mask, 'baseline_auc'])
        .astype(float)
    )

    out_df['raw_degradation_ratio'] = out_df.apply(
        lambda row: _safe_ratio(row['baseline_auc'], row['mean_auc']),
        axis=1,
    )
    out_df['degradation_ratio'] = out_df['raw_degradation_ratio']

    def _degradation_ci_half_width(row):
        baseline_auc = float(row['baseline_auc']) if np.isfinite(row['baseline_auc']) else np.nan
        mean_auc = float(row['mean_auc']) if np.isfinite(row['mean_auc']) else np.nan
        auc_ci = float(row['auc_ci95']) if np.isfinite(row['auc_ci95']) else 0.0
        if not np.isfinite(baseline_auc) or not np.isfinite(mean_auc) or np.isclose(mean_auc, 0.0):
            return np.nan
        lower_mean_auc = max(mean_auc - auc_ci, 1e-12)
        upper_mean_auc = max(mean_auc + auc_ci, 1e-12)
        point_estimate = baseline_auc / mean_auc
        lower_ratio = baseline_auc / upper_mean_auc
        upper_ratio = baseline_auc / lower_mean_auc
        return float(max(point_estimate - lower_ratio, upper_ratio - point_estimate))

    out_df['degradation_ratio_ci95'] = out_df.apply(
        _degradation_ci_half_width,
        axis=1,
    )
    out_df['degradation_ratio_ci95_lower'] = out_df['degradation_ratio'] - out_df['degradation_ratio_ci95']
    out_df['degradation_ratio_ci95_upper'] = out_df['degradation_ratio'] + out_df['degradation_ratio_ci95']

    # At the baseline point of each degradation trajectory the ratio is
    # baseline / baseline by definition. It is therefore fixed to 1 and should
    # not inherit the uncertainty of the underlying AUC/C-index estimate.
    trajectory_baseline_mask = (
        ((out_df['scenario'].isin(['train', 'test']))
         & np.isclose(out_df['train_prop'].astype(float), 0.0)
         & np.isclose(out_df['test_prop'].astype(float), 0.0))
        | ((out_df['scenario'] == 'envelope')
           & np.isclose(out_df['test_prop'].astype(float), 0.0))
    )
    out_df.loc[trajectory_baseline_mask, 'raw_degradation_ratio'] = 1.0
    out_df.loc[trajectory_baseline_mask, 'degradation_ratio'] = 1.0
    out_df.loc[trajectory_baseline_mask, 'degradation_ratio_ci95'] = 0.0
    out_df.loc[trajectory_baseline_mask, 'degradation_ratio_ci95_lower'] = 1.0
    out_df.loc[trajectory_baseline_mask, 'degradation_ratio_ci95_upper'] = 1.0
    out_df['excluded_from_train_degradation'] = (
        (out_df['scenario'] == 'train')
        & out_df['model_name'].astype(str).map(lambda value: _is_distillation_method(value, distillation_model_names))
    )
    train_excluded_mask = out_df['excluded_from_train_degradation']
    out_df.loc[
        train_excluded_mask,
        ['degradation_ratio', 'degradation_ratio_ci95', 'degradation_ratio_ci95_lower', 'degradation_ratio_ci95_upper']
    ] = np.nan

    return out_df.sort_values(['scenario', 'model_name', 'train_prop', 'test_prop']).reset_index(drop=True)


def build_metric_ordering_table(method_level_metrics_df: pd.DataFrame, distillation_model_names=None):
    if method_level_metrics_df.empty:
        return pd.DataFrame()
    metric_specs = [
        ('baseline_auc', 'Baseline AUC', False),
        ('train_time_aupmc', 'Train-time AUPMC', False),
        ('train_degradation_coefficient', 'Train degradation coefficient', True),
        ('test_time_aupmc', 'Test-time AUPMC', False),
        ('test_degradation_coefficient', 'Test degradation coefficient', True),
        ('best_fixed_train_aupmc', 'Best fixed-train AUPMC', False),
        ('minimum_degradation_coefficient', 'Minimum degradation coefficient', True),
    ]
    max_len = int(method_level_metrics_df['model_name'].nunique())
    out = {}
    for metric_col, label, ascending_metric in metric_specs:
        if metric_col not in method_level_metrics_df.columns:
            continue
        metric_df = method_level_metrics_df[['model_name', metric_col]].copy()
        if metric_col in {'train_time_aupmc', 'train_degradation_coefficient'}:
            metric_df = metric_df.loc[
                ~metric_df['model_name'].astype(str).map(lambda value: _is_distillation_method(value, distillation_model_names))
            ].copy()
        ordered = (
            metric_df
            .sort_values([metric_col, 'model_name'], ascending=[ascending_metric, True], na_position='last')
            ['model_name']
            .tolist()
        )
        if len(ordered) < max_len:
            ordered.extend([''] * (max_len - len(ordered)))
        out[label] = ordered
    return pd.DataFrame(out)


def plot_method_line_triplet(
    summary_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    figures_dir: Path,
    file_name: str,
    figsize=None,
    ci_col: str = 'auc_ci95',
    panel_titles=None,
    y_limits=(0.0, 1.0),
    clip_ci=True,
):
    if not HAVE_MPL:
        print(f'Matplotlib is not available. Skipping {metric_col} line plot.')
        return
    if summary_df.empty:
        print(f'No data available for {metric_col} line plot.')
        return

    color_map = _build_color_map(summary_df)
    n_models = summary_df['model_name'].nunique() if not summary_df.empty else 1
    resolved_figsize = tuple(figsize) if figsize is not None else _resolve_plot_figsize('line', n_models)
    fig, axes = plt.subplots(1, 3, figsize=resolved_figsize, sharey=True)

    default_panel_titles = {
        'train': 'Missing at Train',
        'test': 'Missing at Test',
        'envelope': 'Best Fixed Train-Missingness',
    }
    if panel_titles:
        default_panel_titles.update(panel_titles)
    panels = [
        ('train', default_panel_titles['train'], 'train_prop'),
        ('test', default_panel_titles['test'], 'test_prop'),
        ('envelope', default_panel_titles['envelope'], 'test_prop'),
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
            if ci_col in model_df.columns:
                ci = model_df[ci_col].fillna(0.0).to_numpy(dtype=float)
            else:
                ci = np.zeros_like(y, dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            ci = ci[finite]
            if y.size == 0:
                continue
            lower = y - ci
            upper = y + ci
            if clip_ci and y_limits is not None:
                lower = np.clip(lower, y_limits[0], y_limits[1])
                upper = np.clip(upper, y_limits[0], y_limits[1])

            ax.plot(x, y, marker='o', linewidth=2, label=model_name, color=color_map[model_name])
            ax.fill_between(x, lower, upper, color=color_map[model_name], alpha=0.16)
        ax.set_xticks(sorted(scenario_df[x_col].dropna().unique().tolist()))
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.tick_params(axis='both', labelsize=12)

    legend_items = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)
    if legend_items:
        fig.legend(list(legend_items.values()), list(legend_items.keys()), loc='center left', bbox_to_anchor=(0.92, 0.5))

    fig.suptitle(title, fontsize=15, y=0.955)
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.17, top=0.85, wspace=0.18)
    _save_current_figure(figures_dir, file_name, dpi=220)
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
    _save_current_figure(figures_dir, file_name, dpi=220)
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


def _parse_top_equivalent_group_models(row: pd.Series):
    raw = row.get('winner_group_models', '')
    if pd.isna(raw) or not str(raw).strip():
        raw = row.get('winner_group_text', '')
    if pd.isna(raw) or not str(raw).strip():
        raw = row.get('winner_model', '')

    text = str(raw).replace('|', ',')
    models = [part.strip() for part in text.split(',') if part.strip()]
    return list(dict.fromkeys(models))


def build_top_equivalent_group_counts(level2_plot_df: pd.DataFrame):
    """Count how often each method appears in condition-level top-equivalent groups."""
    columns = [
        'model_name',
        'top_equivalent_group_count',
        'top_equivalent_group_fraction',
        'n_condition_cells_with_top_group',
    ]
    if level2_plot_df is None or level2_plot_df.empty:
        return pd.DataFrame(columns=columns)

    counts = {}
    n_cells = int(len(level2_plot_df))
    for _, row in level2_plot_df.iterrows():
        for model_name in _parse_top_equivalent_group_models(row):
            counts[model_name] = counts.get(model_name, 0) + 1

    rows = []
    for model_name, count in counts.items():
        rows.append({
            'model_name': str(model_name),
            'top_equivalent_group_count': int(count),
            'top_equivalent_group_fraction': float(count / n_cells) if n_cells else np.nan,
            'n_condition_cells_with_top_group': n_cells,
        })
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(['top_equivalent_group_count', 'model_name'], ascending=[False, True])
        .reset_index(drop=True)
    )


def _select_best_methods_by_metric(method_level_metrics_df: pd.DataFrame, metric_col: str, ascending=False):
    if method_level_metrics_df is None or method_level_metrics_df.empty or metric_col not in method_level_metrics_df.columns:
        return [], np.nan
    metric_df = method_level_metrics_df[['model_name', metric_col]].copy()
    metric_df[metric_col] = pd.to_numeric(metric_df[metric_col], errors='coerce')
    metric_df = metric_df.dropna(subset=[metric_col])
    if metric_df.empty:
        return [], np.nan
    best_value = metric_df[metric_col].min() if bool(ascending) else metric_df[metric_col].max()
    best_models = (
        metric_df.loc[np.isclose(metric_df[metric_col].astype(float), float(best_value)), 'model_name']
        .astype(str)
        .sort_values()
        .tolist()
    )
    return best_models, float(best_value)


def build_general_results_summary(method_level_metrics_df: pd.DataFrame, level2_plot_df: pd.DataFrame):
    """Build a compact final summary of best methods by evaluation scenario."""
    rows = []
    specs = [
        (
            'Best method complete data',
            'baseline_auc',
            'highest Baseline AUC',
            'Complete train / complete test cell.',
            False,
        ),
        (
            'Best method train-time missingness',
            'train_time_aupmc',
            'highest Train-time AUPMC',
            'Train missingness increases while test data remain complete.',
            False,
        ),
        (
            'Best method test-time missingness',
            'test_time_aupmc',
            'highest Test-time AUPMC',
            'Model trained on complete data and evaluated under increasing test missingness.',
            False,
        ),
        (
            'Best method missing at both',
            'best_fixed_train_aupmc',
            'highest Best fixed-train AUPMC',
            'One train-time missingness setting is selected and evaluated across test-time missingness.',
            False,
        ),
    ]
    for summary_item, metric_col, rank_basis, details, ascending in specs:
        models, value = _select_best_methods_by_metric(
            method_level_metrics_df,
            metric_col=metric_col,
            ascending=ascending,
        )
        rows.append({
            'summary_item': summary_item,
            'selected_method': ', '.join(models),
            'selection_metric': metric_col,
            'selection_value': value,
            'rank_basis': rank_basis,
            'details': details,
        })

    flexibility_df = build_top_equivalent_group_counts(level2_plot_df)
    if flexibility_df.empty:
        rows.append({
            'summary_item': 'Most flexible method',
            'selected_method': '',
            'selection_metric': 'top_equivalent_group_count',
            'selection_value': np.nan,
            'rank_basis': 'highest number of top-equivalent group appearances',
            'details': 'No condition-level top-equivalent group was available after FDR correction.',
        })
    else:
        best_count = int(flexibility_df['top_equivalent_group_count'].max())
        best_models = (
            flexibility_df.loc[
                flexibility_df['top_equivalent_group_count'].astype(int) == best_count,
                'model_name',
            ]
            .astype(str)
            .sort_values()
            .tolist()
        )
        n_cells = int(flexibility_df['n_condition_cells_with_top_group'].max())
        rows.append({
            'summary_item': 'Most flexible method',
            'selected_method': ', '.join(best_models),
            'selection_metric': 'top_equivalent_group_count',
            'selection_value': float(best_count),
            'rank_basis': 'highest number of top-equivalent group appearances',
            'details': f'Appears in the top-equivalent group in {best_count}/{n_cells} condition-level cells.',
        })

    return pd.DataFrame(rows)


def _format_winner_group_text(models_text: str, max_names_per_line=2):
    names = [part.strip() for part in str(models_text).split(',') if part.strip()]
    if not names:
        return ''
    lines = []
    for idx in range(0, len(names), max_names_per_line):
        lines.append(', '.join(names[idx:idx + max_names_per_line]))
    return '\n'.join(lines)


def _winner_group_key(row: pd.Series):
    raw_models = row.get('winner_group_models', '')
    if pd.isna(raw_models) or not str(raw_models).strip():
        raw_models = row.get('winner_group_text', '')
    if pd.isna(raw_models) or not str(raw_models).strip():
        raw_models = row.get('winner_model', '')

    raw_text = str(raw_models).replace('|', ',')
    models = [part.strip() for part in raw_text.split(',') if part.strip()]
    if not models:
        models = [str(row.get('winner_model', '')).strip()]
    # Colour encodes the top-method combination, not the rank order in that cell.
    return ' | '.join(sorted(dict.fromkeys(models)))


def _contrast_text_color(rgba):
    r, g, b = [float(value) for value in rgba[:3]]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'black' if luminance > 0.58 else 'white'


def _split_long_title(title: str, max_chars: int = 82):
    title = str(title)
    if '\n' in title or len(title) <= max_chars:
        return title

    parts = [part.strip() for part in title.split('|')]
    if len(parts) > 1:
        lines = []
        current = parts[0]
        for part in parts[1:]:
            candidate = f'{current} | {part}'
            if len(candidate) <= max_chars:
                current = candidate
            else:
                lines.append(current)
                current = part
        lines.append(current)
        return '\n'.join(lines[:2]) if len(lines) == 2 else '\n'.join([' | '.join(lines[:-1]), lines[-1]])

    midpoint = len(title) // 2
    split_idx = title.rfind(' ', 0, midpoint)
    if split_idx < 0:
        split_idx = midpoint
    return f'{title[:split_idx].rstrip()}\n{title[split_idx:].lstrip()}'


def plot_level2_significant_pairs_heatmap(level2_plot_df: pd.DataFrame, title: str, figures_dir: Path, file_name: str):
    if not HAVE_MPL:
        print('Matplotlib is not available in this environment. Skipping condition-level summary heatmap.')
        return
    if level2_plot_df.empty:
        print('No statistically significant condition-level summary pairs available for plotting after FDR correction.')
        return

    level2_plot_df = level2_plot_df.copy()
    level2_plot_df['winner_group_key'] = level2_plot_df.apply(_winner_group_key, axis=1)
    winner_group_keys = sorted(level2_plot_df['winner_group_key'].astype(str).unique().tolist())
    color_lookup = {name: idx for idx, name in enumerate(winner_group_keys)}
    level2_hex = ['#012a4a', '#013a63', '#01497c', '#014f86', '#2a6f97', '#2c7da0', '#468faf', '#61a5c2', '#89c2d9', '#a9d6e5']
    base_cmap = LinearSegmentedColormap.from_list('level2_custom', level2_hex)
    cmap = ListedColormap(base_cmap(np.linspace(0.0, 0.72, max(len(winner_group_keys), 1))))
    cmap.set_bad('#F2F2F2')

    train_props = sorted(level2_plot_df['train_missing_prop'].unique().tolist())
    test_props = sorted(level2_plot_df['test_missing_prop'].unique().tolist())
    matrix = np.full((len(train_props), len(test_props)), np.nan, dtype=float)
    for _, row in level2_plot_df.iterrows():
        i = train_props.index(float(row['train_missing_prop']))
        j = test_props.index(float(row['test_missing_prop']))
        matrix[i, j] = float(color_lookup[row['winner_group_key']])

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    x_edges = np.arange(len(test_props) + 1, dtype=float)
    y_edges = np.arange(len(train_props) + 1, dtype=float)
    ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(winner_group_keys) - 0.5,
        edgecolors='white',
        linewidth=0.55,
        shading='flat',
    )
    ax.set_xlim(0, len(test_props))
    ax.set_ylim(len(train_props), 0)
    ax.set_aspect('equal')
    ax.set_title(_split_long_title(title), fontsize=13, pad=22, loc='center')
    ax.set_xlabel('Test missing prop', fontsize=12, labelpad=14)
    ax.set_ylabel('Train missing prop', fontsize=12, labelpad=14)
    ax.set_xticks(np.arange(len(test_props), dtype=float) + 0.5)
    ax.set_xticklabels([f'{value:g}' for value in test_props], rotation=0, ha='center', fontsize=11)
    ax.set_yticks(np.arange(len(train_props), dtype=float) + 0.5)
    ax.set_yticklabels([f'{value:g}' for value in train_props], fontsize=11)
    ax.tick_params(axis='both', which='major', length=0, pad=8)
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

    _save_current_figure(figures_dir, file_name, dpi=300)
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
    _save_current_figure(figures_dir, file_name, dpi=300)
    plt.show()
