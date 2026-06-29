#!/usr/bin/env python3
from __future__ import annotations
"""
M3TRICS Interactive Dashboard Generator
Reads progressive_missingness_analysis_outputs CSVs and produces a
self-contained single-file interactive HTML dashboard.

Run from git_exp/:
    python dashboard/generate_dashboard.py
    python dashboard/generate_dashboard.py --output dashboard/m3trics_dashboard.html
"""

import os, sys, json, argparse, base64
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import compute module from same directory
sys.path.insert(0, str(Path(__file__).parent))
try:
    from compute import compute_progressive_dataset as _compute_prog_ds
    _HAS_COMPUTE = True
except ImportError:
    _HAS_COMPUTE = False

# ── CONFIG ──────────────────────────────────────────────────────────────────────
DATASET_META = {
    'mmImmuno_OS_9_label':      {'name': 'mmImmuno',     'endpoint': '9-month OS',  'n': 281, 'modalities': 5, 'scope': 'Pan-cancer immunotherapy',    'task_type': 'Binary classification'},
    'mmColorectal_OS_21_label': {'name': 'mmColorectal', 'endpoint': '21-month OS', 'n': 148, 'modalities': 4, 'scope': 'Metastatic colorectal cancer', 'task_type': 'Binary classification'},
    'mmProstate_OS_27_label':   {'name': 'mmProstate',   'endpoint': '27-month OS', 'n': 83,  'modalities': 3, 'scope': 'Metastatic prostate cancer',   'task_type': 'Binary classification'},
}
METHOD_COLORS = {
    'ZI_MLP':'#00d4ff','KNN_MLP':'#a855f7','VAE_MLP':'#ff2d78',
    'ZI_LR':'#64748b','KNN_LR':'#94a3b8','ZI_RF':'#2563eb','KNN_RF':'#38bdf8',
    'ZI_CoxNet':'#7c2d12','KNN_CoxNet':'#f97316','ZI_RSF':'#166534','KNN_RSF':'#84cc16',
    'pAM':'#00ff9f','Di-PAM':'#4ecdc4',
    'HealNet':'#ff9900','SMILe':'#ffe66d','Di-MMLP':'#ff6b6b',
}
METHOD_DISPLAY = {
    'ZI_MLP':'ZI-MLP','KNN_MLP':'KNN-MLP','VAE_MLP':'VAE-MLP',
    'ZI_LR':'ZI-LR','KNN_LR':'KNN-LR','ZI_RF':'ZI-RF','KNN_RF':'KNN-RF',
    'ZI_CoxNet':'ZI-CoxNet','KNN_CoxNet':'KNN-CoxNet','ZI_RSF':'ZI-RSF','KNN_RSF':'KNN-RSF',
    'pAM':'pAM','Di-PAM':'Di-PAM',
    'HealNet':'HealNet','SMILe':'SMILe','Di-MMLP':'Di-MMLP',
}

# ── DATA LOADING ─────────────────────────────────────────────────────────────────
def _csv(path):
    try:
        df = pd.read_csv(path)
        return json.loads(df.where(pd.notnull(df), None).to_json(orient='records'))
    except Exception:
        return []

METRIC_INFO = {
    'auc': {
        'key': 'auc',
        'label': 'AUC',
        'task_type': 'Binary classification',
        'replicate_file': 'replicate_auc_table.csv',
        'condition_file': 'method_condition_mean_auc_summary.csv',
        'score_col': 'auc',
        'mean_col': 'mean_auc',
        'std_col': 'std_auc',
        'ci_col': 'auc_ci95',
        'baseline_col': 'baseline_auc',
    },
    'cindex': {
        'key': 'cindex',
        'label': 'C-index',
        'task_type': 'Survival analysis',
        'replicate_file': 'replicate_cindex_table.csv',
        'condition_file': 'method_condition_mean_cindex_summary.csv',
        'score_col': 'cindex',
        'mean_col': 'mean_cindex',
        'std_col': 'std_cindex',
        'ci_col': 'cindex_ci95',
        'baseline_col': 'baseline_cindex',
    },
}

CINDEX_TO_AUC_COMPAT = {
    'cindex': 'auc',
    'mean_cindex': 'mean_auc',
    'std_cindex': 'std_auc',
    'cindex_ci95': 'auc_ci95',
    'cindex_ci95_lower': 'auc_ci95_lower',
    'cindex_ci95_upper': 'auc_ci95_upper',
    'baseline_cindex': 'baseline_auc',
    'train_time_cindex_aupmc': 'train_time_aupmc',
    'test_time_cindex_aupmc': 'test_time_aupmc',
    'best_fixed_train_cindex_aupmc': 'best_fixed_train_aupmc',
    'selected_train_cindex_aupmc': 'selected_train_aupmc',
    'envelope_mean_cindex': 'envelope_mean_auc',
    'winner_mean_cindex': 'winner_mean_auc',
    'loser_mean_cindex': 'loser_mean_auc',
    'delta_mean_cindex': 'delta_mean_auc',
}

for _src, _dst in list(CINDEX_TO_AUC_COMPAT.items()):
    if _src.endswith('_cindex_aupmc'):
        for suffix in ('_ci95', '_ci95_lower', '_ci95_upper', '_n_bootstrap_valid'):
            CINDEX_TO_AUC_COMPAT[f'{_src}{suffix}'] = f'{_dst}{suffix}'


def _records_with_aliases(records, metric_key: str):
    """Add AUC-compatible aliases to C-index outputs so the JS plot code stays generic."""
    if metric_key != 'cindex':
        return records
    out = []
    for row in records or []:
        new_row = dict(row)
        for src, dst in CINDEX_TO_AUC_COMPAT.items():
            if src in new_row and dst not in new_row:
                new_row[dst] = new_row[src]
        out.append(new_row)
    return out


def _detect_metric_info(base: Path, fixed: bool = False) -> dict:
    """Infer the modelling task/metric from generated result files, not from hardcoded metadata."""
    if fixed:
        cindex_indicators = [
            base / 'replicate_cindex_table.csv',
            base / 'fixed_dataset_method_summary.csv',
            base / 'fixed_dataset_method_condition_summary.csv',
        ]
        for path in cindex_indicators:
            if path.exists():
                try:
                    cols = pd.read_csv(path, nrows=0).columns
                    if any('cindex' in str(c).lower() for c in cols):
                        return METRIC_INFO['cindex']
                except Exception:
                    if 'cindex' in path.name.lower():
                        return METRIC_INFO['cindex']
        return METRIC_INFO['auc']

    if (base / METRIC_INFO['cindex']['condition_file']).exists() or (base / METRIC_INFO['cindex']['replicate_file']).exists():
        return METRIC_INFO['cindex']
    metrics_path = base / 'method_level_metrics.csv'
    if metrics_path.exists():
        try:
            cols = pd.read_csv(metrics_path, nrows=0).columns
            if any('cindex' in str(c).lower() for c in cols):
                return METRIC_INFO['cindex']
        except Exception:
            pass
    return METRIC_INFO['auc']


def _infer_dataset_meta(ds_key: str) -> dict:
    """Use existing metadata when available, otherwise infer display labels from output folder names."""
    if ds_key in DATASET_META:
        return dict(DATASET_META[ds_key])
    import re
    m = re.match(r'^(?P<cohort>.+?)_(?P<etype>[A-Za-z]+)_(?P<months>\d+)_label$', str(ds_key))
    if m:
        cohort = m.group('cohort')
        etype = m.group('etype').upper()
        months = m.group('months')
        return {
            'name': cohort,
            'endpoint': f'{months}-month {etype}',
            'n': None,
            'modalities': None,
            'scope': cohort,
            'task_type': 'Detected from results',
        }
    return {
        'name': str(ds_key),
        'endpoint': str(ds_key),
        'n': None,
        'modalities': None,
        'scope': str(ds_key),
        'task_type': 'Detected from results',
    }

# mode_key → (subdir under retrainXXX, display label, retrain flag)
MODE_DEFS = [
    ('ensemble',     'retrainfalse', 'ensemble',     'Inner model ensemble'),
    ('inner_models', 'retrainfalse', 'inner_models', 'Retained inner models'),
    ('outer_retrain','retraintrue',  'inner_models', 'Outer models (retrained)'),
    ('outer_retrain_nm','retraintrue','',             'Outer models (retrained)'),
]

def _detect_metric_from_results(results_dir: Path, ds_key: str) -> dict:
    """Detect metric type by peeking at a sample outer_test_metrics.csv in results/."""
    training_runs = results_dir / ds_key / 'training_runs'
    if not training_runs.exists():
        return METRIC_INFO['auc']
    for model_dir in sorted(training_runs.iterdir()):
        if not model_dir.is_dir():
            continue
        tm = model_dir / 'TRAIN_MISSING'
        if not tm.exists():
            continue
        sample = next(
            (sd / 'outer_test_metrics.csv'
             for mod_d in sorted(tm.iterdir()) if mod_d.is_dir()
             for tp_d in sorted(mod_d.iterdir()) if tp_d.is_dir()
             for sd in sorted(tp_d.iterdir()) if sd.is_dir()),
            None,
        )
        if sample and sample.exists():
            try:
                cols = pd.read_csv(sample, nrows=0).columns
                if any('outer_test_cindex' in c.lower() for c in cols):
                    return METRIC_INFO['cindex']
                return METRIC_INFO['auc']
            except Exception:
                pass
    return METRIC_INFO['auc']


def _load_ds(base: Path, meta: dict, rep_type: str) -> dict:
    metric = _detect_metric_info(base, fixed=False)
    mean_auc_data = _records_with_aliases(
        _csv(base / metric['condition_file']),
        metric['key'],
    )
    n_rep = None
    if mean_auc_data:
        reps = [r.get('n_replicates') for r in mean_auc_data if r.get('n_replicates') is not None]
        if reps:
            n_rep = int(max(reps))
    resolved_meta = {
        **meta,
        'rep_type': rep_type,
        'n_replicates': n_rep,
        'metric_key': metric['key'],
        'metric_label': metric['label'],
        'task_type': metric['task_type'],
    }
    return {
        'meta':      resolved_meta,
        'mean_auc':  mean_auc_data,
        'metrics':   _records_with_aliases(_csv(base / 'method_level_metrics.csv'), metric['key']),
        'deg':       _records_with_aliases(_csv(base / 'degradation_curve_summary.csv'), metric['key']),
        'wilcoxon':  _records_with_aliases(_csv(base / 'wilcoxon_significant.csv'), metric['key']),
        'bft':       _records_with_aliases(_csv(base / 'best_fixed_train_curve.csv'), metric['key']),
        'general':   _records_with_aliases(_csv(base / 'general_results_summary.csv'), metric['key']),
        'friedman':  _csv(base / 'level1_global_friedman.csv'),
        'top_counts':_csv(base / 'top_equivalent_group_counts.csv'),
    }

def _load_ds_from_results(
    results_dir: Path, ds_key: str, modality: str, meta: dict,
    rep_type: str, distillation_models: list[str] | None,
    retrain: str = 'false',
    source_filter: str | None = 'probability_averaged_ensemble',
) -> dict:
    """Compute dataset statistics directly from results/ using compute.py."""
    import math as _math
    metric = _detect_metric_from_results(results_dir, ds_key)
    metric_col = 'outer_test_CINDEX' if metric['key'] == 'cindex' else 'outer_test_AUC'
    computed = _compute_prog_ds(
        results_dir, ds_key, modality,
        retrain=retrain,
        distillation_models=set(distillation_models) if distillation_models else None,
        metric_col=metric_col,
        source_filter=source_filter,
    )
    if not computed:
        return {}

    # Add auc_ci95 to mean_auc records (normal approximation)
    mean_auc = computed.get('mean_auc', [])
    for r in mean_auc:
        n = r.get('n_replicates') or 0
        std = r.get('std_auc') or 0.0
        r['auc_ci95'] = 1.96 * std / _math.sqrt(n) if n > 0 else None

    # Rename metrics columns to match analysis CSV column names
    metrics = []
    for r in computed.get('metrics', []):
        metrics.append({
            **r,
            'test_time_aupmc':               r.get('test_aupmc'),
            'train_time_aupmc':              r.get('train_aupmc'),
            'best_fixed_train_aupmc':        r.get('bft_aupmc'),
            'best_fixed_train_missing_prop': r.get('bft_train_prop'),
            'train_degradation_coefficient': r.get('train_degradation_coef'),
            'test_degradation_coefficient':  r.get('test_degradation_coef'),
            'minimum_degradation_coefficient': r.get('bft_degradation_coef'),
        })

    # Normalize degradation rows: keep explicit train/test proportions when
    # compute.py provides them, and fall back to old notebook-style fields.
    deg = []
    for r in computed.get('degradation', []):
        scenario = r.get('scenario', 'test')
        missing = r.get('missing_prop')
        deg.append({
            **r,
            'test_prop':  r.get('test_prop',  missing if scenario in ('test', 'envelope') else 0.0),
            'train_prop': r.get('train_prop', missing if scenario == 'train' else 0.0),
        })

    # Rename bft columns to match analysis CSV column names
    bft = []
    for r in computed.get('bft', []):
        bft.append({
            **r,
            'envelope_mean_auc':           r.get('mean_auc'),
            'envelope_train_missing_prop': r.get('best_train_prop'),
        })

    # Add winner_model/loser_model/significant_fdr_0p05 to wilcoxon records
    wilcoxon = []
    for r in computed.get('wilcoxon', []):
        winner = r.get('winner', '')
        ma, mb = r.get('model_a', ''), r.get('model_b', '')
        loser = mb if winner == ma else ma
        wilcoxon.append({
            **r,
            'winner_model':        winner,
            'loser_model':         loser,
            'significant_fdr_0p05': True,
        })

    n_rep = max((r.get('n_replicates') or 0 for r in mean_auc), default=None)
    if n_rep is not None:
        n_rep = int(n_rep) if n_rep > 0 else None

    resolved_meta = {
        **meta,
        'rep_type': rep_type,
        'n_replicates': n_rep,
        'metric_key': metric['key'],
        'metric_label': metric['label'],
        'task_type': metric['task_type'],
    }
    return {
        'meta':       resolved_meta,
        'mean_auc':   mean_auc,
        'metrics':    metrics,
        'deg':        deg,
        'wilcoxon':   wilcoxon,
        'bft':        bft,
        'general':    [],
        'friedman':   computed.get('friedman', []),
        'top_counts': [],
        'hp_selection': computed.get('hp_selection', []),
    }


def _has_progressive_payload(ds_data: dict) -> bool:
    return any(ds_data.get(k) for k in ('mean_auc', 'metrics', 'deg', 'wilcoxon', 'bft', 'general', 'friedman', 'top_counts'))

def _has_static_payload(ds_data: dict) -> bool:
    return any(ds_data.get(k) for k in ('method_summary', 'method_condition', 'pairwise_wilcoxon', 'pairwise_sig', 'replicates', 'friedman'))

def _img_b64(path: Path) -> str:
    if path.exists():
        return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()
    return ''

def load_static(prog_analysis_dir: Path) -> dict:
    """Load fixed-dataset (static-cohort) outputs from fixed_dataset_analysis_outputs/."""
    fixed_dir = prog_analysis_dir.parent / 'fixed_dataset_analysis_outputs'
    if not fixed_dir.exists():
        return {}
    result = {}
    for ds_dir in sorted(fixed_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_key = ds_dir.name
        meta = _infer_dataset_meta(ds_key)
        base = ds_dir / 'retrainfalse'
        if not base.exists():
            continue
        metric = _detect_metric_info(base, fixed=True)
        figs = base / 'figures'
        fig_perf = _img_b64(figs / 'fixed_inner.png')
        fig_pairwise = _img_b64(figs / 'fixed_pairwise.png')
        # The static-cohort UI currently renders the precomputed figures. Old
        # or partial CSV-only folders should not make the static section visible.
        if not fig_perf or not fig_pairwise:
            continue
        n_rep = None
        ms = _records_with_aliases(_csv(base / 'fixed_dataset_method_summary.csv'), metric['key'])
        if ms:
            reps = [r.get('n_replicates') for r in ms if r.get('n_replicates') is not None]
            if reps: n_rep = int(max(reps))
        result[ds_key] = {
            'meta':             {**meta, 'n_replicates': n_rep, 'metric_key': metric['key'], 'metric_label': metric['label'], 'task_type': metric['task_type']},
            'friedman':         _csv(base / 'fixed_dataset_global_friedman.csv'),
            'method_summary':   ms,
            'method_condition': _records_with_aliases(_csv(base / 'fixed_dataset_method_condition_summary.csv'), metric['key']),
            'pairwise_wilcoxon':_records_with_aliases(_csv(base / 'fixed_dataset_pairwise_wilcoxon.csv'), metric['key']),
            'pairwise_sig':     _records_with_aliases(_csv(base / 'fixed_dataset_pairwise_significant.csv'), metric['key']),
            'replicates':       _records_with_aliases(_csv(base / metric['replicate_file']), metric['key']),
            'fig_perf':         fig_perf,
            'fig_pairwise':     fig_pairwise,
        }
        if not _has_static_payload(result[ds_key]):
            result.pop(ds_key, None)
    return result

MODALITY_ORDER = ['global', 'path', 'radio', 'clin', 'blood', 'radio_report']

def load_all(
    analysis_dir: Path,
    results_dir: Path | None = None,
    distillation_models: list[str] | None = None,
) -> tuple:
    """Returns (modes_data, avail_modes, avail_modalities, cohort_endpoints).
    modes_data = {mode_key: {modality: {ds_key: {...}}}}
    cohort_endpoints = {cohort_name: [{ds_key, label}]}
    """
    from collections import defaultdict

    # Build effective dataset metadata from hardcoded display defaults plus output folders.
    effective_meta = dict(DATASET_META)
    if analysis_dir.exists():
        for d in sorted(analysis_dir.iterdir()):
            if d.is_dir() and d.name not in effective_meta:
                effective_meta[d.name] = _infer_dataset_meta(d.name)
    # Also discover datasets from results/
    if results_dir and results_dir.exists():
        for d in sorted(results_dir.iterdir()):
            if d.is_dir() and d.name not in effective_meta:
                effective_meta[d.name] = _infer_dataset_meta(d.name)

    # Discover all modality subdirectories present across datasets
    all_modalities: set[str] = set()
    for ds_key in effective_meta:
        ds_dir = analysis_dir / ds_key
        if ds_dir.exists():
            all_modalities.update(d.name for d in ds_dir.iterdir() if d.is_dir())
        # Also from results/ training_runs structure
        if results_dir:
            tr = results_dir / ds_key / 'training_runs'
            if tr.exists():
                for model_dir in tr.iterdir():
                    if model_dir.is_dir():
                        tm = model_dir / 'TRAIN_MISSING'
                        if tm.exists():
                            all_modalities.update(
                                d.name.lower() for d in tm.iterdir() if d.is_dir()
                            )
    modalities = [m for m in MODALITY_ORDER if m in all_modalities] + \
                 sorted(all_modalities - set(MODALITY_ORDER))

    modes_data: dict = {}
    seen_modes: dict = {}

    for mode_key, retrain_str, subdir, label in MODE_DEFS:
        canon = mode_key.replace('_nm', '')
        if canon in seen_modes:
            continue
        mode_mod: dict = {}
        for modality in modalities:
            mod_ds: dict = {}
            for ds_key, meta in effective_meta.items():
                ds_data: dict = {}

                # Try computing from results/ first. This keeps the dashboard
                # independent from notebook-generated analysis CSVs.
                if _HAS_COMPUTE and results_dir:
                    source_filter = {
                        'ensemble': 'probability_averaged_ensemble',
                        'inner_models': 'mean_retained_inner_models',
                        'outer_retrain': None,
                    }.get(canon, 'probability_averaged_ensemble')
                    retrain_flag = 'true' if 'retraintrue' in retrain_str else 'false'
                    ds_data = _load_ds_from_results(
                        results_dir, ds_key, modality, meta, label, distillation_models,
                        retrain=retrain_flag,
                        source_filter=source_filter,
                    )

                # Fall back to analysis CSVs if results/ didn't yield data
                if not _has_progressive_payload(ds_data) and analysis_dir.exists():
                    base = analysis_dir / ds_key / modality / retrain_str
                    if subdir:
                        base = base / subdir
                    if base.exists():
                        ds_data = _load_ds(base, meta, label)

                if _has_progressive_payload(ds_data):
                    mod_ds[ds_key] = ds_data
            if mod_ds:
                mode_mod[modality] = mod_ds
        if mode_mod:
            modes_data[canon] = mode_mod
            seen_modes[canon] = label

    # Build cohort_endpoints filtered to ds_keys that actually have data
    avail_ds_keys = {dk for md in modes_data.values() for mod_ds in md.values() for dk in mod_ds}
    cohort_eps: dict = defaultdict(list)
    for ds_key, meta in effective_meta.items():
        if ds_key in avail_ds_keys:
            cohort_eps[meta['name']].append({'ds_key': ds_key, 'label': meta.get('endpoint', ds_key)})
    cohort_endpoints = dict(cohort_eps)

    order = ['ensemble', 'inner_models', 'outer_retrain']
    avail_modes = [{'key': k, 'label': seen_modes[k]} for k in order if k in seen_modes]
    avail_modalities = [m for m in modalities
                        if any(m in md for md in modes_data.values())]
    return modes_data, avail_modes, avail_modalities, cohort_endpoints

def _loaded_dataset_names(mode_data: dict) -> list[str]:
    names = []
    for mod_ds in mode_data.values():
        for ds_key, ds_data in mod_ds.items():
            meta = ds_data.get('meta') or _infer_dataset_meta(ds_key)
            names.append(meta.get('name') or ds_key)
    return sorted(set(names))

# ── MAIN ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis_dir', default=None)
    ap.add_argument('--results_dir', default=None,
                    help='Path to results/ directory. Auto-detected if not specified.')
    ap.add_argument('--output', default='m3trics/dashboard/m3trics_dashboard.html')
    ap.add_argument('--distillation_models', default='',
                    help='Optional comma-separated extra model names treated as distillation methods. '
                         'Models ending in _KD are detected automatically.')
    args = ap.parse_args()

    if args.analysis_dir:
        adir = Path(args.analysis_dir)
    else:
        candidates = [
            Path('m3trics/analysis/progressive_missingness_analysis_outputs'),
            Path('../m3trics/analysis/progressive_missingness_analysis_outputs'),
        ]
        adir = next((p for p in candidates if p.exists()), candidates[0])

    if args.results_dir:
        rdir: Path | None = Path(args.results_dir)
    else:
        r_candidates = [
            Path('m3trics/results'),
            Path('../m3trics/results'),
            adir.parent.parent.parent / 'results',
        ]
        rdir = next((p for p in r_candidates if p.exists()), None)

    print(f'[M3TRICS Dashboard] analysis dir: {adir.resolve()}')
    if rdir and rdir.exists():
        print(f'[M3TRICS Dashboard] results dir:  {rdir.resolve()}')
    else:
        print(f'[M3TRICS Dashboard] results dir:  not found — using analysis CSVs only')
        rdir = None

    distillation_models = [m.strip() for m in args.distillation_models.split(',') if m.strip()]
    data, avail_modes, avail_modalities, cohort_endpoints = load_all(adir, rdir, distillation_models)
    if not data:
        print('  WARNING: no datasets found — dashboard will render in empty state')
    else:
        for mk, mds in data.items():
            print(f'  [{mk}] {", ".join(_loaded_dataset_names(mds))}')

    static_data = load_static(adir)
    if static_data:
        names = sorted({(v.get('meta') or {}).get('name') or _infer_dataset_meta(k).get('name') or k
                        for k, v in static_data.items()})
        print(f'  [static-cohort] {", ".join(names)}')
    else:
        print(f'  [static-cohort] no results found')

    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = _build_html(data, static_data, avail_modes, avail_modalities, cohort_endpoints, distillation_models, ts)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')
    kb = out.stat().st_size // 1024
    print(f'  → {out.resolve()}  ({kb} KB)')
    print(f'  open with: open "{out}"')

# ── HTML ──────────────────────────────────────────────────────────────────────────
def _load_logo() -> str:
    logo_path = Path(__file__).parent / 'm3trics_logo.png'
    if logo_path.exists():
        b64 = base64.b64encode(logo_path.read_bytes()).decode()
        return f'data:image/png;base64,{b64}'
    return ''


def _hex_to_rgb(hex_color: str):
    value = str(hex_color).strip().lstrip('#')
    if len(value) != 6:
        return (136, 136, 136)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(max(0, min(255, int(round(v)))) for v in rgb)


def _mix_hex(hex_color: str, target: str = '#ffffff', amount: float = 0.28):
    r1, g1, b1 = _hex_to_rgb(hex_color)
    r2, g2, b2 = _hex_to_rgb(target)
    a = float(amount)
    return _rgb_to_hex((r1 + (r2 - r1) * a, g1 + (g2 - g1) * a, b1 + (b2 - b1) * a))


def _collect_model_names(data, static_data):
    names = set()
    def add_from_dataset(ds_data):
        if not isinstance(ds_data, dict):
            return
        for key in ['mean_auc', 'metrics', 'method_level_metrics', 'summary']:
            for row in ds_data.get(key, []) or []:
                if isinstance(row, dict) and row.get('model_name'):
                    names.add(str(row['model_name']))
    for mode_data in (data or {}).values():
        for modality_data in (mode_data or {}).values():
            for ds_data in (modality_data or {}).values():
                add_from_dataset(ds_data)
    for ds_data in (static_data or {}).values():
        add_from_dataset(ds_data)
    return names


def _method_style_maps(data, static_data):
    colors = dict(METHOD_COLORS)
    display = dict(METHOD_DISPLAY)
    for model_name in sorted(_collect_model_names(data, static_data)):
        if model_name in colors and model_name in display:
            continue
        if model_name.endswith('_KD'):
            base = model_name[:-3]
            colors[model_name] = _mix_hex(colors.get(base, '#888888'), '#ffffff', 0.30)
            display[model_name] = f"{display.get(base, base)}-KD"
        else:
            colors.setdefault(model_name, '#888888')
            display.setdefault(model_name, model_name)
    return colors, display

def _build_html(data, static_data, avail_modes, avail_modalities, cohort_endpoints, distillation_models, ts):
    dj   = json.dumps(data,              separators=(',',':'), ensure_ascii=False)
    sdj  = json.dumps(static_data,       separators=(',',':'), ensure_ascii=False)
    amj  = json.dumps(avail_modes,       separators=(',',':'))
    alj  = json.dumps(avail_modalities,  separators=(',',':'))
    cej  = json.dumps(cohort_endpoints,  separators=(',',':'))
    method_colors, method_display = _method_style_maps(data, static_data)
    mcj  = json.dumps(method_colors,       separators=(',',':'))
    mdj  = json.dumps(method_display,      separators=(',',':'))
    smj  = json.dumps(DATASET_META,       separators=(',',':'))
    dmj  = json.dumps(distillation_models,separators=(',',':'))
    logo = _load_logo()
    return TEMPLATE.replace('$$DATA$$',              dj)\
                   .replace('$$STATIC_DS_DATA$$',     sdj)\
                   .replace('$$AVAIL_MODES$$',        amj)\
                   .replace('$$AVAIL_MODALITIES$$',   alj)\
                   .replace('$$COHORT_ENDPOINTS$$',   cej)\
                   .replace('$$DISTILLATION$$',       dmj)\
                   .replace('$$MC$$', mcj).replace('$$MD$$', mdj)\
                   .replace('$$SM$$', smj).replace('$$TS$$', ts)\
                   .replace('$$LOGO$$', logo)

# ── TEMPLATE ──────────────────────────────────────────────────────────────────────
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>M3TRICS · Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{font-family:'Inter',system-ui,sans-serif;font-size:16.5px;line-height:1.5;transition:background .3s,color .3s;background:var(--bg)}
:root{--dash-scale:1}
html{color-scheme:dark;scrollbar-color:var(--scroll-thumb) var(--scroll-track)}
[data-theme=light]{color-scheme:light}
*{scrollbar-color:var(--scroll-thumb) var(--scroll-track)}
*::-webkit-scrollbar{background:var(--scroll-track)!important}
*::-webkit-scrollbar-track{background:var(--scroll-track)!important}
*::-webkit-scrollbar-thumb{background:var(--scroll-thumb)!important;border-radius:4px}
*::-webkit-scrollbar-corner{background:var(--scroll-track)!important}

/* ── THEMES ─────────────────────────────── */
:root{
  --bg:#080818;--sb:#0d0d26;--card:#13132e;--card2:#101028;--menu-bg:#1a1a3d;
  --scroll-track:#080818;--scroll-thumb:rgba(130,144,190,.34);
  --hov:rgba(255,255,255,.04);--bd:rgba(255,255,255,.095);--bd2:rgba(255,255,255,.18);
  --t1:#e0e4ff;--t2:#7b85b4;--t3:#4a5080;
  --a1:#00d4ff;--a2:#ff2d78;--a3:#a855f7;--a4:#00ff9f;--a5:#ff9900;
  --sh:0 8px 32px rgba(0,0,0,.5);--sh2:0 2px 12px rgba(0,0,0,.3);
  --r:14px;--rsm:8px;--sw:273px;--ease:.25s cubic-bezier(.4,0,.2,1);--hdr-h:80px;
  --panel-head-h:50px;--panel-pill-h:30px;
  --t3:#8290be;
}
[data-theme=light]{
  --bg:#f0f4ff;--sb:#fff;--card:#fff;--card2:#f8faff;--menu-bg:#f8f6ff;
  --scroll-track:#f0f4ff;--scroll-thumb:rgba(74,80,128,.30);
  --hov:rgba(0,0,0,.04);--bd:rgba(0,0,0,.105);--bd2:rgba(0,0,0,.19);
  --t1:#0b0b2e;--t2:#4a5080;--t3:#9ba3c0;
  --a1:#0099cc;--a2:#cc0066;--a3:#6d28d9;--a4:#00aa66;--a5:#b45309;
  --sh:0 8px 32px rgba(0,0,30,.09);--sh2:0 2px 12px rgba(0,0,30,.06);
}

/* ── LAYOUT ──────────────────────────────── */
.app-shell{position:fixed;inset:0;width:100vw;height:100vh;overflow:hidden;background:var(--bg)}
.app{display:flex;width:calc(100vw / var(--dash-scale));height:calc(100vh / var(--dash-scale));
     background:var(--bg);color:var(--t1);transform:scale(var(--dash-scale));
     transform-origin:top left;will-change:transform}

/* ── SIDEBAR ─────────────────────────────── */
.sb{width:var(--sw);min-width:var(--sw);background:var(--sb);border-right:1px solid var(--bd);
    display:flex;flex-direction:column;overflow:hidden;z-index:20;transition:background var(--ease)}
.sb-brand{height:var(--hdr-h);padding:12px 16px 11px;display:flex;flex-direction:column;
          align-items:flex-start;justify-content:center;gap:3px;border-bottom:1px solid var(--bd);
          flex-shrink:0;position:relative;overflow:hidden}
.sb-brand::after{content:'';position:absolute;right:-78px;top:-82px;width:168px;height:168px;
          background:radial-gradient(circle,rgba(168,85,247,.12),rgba(0,212,255,.025) 44%,transparent 72%);
          pointer-events:none}
.sb-logo-img{height:39px;width:auto;max-width:166px;object-fit:contain;
             display:block;transition:filter var(--ease);position:relative;z-index:1}
[data-theme=dark]  .sb-logo-img{filter:brightness(0) invert(1) drop-shadow(0 0 5px rgba(168,85,247,.42))}
[data-theme=light] .sb-logo-img{filter:none}
.sb-tag{font-size:10.5px;font-weight:800;letter-spacing:1.45px;text-transform:uppercase;
        color:var(--t3);line-height:1;position:relative;z-index:1;margin-left:2px}
.sb-name{display:none}

.sb-nav{flex:1;padding:10px 7px 8px;overflow-y:auto;scrollbar-width:none}
.sb-nav::-webkit-scrollbar{display:none}
.sec-lbl{font-size:11.5px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;
         color:var(--t3);padding:10px 9px 4px}
.ni{display:flex;align-items:center;gap:9px;padding:8px 9px;border-radius:var(--rsm);
    cursor:pointer;color:var(--t2);font-size:15px;font-weight:500;
    position:relative;overflow:hidden;transition:all var(--ease);user-select:none}
.ni:hover{background:var(--hov);color:var(--t1)}
.ni.on{background:linear-gradient(90deg,rgba(124,58,237,.2),rgba(124,58,237,.04));color:var(--t1)}
.ni.on::before{content:'';position:absolute;left:0;top:4px;bottom:4px;
               width:3px;background:linear-gradient(180deg,var(--a3),var(--a1));border-radius:0 3px 3px 0}
.ni svg{width:15px;height:15px;flex-shrink:0;opacity:.7;transition:opacity var(--ease)}
.ni.on svg,.ni:hover svg{opacity:1}

.sb-cohorts{padding:6px 7px 4px;border-top:1px solid var(--bd);flex-shrink:0}
.cb{display:flex;align-items:center;gap:8px;width:100%;padding:7px 9px;border:none;
    background:transparent;border-radius:var(--rsm);cursor:pointer;color:var(--t2);
    font-size:14.5px;font-weight:500;text-align:left;transition:all var(--ease)}
.cb:hover{background:var(--hov);color:var(--t1)}
.cb.on{background:rgba(255,255,255,.06);color:var(--t1)}
.cdot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
[data-theme=light] .cb.on{background:rgba(0,0,0,.05)}

.sb-foot{padding:8px 14px 10px;border-top:1px solid var(--bd);
         font-size:12.5px;color:var(--t3);flex-shrink:0}

/* ── MAIN ────────────────────────────────── */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
.hdr{display:flex;align-items:center;
     justify-content:space-between;height:var(--hdr-h);padding:0 18px;
     border-bottom:1px solid var(--bd);background:var(--sb);gap:10px;flex-shrink:0}
.hdr-t{font-size:17.5px;font-weight:700;color:var(--t1)}
.hdr-m{font-size:13.5px;color:var(--t3);margin-top:1px}
.hdr-r{display:flex;align-items:center;gap:7px;flex-shrink:0}
.tpill{display:flex;align-items:center;gap:7px;padding:8px 13px;border-radius:20px;
       border:1px solid var(--bd);background:var(--card);cursor:pointer;color:var(--t2);
       font-size:14.5px;font-weight:500;line-height:1;transition:all var(--ease);white-space:nowrap}
.tpill:hover{border-color:var(--bd2);color:var(--t1)}
.theme-toggle{position:relative}
.hdr-sep{width:1px;height:24px;background:var(--bd);flex:0 0 auto;align-self:center;margin:0 5px}
.ddrop{position:relative}
.dd-btn{display:flex;align-items:center;gap:8px;padding:8px 12px 8px 11px;
        border:1px solid var(--bd);border-radius:20px;cursor:pointer;line-height:1;
        background:var(--card);transition:all var(--ease);user-select:none;white-space:nowrap}
.dd-btn:hover{border-color:var(--bd2)}
.dd-lbl{display:flex;align-items:center;font-size:11.5px;font-weight:700;line-height:1;color:var(--t3);letter-spacing:.5px;text-transform:uppercase}
.dd-sep{width:1px;height:13px;background:var(--bd);flex-shrink:0}
.dd-val{display:flex;align-items:center;font-size:13.5px;font-weight:600;line-height:1;
        background:linear-gradient(90deg,var(--a3),var(--a1));
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
[data-theme=light] .dd-val{background:linear-gradient(90deg,#6d28d9,#0099cc);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.dd-caret{width:9px;height:9px;flex-shrink:0;color:var(--t3)}
.dd-menu{position:absolute;top:calc(100% + 6px);right:0;min-width:100%;
         background:var(--menu-bg);border:1px solid var(--bd2);border-radius:var(--rsm);
         box-shadow:var(--sh);z-index:200;overflow:hidden;display:none}
.dd-menu.open{display:block}
.dd-item{padding:8px 16px;cursor:pointer;font-size:13.5px;font-weight:500;color:var(--t2);
         transition:background var(--ease),color var(--ease);white-space:nowrap}
.dd-item:hover{background:var(--hov);color:var(--t1)}
.dd-item.on{background:linear-gradient(90deg,var(--a3),var(--a1));color:#fff;font-weight:600}

/* Panel visibility toggle pills */
.pv-btns{display:flex;gap:7px;flex-wrap:wrap}
.ci-sep{width:1px;height:24px;background:var(--bd);align-self:center;margin:0 5px;flex:0 0 auto}
.pvbtn.ci-toggle{margin-left:0;color:var(--t3);border-color:var(--bd);background:linear-gradient(180deg,rgba(255,255,255,.045),rgba(255,255,255,.015))}
.pvbtn.ci-toggle:hover{color:var(--t1);border-color:rgba(0,212,255,.34);background:rgba(0,153,204,.09)}
.pvbtn.ci-toggle.on{color:#ffffff;border-color:rgba(0,212,255,.58);background:rgba(0,153,204,.24);box-shadow:0 0 0 1px rgba(0,212,255,.08) inset}
.pvbtn{position:relative;font-size:12px;font-weight:700;padding:0 12px;height:var(--panel-pill-h);
       display:inline-flex;align-items:center;justify-content:center;border-radius:20px;
       cursor:pointer;border:1px solid var(--bd);color:var(--t3);
       background:linear-gradient(180deg,rgba(255,255,255,.045),rgba(255,255,255,.015));
       transition:all var(--ease);user-select:none;white-space:nowrap;letter-spacing:.01em}
.pvbtn:hover{color:var(--t1);border-color:rgba(168,85,247,.34);
       background:rgba(168,85,247,.075)}
.pvbtn.on{color:#fff;border-color:rgba(168,85,247,.48);
       background:rgba(109,40,217,.22);
       box-shadow:none}
[data-theme=light] .pvbtn{background:linear-gradient(180deg,rgba(255,255,255,.95),rgba(246,248,255,.88))}
[data-theme=light] .pvbtn:hover{border-color:rgba(109,40,217,.32);
       background:rgba(109,40,217,.07)}
[data-theme=light] .pvbtn.on{color:#fff;border-color:rgba(109,40,217,.42);
       background:rgba(109,40,217,.72);
       box-shadow:none}
.method-filter{position:relative}
.mf-btn{display:flex;align-items:center;gap:8px;padding:8px 12px 8px 11px;
        border:1px solid var(--bd);border-radius:20px;cursor:pointer;line-height:1;
        background:var(--card);transition:all var(--ease);user-select:none;white-space:nowrap}
.mf-btn:hover{border-color:var(--bd2)}
.mf-caret{width:9px;height:9px;flex-shrink:0;color:var(--t3)}
.mf-menu{position:fixed;min-width:190px;z-index:5000;
         background:var(--menu-bg);border:1px solid var(--bd2);border-radius:14px;
         box-shadow:var(--sh);padding:7px;display:none;gap:7px;flex-direction:column}
.mf-menu.open{display:flex}
.mf-item{display:flex;align-items:center;gap:8px;padding:7px 9px;border-radius:10px;
         cursor:pointer;font-size:12.5px;font-weight:700;color:rgba(255,255,255,.94);
         border:1px solid transparent;user-select:none;white-space:nowrap}
.mf-item:hover{background:var(--hov);color:#fff}
.mf-dot{width:8px;height:8px;border-radius:50%;background:var(--mc);box-shadow:0 0 0 2px var(--mcb)}
.mf-name{flex:1}
.mf-check{font-size:12px;color:var(--mc);opacity:1}
.mf-item.off{opacity:.62;color:rgba(255,255,255,.68)}
.mf-item.off .mf-check{opacity:0}
.mf-item.on{border-color:var(--mcb);background:var(--mcb)}
[data-theme=light] .mf-item.on{background:var(--mcb)}
.dd-btn,.dd-item,.cb,.ni,.pvbtn,.mf-btn,.mf-item,.hp-item,.tpill,.cg-cell,.mt th,.mt tr{
  -webkit-user-select:none;
  user-select:none;
}
#ch-tr .hoverlayer .hovertext,#ch-dg .hoverlayer .hovertext{display:none!important}
.m3tip{position:fixed;z-index:6000;pointer-events:none;display:none;
       background:var(--tip-bg,rgba(19,19,46,.97));border:1.15px solid var(--tip-c,#a855f7);
       border-radius:9px;padding:10px 12px;box-shadow:0 10px 28px rgba(0,0,0,.32);
       color:var(--t1);font-size:13.5px;font-weight:650;line-height:1.28;white-space:nowrap}
.m3tip b{font-weight:800;color:var(--t1)}
.m3tip::before{content:'';position:absolute;left:-8px;top:50%;transform:translateY(-50%);
                border-top:7px solid transparent;border-bottom:7px solid transparent;border-right:8px solid var(--tip-c,#a855f7)}
.m3tip::after{content:'';position:absolute;left:-6.4px;top:50%;transform:translateY(-50%);
               border-top:6px solid transparent;border-bottom:6px solid transparent;border-right:7px solid var(--tip-bg,rgba(19,19,46,.97))}
.m3tip.flip::before{left:auto;right:-8px;border-right:0;border-left:8px solid var(--tip-c,#a855f7)}
.m3tip.flip::after{left:auto;right:-6.4px;border-right:0;border-left:7px solid var(--tip-bg,rgba(19,19,46,.97))}
[data-theme=light] .m3tip{--tip-bg:rgba(255,255,255,.98);box-shadow:0 10px 26px rgba(20,20,60,.16)}

/* ── CONTENT ──────────────────────────────── */
.ct{flex:1;overflow-y:auto;padding:14px 19px 14px 14px;
    scrollbar-width:none;scrollbar-color:transparent transparent}
.ct::-webkit-scrollbar{width:0;height:0;background:transparent!important}
.ct::-webkit-scrollbar-track{background:transparent!important}
.ct::-webkit-scrollbar-thumb{background:transparent!important}
.ct::-webkit-scrollbar-thumb:hover{background:transparent!important}
.sec{display:none}.sec.on{display:block}
/* ── STAT CARDS ───────────────────────────── */
.sr{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:12px}
.sc{border-radius:var(--r);padding:14px 16px;position:relative;overflow:hidden;box-shadow:var(--sh2)}
.sc1{background:linear-gradient(135deg,#7c3aed,#a855f7)}
.sc2{background:linear-gradient(135deg,#0e7490,#00d4ff)}
.sc3{background:linear-gradient(135deg,#047857,#00c47a 60%,#00d4ff 100%)}
.sc4{background:linear-gradient(135deg,#b45309,#ff9900)}
.sc5{background:linear-gradient(135deg,#0e4a6e,#0ea5e9)}
.sc::after{content:'';position:absolute;top:-24px;right:-20px;width:80px;height:80px;
           border-radius:50%;background:rgba(255,255,255,.1)}
.sc-l{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;
      color:rgba(255,255,255,.65);margin-bottom:4px}
.sc-v{font-size:20px;font-weight:800;color:#fff;line-height:1.08;letter-spacing:-.35px}
.sc-s{font-size:13.5px;color:rgba(255,255,255,.55);margin-top:4px}

/* ── CARDS ────────────────────────────────── */
.g1{margin-bottom:10px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.g3{display:grid;grid-template-columns:2fr 1fr;gap:10px;margin-bottom:10px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:var(--r);
      overflow:hidden;box-shadow:var(--sh2);transition:border-color var(--ease)}
.card:hover{border-color:var(--bd2)}
.ch{height:var(--panel-head-h);padding:0 14px;border-bottom:1px solid var(--bd);
    display:flex;align-items:center;justify-content:space-between;gap:8px}
.ct2{font-size:14.5px;font-weight:700;color:var(--t1);letter-spacing:.15px}
.badge{font-size:12px;font-weight:600;padding:2px 8px;border-radius:20px;
       background:rgba(168,85,247,.12);color:var(--a3);border:1px solid rgba(168,85,247,.22);
       white-space:nowrap}
.badge-c{background:rgba(0,212,255,.1);color:var(--a1);border-color:rgba(0,212,255,.2)}
.cp{padding:6px 8px 4px}
#ch-hm .modebar{right:12px!important}

.traj-card{position:relative}
.traj-card .cp{padding-bottom:12px}
.traj-card .cp{min-width:0;overflow:hidden}
.traj-card #ch-tr,.traj-card #ch-dg{width:100%!important;max-width:100%!important;min-width:0!important;overflow:hidden}
.traj-card .js-plotly-plot,.traj-card .plot-container,.traj-card .svg-container{max-width:100%!important;min-width:0!important}
.traj-resize-handle{position:absolute;left:0;right:0;bottom:0;height:13px;cursor:ns-resize;
  display:flex;align-items:center;justify-content:center;opacity:.52;transition:opacity var(--ease),background var(--ease);
  background:linear-gradient(180deg,transparent,rgba(168,85,247,.08));touch-action:none;user-select:none}
.traj-resize-handle::before{content:'';width:54px;height:3px;border-radius:999px;
  background:linear-gradient(90deg,transparent,rgba(168,85,247,.55),transparent)}
.traj-resize-handle:hover,.traj-card.resizing .traj-resize-handle{opacity:1;background:linear-gradient(180deg,transparent,rgba(168,85,247,.14))}
body.traj-resizing{cursor:ns-resize!important;user-select:none!important}

.traj-widget-board{position:relative;display:grid;grid-template-columns:1fr;gap:10px;margin-bottom:10px;transition:grid-template-columns .24s cubic-bezier(.22,1,.36,1),gap .2s ease}
.traj-widget-board.traj-side{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
.traj-slot{min-width:0;transition:transform .22s cubic-bezier(.22,1,.36,1),opacity .18s ease}
.traj-card{min-width:0;transition:transform .22s cubic-bezier(.22,1,.36,1),box-shadow .2s ease,border-color .2s ease,opacity .18s ease}
.traj-card .traj-drag-handle{cursor:grab;position:relative;background:linear-gradient(180deg,rgba(255,255,255,.018),rgba(255,255,255,0))}
.traj-card .traj-drag-handle:active{cursor:grabbing}
.traj-card .pv-btns,.traj-card .pvbtn,.traj-card .ci-toggle{cursor:default}
.traj-card.dragging{z-index:50;transform:scale(1.012);border-color:rgba(168,85,247,.55);box-shadow:0 18px 50px rgba(0,0,0,.38),0 0 0 1px rgba(168,85,247,.16) inset}
.traj-widget-board.widget-previewing .traj-slot:not(.drag-source) .traj-card{opacity:.86;transform:scale(.994)}
.traj-drop-preview{position:absolute;z-index:8;pointer-events:none;border:1.5px solid rgba(224,228,255,.46);
  background:linear-gradient(180deg,rgba(224,228,255,.105),rgba(168,85,247,.07));border-radius:var(--r);
  box-shadow:0 18px 50px rgba(0,0,0,.2),0 0 0 1px rgba(168,85,247,.12) inset;opacity:0;
  transform:scale(.985);transition:left .16s cubic-bezier(.22,1,.36,1),top .16s cubic-bezier(.22,1,.36,1),
  width .16s cubic-bezier(.22,1,.36,1),height .16s cubic-bezier(.22,1,.36,1),opacity .12s ease,transform .16s cubic-bezier(.22,1,.36,1)}
.traj-drop-preview.on{opacity:1;transform:scale(1)}
[data-theme=light] .traj-drop-preview{border-color:rgba(109,40,217,.34);background:linear-gradient(180deg,rgba(109,40,217,.08),rgba(0,153,204,.045))}
.traj-widget-board.traj-side .traj-card .ct2{font-size:13.5px;letter-spacing:0}
.traj-widget-board.traj-side .pv-btns{gap:5px}
.traj-widget-board.traj-side .pvbtn{padding:0 9px;font-size:11.5px;height:var(--panel-pill-h)}
.traj-widget-board.traj-side .ci-sep{margin:0 3px}
body.widget-dragging{cursor:grabbing!important;user-select:none!important}

/* ── TABLE ────────────────────────────────── */
.tw{overflow-x:auto;padding:0 12px 12px}
.hp-card{position:relative}
.hp-card .ch{align-items:center;height:var(--panel-head-h);min-height:var(--panel-head-h);padding-top:0;padding-bottom:0}
.hp-card .tw{padding:6px 0 38px}
.hp-card .tw.hpw{padding:0}
.hp-header-filters{display:flex;align-items:center;gap:6px}
.hp-tp-filter{position:relative;transform:translateZ(0)}
.hp-tp-filter .dd-btn{min-width:130px;transition:none;justify-content:center}
.hp-tp-filter .dd-val{max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hp-tp-filter .dd-caret{transform:translateZ(0)}
.hp-tp-menu{position:fixed;min-width:130px;z-index:5001;
  background:var(--menu-bg);border:1px solid var(--bd2);border-radius:var(--rsm);
  box-shadow:var(--sh);overflow:hidden;display:none}
.hp-tp-menu.open{display:block}
.hp-tp-item{padding:8px 16px;cursor:pointer;font-size:13.5px;font-weight:600;color:var(--t2);
  transition:background var(--ease),color var(--ease);white-space:nowrap;user-select:none;text-align:left}
.hp-tp-item:hover{background:var(--hov);color:var(--t1)}
.hp-tp-item.on{background:linear-gradient(90deg,var(--a3),var(--a1));color:#fff;font-weight:750}
.hp-method-filter{position:relative;transform:translateZ(0)}
.hp-method-filter .dd-btn{min-width:190px;transition:none;justify-content:center}
.hp-method-filter .dd-val{max-width:190px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hp-method-filter .dd-caret{transform:translateZ(0)}
.hp-menu{position:fixed;min-width:210px;z-index:5000;
  background:var(--menu-bg);border:1px solid var(--bd2);border-radius:var(--rsm);
  box-shadow:var(--sh);overflow:hidden;display:none}
.hp-menu.open{display:block}
.hp-item{padding:8px 16px;cursor:pointer;font-size:13.5px;font-weight:600;color:var(--t2);
  transition:background var(--ease),color var(--ease);white-space:nowrap;user-select:none;text-align:left}
.hp-item:hover{background:var(--hov);color:var(--t1)}
.hp-item.on{background:linear-gradient(90deg,var(--a3),var(--a1));color:#fff;font-weight:750}
.hp-item.disabled{opacity:.55;cursor:not-allowed}
.hpw{position:relative;max-height:360px;overflow-y:auto;overflow-x:hidden}
.hpw::after{content:'';display:block;height:18px;flex:0 0 auto}
.hp-table-shell{position:relative;min-width:0}
.hp-head-overlay{position:sticky;top:0;z-index:20;display:flex;align-items:flex-start;
  min-width:0;padding-left:0;background:var(--card);isolation:isolate}
.hp-head-overlay::before{content:'';position:absolute;left:0;right:0;top:-4px;bottom:0;
  background:var(--card);z-index:-1;pointer-events:none}
.hp-table-wrap{display:flex;align-items:flex-start;min-width:0;padding-left:0}
.hp-scroll{min-width:0;flex:1 1 auto;overflow-x:auto;overflow-y:hidden}
.hp-head-main{overflow:hidden}
.hp-fixed{flex:0 0 176px;width:176px;z-index:5;background:var(--card);
  border-left:1px solid var(--bd2);box-shadow:-18px 0 22px -18px rgba(0,0,0,.9)}
.hpw .mt{border-collapse:separate;border-spacing:0;background:var(--card)}
.hpw .mt td,.hpw .mt th{background:var(--card)}
.hp-scroll .mt{min-width:900px;margin-left:0}
.hp-fixed .mt{width:176px;min-width:176px;margin-left:0}
.mt.hp-main,.mt.hp-selected{table-layout:fixed}
.hp-scroll .mt.hp-main td{overflow:hidden;text-overflow:ellipsis}
.hp-fixed th,.hp-fixed td{text-align:center;background:var(--card)!important}
.hpw td.hp-combo{font-family:'JetBrains Mono','SFMono-Regular',Menlo,monospace;font-size:12px;color:var(--t3);max-width:360px;overflow:hidden;text-overflow:ellipsis;position:relative;cursor:default}
.hp-combo-text{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;pointer-events:none}
.hp-combo-copy{display:none;position:absolute;inset:0;align-items:center;justify-content:center;
  gap:5px;cursor:pointer;font-size:12px;font-weight:600;color:var(--a1);
  background:var(--card);font-family:inherit;user-select:none}
.hp-combo:hover .hp-combo-copy{display:flex}
.hp-combo:hover .hp-combo-text{visibility:hidden}
.hpw td.hp-method{font-weight:800}
.hpw .mt thead th{position:relative;z-index:7;
  background:linear-gradient(0deg,rgba(255,255,255,.018),rgba(255,255,255,.018)),var(--card)!important;
  box-shadow:none;height:var(--panel-head-h);vertical-align:middle}
.hp-fixed th{z-index:9;background:linear-gradient(0deg,rgba(255,255,255,.018),rgba(255,255,255,.018)),var(--card)!important}
.hp-table-wrap tr.hp-row-hover td,.hp-table-wrap tr[data-m]:hover td{background:var(--hov)}
.hp-fixed tr.hp-row-hover td,.hp-fixed tr[data-m]:hover td{background:linear-gradient(0deg,var(--hov),var(--hov)),var(--card)!important}
.hp-resize-handle{position:absolute;left:0;right:0;bottom:0;height:18px;cursor:ns-resize;z-index:8;
  display:flex;align-items:center;justify-content:center;
  transition:background var(--ease);
  background:var(--card);
  touch-action:none;user-select:none}
.hp-resize-handle::before{content:'';width:54px;height:3px;border-radius:999px;
  opacity:.58;transition:opacity var(--ease);
  background:linear-gradient(90deg,transparent,rgba(168,85,247,.62),transparent)}
.hp-resize-handle:hover,.hp-card.resizing .hp-resize-handle{background:linear-gradient(180deg,var(--card),var(--card))}
.hp-resize-handle:hover::before,.hp-card.resizing .hp-resize-handle::before{opacity:1}
body.hp-resizing{cursor:ns-resize!important;user-select:none!important}
.mt{width:100%;border-collapse:collapse;font-size:14.5px}
.mt th{padding:8px 12px;text-align:left;font-size:12px;font-weight:700;
       text-transform:uppercase;letter-spacing:.4px;color:var(--t3);
       border-bottom:1px solid var(--bd);cursor:pointer;white-space:nowrap;user-select:none}
.mt th:hover{color:var(--t2)}
.mt td{padding:8px 12px;color:var(--t2);border-bottom:1px solid var(--bd);white-space:nowrap}
.mt tr:last-child td{border-bottom:none}
.hp-spacer td{height:14px;padding:0;border:none;background:transparent!important;pointer-events:none}
.hpw .mt tr:nth-last-child(2) td{border-bottom:none}
.mt tr[data-m]:hover td{background:var(--hov);cursor:pointer}
.mt tr.hl td{background:rgba(168,85,247,.07)!important}
.hp-scroll .mt th:first-child,.hp-scroll .mt td:first-child{padding-left:15px}
.best{font-weight:700;color:var(--a1)!important}
.si{opacity:.3;font-size:9px;margin-left:2px}

/* ── CONDITION GRID ───────────────────────── */
.cgw{padding:10px 12px 14px;overflow-x:auto}
.cg{display:grid;gap:5px}
.cg-ch,.cg-rh{font-size:12.5px;font-weight:700;color:var(--t3);letter-spacing:.4px}
.cg-rh{text-align:right;padding-right:6px;white-space:nowrap}
.cg-ch{text-align:center}
.cg-cell{border-radius:var(--rsm);padding:7px 4px;cursor:pointer;
         text-align:center;transition:all var(--ease);border:1px solid transparent}
.cg-cell:hover{transform:scale(1.05);box-shadow:0 4px 18px rgba(0,0,0,.3)}
.cg-mn{font-size:13.5px;font-weight:700;line-height:1.2}
.cg-av{font-size:12px;margin-top:2px;opacity:.65}

/* ── MISC ────────────────────────────────── */
.es{display:flex;flex-direction:column;align-items:center;justify-content:center;
    height:200px;color:var(--t3);font-size:12.5px;gap:8px;text-align:center}
.es-i{font-size:32px;opacity:.35}
/* ── STATIC COHORT FIGURES ───────────────── */
.sc-static-cards{grid-template-columns:repeat(3,1fr)!important;margin-bottom:12px}
.sc-fig-wrap{padding:8px 12px 12px;text-align:center}
.sc-fig{max-width:100%;height:auto;border-radius:var(--rsm);display:block;margin:0 auto}
.sc-fig[src=""]{display:none}
/* ── NO RESULTS PANEL ────────────────────── */
.noresults-wrap{display:flex;align-items:center;justify-content:center;
                height:calc(100vh - 120px);flex-direction:column;gap:18px;
                color:var(--t3);text-align:center;padding:40px}
.noresults-title{font-size:20px;font-weight:700;color:var(--t2)}
.noresults-sub{font-size:14.5px;max-width:360px;line-height:1.6}
/* ── PLOTLY HOVER TOOLTIP ──────────────── */
</style>
</head>
<body>
<div class="app-shell">
<div class="app">

<!-- SIDEBAR -->
<aside class="sb">
  <div class="sb-brand">
    <img class="sb-logo-img" src="$$LOGO$$" alt="M3TRICS">
    <div class="sb-tag">Results dashboard</div>
  </div>

  <nav class="sb-nav">
    <div class="sec-lbl">Progressive Missingness</div>
    <div class="ni on" data-s="global" data-study="progressive" onclick="nav('global','progressive')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 010 20M12 2a15.3 15.3 0 000 20"/></svg>
      <span>Global Results</span>
    </div>
    <div class="ni" data-s="metrics" data-study="progressive" onclick="nav('metrics','progressive')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      <span>Method Metrics</span>
    </div>
    <div class="ni" data-s="conds" data-study="progressive" onclick="nav('conds','progressive')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 21H3M21 3H3"/><path d="M7 3v18M17 3v18M3 7h18M3 17h18"/></svg>
      <span>Condition Analysis</span>
    </div>
    <div class="ni" data-s="summary" data-study="progressive" onclick="nav('summary','progressive')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      <span>Cross-Cohort Summary</span>
    </div>

    <div id="nav-static-lbl" class="sec-lbl" style="margin-top:12px">Static-Cohort</div>
    <div class="ni" data-s="sc-perf" data-study="static" onclick="nav('sc-perf','static')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      <span>Performance Results</span>
    </div>
    <div class="ni" data-s="sc-pairwise" data-study="static" onclick="nav('sc-pairwise','static')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>
      <span>Pairwise Tests</span>
    </div>
  </nav>

  <div class="sb-cohorts">
    <div class="sec-lbl" style="padding:6px 9px 5px">Dataset</div>
    <div id="cbs"></div>
  </div>
  <div class="sb-foot">Generated $$TS$$</div>
</aside>

<!-- MAIN -->
<main class="main">
  <header class="hdr">
    <div>
      <div class="hdr-t" id="ht">Global Results</div>
      <div class="hdr-m" id="hm"></div>
    </div>
    <div class="hdr-r">
      <div class="ddrop" id="ddrop-ep">
        <div class="dd-btn" onclick="toggleDd('ep')">
          <span class="dd-lbl">Endpoint</span>
          <div class="dd-sep"></div>
          <span class="dd-val" id="dd-ep-val">–</span>
          <svg class="dd-caret" viewBox="0 0 10 6" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M1 1l4 4 4-4"/></svg>
        </div>
        <div class="dd-menu" id="dd-ep-menu"></div>
      </div>
      <div class="ddrop" id="ddrop-mod">
        <div class="dd-btn" onclick="toggleDd('mod')">
          <span class="dd-lbl">Degrading modality</span>
          <div class="dd-sep"></div>
          <span class="dd-val" id="dd-mod-val">–</span>
          <svg class="dd-caret" viewBox="0 0 10 6" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M1 1l4 4 4-4"/></svg>
        </div>
        <div class="dd-menu" id="dd-mod-menu"></div>
      </div>
      <div class="ddrop" id="ddrop-rep">
        <div class="dd-btn" onclick="toggleDd('rep')">
          <span class="dd-lbl">Replicate definition</span>
          <div class="dd-sep"></div>
          <span class="dd-val" id="dd-rep-val">–</span>
          <svg class="dd-caret" viewBox="0 0 10 6" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M1 1l4 4 4-4"/></svg>
        </div>
        <div class="dd-menu" id="dd-rep-menu"></div>
      </div>
      <div class="method-filter" id="method-filter"></div>
      <div class="hdr-sep" aria-hidden="true"></div>
      <div class="tpill theme-toggle" onclick="toggleTheme()">
        <span id="ti">🌙</span><span id="tl">Dark</span>
      </div>
    </div>
  </header>

  <div class="ct">

    <!-- GLOBAL -->
    <div class="sec on" id="s-global">
      <div class="sr">
        <div class="sc sc1">
          <div class="sc-l">Dataset</div>
          <div class="sc-v" id="s-dsname" style="margin-top:4px">–</div>
          <div class="sc-s"><span id="s-nm">–</span> modalities · <span id="s-np">–</span> patients</div>
        </div>
        <div class="sc sc2">
          <div class="sc-l">Task — Endpoint</div>
          <div class="sc-v" id="s-ep" style="margin-top:4px">–</div>
          <div class="sc-s" id="s-tasktype">–</div>
        </div>
        <div class="sc sc3">
          <div class="sc-l">Methods evaluated</div>
          <div class="sc-v" id="s-nme">–</div>
          <div class="sc-s" id="s-nc">–</div>
        </div>
        <div class="sc sc4">
          <div class="sc-l">Friedman test p-value</div>
          <div class="sc-v" id="s-fp">–</div>
          <div class="sc-s" id="s-fs">–</div>
        </div>
        <div class="sc sc5">
          <div class="sc-l">Replicate definition</div>
          <div class="sc-v" id="s-rtype" style="margin-top:4px;line-height:1.25">–</div>
          <div class="sc-s" id="s-nrep">– replicates per condition</div>
        </div>
      </div>
      <div class="g1"><div class="card">
        <div class="ch"><span class="ct2">Performance Heatmaps across the m<sub>train</sub> × m<sub>test</sub> Missingness Grid &mdash; Mean <span id="hm-metric-lbl">AUC</span> &plusmn; 95% C.I.</span></div>
        <div class="cp" style="padding-top:0"><div id="ch-hm"></div></div>
      </div></div>
      <div class="traj-widget-board traj-stack" id="traj-widgets">
        <div class="traj-slot" data-widget="perf">
          <div class="card traj-card" data-widget="perf" data-plot="ch-tr">
            <div class="ch traj-drag-handle"><span class="ct2">Method-level Performance Trajectories</span>
              <div class="pv-btns">
                <span class="pvbtn on" data-chart="traj" data-p="train" onclick="togglePanel('traj','train')">Train-time</span>
                <span class="pvbtn on" data-chart="traj" data-p="test"  onclick="togglePanel('traj','test')">Test-time</span>
                <span class="pvbtn on" data-chart="traj" data-p="bft"   onclick="togglePanel('traj','bft')">Best fixed-train</span>
                <span class="ci-sep" aria-hidden="true"></span>
                <span class="pvbtn ci-toggle on" data-chart="traj" data-ci="1" onclick="toggleCI('traj')">95% C.I.</span>
              </div></div>
            <div class="cp"><div id="ch-tr"></div></div>
            <div class="traj-resize-handle" data-plot="ch-tr" title="Click or drag to resize panel"></div>
          </div>
        </div>
        <div class="traj-slot" data-widget="deg">
          <div class="card traj-card" data-widget="deg" data-plot="ch-dg">
            <div class="ch traj-drag-handle"><span class="ct2">Method-level Degradation Trajectories</span>
              <div class="pv-btns">
                <span class="pvbtn on" data-chart="deg" data-p="train" onclick="togglePanel('deg','train')">Train-time</span>
                <span class="pvbtn on" data-chart="deg" data-p="test"  onclick="togglePanel('deg','test')">Test-time</span>
                <span class="pvbtn on" data-chart="deg" data-p="bft"   onclick="togglePanel('deg','bft')">Best fixed-train</span>
                <span class="ci-sep" aria-hidden="true"></span>
                <span class="pvbtn ci-toggle on" data-chart="deg" data-ci="1" onclick="toggleCI('deg')">95% C.I.</span>
              </div></div>
            <div class="cp"><div id="ch-dg"></div></div>
            <div class="traj-resize-handle" data-plot="ch-dg" title="Click or drag to resize panel"></div>
          </div>
        </div>
      </div>
      <div class="g1"><div class="card hp-card">
        <div class="ch">
          <span class="ct2">Hyperparameter Selection Summary</span>
          <div class="hp-header-filters">
            <div class="hp-tp-filter" id="hp-trainprop-filter"></div>
            <div class="hp-method-filter" id="hp-method-filter"></div>
          </div>
        </div>
        <div class="tw hpw" id="hp-sel"></div>
        <div class="hp-resize-handle" title="Click or drag to resize panel"></div>
      </div></div>
    </div>

    <!-- METRICS -->
    <div class="sec" id="s-metrics">
      <div class="g3">
        <div class="card"><div class="ch"><span class="ct2">AUPMC Comparison</span></div><div class="cp"><div id="ch-br"></div></div></div>
        <div class="card"><div class="ch"><span class="ct2">Multi-Metric Radar</span><span class="badge">normalised</span></div><div class="cp"><div id="ch-rd"></div></div></div>
      </div>
      <div class="g1"><div class="card">
        <div class="ch"><span class="ct2">Method-Level Metrics Table</span><span class="badge">click header to sort · click row to highlight</span></div>
        <div class="tw" id="tbl"></div>
      </div></div>
    </div>

    <!-- CONDITIONS -->
    <div class="sec" id="s-conds">
      <div class="g1"><div class="card">
        <div class="ch"><span class="ct2">Top Method by Condition</span><span class="badge">click a cell to drill down</span></div>
        <div class="cgw" id="cgc"></div>
      </div></div>
      <div class="g1"><div class="card">
        <div class="ch">
          <span class="ct2">Pairwise Wilcoxon — Significant wins (FDR p&lt;0.05)</span>
          <span class="badge" id="clbl">0% train / 0% test</span>
        </div>
        <div class="cp"><div id="ch-wl"></div></div>
      </div></div>
    </div>

    <!-- SUMMARY -->
    <div class="sec" id="s-summary">
      <div class="g1"><div class="card">
        <div class="ch"><span class="ct2">Cross-Cohort AUPMC</span><span class="badge badge-c">BFT AUPMC per cohort</span></div>
        <div class="cp"><div id="ch-xc"></div></div>
      </div></div>
      <div class="g2">
        <div class="card"><div class="ch"><span class="ct2">Method Rankings Across Cohorts</span><span class="badge">by BFT AUPMC</span></div><div class="tw" id="tbl-rk"></div></div>
        <div class="card"><div class="ch"><span class="ct2">Top-Group Frequency</span><span class="badge">fraction of conditions</span></div><div class="cp"><div id="ch-tg"></div></div></div>
      </div>
    </div>

    <!-- STATIC COHORT — PERFORMANCE -->
    <div class="sec" id="s-sc-perf">
      <div class="sr sc-static-cards">
        <div class="sc sc1">
          <div class="sc-l">Dataset</div>
          <div class="sc-v" id="sc-dsname" style="margin-top:4px">–</div>
          <div class="sc-s"><span id="sc-nm">–</span> modalities · <span id="sc-np">–</span> patients</div>
        </div>
        <div class="sc sc4">
          <div class="sc-l">Friedman test p-value</div>
          <div class="sc-v" id="sc-fp">–</div>
          <div class="sc-s" id="sc-fs">–</div>
        </div>
        <div class="sc sc5">
          <div class="sc-l">Replicates per model</div>
          <div class="sc-v" id="sc-nrep">–</div>
          <div class="sc-s">retained inner models</div>
        </div>
      </div>
      <div class="g1"><div class="card">
        <div class="ch"><span class="ct2">Method Performance — <span id="sc-metric-dist-lbl">AUC</span> Distribution</span><span class="badge">Static-cohort · Retained inner models</span></div>
        <div class="cp sc-fig-wrap"><img id="sc-fig-perf" class="sc-fig" src="" alt="Performance figure"></div>
      </div></div>
    </div>

    <!-- STATIC COHORT — PAIRWISE TESTS -->
    <div class="sec" id="s-sc-pairwise">
      <div class="g1"><div class="card">
        <div class="ch"><span class="ct2">Pairwise Wilcoxon Tests — Significant Comparisons</span><span class="badge">FDR p&lt;0.05</span></div>
        <div class="cp sc-fig-wrap"><img id="sc-fig-pairwise" class="sc-fig" src="" alt="Pairwise tests figure"></div>
      </div></div>
    </div>

    <!-- NO RESULTS -->
    <div class="sec" id="s-noresults">
      <div class="noresults-wrap">
        <div class="noresults-title">No results found for this analysis mode</div>
        <div class="noresults-sub">Run the M3TRICS pipeline for this study type first, then regenerate the dashboard.</div>
      </div>
    </div>


  </div>
</main>
</div>
</div>

<script>
// ── DATA ───────────────────────────────────────────────────────────────────
const MODES_DATA         = $$DATA$$;
const STATIC_DS_DATA     = $$STATIC_DS_DATA$$;
const AVAIL_MODES        = $$AVAIL_MODES$$;
const AVAIL_MODALITIES   = $$AVAIL_MODALITIES$$;
const COHORT_ENDPOINTS   = $$COHORT_ENDPOINTS$$;
const DISTILLATION_MODELS= new Set($$DISTILLATION$$);
// ds_key → cohort name lookup derived from COHORT_ENDPOINTS
const DS_COHORT = {};
Object.entries(COHORT_ENDPOINTS).forEach(([c,eps])=>eps.forEach(e=>DS_COHORT[e.ds_key]=c));
const MODALITY_LABELS    = {global:'Global',path:'Pathology',radio:'Radiology',
                            clin:'Clinical',blood:'Blood',radio_report:'Radiology report'};
const MC  = $$MC$$;
const MD  = $$MD$$;
const SM  = $$SM$$;

// ── RESPONSIVE DASHBOARD SCALE ─────────────────────────────────────────────
// The dashboard is designed on large monitors. On MacBook Pro Retina screens,
// the logical viewport is smaller, so we start slightly zoomed out. External
// monitors keep scale=1. Override manually with ?scale=1 or ?scale=0.82.
let DASH_SCALE_LAST = null;
let DASH_SCALE_TIMER = null;
function preferredDashboardScale(){
  const qs = new URLSearchParams(window.location.search);
  const forced = qs.get('scale');
  if(forced && forced.toLowerCase() !== 'auto'){
    const v = Number(forced);
    if(Number.isFinite(v)) return Math.max(0.68, Math.min(1.12, v));
  }
  const dpr = window.devicePixelRatio || 1;
  const sw = Math.max(window.screen?.width || 0, window.screen?.height || 0);
  const sh = Math.min(window.screen?.width || 0, window.screen?.height || 0);
  const vw = window.innerWidth || 0;
  const vh = window.innerHeight || 0;

  // MacBook Pro M3 14-inch commonly reports ~1512 x 982 CSS pixels.
  // MacBook Pro 16-inch commonly reports ~1728 x 1117 CSS pixels.
  const retinaLaptop = dpr >= 1.8 && sw <= 1800 && sh <= 1180;
  if(retinaLaptop){
    if(sw <= 1530 || vw <= 1530) return 0.77;
    return 0.83;
  }

  // Fallback for very small browser windows, without affecting normal external monitors.
  if(vw && vh && (vw < 1320 || vh < 760)) return 0.84;
  return 1;
}
function dashboardScale(){
  const css = Number(getComputedStyle(document.documentElement).getPropertyValue('--dash-scale'));
  return DASH_SCALE_LAST || (Number.isFinite(css) && css > 0 ? css : 1);
}
function applyDashboardScale(){
  const scale = preferredDashboardScale();
  if(DASH_SCALE_LAST !== null && Math.abs(scale - DASH_SCALE_LAST) < 0.01) return;
  DASH_SCALE_LAST = scale;
  document.documentElement.style.setProperty('--dash-scale', scale.toFixed(3));
  document.body?.setAttribute('data-dashboard-scale', scale.toFixed(2));
  if(DASH_SCALE_TIMER) clearTimeout(DASH_SCALE_TIMER);
  DASH_SCALE_TIMER = setTimeout(()=>{
    if(window.Plotly){
      document.querySelectorAll('.js-plotly-plot').forEach(gd=>Plotly.Plots.resize(gd));
    }
    if(typeof scheduleTrajectoryResize === 'function') scheduleTrajectoryResize();
    if(typeof positionMethodMenu === 'function') positionMethodMenu();
    if(typeof positionHpMethodMenu === 'function') positionHpMethodMenu();
  }, 120);
}
applyDashboardScale();
window.addEventListener('resize', applyDashboardScale, {passive:true});

// ── STATE ──────────────────────────────────────────────────────────────────
const S = {ds:null, sec:'global', theme:'dark', mode: AVAIL_MODES[0]?.key||'ensemble',
           modality: AVAIL_MODALITIES[0]||'global',
           study:'progressive', sk:'baseline_auc', sa:false, hl:null, cond:{tr:0,te:0},
           trajVis:{train:true,test:true,bft:true},
           degVis:{train:true,test:true,bft:true},
           ciVis:{traj:true,deg:true},
           hmTextSize:11.5,
           trajHeights:{},
           trajWidgetLayout:'stack',
           trajWidgetOrder:['perf','deg'],
           trajCompactPanel:'train',
           hpMethod:null,hpTrainProp:null,
           hpHeight:360,
           methodVis:{}};

// ── HELPERS ────────────────────────────────────────────────────────────────
const curMod = () => ((MODES_DATA[S.mode]||{})[S.modality]||{});
const D   = () => curMod()[S.ds] || {};
const SD  = () => STATIC_DS_DATA[S.ds] || {};
const isDistill = m => DISTILLATION_MODELS.has(m) || /_KD$/i.test(String(m||''));
const notDistill = m => !isDistill(m);
const dk  = () => S.theme === 'dark';
const f2  = v  => v == null ? '–' : v.toFixed(3);
const pct = v  => v == null ? '–' : `${(v*100).toFixed(0)}%`;
const clamp = (v,lo,hi) => Math.max(lo, Math.min(hi, v));
const metricLabel = (obj=null) => ((obj||D()).meta||{}).metric_label || 'AUC';
const metricDeltaLabel = (obj=null) => `Δ${metricLabel(obj)}`;
const activeMethods = () => [...new Set((D().mean_auc||[]).map(r=>r.model_name))].sort()
  .filter(m=>S.methodVis[m]!==false);
const allMethods = () => [...new Set((D().mean_auc||[]).map(r=>r.model_name))].sort();
const anyDropdownOpen = () => !!document.querySelector('.dd-menu.open,.mf-menu.open,.hp-menu.open,.hp-tp-menu.open');
function clearPlotHovers(){
  ['ch-hm','ch-tr','ch-dg','ch-bars','ch-radar','ch-pw','ch-top','sc-heat','sc-bars'].forEach(id=>{
    if(window.Plotly) Plotly.Fx.unhover(id);
  });
}
function guardHoverWhenDropdownOpen(gd){
  if(!gd||!gd.on) return;
  if(gd._m3DropdownHoverGuard&&gd.removeListener) gd.removeListener('plotly_beforehover',gd._m3DropdownHoverGuard);
  gd._m3DropdownHoverGuard=()=>!anyDropdownOpen();
  gd.on('plotly_beforehover',gd._m3DropdownHoverGuard);
}

function escHtml(v){
  return String(v??'–').replace(/[&<>"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
}
function ensureTrajectoryTip(){
  let tip=document.getElementById('m3-traj-tip');
  if(!tip){tip=document.createElement('div');tip.id='m3-traj-tip';tip.className='m3tip';document.body.appendChild(tip);}
  return tip;
}
function hideTrajectoryTip(){
  const tip=document.getElementById('m3-traj-tip');
  if(tip)tip.style.display='none';
}
function attachTrajectoryTooltip(gd,kind){
  if(!gd||!gd.on)return;
  if(gd._m3TipHover&&gd.removeListener)gd.removeListener('plotly_hover',gd._m3TipHover);
  if(gd._m3TipUnhover&&gd.removeListener)gd.removeListener('plotly_unhover',gd._m3TipUnhover);
  gd._m3TipHover=function(ev){
    if(anyDropdownOpen()){hideTrajectoryTip();return false;}
    const pt=(ev.points||[])[0];
    if(!pt||!pt.data||pt.data.hoverinfo==='skip'){hideTrajectoryTip();return;}
    const cd=pt.customdata||[];
    const color=(pt.data.line&&pt.data.line.color)||pt.data.marker?.color||'#a855f7';
    const label=kind==='deg'?'Ratio':`Mean ${metricLabel()}`;
    const tip=ensureTrajectoryTip();
    tip.style.setProperty('--tip-c',color);
    tip.innerHTML=`<b>${escHtml(pt.data.name)}</b><br>Train m: ${escHtml(cd[0])}<br>Test m: ${escHtml(cd[1])}<br>${label}: ${Number(pt.y).toFixed(3)}<br>95% C.I.: ${escHtml(cd[2])}`;
    const e=ev.event||window.event||{};
    tip.style.display='block';
    const w=tip.offsetWidth||0,h=tip.offsetHeight||0;
    const rect=gd.getBoundingClientRect();
    const ax=pt.xaxis, ay=pt.yaxis;
    const sc=dashboardScale();
    const px=rect.left+(ax&&ax.l2p?(ax._offset+ax.l2p(pt.x))*sc:(e.clientX||rect.left)-rect.left);
    const py=rect.top +(ay&&ay.l2p?(ay._offset+ay.l2p(pt.y))*sc:(e.clientY||rect.top)-rect.top);
    let flip=false;
    let x=px+14;
    let y=py-h/2;
    if(x+w>window.innerWidth-8){x=px-w-14;flip=true;}
    y=Math.max(8,Math.min(window.innerHeight-h-8,y));
    tip.classList.toggle('flip',flip);
    tip.style.left=`${x}px`;tip.style.top=`${y}px`;
  };
  gd._m3TipUnhover=hideTrajectoryTip;
  gd.on('plotly_hover',gd._m3TipHover);
  gd.on('plotly_unhover',gd._m3TipUnhover);
}

function isVisibleMethodTrace(tr){
  return tr && tr.hoverinfo!=='skip' && tr.name!=='No degradation' && tr.visible!==false && tr.visible!=='legendonly';
}
function compactTrajectoryLayout(gd,msg){
  const patch={
    height:TRAJ_EMPTY_HEIGHT,
    margin:{t:76,b:26,l:24,r:24},
    plot_bgcolor:'rgba(0,0,0,0)',
    paper_bgcolor:'rgba(0,0,0,0)',
    annotations:[{x:.5,y:.42,xref:'paper',yref:'paper',showarrow:false,
      text:msg,align:'center',font:{size:15,color:dk()?'rgba(224,228,255,.68)':'rgba(11,11,46,.6)'}}],
  };
  (gd._m3AxisKeys||[]).forEach(k=>{patch[k]={visible:false,showticklabels:false,showgrid:false,zeroline:false};});
  return patch;
}
function setNoDegradationVisible(gd,visible){
  const idx=(gd.data||[]).map((tr,i)=>tr.name==='No degradation'?i:null).filter(i=>i!=null);
  if(!idx.length)return Promise.resolve();
  const needs=idx.some(i=>(gd.data[i].visible!==visible));
  if(!needs)return Promise.resolve();
  gd._m3LegendEmptySyncing=true;
  return Plotly.restyle(gd,{visible},idx).then(()=>{gd._m3LegendEmptySyncing=false;});
}
function syncTrajectoryLegendEmpty(gd,msg){
  if(!gd||!gd.data||!window.Plotly||gd._m3LegendEmptySyncing)return;
  const hasVisible=gd.data.some(isVisibleMethodTrace);
  if(hasVisible){
    setNoDegradationVisible(gd,true).then(()=>{
      Plotly.relayout(gd,gd._m3FullLayoutForEmpty||{});
    });
  }else{
    setNoDegradationVisible(gd,false).then(()=>{
      Plotly.relayout(gd,compactTrajectoryLayout(gd,msg));
    });
  }
}
function attachTrajectoryLegendEmpty(gd,msg){
  if(!gd||!gd.on)return;
  if(gd._m3EmptyHandler&&gd.removeListener)gd.removeListener('plotly_restyle',gd._m3EmptyHandler);
  gd._m3EmptyHandler=()=>setTimeout(()=>{
    syncTrajectoryLegendEmpty(gd,msg);
    scheduleTrajectoryLegendCenter(gd.id);
  },0);
  gd.on('plotly_restyle',gd._m3EmptyHandler);
  syncTrajectoryLegendEmpty(gd,msg);
  scheduleTrajectoryLegendCenter(gd.id);
}

function hexRgb(h){h=h.replace('#','');return[parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)]}
function rgba(h,a=1){const[r,g,b]=hexRgb(h);return`rgba(${r},${g},${b},${a})`}
function axisDomains(n,g=.045){
  const w=(1-g*(n-1))/n;
  return Array.from({length:n},(_,i)=>[i*(w+g),i*(w+g)+w]);
}
function changeHeatmapTextSize(delta){
  S.hmTextSize=clamp(S.hmTextSize+delta,8,18);
  rHeatmaps();
}
function resetHeatmapTextSize(){
  S.hmTextSize=11.5;
  rHeatmaps();
}
const HM_PLUS_ICON={width:24,height:24,path:'M11 4h2v7h7v2h-7v7h-2v-7H4v-2h7z'};
const HM_MINUS_ICON={width:24,height:24,path:'M5 11h14v2H5z'};
const HM_RESET_ICON={width:24,height:24,path:'M12 5a7 7 0 1 1-6.3 4H3l3.8-3.8L10.6 9H7.8A5 5 0 1 0 12 7z'};
const CFG  = {displayModeBar:true,modeBarButtonsToRemove:['lasso2d','select2d','toImage'],displaylogo:false,responsive:true,scrollZoom:false};
const CFG_HM = {displayModeBar:true,displaylogo:false,responsive:true,scrollZoom:false,doubleClick:false,
  modeBarButtons:[[{
    name:'Increase cell text size',title:'Increase cell text size',icon:HM_PLUS_ICON,click:()=>changeHeatmapTextSize(1.0)
  },{
    name:'Reset cell text size',title:'Reset cell text size',icon:HM_RESET_ICON,click:()=>resetHeatmapTextSize()
  },{
    name:'Decrease cell text size',title:'Decrease cell text size',icon:HM_MINUS_ICON,click:()=>changeHeatmapTextSize(-1.0)
  }]]};
const CFGs = {displayModeBar:false,responsive:true};
const TRAJ_HEIGHT = 530;
const TRAJ_EMPTY_HEIGHT = 190;
const TRAJ_MARGIN = {t:140,b:72,l:68,r:32};
const TRAJ_LEGEND_TOP_PX = 38;
const TRAJ_LEGEND = {orientation:'h',x:0.5,xanchor:'center',y:1.18,yanchor:'top',
  font:{size:13.5},bgcolor:'rgba(0,0,0,0)',groupclick:'togglegroup'};
const TRAJ_LEGEND_ENTRY_W = 110;
function trajMarginTop(divId){
  const w=Math.max(200,(document.getElementById(divId)?.offsetWidth||600)-TRAJ_MARGIN.l-TRAJ_MARGIN.r);
  const nMeth=activeMethods().length;
  const perRow=Math.max(1,Math.floor(w/TRAJ_LEGEND_ENTRY_W));
  const nRows=Math.ceil(nMeth/perRow);
  return TRAJ_MARGIN.t + Math.max(0,nRows-1)*26;
}
function _legendTranslate(el){
  const m=(el.getAttribute('transform')||'').match(/translate\(([-\d.]+)[,\\s]+([-\d.]+)\)/);
  return m ? {x:+m[1],y:+m[2]} : {x:0,y:0};
}
function _setLegendTranslate(el,x,y){
  el.setAttribute('transform',`translate(${x},${y})`);
}
function centerLastLegendRow(divId){
  const gd=document.getElementById(divId);
  if(!gd)return;
  const legend=gd.querySelector('.legend');
  const svg=gd.querySelector('svg.main-svg');
  if(!legend||!svg)return;
  const legendRect=legend.getBoundingClientRect();
  const svgScale=Math.abs(svg.getScreenCTM?.().a||1)||1;
  const targetCenter=(legendRect.left+legendRect.right)/2;
  const parsed=[...gd.querySelectorAll('.legend .traces')].map(el=>{
    const rect=el.getBoundingClientRect();
    if(!rect.width||!rect.height)return null;
    const tr=_legendTranslate(el);
    return {el,rect,x:tr.x,y:tr.y};
  }).filter(Boolean);
  if(parsed.length<2)return;
  parsed.sort((a,b)=>a.rect.top-b.rect.top||a.rect.left-b.rect.left);
  const rows=[];
  let cur=[parsed[0]];
  for(let i=1;i<parsed.length;i++){
    if(Math.abs(parsed[i].rect.top-cur[0].rect.top)<5) cur.push(parsed[i]);
    else{rows.push(cur);cur=[parsed[i]];}
  }
  rows.push(cur);
  const rowCenter=row=>{
    const left=Math.min(...row.map(e=>e.rect.left));
    const right=Math.max(...row.map(e=>e.rect.right));
    return (left+right)/2;
  };
  rows.forEach(row=>{
    const shiftPx=targetCenter-rowCenter(row);
    if(Math.abs(shiftPx)<1)return;
    const shiftSvg=shiftPx/svgScale;
    row.forEach(({el,x,y})=>_setLegendTranslate(el,x+shiftSvg,y));
  });
}
function scheduleTrajectoryLegendCenter(divId){
  requestAnimationFrame(()=>centerLastLegendRow(divId));
  setTimeout(()=>centerLastLegendRow(divId),60);
  setTimeout(()=>centerLastLegendRow(divId),180);
}
function trajectoryLegendY(h,mt){
  mt=mt!=null?mt:TRAJ_MARGIN.t;
  return 1 + (mt - TRAJ_LEGEND_TOP_PX) / Math.max(160,h - mt - TRAJ_MARGIN.b);
}
function trajectoryLegend(h,mt){return {...TRAJ_LEGEND,y:trajectoryLegendY(h,mt)}}
const TRAJ_PENDING_HEIGHTS = {};
let TRAJ_HEIGHT_RAF = null;
let TRAJ_RESIZE_RAF = null;
let TRAJ_RESIZE_TIMER = null;
const HP_HEIGHT_MIN = 360;
const HP_HEIGHT_MAX = 720;

function trajectoryHeight(divId){return clamp(S.trajHeights[divId]||TRAJ_HEIGHT,TRAJ_HEIGHT,TRAJ_HEIGHT*2)}
function setTrajectoryHeight(divId,h){S.trajHeights[divId]=clamp(h,TRAJ_HEIGHT,TRAJ_HEIGHT*2);return S.trajHeights[divId]}
function syncTrajectoryPlotHeight(divId,h=null){
  const gd=document.getElementById(divId);
  if(!gd)return;
  const nh=clamp(h==null?trajectoryHeight(divId):h,TRAJ_HEIGHT,TRAJ_HEIGHT*2);
  gd.style.height=nh+'px';
  gd._m3LastHeight=nh;
}
function resetTrajectoryHeightsToMin(){
  ['ch-tr','ch-dg'].forEach(id=>{
    setTrajectoryHeight(id,TRAJ_HEIGHT);
    syncTrajectoryPlotHeight(id,TRAJ_HEIGHT);
  });
}
function resetTrajectoryWidgetArtifacts(board=document.getElementById('traj-widgets')){
  board?.querySelectorAll('.traj-card').forEach(card=>{
    card.classList.remove('dragging','resizing');
    card.style.transform='';
    card.style.transition='';
  });
  document.body.classList.remove('widget-dragging','traj-resizing');
}
function applyTrajectoryHeight(divId,h){
  const nh=clamp(h,TRAJ_HEIGHT,TRAJ_HEIGHT*2);
  const targets=S.trajWidgetLayout==='side' ? ['ch-tr','ch-dg'] : [divId];
  targets.forEach(id=>{
    setTrajectoryHeight(id,nh);
    TRAJ_PENDING_HEIGHTS[id]=nh;
  });
  if(TRAJ_HEIGHT_RAF)return;
  TRAJ_HEIGHT_RAF=requestAnimationFrame(()=>{
    TRAJ_HEIGHT_RAF=null;
    const entries=Object.entries(TRAJ_PENDING_HEIGHTS);
    Object.keys(TRAJ_PENDING_HEIGHTS).forEach(k=>delete TRAJ_PENDING_HEIGHTS[k]);
    entries.forEach(([id,height])=>{
      const gd=document.getElementById(id);
      if(!gd||!window.Plotly)return;
      gd.style.height=height+'px';
      if(Math.abs((gd._m3LastHeight||0)-height)<1){
        scheduleTrajectoryLegendCenter(id);
        return;
      }
      gd._m3LastHeight=height;
      const mt=trajMarginTop(id);
      Plotly.relayout(gd,{height,'margin.t':mt,'legend.y':trajectoryLegendY(height,mt)})
        .then(()=>scheduleTrajectoryLegendCenter(id));
      if(gd._m3FullLayoutForEmpty){
        gd._m3FullLayoutForEmpty.height=height;
        gd._m3FullLayoutForEmpty.legend=trajectoryLegend(height,mt);
        gd._m3FullLayoutForEmpty.margin={...TRAJ_MARGIN,t:mt};
      }
    });
  });
}
function initTrajectoryResize(){
  document.querySelectorAll('.traj-resize-handle').forEach(handle=>{
    if(handle._m3ResizeReady)return;
    handle._m3ResizeReady=true;
    handle.addEventListener('click',ev=>{
      if(handle._m3SuppressClick){
        handle._m3SuppressClick=false;
        ev.preventDefault();
        ev.stopPropagation();
        return;
      }
      const divId=handle.dataset.plot;
      const cur=trajectoryHeight(divId);
      const target=cur < TRAJ_HEIGHT*1.5 ? TRAJ_HEIGHT*2 : TRAJ_HEIGHT;
      applyTrajectoryHeight(divId,target);
      ev.preventDefault();
      ev.stopPropagation();
    });
    handle.addEventListener('pointerdown',ev=>{
      const divId=handle.dataset.plot;
      const card=handle.closest('.traj-card');
      const scroller=document.querySelector('.ct');
      const startY=ev.clientY;
      const startH=trajectoryHeight(divId);
      const startScroll=scroller?scroller.scrollTop:0;
      let lastY=ev.clientY;
      let moved=false;
      let raf=null;
      document.body.classList.add('traj-resizing');
      if(card)card.classList.add('resizing');
      handle.setPointerCapture?.(ev.pointerId);
      const update=()=>{
        const scrollDelta=scroller?(scroller.scrollTop-startScroll):0;
        applyTrajectoryHeight(divId,startH+(lastY-startY)+scrollDelta);
      };
      const autoScroll=()=>{
        if(!scroller||!document.body.classList.contains('traj-resizing'))return;
        const r=scroller.getBoundingClientRect();
        let dy=0;
        if(lastY>r.bottom-44)dy=Math.min(22,(lastY-(r.bottom-44))*.38+4);
        else if(lastY<r.top+44)dy=-Math.min(22,((r.top+44)-lastY)*.38+4);
        if(dy){scroller.scrollTop+=dy;update();raf=requestAnimationFrame(autoScroll);}
        else raf=null;
      };
      const move=e=>{
        lastY=e.clientY;
        if(Math.abs(lastY-startY)>3)moved=true;
        update();
        if(!raf)raf=requestAnimationFrame(autoScroll);
      };
      const up=()=>{
        document.body.classList.remove('traj-resizing');
        if(card)card.classList.remove('resizing');
        if(raf)cancelAnimationFrame(raf);
        handle._m3SuppressClick=moved;
        if(moved)setTimeout(()=>{handle._m3SuppressClick=false;},120);
        window.removeEventListener('pointermove',move);
        window.removeEventListener('pointerup',up);
        window.removeEventListener('pointercancel',up);
      };
      window.addEventListener('pointermove',move);
      window.addEventListener('pointerup',up,{once:true});
      window.addEventListener('pointercancel',up,{once:true});
      ev.preventDefault();
    });
  });
}

function hpHeight(){return clamp(S.hpHeight||HP_HEIGHT_MIN,HP_HEIGHT_MIN,HP_HEIGHT_MAX)}
function applyHpHeight(h=null){
  if(h!=null)S.hpHeight=clamp(h,HP_HEIGHT_MIN,HP_HEIGHT_MAX);
  const el=document.getElementById('hp-sel');
  if(el)el.style.maxHeight=hpHeight()+'px';
}
function initHpResize(){
  const handle=document.querySelector('.hp-resize-handle');
  if(!handle||handle._m3HpResizeReady)return;
  handle._m3HpResizeReady=true;
  handle.addEventListener('click',ev=>{
    if(handle._m3SuppressClick){
      handle._m3SuppressClick=false;
      ev.preventDefault();
      ev.stopPropagation();
      return;
    }
    applyHpHeight(hpHeight()<HP_HEIGHT_MIN*1.5?HP_HEIGHT_MAX:HP_HEIGHT_MIN);
    ev.preventDefault();
    ev.stopPropagation();
  });
  handle.addEventListener('pointerdown',ev=>{
    const card=handle.closest('.hp-card');
    const scroller=document.querySelector('.ct');
    const startY=ev.clientY;
    const startH=hpHeight();
    const startScroll=scroller?scroller.scrollTop:0;
    let lastY=ev.clientY;
    let moved=false;
    let raf=null;
    document.body.classList.add('hp-resizing');
    if(card)card.classList.add('resizing');
    handle.setPointerCapture?.(ev.pointerId);
    const update=()=>{
      const scrollDelta=scroller?(scroller.scrollTop-startScroll):0;
      applyHpHeight(startH+(lastY-startY)+scrollDelta);
    };
    const autoScroll=()=>{
      if(!scroller||!document.body.classList.contains('hp-resizing'))return;
      const r=scroller.getBoundingClientRect();
      let dy=0;
      if(lastY>r.bottom-44)dy=Math.min(22,(lastY-(r.bottom-44))*.38+4);
      else if(lastY<r.top+44)dy=-Math.min(22,((r.top+44)-lastY)*.38+4);
      if(dy){scroller.scrollTop+=dy;update();raf=requestAnimationFrame(autoScroll);}
      else raf=null;
    };
    const move=e=>{
      lastY=e.clientY;
      if(Math.abs(lastY-startY)>3)moved=true;
      update();
      if(!raf)raf=requestAnimationFrame(autoScroll);
    };
    const up=()=>{
      document.body.classList.remove('hp-resizing');
      if(card)card.classList.remove('resizing');
      if(raf)cancelAnimationFrame(raf);
      handle._m3SuppressClick=moved;
      if(moved)setTimeout(()=>{handle._m3SuppressClick=false;},120);
      window.removeEventListener('pointermove',move);
      window.removeEventListener('pointerup',up);
      window.removeEventListener('pointercancel',up);
    };
    window.addEventListener('pointermove',move);
    window.addEventListener('pointerup',up,{once:true});
    window.addEventListener('pointercancel',up,{once:true});
    ev.preventDefault();
  });
}


function ptBase(){
  const d=dk();
  return{
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor: d?'rgba(255,255,255,0.015)':'rgba(0,0,0,0.012)',
    font:{color:d?'#e0e4ff':'#0b0b2e',family:'Inter,system-ui,sans-serif',size:14.5},
    hoverlabel:{
      borderradius:5,
      bgcolor:d?'rgba(19,19,46,.96)':'rgba(255,255,255,.97)',
      bordercolor:d?'rgba(168,85,247,.45)':'rgba(109,40,217,.28)',
      font:{color:d?'#e0e4ff':'#0b0b2e',family:'Inter,system-ui,sans-serif',size:13.5},
    },
    xaxis:{gridcolor:d?'rgba(255,255,255,.06)':'rgba(0,0,0,.06)',zeroline:false,linecolor:d?'rgba(255,255,255,.1)':'rgba(0,0,0,.1)'},
    yaxis:{gridcolor:d?'rgba(255,255,255,.06)':'rgba(0,0,0,.06)',zeroline:false,linecolor:d?'rgba(255,255,255,.1)':'rgba(0,0,0,.1)'},
  };
}

function emptyPlot(divId,msg,height=TRAJ_EMPTY_HEIGHT){
  const th=ptBase(), d=dk();
  const gd=document.getElementById(divId);
  if(gd){
    gd.style.height=height+'px';
    gd._m3LastHeight=height;
  }
  Plotly.react(divId,[],{...th,height,margin:{t:14,b:14,l:20,r:20},
    paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    xaxis:{visible:false},yaxis:{visible:false},
    annotations:[{x:.5,y:.5,xref:'paper',yref:'paper',showarrow:false,
      text:msg,font:{size:15,color:d?'rgba(224,228,255,.62)':'rgba(11,11,46,.58)'},
      align:'center'}],
  },CFGs);
}

// ── NAV ───────────────────────────────────────────────────────────────────
const STITLE = {
  global:'Global Results', metrics:'Method Metrics', conds:'Condition Analysis',
  summary:'Cross-Cohort Summary', 'sc-perf':'Performance Results', 'sc-pairwise':'Pairwise Tests',
};
const SLABEL = {progressive:'Progressive Missingness Study', static:'Static-Cohort'};

function rebuildCohorts(study){
  const dcs=['#00d4ff','#a855f7','#ff2d78'];
  const cbEl=document.getElementById('cbs');
  cbEl.innerHTML='';
  // Always show all progressive cohorts — dataset buttons never change between study modes
  const dks=new Set(Object.keys(curMod()));
  const cohorts=Object.entries(COHORT_ENDPOINTS)
    .filter(([c,eps])=>eps.some(e=>dks.has(e.ds_key)));
  if(!cohorts.length){
    cbEl.innerHTML='<div style="font-size:10px;color:var(--t3);padding:8px 10px">No datasets found.</div>';
    return;
  }
  if(!dks.has(S.ds)){
    const firstDk=cohorts[0][1].find(e=>dks.has(e.ds_key))?.ds_key||null;
    S.ds=firstDk;
  }
  const curCohort=DS_COHORT[S.ds]||cohorts[0][0];
  cohorts.forEach(([cohort,eps],i)=>{
    const firstDk=eps.find(e=>dks.has(e.ds_key))?.ds_key;
    const meta=firstDk?(curMod()[firstDk]||{}).meta||{}:{};
    const b=document.createElement('button');
    b.className='cb'+(cohort===curCohort?' on':''); b.dataset.cohort=cohort;
    b.onclick=()=>switchCohort(cohort);
    b.innerHTML=`<span class="cdot" style="background:${dcs[i%3]}"></span><span>${meta.name||cohort}</span>`;
    cbEl.appendChild(b);
  });
  updateEndpointDropdown(curCohort);
}

function updateStaticNavVisibility(){
  const show=Object.keys(STATIC_DS_DATA[S.ds]||{}).length>0;
  const d=show?'':'none';
  document.getElementById('nav-static-lbl').style.display=d;
  document.querySelectorAll('.ni[data-study="static"]').forEach(el=>el.style.display=d);
  return show;
}

function switchCohort(cohort){
  const dks=new Set(Object.keys(curMod()));
  const eps=(COHORT_ENDPOINTS[cohort]||[]).filter(e=>dks.has(e.ds_key));
  if(!eps.length) return;
  // Keep current endpoint if still available for this cohort, else pick first
  S.ds=(eps.find(e=>e.ds_key===S.ds)||eps[0]).ds_key;
  document.querySelectorAll('.cb').forEach(e=>e.classList.toggle('on',e.dataset.cohort===cohort));
  updateEndpointDropdown(cohort);
  const hasStatic=updateStaticNavVisibility();
  if(S.study==='static'&&!hasStatic){
    nav('global','progressive');
    return;
  }
  updateDdLabels(); updateMeta(); render();
}

function updateEndpointDropdown(cohort){
  const menu=document.getElementById('dd-ep-menu');
  if(!menu) return;
  menu.innerHTML='';
  if(S.study==='static'){
    const meta=(STATIC_DS_DATA[S.ds]||{}).meta||{};
    if(meta.endpoint){
      const it=document.createElement('div');
      it.className='dd-item on'; it.textContent=meta.endpoint;
      menu.appendChild(it);
    }
    const epEl=document.getElementById('dd-ep-val');
    if(epEl) epEl.textContent=meta.endpoint||'–';
    return;
  }
  const dks=new Set(Object.keys(curMod()));
  const eps=(COHORT_ENDPOINTS[cohort]||[]).filter(e=>dks.has(e.ds_key));
  eps.forEach(({ds_key,label})=>{
    const it=document.createElement('div');
    it.className='dd-item'+(ds_key===S.ds?' on':'');
    it.dataset.v=ds_key; it.textContent=label;
    it.onclick=()=>{
      S.ds=ds_key;
      updateEndpointDropdown(cohort);
      updateDdLabels(); updateMeta(); render();
      toggleDd('ep');
    };
    menu.appendChild(it);
  });
}

function nav(s, study='progressive'){
  const studyChanged = S.study !== study;
  S.sec=s; S.study=study;
  document.querySelectorAll('.ni').forEach(e=>
    e.classList.toggle('on', e.dataset.s===s && e.dataset.study===study));
  document.querySelectorAll('.sec').forEach(e=>e.classList.remove('on'));
  document.getElementById('ht').textContent=STITLE[s]||s;
  if(studyChanged) rebuildCohorts(study);
  updateStaticNavVisibility();
  const hasData = study==='static'
    ? Object.keys(STATIC_DS_DATA[S.ds]||{}).length > 0
    : Object.keys(curMod()).length > 0;
  if(!hasData){
    document.getElementById('s-noresults').classList.add('on');
  } else {
    document.getElementById(`s-${s}`).classList.add('on');
    render();
  }
}

function switchDs(k){
  S.ds=k;
  document.querySelectorAll('.cb').forEach(e=>e.classList.toggle('on',e.dataset.k===k));
  updateEndpointDropdown();
  updateStaticNavVisibility();
  updateDdLabels(); updateMeta(); render();
}

function toggleDd(id){
  const menu=document.getElementById(`dd-${id}-menu`);
  const wasOpen=menu.classList.contains('open');
  document.querySelectorAll('.dd-menu').forEach(m=>m.classList.remove('open'));
  closeHpMenu();closeHpTpMenu();
  if(!wasOpen){
    menu.classList.add('open');
    clearPlotHovers();
  }
}
document.addEventListener('click',e=>{
  if(!e.target.closest('.ddrop')) document.querySelectorAll('.dd-menu').forEach(m=>m.classList.remove('open'));
  if(!e.target.closest('.method-filter')&&!e.target.closest('.mf-menu')) document.querySelectorAll('.mf-menu').forEach(m=>m.classList.remove('open'));
  if(!e.target.closest('.hp-method-filter')&&!e.target.closest('.hp-menu')){closeHpMenu();}
  if(!e.target.closest('#hp-trainprop-filter')&&!e.target.closest('.hp-tp-menu')){closeHpTpMenu();}
});
window.addEventListener('resize',positionMethodMenu);
document.querySelector('.ct')?.addEventListener('scroll',positionMethodMenu);


function trajectoryPanelVis(chart){
  if(S.trajWidgetLayout==='side') return {train:S.trajCompactPanel==='train',test:S.trajCompactPanel==='test',bft:S.trajCompactPanel==='bft'};
  return S[chart+'Vis'];
}
function updateTrajectoryPanelButtons(){
  ['traj','deg'].forEach(chart=>{
    ['train','test','bft'].forEach(panel=>{
      const on=S.trajWidgetLayout==='side' ? S.trajCompactPanel===panel : !!S[chart+'Vis'][panel];
      document.querySelectorAll(`.pvbtn[data-chart="${chart}"][data-p="${panel}"]`).forEach(e=>e.classList.toggle('on',on));
    });
    document.querySelectorAll(`.pvbtn[data-chart="${chart}"][data-ci="1"]`).forEach(e=>e.classList.toggle('on',S.ciVis[chart]));
  });
}
function syncTrajectoryWidgets(){
  const board=document.getElementById('traj-widgets');
  if(!board)return;
  board.classList.toggle('traj-side',S.trajWidgetLayout==='side');
  board.classList.toggle('traj-stack',S.trajWidgetLayout!=='side');
  S.trajWidgetOrder.forEach(w=>{
    const slot=board.querySelector(`.traj-slot[data-widget="${w}"]`);
    if(slot)board.appendChild(slot);
  });
  updateTrajectoryPanelButtons();
}

function resizeTrajectoryPlots(){
  if(TRAJ_RESIZE_RAF)return;
  TRAJ_RESIZE_RAF=requestAnimationFrame(()=>{
    TRAJ_RESIZE_RAF=null;
    ['ch-tr','ch-dg'].forEach(id=>{
      const gd=document.getElementById(id);
      if(!gd||!window.Plotly)return;
      gd.style.width='100%';
      gd.style.maxWidth='100%';
      gd.style.minWidth='0';
      const card=gd.closest('.traj-card');
      const w=Math.max(260,Math.floor((card?.querySelector('.cp')||card||gd).clientWidth));
      if(Math.abs((gd._m3LastWidth||0)-w)>2){
        gd._m3LastWidth=w;
        Plotly.relayout(gd,{width:w}).then(()=>scheduleTrajectoryLegendCenter(id));
      }else{
        Plotly.Plots.resize(gd);
        scheduleTrajectoryLegendCenter(id);
      }
    });
  });
}
function scheduleTrajectoryResize(){
  resizeTrajectoryPlots();
  if(TRAJ_RESIZE_TIMER)clearTimeout(TRAJ_RESIZE_TIMER);
  TRAJ_RESIZE_TIMER=setTimeout(resizeTrajectoryPlots,260);
}
function initTrajectoryResizeObserver(){
  if(window._m3TrajectoryResizeObserverReady)return;
  window._m3TrajectoryResizeObserverReady=true;
  if(!window.ResizeObserver)return;
  const ro=new ResizeObserver(()=>scheduleTrajectoryResize());
  document.querySelectorAll('.traj-card,.traj-slot,#traj-widgets').forEach(el=>ro.observe(el));
  document.getElementById('traj-widgets')?.addEventListener('transitionend',scheduleTrajectoryResize);
}
function setTrajectoryWidgetLayout(layout,order=null){
  S.trajWidgetLayout=layout;
  if(order)S.trajWidgetOrder=order;
  resetTrajectoryWidgetArtifacts();
  if(layout==='side') resetTrajectoryHeightsToMin();
  else ['ch-tr','ch-dg'].forEach(id=>syncTrajectoryPlotHeight(id,trajectoryHeight(id)));
  syncTrajectoryWidgets();
  rTraj();
  rDeg();
  scheduleTrajectoryResize();
}

function ensureTrajectoryDropPreview(board){
  let p=board.querySelector('.traj-drop-preview');
  if(!p){p=document.createElement('div');p.className='traj-drop-preview';board.appendChild(p);}
  return p;
}
function hideTrajectoryDropPreview(board){
  const p=board?.querySelector('.traj-drop-preview');
  if(p)p.classList.remove('on');
  board?.classList.remove('widget-previewing');
  board?.querySelectorAll('.traj-slot').forEach(s=>s.classList.remove('drag-source'));
}
function trajectoryCardHeightForPlot(card,plotHeight){
  if(!card)return plotHeight;
  const plotId=card.dataset.plot;
  const plot=document.getElementById(plotId);
  const currentPlotH=plot?.getBoundingClientRect().height||trajectoryHeight(plotId);
  const currentCardH=card.getBoundingClientRect().height;
  const chrome=Math.max(0,currentCardH-currentPlotH);
  return chrome+plotHeight;
}
function showTrajectoryDropPreview(board,card,widget,mode,dx,dy){
  const preview=ensureTrajectoryDropPreview(board);
  const gap=parseFloat(getComputedStyle(board).gap)||10;
  const slot=card.closest('.traj-slot');
  const otherWidget=widget==='perf'?'deg':'perf';
  const otherCard=board.querySelector(`.traj-card[data-widget="${otherWidget}"]`);
  const draggedH=trajectoryCardHeightForPlot(card,trajectoryHeight(card.dataset.plot));
  const otherH=trajectoryCardHeightForPlot(otherCard,trajectoryHeight(otherCard?.dataset.plot||card.dataset.plot));
  const minSideH=Math.max(
    trajectoryCardHeightForPlot(card,TRAJ_HEIGHT),
    trajectoryCardHeightForPlot(otherCard,TRAJ_HEIGHT)
  );
  const bw=board.clientWidth;
  const sideW=(bw-gap)/2;
  let left=0,top=0,width=bw,height=draggedH;
  if(mode==='side'){
    width=sideW;height=minSideH;top=0;left=dx<0?0:sideW+gap;
  }else{
    width=bw;height=draggedH;left=0;top=dy<0?0:otherH+gap;
  }
  preview.style.left=`${left}px`;
  preview.style.top=`${top}px`;
  preview.style.width=`${width}px`;
  preview.style.height=`${height}px`;
  preview.classList.add('on');
  board.classList.add('widget-previewing');
  board.querySelectorAll('.traj-slot').forEach(s=>s.classList.toggle('drag-source',s===slot));
}
function initTrajectoryWidgetDrag(){
  const board=document.getElementById('traj-widgets');
  if(!board||board._m3DragReady)return;
  board._m3DragReady=true;
  board.querySelectorAll('.traj-drag-handle').forEach(handle=>{
    handle.addEventListener('pointerdown',ev=>{
      if(ev.target.closest('.pv-btns,.pvbtn,.ci-toggle'))return;
      const card=handle.closest('.traj-card');
      const widget=card?.dataset.widget;
      if(!card||!widget)return;
      const startX=ev.clientX,startY=ev.clientY;
      card.classList.add('dragging');
      document.body.classList.add('widget-dragging');
      handle.setPointerCapture?.(ev.pointerId);
      let lastX=startX,lastY=startY;
      let lastGhostSide=null;
      const move=e=>{
        lastX=e.clientX;lastY=e.clientY;
        const dx=lastX-startX,dy=lastY-startY;
        const side=Math.abs(dx)>70&&Math.abs(dx)>Math.abs(dy);
        const stack=Math.abs(dy)>55&&!side;
        // 3 card positions: zoom only · ghost-left · ghost-right
        const ghostSide=side?(dx>0?'right':'left'):null;
        const tx=ghostSide==='right'?14:ghostSide==='left'?-14:0;
        if(ghostSide!==lastGhostSide){lastGhostSide=ghostSide;card.style.transition='transform .14s cubic-bezier(.22,1,.36,1)';}
        card.style.transform=`translate3d(${tx}px,0,0) scale(1.012)`;
        if(side) showTrajectoryDropPreview(board,card,widget,'side',dx,dy);
        else if(stack) showTrajectoryDropPreview(board,card,widget,'stack',dx,dy);
        else hideTrajectoryDropPreview(board);
      };
      const up=()=>{
        const dx=lastX-startX,dy=lastY-startY;
        card.classList.remove('dragging');
        card.style.transform='';
        card.style.transition='';
        hideTrajectoryDropPreview(board);
        document.body.classList.remove('widget-dragging');
        window.removeEventListener('pointermove',move);
        window.removeEventListener('pointerup',up);
        window.removeEventListener('pointercancel',up);
        const other=widget==='perf'?'deg':'perf';
        if(Math.abs(dx)>70&&Math.abs(dx)>Math.abs(dy)){
          setTrajectoryWidgetLayout('side',dx<0?[widget,other]:[other,widget]);
        }else if(Math.abs(dy)>55){
          setTrajectoryWidgetLayout('stack',dy<0?[widget,other]:[other,widget]);
        }
      };
      window.addEventListener('pointermove',move);
      window.addEventListener('pointerup',up,{once:true});
      window.addEventListener('pointercancel',up,{once:true});
      ev.preventDefault();
    });
  });
}
function togglePanel(chart,panel){
  if(S.trajWidgetLayout==='side'){
    S.trajCompactPanel=panel;
    updateTrajectoryPanelButtons();
    rTraj();
    rDeg();
    return;
  }
  S[chart+'Vis'][panel]=!S[chart+'Vis'][panel];
  updateTrajectoryPanelButtons();
  chart==='traj'?rTraj():rDeg();
}
function toggleCI(chart){
  S.ciVis[chart]=!S.ciVis[chart];
  document.querySelectorAll(`.pvbtn[data-chart="${chart}"][data-ci="1"]`)
    .forEach(e=>e.classList.toggle('on',S.ciVis[chart]));
  chart==='traj'?rTraj():rDeg();
}

function renderMethodFilter(){
  const el=document.getElementById('method-filter');
  if(!el)return;
  const methods=allMethods();
  methods.forEach(m=>{if(S.methodVis[m]===undefined)S.methodVis[m]=true;});
  Object.keys(S.methodVis).forEach(m=>{if(!methods.includes(m))delete S.methodVis[m];});
  if(!methods.length){el.innerHTML='';return;}
  const nOn=methods.filter(m=>S.methodVis[m]!==false).length;
  const label=nOn===methods.length?'All':`${nOn}/${methods.length}`;
  const items=methods.map(m=>{
    const c=MC[m]||'#a855f7';
    const on=S.methodVis[m]!==false;
    return `<div class="mf-item ${on?'on':'off'}" data-m="${m}"
      style="--mc:${c};--mcb:${rgba(c,.16)}"
      onclick="toggleMethod('${m.replace(/'/g,"\\'")}',event)">
      <span class="mf-dot"></span><span class="mf-name">${MD[m]||m}</span><span class="mf-check">✓</span>
    </div>`;
  }).join('');
  el.className='method-filter';
  el.innerHTML=`<div class="mf-btn" onclick="toggleMethodMenu(event)">
      <span class="dd-lbl">Methods</span>
      <div class="dd-sep"></div>
      <span class="dd-val">${label}</span>
      <svg class="mf-caret" viewBox="0 0 10 6" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M1 1l4 4 4-4"/></svg>
    </div>
    <div class="mf-menu" id="method-menu">${items}</div>`;
}

function positionMethodMenu(){
  const menu=document.getElementById('method-menu');
  const btn=document.querySelector('#method-filter .mf-btn');
  if(!menu||!btn||!menu.classList.contains('open'))return;
  const r=btn.getBoundingClientRect();
  const sc=dashboardScale();
  const menuW=Math.max(menu.offsetWidth||190,190);
  const menuH=Math.max(menu.offsetHeight||0,120);
  const margin=8/sc;
  const center=(r.left+r.width/2)/sc;
  const left=Math.min(Math.max(margin,center-menuW/2),window.innerWidth/sc-menuW-margin);
  const top=Math.min(r.bottom/sc+7/sc,window.innerHeight/sc-menuH-margin);
  menu.style.left=`${left}px`;
  menu.style.top=`${top}px`;
}

function toggleMethodMenu(ev){
  if(ev) ev.stopPropagation();
  const menu=document.getElementById('method-menu');
  if(!menu)return;
  const wasOpen=menu.classList.contains('open');
  document.querySelectorAll('.dd-menu').forEach(m=>m.classList.remove('open'));
  closeHpMenu();closeHpTpMenu();
  if(wasOpen){
    menu.classList.remove('open');
  }else{
    menu.classList.add('open');
    positionMethodMenu();
    if(window.Plotly) Plotly.Fx.unhover('ch-hm');
  }
}

function toggleMethod(m,ev=null){
  if(ev) ev.stopPropagation();
  S.methodVis[m]=!(S.methodVis[m]!==false);
  const menuWasOpen=document.getElementById('method-menu')?.classList.contains('open');
  renderMethodFilter();
  if(menuWasOpen){
    document.getElementById('method-menu')?.classList.add('open');
    positionMethodMenu();
  }
  rHeatmaps();
  rTraj();
  rDeg();
  rHpSelection();
}

function updateDdLabels(){
  const repEl=document.getElementById('dd-rep-val');
  if(repEl) repEl.textContent=(AVAIL_MODES.find(m=>m.key===S.mode)||{}).label||S.mode;
  const modEl=document.getElementById('dd-mod-val');
  if(modEl) modEl.textContent=MODALITY_LABELS[S.modality]||S.modality;
  const epEl=document.getElementById('dd-ep-val');
  if(epEl){
    if(S.study==='static'){
      epEl.textContent=(STATIC_DS_DATA[S.ds]||{}).meta?.endpoint||'–';
    } else {
      const cohort=DS_COHORT[S.ds];
      const ep=cohort?(COHORT_ENDPOINTS[cohort]||[]).find(e=>e.ds_key===S.ds):null;
      epEl.textContent=ep?ep.label:(S.ds||'–');
    }
  }
  document.querySelectorAll('#dd-rep-menu .dd-item').forEach(e=>e.classList.toggle('on',e.dataset.v===S.mode));
  document.querySelectorAll('#dd-mod-menu .dd-item').forEach(e=>e.classList.toggle('on',e.dataset.v===S.modality));
  document.querySelectorAll('#dd-ep-menu .dd-item').forEach(e=>e.classList.toggle('on',e.dataset.v===S.ds));
}

function switchMode(m){
  S.mode=m;
  const modalities=Object.keys(MODES_DATA[m]||{});
  if(!modalities.includes(S.modality)) S.modality=modalities[0]||S.modality;
  rebuildCohorts('progressive');  // handles S.ds + endpoint dropdown
  updateDdLabels(); updateMeta(); render();
}

function switchModality(m){
  S.modality=m;
  rebuildCohorts('progressive');  // handles S.ds + endpoint dropdown
  updateDdLabels(); updateMeta(); render();
}

function toggleTheme(){
  S.theme=S.theme==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme',S.theme);
  document.getElementById('ti').textContent=dk()?'🌙':'☀️';
  document.getElementById('tl').textContent=dk()?'Dark':'Light';
  render();
}

function updateMeta(){
  const m = S.study==='static' ? SD().meta : D().meta;
  if(!m)return;
  document.getElementById('hm').textContent=`${m.name||m.scope} · ${m.n} patients · ${m.modalities} modalities · ${m.endpoint}`;
}

function friedmanSig(f){
  if(!f) return null;
  const v=f.significant_p0_05;
  if(typeof v==='boolean') return v;
  if(typeof v==='number') return v!==0;
  if(typeof v==='string'){
    const s=v.trim().toLowerCase();
    if(['true','1','yes','y','significant'].includes(s)) return true;
    if(['false','0','no','n','not significant'].includes(s)) return false;
  }
  const pv=Number(f.p_value);
  return Number.isFinite(pv) ? pv<0.05 : null;
}

function render(){
  switch(S.sec){
    case'global':      rGlobal();       break;
    case'metrics':     rMetrics();      break;
    case'conds':       rConds();        break;
    case'summary':     rSummary();      break;
    case'sc-perf':     rStaticPerf();   break;
    case'sc-pairwise': rStaticPairwise();break;
  }
}

// ── STATIC COHORT ─────────────────────────────────────────────────────────
function rStaticPerf(){
  const d=SD(); const meta=d.meta||{};
  const ml=metricLabel(d);
  // stat cards
  document.getElementById('sc-dsname').textContent=meta.name||'–';
  document.getElementById('sc-nm').textContent=meta.modalities||'–';
  document.getElementById('sc-np').textContent=meta.n||'–';
  const distLbl=document.getElementById('sc-metric-dist-lbl');
  if(distLbl) distLbl.textContent=ml;
  const f=(d.friedman||[])[0];
  const pv=f?.p_value; const sig=friedmanSig(f);
  document.getElementById('sc-fp').textContent=pv!=null?pv.toFixed(4):'–';
  document.getElementById('sc-fs').textContent=sig!=null?(sig?'Significant (p<0.05)':'Not significant'):'–';
  document.getElementById('sc-nrep').textContent=meta.n_replicates??'–';
  // figure
  const img=document.getElementById('sc-fig-perf');
  if(d.fig_perf){img.src=d.fig_perf;img.style.display='block';}
  else{img.style.display='none';img.closest('.cp').innerHTML='<div class="es"><p>Figure not generated yet — run the fixed-dataset analysis notebook first.</p></div>';}
  updateMeta();
}

function rStaticPairwise(){
  const d=SD();
  const img=document.getElementById('sc-fig-pairwise');
  if(d.fig_pairwise){img.src=d.fig_pairwise;img.style.display='block';}
  else{img.style.display='none';img.closest('.cp').innerHTML='<div class="es"><p>Figure not generated yet — run the fixed-dataset analysis notebook first.</p></div>';}
  updateMeta();
}

// ── GLOBAL ────────────────────────────────────────────────────────────────
function rGlobal(){rCards();renderMethodFilter();syncTrajectoryWidgets();rHeatmaps();rTraj();rDeg();rHpSelection();}

function rCards(){
  const d=D(); if(!d.meta)return;
  const f=(d.friedman||[])[0];
  const ml=metricLabel(d);
  const hmMetricLbl=document.getElementById('hm-metric-lbl');
  if(hmMetricLbl) hmMetricLbl.textContent=ml;

  // Card 1 — Dataset
  document.getElementById('s-dsname').textContent=d.meta.name||'–';
  document.getElementById('s-nm').textContent=d.meta.modalities||'–';
  document.getElementById('s-np').textContent=d.meta.n||'–';

  // Card 2 — Task / Endpoint
  document.getElementById('s-ep').textContent=d.meta.endpoint||'–';
  document.getElementById('s-tasktype').textContent=d.meta.task_type||'–';

  // Card 3 — Methods
  const ma=d.mean_auc||[];
  const tps=new Set(ma.map(r=>r.train_missing_prop));
  const mps=new Set(ma.map(r=>r.test_missing_prop));
  const meths=new Set(ma.map(r=>r.model_name));
  document.getElementById('s-nme').textContent=meths.size||'–';
  document.getElementById('s-nc').textContent=`${tps.size*mps.size||0} missingness conditions`;

  // Card 4 — Friedman
  if(f){
    const pv=f.p_value;
    const sig=friedmanSig(f);
    document.getElementById('s-fp').textContent=pv<.001?'<0.001':pv.toFixed(4);
    document.getElementById('s-fs').textContent=sig?'✓ Significant (p<0.05)':'Not significant';
  }

  // Card 5 — Replicates
  document.getElementById('s-rtype').textContent=d.meta.rep_type||'–';
  const nrep=d.meta.n_replicates;
  document.getElementById('s-nrep').textContent=nrep?`${nrep} replicates per condition`:'– replicates per condition';
}

function heatCS(){
  return dk()?[[0,'#1a003a'],[.25,'#4a0080'],[.5,'#7c3aed'],[.75,'#0099bb'],[1,'#00d4ff']]
             :[[0,'#fff0f8'],[.3,'#dbb0ff'],[.6,'#7c3aed'],[.8,'#0088aa'],[1,'#006699']];
}

function rHeatmaps(){
  const ma=D().mean_auc||[];
  if(!ma.length){Plotly.purge('ch-hm');return;}
  const ml=metricLabel();
  const methods=activeMethods();
  if(!methods.length){emptyPlot('ch-hm','Select at least one method to visualise it.');return;}
  const trainPs=[...new Set(ma.map(r=>r.train_missing_prop))].sort((a,b)=>a-b);
  const testPs=[...new Set(ma.map(r=>r.test_missing_prop))].sort((a,b)=>a-b);
  const nC=4, nR=Math.ceil(methods.length/nC);
  const nTr=trainPs.length, nTe=testPs.length;
  const all=ma.map(r=>r.mean_auc).filter(v=>v!=null);
  const zmin=Math.min(...all)-.005, zmax=Math.max(...all)+.005;
  const cs=heatCS(), th=ptBase(), d=dk();
  const traces=[], anns=[];
  const tc=d?'rgba(255,255,255,0.93)':'rgba(0,0,0,0.88)';

  // Use numeric coords so annotations land precisely on cells
  const xi=testPs.map((_,i)=>i);   // 0,1,2,...
  const yi=trainPs.map((_,i)=>i);
  const xtick=testPs.map(v=>`${(v*100).toFixed(0)}%`);
  const ytick=trainPs.map(v=>`${(v*100).toFixed(0)}%`);

  methods.forEach((m,idx)=>{
    const axN=idx===0?'':idx+1;
    const z=trainPs.map(tp=>testPs.map(mp=>{
      const r=ma.find(x=>x.model_name===m&&Math.abs(x.train_missing_prop-tp)<.001&&Math.abs(x.test_missing_prop-mp)<.001);
      return r?r.mean_auc:null;
    }));
    const cust=trainPs.map((tp,ri)=>testPs.map((mp,ci)=>{
      const r=ma.find(x=>x.model_name===m&&Math.abs(x.train_missing_prop-tp)<.001&&Math.abs(x.test_missing_prop-mp)<.001);
      const ci95=(r&&r.std_auc!=null&&r.n_replicates>1)?1.96*r.std_auc/Math.sqrt(r.n_replicates):null;
      return [ytick[ri], xtick[ci], ci95!=null?ci95.toFixed(3):''];
    }));
    traces.push({
      type:'heatmap',z,x:xi,y:yi,
      colorscale:cs,zmin,zmax,
      showscale:idx===methods.length-1,
      colorbar:{thickness:10,len:1,y:0.5,yanchor:'middle',tickfont:{size:11.5},title:{text:ml,font:{size:12}},x:1.01},
      xaxis:`x${axN}`,yaxis:`y${axN}`,
      customdata:cust,
      hoverinfo:'skip',
    });
    // Method title: same title-to-plot spacing as trajectory panels.
    anns.push({x:.5,y:1.11,xref:`x${axN} domain`,yref:`y${axN} domain`,
               text:`<b>${MD[m]||m}</b>`,yanchor:'middle',
               font:{color:MC[m]||'#aaa',size:14.5},showarrow:false,xanchor:'center'});
    // Per-cell text: bold mean + <br> CI
    trainPs.forEach((tp,ri)=>testPs.forEach((mp,ci)=>{
      const r=ma.find(x=>x.model_name===m&&Math.abs(x.train_missing_prop-tp)<.001&&Math.abs(x.test_missing_prop-mp)<.001);
      if(!r||r.mean_auc==null) return;
      const ci95=(r.std_auc!=null&&r.n_replicates>1)
        ?1.96*r.std_auc/Math.sqrt(r.n_replicates):null;
      const label=ci95!=null
        ?`<b>${r.mean_auc.toFixed(3)}</b><br>±${ci95.toFixed(3)}`
        :`<b>${r.mean_auc.toFixed(3)}</b>`;
      anns.push({x:ci,y:ri,xref:`x${axN}`,yref:`y${axN}`,
                 text:label,font:{size:S.hmTextSize,color:tc},
                 showarrow:false,align:'center'});
    }));
  });

  const cellH=46;
  const marginT=68, marginB=72;
  // Fixed cell scale: one-row plot area is the reference, extra rows add space instead of rescaling cells.
  const rowPlotH=nTr*cellH+40;
  const rowGapPx=86;
  const plotAreaH=nR*rowPlotH+(nR-1)*rowGapPx;
  const hmHeight=marginT+marginB+plotAreaH;
  const xTitleStandoff=14;
  const yTitleStandoff=6;

  const layout={...th,dragmode:false,
    grid:{rows:nR,columns:nC,pattern:'independent',ygap:0,xgap:.13},
    height:hmHeight,margin:{t:marginT,b:marginB,l:68,r:28},
    annotations:anns,showlegend:false,
  };
  const xDomains=axisDomains(nC,.045);
  const rowDomainH=rowPlotH/plotAreaH;
  const rowGap=rowGapPx/plotAreaH;
  let colorbarBottom=1;
  let colorbarTop=0;
  methods.forEach((_,idx)=>{
    const axN=idx===0?'':idx+1;
    const row=Math.floor(idx/nC);
    const col=idx%nC;
    const yTop=1-row*(rowDomainH+rowGap);
    const yBot=yTop-rowDomainH;
    colorbarBottom=Math.min(colorbarBottom,yBot);
    colorbarTop=Math.max(colorbarTop,yTop);
    layout[`xaxis${axN}`]={...th.xaxis,tickfont:{size:11},tickvals:xi,ticktext:xtick,
      domain:xDomains[col],
      title:row===nR-1?{text:'Test missing %',font:{size:12},standoff:xTitleStandoff}:{},range:[-.5,nTe-.5],fixedrange:true};
    layout[`yaxis${axN}`]={...th.yaxis,tickfont:{size:11},tickvals:yi,ticktext:ytick,
      domain:[yBot,yTop],
      title:idx%nC===0?{text:'Train missing %',font:{size:12},standoff:yTitleStandoff}:{},range:[nTr-.5,-.5],fixedrange:true};
  });
  const scaleIdx=traces.findIndex(t=>t.showscale);
  if(scaleIdx>=0){
    const span=Math.max(0.01,colorbarTop-colorbarBottom);
    traces[scaleIdx].colorbar={...traces[scaleIdx].colorbar,
      y:colorbarBottom+span/2,
      len:Math.max(0.01,span*.96),
      yanchor:'middle',
      ypad:0,
      ticks:'outside'};
  }
  const gd=document.getElementById('ch-hm');
  Plotly.react(gd,traces,layout,CFG_HM).then(()=>{
    guardHoverWhenDropdownOpen(gd);
  });
}

function rTraj(){
  const ma=D().mean_auc||[]; const bft=D().bft||[];
  if(!ma.length)return;
  const ml=metricLabel();
  const methods=activeMethods();
  if(!methods.length){emptyPlot('ch-tr','Select at least one method to visualise it.');return;}
  const th=ptBase(),d=dk(),tc=d?'#e0e4ff':'#0b0b2e';

  const ciFromRow=r=>(r&&r.std_auc!=null&&r.n_replicates>1)?1.96*r.std_auc/Math.sqrt(r.n_replicates):null;
  const pctLab=v=>v!=null?`${(v*100).toFixed(0)}%`:'–';
  const ciLab=v=>v!=null?`±${v.toFixed(3)}`:'–';
  const aucs=ma.flatMap(r=>{
    if(r.mean_auc==null) return [];
    const ci=ciFromRow(r);
    return ci!=null?[r.mean_auc-ci,r.mean_auc+ci,r.mean_auc]:[r.mean_auc];
  }).filter(v=>v!=null);
  const yr=[clamp(Math.min(...aucs)-.01,.35,.9),clamp(Math.max(...aucs)+.01,.5,1)];
  const maxTr=Math.max(...[...new Set(ma.map(r=>r.train_missing_prop))]);
  const maxTe=Math.max(...[...new Set(ma.map(r=>r.test_missing_prop))]);
  const pad=maxTr*0.06;
  const xPad=(rng)=>Math.max(rng*.04,.015);
  const xA=(t,rng)=>({...th.xaxis,title:{text:t,font:{size:12.5},standoff:14},tickformat:',.0%',range:[-xPad(rng),rng+xPad(rng)],autorange:false,automargin:false,tickfont:{size:12}});
  const yA=(showTitle)=>({...th.yaxis,title:showTitle?{text:ml,font:{size:12.5},standoff:6}:{},autorange:true,automargin:false,tickfont:{size:12}});
  const fillCol=(m,a=.11)=>{
    const[ri,gi,bi]=hexRgb(MC[m]||'#888888');
    return`rgba(${ri},${gi},${bi},${a})`;
  };
  const ciBand=(m,pts,xfn,yfn,cifn,ax,ay)=>{
    if(!S.ciVis.traj || !pts.some(p=>cifn(p)!=null)) return [];
    return[
      {type:'scatter',mode:'lines',name:`${MD[m]||m} 95% CI`,
       x:pts.map(xfn),y:pts.map(p=>{const ci=cifn(p);return ci!=null?yfn(p)+ci:null}),
       line:{width:0},hoverinfo:'skip',showlegend:false,legendgroup:m,xaxis:ax,yaxis:ay},
      {type:'scatter',mode:'lines',name:`${MD[m]||m} 95% CI`,
       x:pts.map(xfn),y:pts.map(p=>{const ci=cifn(p);return ci!=null?yfn(p)-ci:null}),
       line:{width:0},fill:'tonexty',fillcolor:fillCol(m),hoverinfo:'skip',
       showlegend:false,legendgroup:m,xaxis:ax,yaxis:ay},
    ];
  };

  const mkLines=(filter,xfn,yfn,src,ax,ay,leg,methFilt)=>{
    const mths=methFilt?methods.filter(methFilt):methods;
    return mths.flatMap(m=>{
      const pts=(src||ma).filter(r=>r.model_name===m&&filter(r)).sort((a,b)=>xfn(a)-xfn(b));
      if(!pts.length)return[];
      return[...ciBand(m,pts,xfn,yfn,ciFromRow,ax,ay),
        {type:'scatter',mode:'lines+markers',name:MD[m]||m,
        x:pts.map(xfn),y:pts.map(yfn),
        customdata:pts.map(r=>[pctLab(r.train_missing_prop),pctLab(r.test_missing_prop),ciLab(ciFromRow(r))]),
        line:{color:MC[m]||'#888',width:2.2,shape:'spline'},
        marker:{color:MC[m]||'#888',size:5,line:{color:d?'rgba(0,0,0,.3)':'rgba(255,255,255,.5)',width:1}},
        xaxis:ax,yaxis:ay,showlegend:leg,legendgroup:m,
        hoverlabel:{bordercolor:MC[m]||'#888'},
        hovertemplate:`&nbsp;<b>${MD[m]||m}</b>&nbsp;<br>&nbsp;Train m: %{customdata[0]}&nbsp;<br>&nbsp;Test m: %{customdata[1]}&nbsp;<br>&nbsp;Mean ${ml}: %{y:.3f}&nbsp;<br>&nbsp;95% C.I.: %{customdata[2]}&nbsp;<extra></extra>`,
      }];
    });
  };
  const mkBftLines=(ax,ay,leg)=>{
    if(bft.length) return methods.map(m=>{
      const pts=bft.filter(r=>r.model_name===m).sort((a,b)=>a.test_missing_prop-b.test_missing_prop)
        .map(r=>{
          const mr=ma.find(x=>x.model_name===m
            &&Math.abs(x.train_missing_prop-r.envelope_train_missing_prop)<.001
            &&Math.abs(x.test_missing_prop-r.test_missing_prop)<.001);
          return {...r,_ci:ciFromRow(mr)};
        });
      return[...ciBand(m,pts,r=>r.test_missing_prop,r=>r.envelope_mean_auc,r=>r._ci,ax,ay),
        {type:'scatter',mode:'lines+markers',name:MD[m]||m,
        x:pts.map(r=>r.test_missing_prop),y:pts.map(r=>r.envelope_mean_auc),
        customdata:pts.map(r=>[pctLab(r.envelope_train_missing_prop),pctLab(r.test_missing_prop),ciLab(r._ci)]),
        line:{color:MC[m]||'#888',width:2.2,shape:'spline'},
        marker:{color:MC[m]||'#888',size:5,line:{color:d?'rgba(0,0,0,.3)':'rgba(255,255,255,.5)',width:1}},
        xaxis:ax,yaxis:ay,showlegend:leg,legendgroup:m,
        hoverlabel:{bordercolor:MC[m]||'#888'},
        hovertemplate:`&nbsp;<b>${MD[m]||m}</b>&nbsp;<br>&nbsp;Train m: %{customdata[0]}&nbsp;<br>&nbsp;Test m: %{customdata[1]}&nbsp;<br>&nbsp;Mean ${ml}: %{y:.3f}&nbsp;<br>&nbsp;95% C.I.: %{customdata[2]}&nbsp;<extra></extra>`,
      }];
    }).flat();
    return mkLines(r=>Math.abs(r.train_missing_prop)<.001,r=>r.test_missing_prop,r=>r.mean_auc,null,ax,ay,leg);
  };

  // Determine visible panels
  const vis=trajectoryPanelVis('traj');
  const PANELS=[
    {key:'train',lbl:'Train-time',     build:(ax,ay,leg)=>mkLines(r=>Math.abs(r.test_missing_prop)<.001,r=>r.train_missing_prop,r=>r.mean_auc,null,ax,ay,leg,notDistill)},
    {key:'test', lbl:'Test-time',      build:(ax,ay,leg)=>mkLines(r=>Math.abs(r.train_missing_prop)<.001,r=>r.test_missing_prop,r=>r.mean_auc,null,ax,ay,leg)},
    {key:'bft',  lbl:'Best fixed-train',build:(ax,ay,leg)=>mkBftLines(ax,ay,leg)},
  ];
  const hasTrainCompatible=methods.some(notDistill);
  const vp=PANELS.filter(p=>vis[p.key] && (p.key!=='train' || hasTrainCompatible));
  if(!vp.length){emptyPlot('ch-tr','Select at least one compatible trajectory to visualise it.');return;}

  const axId=i=>({x:i===0?'x':`x${i+1}`,y:i===0?'y':`y${i+1}`});
  const allTraces=[];
  const annotations=[];
  const legendSeen=new Set();
  vp.forEach((p,i)=>{
    const {x:ax,y:ay}=axId(i);
    const traces=p.build(ax,ay,true).map(tr=>{
      if(tr.showlegend===false) return tr;
      const key=tr.name||'';
      if(!key || legendSeen.has(key)){
        tr.showlegend=false;
      }else{
        tr.showlegend=true;
        legendSeen.add(key);
      }
      return tr;
    });
    allTraces.push(...traces);
    annotations.push({text:`<b>${p.lbl}</b>`,x:.5,y:1,xref:`${ax} domain`,yref:`${ay} domain`,yshift:28,showarrow:false,font:{size:14.5,color:tc},xanchor:'center',yanchor:'bottom'});
  });
  if(!allTraces.some(tr=>tr.hoverinfo!=='skip' && tr.name!=='No degradation')){
    emptyPlot('ch-tr','Select at least one method to visualise it.');
    return;
  }

  const trH=trajectoryHeight('ch-tr');
  const trMt=trajMarginTop('ch-tr');
  const layout={...th,
    grid:{rows:1,columns:vp.length,pattern:'independent',xgap:.12},height:trH,
    margin:{...TRAJ_MARGIN,t:trMt},showlegend:true,
    legend:trajectoryLegend(trH,trMt),
    annotations,
  };
  const domains=axisDomains(vp.length);
  vp.forEach((p,i)=>{
    const sfx=i===0?'':i+1;
    const isTrain=p.key==='train';
    layout[`xaxis${sfx}`]=xA(isTrain?'Train missing %':'Test missing %',isTrain?maxTr:maxTe);
    layout[`xaxis${sfx}`].domain=domains[i];
    layout[`yaxis${sfx}`]=yA(i===0);
    layout[`yaxis${sfx}`].domain=[0,1];
  });
  const gd=document.getElementById('ch-tr');
  syncTrajectoryPlotHeight('ch-tr',trH);
  Plotly.react(gd,allTraces,layout,CFG).then(()=>{
    gd.style.height=trH+'px';
    gd._m3LastHeight=trH;
    gd._m3BaseAnnotations=annotations;
    gd._m3FullLayoutForEmpty=JSON.parse(JSON.stringify(layout));
    gd._m3AxisKeys=Object.keys(layout).filter(k=>/^xaxis\d*$/.test(k)||/^yaxis\d*$/.test(k));
    guardHoverWhenDropdownOpen(gd);
    attachTrajectoryTooltip(gd,'perf');
    attachTrajectoryLegendEmpty(gd,'Select at least one method to visualise it.');
    scheduleTrajectoryLegendCenter('ch-tr');
  });
}

function rDeg(){
  const deg=D().deg||[];
  if(!deg.length)return;
  const methods=activeMethods().filter(m=>deg.some(r=>r.model_name===m));
  if(!methods.length){emptyPlot('ch-dg','Select at least one method to visualise it.');return;}
  const th=ptBase(),d=dk(),tc=d?'#e0e4ff':'#0b0b2e';

  const allRCI=deg.flatMap(r=>[
    r.degradation_ratio,
    r.degradation_ratio_ci95_lower,
    r.degradation_ratio_ci95_upper,
  ]).filter(v=>v!=null);
  const rmin=allRCI.length?Math.max(Math.min(...allRCI,1)-.03,.5):.9;
  const rmax=allRCI.length?Math.min(Math.max(...allRCI,1)+.03,1.4):1.25;
  const trainMaxD=Math.max(...deg.filter(r=>r.scenario==='train'&&r.train_prop!=null).map(r=>r.train_prop).concat([0]));
  const testMaxD =Math.max(...deg.filter(r=>r.scenario==='test' &&r.test_prop !=null).map(r=>r.test_prop ).concat([0]));
  const envMaxD  =Math.max(...deg.filter(r=>r.scenario==='envelope'&&r.test_prop!=null).map(r=>r.test_prop).concat([0]));
  const pad=Math.max(trainMaxD,testMaxD)*0.06;
  const xPad=(mx)=>Math.max(mx*.04,.015);
  const xA=(t,mx)=>({...th.xaxis,title:{text:t,font:{size:12.5},standoff:14},tickformat:',.0%',range:[-xPad(mx),mx+xPad(mx)],autorange:false,automargin:false,tickfont:{size:12}});
  const yA=(showTitle)=>({...th.yaxis,title:showTitle?{text:'Degradation ratio',font:{size:12.5},standoff:6}:{},autorange:true,automargin:false,tickfont:{size:12}});
  const pctLab=v=>v!=null?`${(v*100).toFixed(0)}%`:'–';
  const ciLab=v=>v!=null?`±${v.toFixed(3)}`:'–';
  const fillCol=(m,a=.11)=>{
    const[ri,gi,bi]=hexRgb(MC[m]||'#888888');
    return`rgba(${ri},${gi},${bi},${a})`;
  };
  const degBand=(m,pts,xf,ax,ay)=>{
    if(!S.ciVis.deg || !pts.some(p=>p.degradation_ratio_ci95_lower!=null&&p.degradation_ratio_ci95_upper!=null)) return [];
    return[
      {type:'scatter',mode:'lines',name:`${MD[m]||m} 95% CI`,
       x:pts.map(r=>r[xf]),y:pts.map(r=>r.degradation_ratio_ci95_upper??null),
       line:{width:0},hoverinfo:'skip',showlegend:false,legendgroup:m,xaxis:ax,yaxis:ay},
      {type:'scatter',mode:'lines',name:`${MD[m]||m} 95% CI`,
       x:pts.map(r=>r[xf]),y:pts.map(r=>r.degradation_ratio_ci95_lower??null),
       line:{width:0},fill:'tonexty',fillcolor:fillCol(m),hoverinfo:'skip',
       showlegend:false,legendgroup:m,xaxis:ax,yaxis:ay},
    ];
  };
  const mkDegLines=(sfn,xf,ax,ay,leg,methFilt)=>{
    const mths=methFilt?methods.filter(methFilt):methods;
    const panelPts=deg.filter(r=>sfn(r)&&r.degradation_ratio!=null&&r[xf]!=null);
    const panelXs=panelPts.map(r=>r[xf]);
    const xMin=panelXs.length?Math.min(...panelXs):0;
    const xMax=panelXs.length?Math.max(...panelXs):1;
    const tr=mths.flatMap(m=>{
      const pts=deg.filter(r=>r.model_name===m&&sfn(r)&&r.degradation_ratio!=null).sort((a,b)=>a[xf]-b[xf]);
      if(!pts.length)return[];
      return[...degBand(m,pts,xf,ax,ay),
        {type:'scatter',mode:'lines+markers',name:MD[m]||m,
        x:pts.map(r=>r[xf]),y:pts.map(r=>r.degradation_ratio),
        customdata:pts.map(r=>[pctLab(r.train_prop),pctLab(r.test_prop),ciLab(r.degradation_ratio_ci95)]),
        line:{color:MC[m]||'#888',width:2,shape:'spline'},marker:{color:MC[m]||'#888',size:5},
        xaxis:ax,yaxis:ay,showlegend:leg,legendgroup:m,legendrank:100,
        hoverlabel:{bordercolor:MC[m]||'#888'},
        hovertemplate:`&nbsp;<b>${MD[m]||m}</b>&nbsp;<br>&nbsp;Train m: %{customdata[0]}&nbsp;<br>&nbsp;Test m: %{customdata[1]}&nbsp;<br>&nbsp;Ratio: %{y:.3f}&nbsp;<br>&nbsp;95% C.I.: %{customdata[2]}&nbsp;<extra></extra>`,
      }];
    });
    tr.push({type:'scatter',mode:'lines',name:'No degradation',
      x:[xMin,xMax],y:[1,1],line:{color:d?'rgba(255,255,255,.22)':'rgba(0,0,0,.18)',width:1.5,dash:'dot'},
      xaxis:ax,yaxis:ay,showlegend:leg,legendgroup:'__no_degradation__',legendrank:9999,hoverinfo:'skip'});
    return tr;
  };

  const vis=trajectoryPanelVis('deg');
  const PANELS=[
    {key:'train',lbl:'Train-time',        build:(ax,ay,leg)=>mkDegLines(r=>r.scenario==='train','train_prop',ax,ay,leg,notDistill)},
    {key:'test', lbl:'Test-time',         build:(ax,ay,leg)=>mkDegLines(r=>r.scenario==='test', 'test_prop', ax,ay,leg)},
    {key:'bft',  lbl:'Best fixed-train',  build:(ax,ay,leg)=>mkDegLines(r=>r.scenario==='envelope','test_prop',ax,ay,leg)},
  ];
  const hasTrainCompatible=methods.some(notDistill);
  const vp=PANELS.filter(p=>vis[p.key] && (p.key!=='train' || hasTrainCompatible));
  if(!vp.length){emptyPlot('ch-dg','Select at least one compatible trajectory to visualise it.');return;}

  const axId=i=>({x:i===0?'x':`x${i+1}`,y:i===0?'y':`y${i+1}`});
  const allTraces=[];
  const annotations=[];
  const legendSeen=new Set();
  vp.forEach((p,i)=>{
    const {x:ax,y:ay}=axId(i);
    const traces=p.build(ax,ay,true).map(tr=>{
      if(tr.showlegend===false) return tr;
      const key=tr.name||'';
      if(!key || legendSeen.has(key)){
        tr.showlegend=false;
      }else{
        tr.showlegend=true;
        legendSeen.add(key);
      }
      return tr;
    });
    allTraces.push(...traces);
    annotations.push({text:`<b>${p.lbl}</b>`,x:.5,y:1,xref:`${ax} domain`,yref:`${ay} domain`,yshift:28,showarrow:false,font:{size:14.5,color:tc},xanchor:'center',yanchor:'bottom'});
  });
  if(!allTraces.some(tr=>tr.hoverinfo!=='skip' && tr.name!=='No degradation')){
    emptyPlot('ch-dg','Select at least one method to visualise it.');
    return;
  }

  const xLabels={train:'Train missing %',test:'Test missing %',bft:'Test missing %'};
  const xMaxes={train:trainMaxD,test:testMaxD,bft:testMaxD};
  const dgH=trajectoryHeight('ch-dg');
  const dgMt=trajMarginTop('ch-dg');
  const layout={...th,
    grid:{rows:1,columns:vp.length,pattern:'independent',xgap:.12},height:dgH,
    margin:{...TRAJ_MARGIN,t:dgMt},showlegend:true,
    legend:trajectoryLegend(dgH,dgMt),
    annotations,
  };
  const domains=axisDomains(vp.length);
  vp.forEach((p,i)=>{
    const sfx=i===0?'':i+1;
    layout[`xaxis${sfx}`]=xA(xLabels[p.key],xMaxes[p.key]);
    layout[`xaxis${sfx}`].domain=domains[i];
    layout[`yaxis${sfx}`]=yA(i===0);
    layout[`yaxis${sfx}`].domain=[0,1];
  });
  const gd=document.getElementById('ch-dg');
  syncTrajectoryPlotHeight('ch-dg',dgH);
  Plotly.react(gd,allTraces,layout,CFG).then(()=>{
    gd.style.height=dgH+'px';
    gd._m3LastHeight=dgH;
    gd._m3BaseAnnotations=annotations;
    gd._m3FullLayoutForEmpty=JSON.parse(JSON.stringify(layout));
    gd._m3AxisKeys=Object.keys(layout).filter(k=>/^xaxis\d*$/.test(k)||/^yaxis\d*$/.test(k));
    guardHoverWhenDropdownOpen(gd);
    attachTrajectoryTooltip(gd,'deg');
    attachTrajectoryLegendEmpty(gd,'Select at least one method to visualise it.');
    const noDegHandler=ev=>{
      const tr=(gd.data||[])[ev.curveNumber]||{};
      if(tr.name==='No degradation') return false;
    };
    if(gd._m3NoDegHandler&&gd.removeListener) gd.removeListener('plotly_legendclick',gd._m3NoDegHandler);
    gd._m3NoDegHandler=noDegHandler;
    if(gd.on) gd.on('plotly_legendclick',noDegHandler);
    scheduleTrajectoryLegendCenter('ch-dg');
  });
}

function renderHpTrainPropSelect(props){
  const el=document.getElementById('hp-trainprop-filter');
  if(!el)return;
  if(!props.length){el.innerHTML='';return;}
  const cur=S.hpTrainProp;
  const pct=v=>Math.round(v*100)+'%';
  const valLabel=cur===null?'All':pct(cur);
  const menuWasOpen=document.getElementById('hp-trainprop-menu')?.classList.contains('open');
  el.innerHTML=`<div class="dd-btn" onclick="toggleHpTpMenu(event)">
    <span class="dd-lbl">Train</span>
    <div class="dd-sep"></div>
    <span class="dd-val">${valLabel}</span>
    <svg class="dd-caret" viewBox="0 0 10 6" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M1 1l4 4 4-4"/></svg>
  </div>`;
  const items=[{v:null,l:'All'},...props.map(v=>({v,l:pct(v)}))].map(({v,l})=>{
    const on=(v===null&&cur===null)||(v!==null&&cur!==null&&Math.abs(v-cur)<1e-6)?'on':'';
    return`<div class="hp-tp-item ${on}" onclick="selectHpTrainProp(${v===null?'null':v},event)">${l}</div>`;
  }).join('');
  let menu=document.getElementById('hp-trainprop-menu');
  if(!menu){menu=document.createElement('div');menu.id='hp-trainprop-menu';(document.querySelector('.app')||document.body).appendChild(menu);}
  menu.className='hp-tp-menu';
  menu.innerHTML=items;
  if(menuWasOpen){menu.classList.add('open');positionHpTpMenu();}
}
function selectHpTrainProp(v,ev=null){
  if(ev)ev.stopPropagation();
  S.hpTrainProp=(v===null||v===undefined)?null:Number(v);
  closeHpTpMenu();
  rHpSelection();
}
function rHpSelection(){
  const el=document.getElementById('hp-sel');
  if(!el)return;
  applyHpHeight();
  const active=new Set(activeMethods());
  let allRows=[...(D().hp_selection||[])].filter(r=>active.has(r.model_name));
  const methods=[...new Set(allRows.map(r=>r.model_name).filter(Boolean))].sort();
  const trainProps=[...new Set(allRows.map(r=>r.train_prop).filter(v=>v!=null))].sort((a,b)=>a-b);
  renderHpMethodSelect(methods);
  renderHpTrainPropSelect(trainProps);
  if(!allRows.length){
    el.innerHTML='<div class="es"><div class="es-i">⚙️</div><div>No hyperparameter selection data found.</div><p>This widget requires inner_hp_eval.csv and selected_inner_hp_names in outer_test_metrics.csv.</p></div>';
    return;
  }
  if(!S.hpMethod||!methods.includes(S.hpMethod))S.hpMethod=methods[0];
  renderHpMethodSelect(methods);
  renderHpTrainPropSelect(trainProps);
  // Filter by method
  let rows=allRows.filter(r=>r.model_name===S.hpMethod);
  // Aggregate or filter by train prop
  if(S.hpTrainProp===null){
    // total events = sum of events per unique train_prop (n_seeds × n_outer_folds per train_prop)
    const seenTp=new Set();
    let totalEvents=0;
    rows.forEach(r=>{
      const tpKey=r.train_prop==null?'null':String(r.train_prop);
      if(!seenTp.has(tpKey)){seenTp.add(tpKey);totalEvents+=(r.selection_events||0);}
    });
    // aggregate by hp_combination with the same shared denominator for all configs
    const agg={};
    rows.forEach(r=>{
      const k=r.hp_combination;
      if(!agg[k]){agg[k]={...r,selected_count:0,selection_events:totalEvents};}
      agg[k].selected_count+=(r.selected_count||0);
    });
    rows=Object.values(agg).map(r=>({...r,selected_pct:r.selection_events?100*r.selected_count/r.selection_events:null}));
  } else {
    rows=rows.filter(r=>r.train_prop!=null&&Math.abs(r.train_prop-S.hpTrainProp)<1e-6);
  }
  rows.sort((a,b)=>
    (b.selected_count||0)-(a.selected_count||0) ||
    String(a.model_name||'').localeCompare(String(b.model_name||'')) ||
    String(a.hp_combination||'').localeCompare(String(b.hp_combination||'')));

  const baseCols=new Set(['model_name','hp_combination','train_prop','selected_count','selection_events','selected_pct']);
  const preferred=['lr_C','lr_penalty','lr_solver','lr_class_weight','lr_max_iter',
    'rf_n_estimators','rf_max_depth','rf_min_samples_split','rf_min_samples_leaf','rf_max_features','rf_class_weight',
    'coxnet_alpha','coxnet_l1_ratio','coxnet_max_iter','coxnet_tol',
    'rsf_n_estimators','rsf_max_depth','rsf_min_samples_split','rsf_min_samples_leaf','rsf_max_features',
    'batch_size','learning_rate','weight_decay','dropout',
    'modality_hidden_layers','fusion_hidden_dim','fusion_hidden_layers','fusion_batchnorm',
    'distill_alpha','distill_beta','vae_latent_dim','knn_neighbors',
    'n_estimators','max_depth','l1_ratio','alpha'];
  const hasHpValue=v=>{
    const s=String(v??'').trim();
    return s!==''&&s!=='–'&&s.toLowerCase()!=='nan'&&s.toLowerCase()!=='none'&&s.toLowerCase()!=='null';
  };
  const present=new Set([...new Set(rows.flatMap(r=>Object.keys(r)))]
    .filter(k=>!baseCols.has(k)&&rows.some(r=>hasHpValue(r[k]))));
  const paramCols=[
    ...preferred.filter(k=>present.has(k)),
    ...[...present].filter(k=>!preferred.includes(k)).sort()
  ];
  const label=k=>{
    const explicit={
      lr_C:'C',lr_penalty:'Penalty',lr_solver:'Solver',lr_class_weight:'Class weight',lr_max_iter:'Max iter',
      rf_n_estimators:'Trees',rf_max_depth:'Max depth',rf_min_samples_split:'Min split',
      rf_min_samples_leaf:'Min leaf',rf_max_features:'Max features',rf_class_weight:'Class weight',
      coxnet_alpha:'Alpha',coxnet_l1_ratio:'L1 ratio',coxnet_max_iter:'Max iter',coxnet_tol:'Tolerance',
      rsf_n_estimators:'Trees',rsf_max_depth:'Max depth',rsf_min_samples_split:'Min split',
      rsf_min_samples_leaf:'Min leaf',rsf_max_features:'Max features',
      batch_size:'Batch size',learning_rate:'LR',weight_decay:'Weight decay',
      modality_hidden_layers:'Modality layers',fusion_hidden_dim:'Fusion dim',
      fusion_hidden_layers:'Fusion layers',fusion_batchnorm:'Fusion BN',
      distill_alpha:'KD α',distill_beta:'KD β',vae_latent_dim:'VAE latent',
      knn_neighbors:'KNN k',n_estimators:'Trees',max_depth:'Max depth',l1_ratio:'L1 ratio',
    }[k];
    if(explicit) return explicit;
    const stripped=k.replace(/^(healnet_|pam_|smil_e_)/,'');
    return stripped.replaceAll('_',' ');
  };

  const cols=[
    {k:'model_name',l:'Method'},
    {k:'hp_combination',l:'HP combination name'},
    ...paramCols.map(k=>({k,l:label(k)})),
  ];
  const colWidth=c=>{
    if(c.k==='model_name')return 130;
    if(c.k==='hp_combination')return 360;
    return 140;
  };
  const rawColSum=cols.reduce((s,c)=>s+colWidth(c),0);
  const mainTableWidth=Math.max(900,rawColSum);
  const hpComboBonus=mainTableWidth-rawColSum;
  const mainColgroup=`<colgroup>${cols.map(c=>{const w=colWidth(c)+(c.k==='hp_combination'?hpComboBonus:0);return`<col style="width:${w}px">`;}).join('')}</colgroup>`;
  const fixedColgroup=`<colgroup><col style="width:104px"><col style="width:72px"></colgroup>`;
  const ths=cols.map(c=>`<th>${escHtml(c.l)}</th>`).join('');
  const fixedTh='<th>Selected</th><th>%</th>';
  const trs=rows.map((r,i)=>{
    const cells=cols.map(c=>{
      if(c.k==='model_name'){
        return `<td class="hp-method" style="color:${MC[r.model_name]||'#aaa'}">${escHtml(MD[r.model_name]||r.model_name)}</td>`;
      }
      if(c.k==='hp_combination'){
        const clipSvg=`<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>`;
        return `<td class="hp-combo" title="${escHtml(r.hp_combination)}"><span class="hp-combo-text">${escHtml(r.hp_combination)}</span><span class="hp-combo-copy" data-v="${escHtml(r.hp_combination)}" onclick="copyHpCombo(this)">${clipSvg} Copy to clipboard</span></td>`;
      }
      return `<td>${escHtml(r[c.k]??'–')}</td>`;
    }).join('');
    return `<tr data-i="${i}" data-m="${escHtml(r.model_name)}">${cells}</tr>`;
  }).join('');
  const fixedTrs=rows.map((r,i)=>{
    const pct=r.selected_pct==null?'–':Number(r.selected_pct).toFixed(1)+'%';
    return `<tr data-i="${i}" data-m="${escHtml(r.model_name)}"><td class="best">${escHtml(r.selected_count)}/${escHtml(r.selection_events)}</td><td>${pct}</td></tr>`;
  }).join('');
  el.innerHTML=`<div class="hp-table-shell">
    <div class="hp-head-overlay">
      <div class="hp-scroll hp-head-main"><table class="mt hp-main" style="width:${mainTableWidth}px">${mainColgroup}<thead><tr>${ths}</tr></thead></table></div>
      <div class="hp-fixed hp-head-fixed"><table class="mt hp-selected">${fixedColgroup}<thead><tr>${fixedTh}</tr></thead></table></div>
    </div>
    <div class="hp-table-wrap hp-body-wrap">
      <div class="hp-scroll hp-body-main"><table class="mt hp-main" style="width:${mainTableWidth}px">${mainColgroup}<tbody>${trs}<tr class="hp-spacer"><td colspan="999"></td></tr></tbody></table></div>
      <div class="hp-fixed hp-body-fixed"><table class="mt hp-selected">${fixedColgroup}<tbody>${fixedTrs}<tr class="hp-spacer"><td colspan="2"></td></tr></tbody></table></div>
    </div>
  </div>`;
  syncHpRowHover(el);
  syncHpHeaderScroll(el);
}

function syncHpRowHover(el){
  const rows=[...el.querySelectorAll('.hp-table-wrap tr[data-i]')];
  rows.forEach(row=>{
    row.addEventListener('mouseenter',()=>{
      const idx=row.dataset.i;
      el.querySelectorAll(`.hp-table-wrap tr[data-i="${idx}"]`).forEach(r=>r.classList.add('hp-row-hover'));
    });
    row.addEventListener('mouseleave',()=>{
      const idx=row.dataset.i;
      el.querySelectorAll(`.hp-table-wrap tr[data-i="${idx}"]`).forEach(r=>r.classList.remove('hp-row-hover'));
    });
  });
}

function copyHpCombo(el){
  const text=el.dataset.v||'';
  const clipSvg=`<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>`;
  const checkSvg=`<svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:1px"><polyline points="2 8 6 12 14 4"/></svg>`;
  navigator.clipboard.writeText(text).then(()=>{
    el.innerHTML=checkSvg+' Copied!';
    setTimeout(()=>{el.innerHTML=clipSvg+' Copy to clipboard';},1600);
  }).catch(()=>{
    el.textContent='Error';
    setTimeout(()=>{el.innerHTML=clipSvg+' Copy to clipboard';},1600);
  });
}
function syncHpHeaderScroll(el){
  const body=el.querySelector('.hp-body-main');
  const head=el.querySelector('.hp-head-main');
  if(!body||!head)return;
  const apply=()=>{head.scrollLeft=body.scrollLeft||0;};
  body.addEventListener('scroll',apply,{passive:true});
  apply();
}

function renderHpMethodSelect(methods){
  const el=document.getElementById('hp-method-filter');
  if(!el)return;
  if(!methods.length){
    el.innerHTML=`<div class="dd-btn" style="opacity:.55;cursor:not-allowed">
      <span class="dd-lbl">Method</span>
      <div class="dd-sep"></div>
      <span class="dd-val">None</span>
    </div>`;
    S.hpMethod=null;
    return;
  }
  if(!S.hpMethod||!methods.includes(S.hpMethod))S.hpMethod=methods[0];
  const menuWasOpen=document.getElementById('hp-method-menu')?.classList.contains('open');
  const items=methods.map(m=>`
    <div class="hp-item ${m===S.hpMethod?'on':''}" data-m="${escHtml(m)}" onclick="selectHpMethod('${m.replace(/'/g,"\\'")}',event)">
      ${escHtml(MD[m]||m)}
    </div>`).join('');
  el.innerHTML=`<div class="dd-btn" onclick="toggleHpMethodMenu(event)">
      <span class="dd-lbl">Method</span>
      <div class="dd-sep"></div>
      <span class="dd-val">${escHtml(MD[S.hpMethod]||S.hpMethod)}</span>
      <svg class="dd-caret" viewBox="0 0 10 6" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M1 1l4 4 4-4"/></svg>
    </div>`;
  // Portal menu outside hp-method-filter so contain:paint doesn't clip it
  let menu=document.getElementById('hp-method-menu');
  if(!menu){menu=document.createElement('div');menu.id='hp-method-menu';(document.querySelector('.app')||document.body).appendChild(menu);}
  menu.className='hp-menu';
  menu.innerHTML=items;
  if(menuWasOpen){menu.classList.add('open');positionHpMethodMenu();}
}
let _hpRafId=null,_hpLastL=null,_hpLastT=null;
function _hpMenuTick(){
  const menu=document.getElementById('hp-method-menu');
  if(!menu||!menu.classList.contains('open')){_hpRafId=null;return;}
  const btn=document.querySelector('#hp-method-filter .dd-btn');
  if(btn){
    const r=btn.getBoundingClientRect();
    const sc=dashboardScale();
    const btnW=Math.round(r.width/sc);
    const menuW=btnW;
    const left=Math.round(r.left/sc);
    const top=Math.round(r.bottom/sc+7/sc);
    if(left!==_hpLastL||top!==_hpLastT||menu.style.width!==btnW+'px'){
      menu.style.left=`${left}px`;
      menu.style.top=`${top}px`;
      menu.style.width=`${btnW}px`;
      menu.style.minWidth=`${btnW}px`;
      _hpLastL=left;_hpLastT=top;
    }
  }
  _hpRafId=requestAnimationFrame(_hpMenuTick);
}
function positionHpMethodMenu(){
  _hpLastL=null;_hpLastT=null;
  if(!_hpRafId)_hpRafId=requestAnimationFrame(_hpMenuTick);
}
function closeHpMenu(){
  document.querySelectorAll('.hp-menu').forEach(m=>m.classList.remove('open'));
  if(_hpRafId){cancelAnimationFrame(_hpRafId);_hpRafId=null;}
  _hpLastL=null;_hpLastT=null;
}
function toggleHpMethodMenu(ev){
  if(ev)ev.stopPropagation();
  const menu=document.getElementById('hp-method-menu');
  if(!menu)return;
  const wasOpen=menu.classList.contains('open');
  document.querySelectorAll('.dd-menu,.mf-menu').forEach(m=>m.classList.remove('open'));
  closeHpMenu();closeHpTpMenu();
  if(!wasOpen){
    menu.classList.add('open');
    positionHpMethodMenu();
    clearPlotHovers();
  }
}
function selectHpMethod(m,ev=null){
  if(ev)ev.stopPropagation();
  S.hpMethod=m||null;
  closeHpMenu();
  rHpSelection();
}
let _hpTpRafId=null,_hpTpLastL=null,_hpTpLastT=null;
function _hpTpMenuTick(){
  const menu=document.getElementById('hp-trainprop-menu');
  if(!menu||!menu.classList.contains('open')){_hpTpRafId=null;return;}
  const btn=document.querySelector('#hp-trainprop-filter .dd-btn');
  if(btn){
    const r=btn.getBoundingClientRect();
    const sc=dashboardScale();
    const btnW=Math.round(r.width/sc);
    const left=Math.round(r.left/sc);
    const top=Math.round(r.bottom/sc+7/sc);
    if(left!==_hpTpLastL||top!==_hpTpLastT||menu.style.width!==btnW+'px'){
      menu.style.left=`${left}px`;
      menu.style.top=`${top}px`;
      menu.style.width=`${btnW}px`;
      menu.style.minWidth=`${btnW}px`;
      _hpTpLastL=left;_hpTpLastT=top;
    }
  }
  _hpTpRafId=requestAnimationFrame(_hpTpMenuTick);
}
function positionHpTpMenu(){
  _hpTpLastL=null;_hpTpLastT=null;
  if(!_hpTpRafId)_hpTpRafId=requestAnimationFrame(_hpTpMenuTick);
}
function closeHpTpMenu(){
  const menu=document.getElementById('hp-trainprop-menu');
  if(menu)menu.classList.remove('open');
  if(_hpTpRafId){cancelAnimationFrame(_hpTpRafId);_hpTpRafId=null;}
  _hpTpLastL=null;_hpTpLastT=null;
}
function toggleHpTpMenu(ev){
  if(ev)ev.stopPropagation();
  const menu=document.getElementById('hp-trainprop-menu');
  if(!menu)return;
  const wasOpen=menu.classList.contains('open');
  document.querySelectorAll('.dd-menu,.mf-menu').forEach(m=>m.classList.remove('open'));
  closeHpMenu();closeHpTpMenu();
  if(!wasOpen){
    menu.classList.add('open');
    positionHpTpMenu();
    clearPlotHovers();
  }
}

// ── METRICS ───────────────────────────────────────────────────────────────
function rMetrics(){rBars();rRadar();rTable();}

function metricCols(){
  const ml=metricLabel();
  return [
  {k:'model_name',l:'Method',fmt:v=>`<span style="color:${MC[v]||'#aaa'};font-weight:700">${MD[v]||v}</span>`,hb:null},
  {k:'baseline_auc',l:`Baseline ${ml}`,fmt:f2,hb:true},
  {k:'train_time_aupmc',l:'Train AUPMC',fmt:f2,hb:true},
  {k:'test_time_aupmc',l:'Test AUPMC',fmt:f2,hb:true},
  {k:'best_fixed_train_aupmc',l:'BFT AUPMC',fmt:f2,hb:true},
  {k:'train_degradation_coefficient',l:'Train DC',fmt:f2,hb:false},
  {k:'test_degradation_coefficient',l:'Test DC',fmt:f2,hb:false},
  {k:'minimum_degradation_coefficient',l:'Best DC',fmt:f2,hb:false},
  ];
}

function sortedMet(){
  const r=[...(D().metrics||[])];
  return r.sort((a,b)=>{
    const av=a[S.sk],bv=b[S.sk];
    if(av==null&&bv==null)return 0; if(av==null)return S.sa?-1:1; if(bv==null)return S.sa?1:-1;
    return S.sa?(av>bv?1:-1):(bv>av?1:-1);
  });
}

function rTable(){
  const r=sortedMet();
  if(!r.length){document.getElementById('tbl').innerHTML='<div class="es"><div class="es-i">📊</div><div>No metrics data</div></div>';return;}
  const cols=metricCols();
  const best={};
  cols.slice(1).forEach(c=>{
    const v=r.map(x=>x[c.k]).filter(v=>v!=null);
    if(!v.length)return;
    best[c.k]=c.hb?Math.max(...v):Math.min(...v);
  });
  const ths=cols.map(c=>`<th onclick="srt('${c.k}')">${c.l}<span class="si">${S.sk===c.k?(S.sa?'↑':'↓'):'⇅'}</span></th>`).join('');
  const trs=r.map(x=>{
    const cells=cols.map(c=>{
      const v=x[c.k]; const fv=c.fmt?c.fmt(v):(v!=null?v:'–');
      const b=c.k!=='model_name'&&v!=null&&best[c.k]!=null&&Math.abs(v-best[c.k])<.00001;
      return`<td class="${b?'best':''}">${fv}</td>`;
    }).join('');
    const hl=S.hl===x.model_name?' class="hl"':'';
    return`<tr${hl} data-m="${x.model_name}" onclick="hlM('${x.model_name}')">${cells}</tr>`;
  }).join('');
  document.getElementById('tbl').innerHTML=`<table class="mt"><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
}

function srt(k){if(S.sk===k)S.sa=!S.sa;else{S.sk=k;S.sa=k==='model_name';}rTable();}
function hlM(m){S.hl=S.hl===m?null:m;rTable();}

function rRadar(){
  const r=D().metrics||[]; if(!r.length)return;
  const th=ptBase(),d=dk();
  const ml=metricLabel();
  const axes=[
    {k:'baseline_auc',l:`Baseline ${ml}`,hb:true},{k:'train_time_aupmc',l:'Train AUPMC',hb:true},
    {k:'test_time_aupmc',l:'Test AUPMC',hb:true},{k:'best_fixed_train_aupmc',l:'BFT AUPMC',hb:true},
    {k:'test_degradation_coefficient',l:'Test Robust.',hb:false},{k:'minimum_degradation_coefficient',l:'Min Robust.',hb:false},
  ];
  const rng={};
  axes.forEach(a=>{const v=r.map(x=>x[a.k]).filter(v=>v!=null);rng[a.k]={mn:Math.min(...v),mx:Math.max(...v)};});
  const norm=(v,k,hb)=>{const{mn,mx}=rng[k];if(mx===mn)return.5;const n=(v-mn)/(mx-mn);return hb?n:1-n;};
  const traces=r.map(x=>{
    const[ri,gi,bi]=hexRgb(MC[x.model_name]||'#888');
    const vals=axes.map(a=>x[a.k]!=null?norm(x[a.k],a.k,a.hb):0);
    return{type:'scatterpolar',name:MD[x.model_name]||x.model_name,
      r:[...vals,vals[0]],theta:[...axes.map(a=>a.l),axes[0].l],
      fill:'toself',fillcolor:`rgba(${ri},${gi},${bi},.06)`,
      line:{color:MC[x.model_name]||'#888',width:1.8},
      hovertemplate:`<b>${MD[x.model_name]||x.model_name}</b><br>%{theta}: %{r:.2f}<extra></extra>`,
    };
  });
  const gc=d?'rgba(255,255,255,.08)':'rgba(0,0,0,.08)';
  const layout={...th,
    polar:{bgcolor:d?'rgba(255,255,255,.015)':'rgba(0,0,0,.015)',
      radialaxis:{visible:true,range:[0,1],gridcolor:gc,tickfont:{size:10.5},tickvals:[.25,.5,.75,1]},
      angularaxis:{gridcolor:gc,tickfont:{size:12}}},
    height:300,margin:{t:14,b:10,l:14,r:14},showlegend:true,
    legend:{font:{size:12},x:1.06,y:.5,orientation:'v',bgcolor:'rgba(0,0,0,0)'},
  };
  Plotly.react('ch-rd',traces,layout,CFGs);
}

function rBars(){
  const r=D().metrics||[]; if(!r.length)return;
  const th=ptBase(); const sorted=[...r].sort((a,b)=>(b.best_fixed_train_aupmc||0)-(a.best_fixed_train_aupmc||0));
  const ml=metricLabel();
  const xl=sorted.map(x=>MD[x.model_name]||x.model_name);
  const ms=[{k:'baseline_auc',l:`Baseline ${ml}`,c:'#00d4ff'},{k:'test_time_aupmc',l:'Test AUPMC',c:'#a855f7'},{k:'best_fixed_train_aupmc',l:'BFT AUPMC',c:'#ff2d78'}];
  const traces=ms.map(m=>({type:'bar',name:m.l,x:xl,y:sorted.map(x=>x[m.k]!=null?x[m.k]:0),
    marker:{color:m.c,opacity:.9},hovertemplate:`<b>%{x}</b><br>${m.l}: %{y:.3f}<extra></extra>`}));
  const aucs=r.flatMap(x=>[x.baseline_auc,x.test_time_aupmc,x.best_fixed_train_aupmc]).filter(v=>v!=null);
  const ymin=Math.max(Math.min(...aucs)-.03,.3),ymax=Math.min(Math.max(...aucs)+.02,.9);
  const layout={...th,barmode:'group',height:265,margin:{t:14,b:72,l:58,r:12},
    xaxis:{...th.xaxis,tickangle:-30,tickfont:{size:12.5}},
    yaxis:{...th.yaxis,title:{text:'Score',font:{size:12}},range:[ymin,ymax]},
    showlegend:true,legend:{orientation:'h',x:0,y:-0.52,font:{size:12},bgcolor:'rgba(0,0,0,0)'}};
  Plotly.react('ch-br',traces,layout,CFG);
}

// ── CONDITIONS ─────────────────────────────────────────────────────────────
function rConds(){buildGrid();rWilcoxon(S.cond.tr,S.cond.te);}

function topMethod(tp,mp){
  const ma=D().mean_auc||[];
  const c=ma.filter(r=>Math.abs(r.train_missing_prop-tp)<.001&&Math.abs(r.test_missing_prop-mp)<.001)
    .sort((a,b)=>b.mean_auc-a.mean_auc);
  return c.length?c[0]:null;
}

function buildGrid(){
  const ma=D().mean_auc||[];
  if(!ma.length){document.getElementById('cgc').innerHTML='<div class="es"><div class="es-i">🔍</div><div>No data</div></div>';return;}
  const tps=[...new Set(ma.map(r=>r.train_missing_prop))].sort((a,b)=>a-b);
  const mps=[...new Set(ma.map(r=>r.test_missing_prop))].sort((a,b)=>a-b);

  let html=`<div class="cg" style="grid-template-columns:66px ${'1fr '.repeat(mps.length).trim()}">`;
  html+=`<div class="cg-rh" style="text-align:center;padding:4px 0;font-size:9px;color:var(--t3)">Train↓ Test→</div>`;
  mps.forEach(mp=>`<div class="cg-ch" style="padding:4px 2px">${pct(mp)}</div>`).forEach(s=>html+=s);
  tps.forEach(tp=>{
    html+=`<div class="cg-rh">${pct(tp)}</div>`;
    mps.forEach(mp=>{
      const top=topMethod(tp,mp);
      const color=top?(MC[top.model_name]||'#888'):'rgba(120,120,120,.3)';
      const label=top?(MD[top.model_name]||top.model_name):'–';
      const auc=top?top.mean_auc.toFixed(3):'';
      const[ri,gi,bi]=top?hexRgb(color):[120,120,120];
      html+=`<div class="cg-cell"
        style="background:rgba(${ri},${gi},${bi},.12);border-color:rgba(${ri},${gi},${bi},.3)"
        onclick="drillCond(${tp},${mp})"
        onmouseover="this.style.background='rgba(${ri},${gi},${bi},.25)'"
        onmouseout="this.style.background='rgba(${ri},${gi},${bi},.12)'">
        <div class="cg-mn" style="color:${color}">${label}</div>
        <div class="cg-av">${auc}</div>
      </div>`;
    });
  });
  html+='</div>';
  document.getElementById('cgc').innerHTML=html;
}

function drillCond(tp,mp){
  S.cond={tr:tp,te:mp};
  document.getElementById('clbl').textContent=`Train ${pct(tp)} · Test ${pct(mp)}`;
  rWilcoxon(tp,mp);
}

function rWilcoxon(tp,mp){
  const wlx=D().wilcoxon||[]; const ma=D().mean_auc||[];
  const th=ptBase(), d=dk();
  const dl=metricDeltaLabel();
  const cond=ma.filter(r=>Math.abs(r.train_missing_prop-tp)<.001&&Math.abs(r.test_missing_prop-mp)<.001)
    .sort((a,b)=>b.mean_auc-a.mean_auc);
  const condW=wlx.filter(r=>Math.abs(r.train_missing_prop-tp)<.001&&Math.abs(r.test_missing_prop-mp)<.001);
  if(!cond.length)return;
  const meths=cond.map(r=>r.model_name);
  const z=meths.map(()=>meths.map(()=>null));
  const txt=meths.map(()=>meths.map(()=>'ns'));
  meths.forEach((mr,i)=>meths.forEach((mc,j)=>{
    if(i===j){z[i][j]=0;txt[i][j]='—';return;}
    const w=condW.find(x=>x.winner_model===mr&&x.loser_model===mc&&x.significant_fdr_0p05);
    if(w){z[i][j]=w.delta_mean_auc;txt[i][j]=`+${w.delta_mean_auc.toFixed(3)}`;}
  }));
  const yl=cond.map(r=>`${MD[r.model_name]||r.model_name} (${r.mean_auc.toFixed(3)})`);
  const xl=meths.map(m=>MD[m]||m);
  const cs=d?[[0,'rgba(30,0,60,.85)'],[.4,'rgba(80,20,200,.85)'],[1,'rgba(0,210,255,.9)']]
            :[[0,'rgba(255,230,255,.95)'],[.4,'rgba(160,100,255,.95)'],[1,'rgba(0,100,200,.95)']];
  Plotly.react('ch-wl',[{
    type:'heatmap',z,x:xl,y:yl,text:txt,texttemplate:'%{text}',textfont:{size:10},
    colorscale:cs,showscale:true,hoverinfo:'skip',
    colorbar:{title:{text:dl,font:{size:12}},thickness:10,tickfont:{size:11.5},len:.85},
  }],{...th,
    height:360,margin:{t:14,b:68,l:150,r:28},
    xaxis:{...th.xaxis,title:{text:'Lower-ranked',font:{size:12}},tickangle:-25,tickfont:{size:12.5}},
    yaxis:{...th.yaxis,title:{text:'Higher-ranked',font:{size:12}},tickfont:{size:12.5},autorange:'reversed'},
    annotations:[{x:.5,y:-0.26,xref:'paper',yref:'paper',showarrow:false,xanchor:'center',
      text:`Coloured cells = significant win (Wilcoxon, FDR p<0.05). Value = ${dl}. "ns" = not significant.`,
      font:{size:11.5,color:d?'rgba(255,255,255,.3)':'rgba(0,0,0,.35)'}}],
  },CFGs);
}

// ── SUMMARY ───────────────────────────────────────────────────────────────
function rSummary(){rXCohort();rRankTable();rTopGroup();}

const DSC=['#00d4ff','#a855f7','#ff2d78'];

function rXCohort(){
  const dks=Object.keys(curMod()||{}); if(!dks.length)return;
  const th=ptBase();
  const ml=metricLabel(curMod()[dks[0]]||null);
  const mets=[{k:'best_fixed_train_aupmc',l:'BFT AUPMC'},{k:'test_time_aupmc',l:'Test AUPMC'},{k:'baseline_auc',l:`Baseline ${ml}`}];
  const traces=dks.flatMap((dk2,di)=>{
    const r=(curMod()[dk2]||{}).metrics||[]; const meta=(curMod()[dk2]||{}).meta||{};
    const sorted=[...r].sort((a,b)=>(b.best_fixed_train_aupmc||0)-(a.best_fixed_train_aupmc||0));
    const c=DSC[di%3]; const[ri,gi,bi]=hexRgb(c);
    return mets.map((m,mi)=>({
      type:'bar',name:`${meta.name||dk2} · ${m.l}`,legendgroup:meta.name||dk2,
      x:sorted.map(x=>MD[x.model_name]||x.model_name),y:sorted.map(x=>x[m.k]!=null?x[m.k]:0),
      marker:{color:`rgba(${ri},${gi},${bi},${.9-mi*.28})`},
      hovertemplate:`<b>%{x}</b><br>${meta.name||dk2} ${m.l}: %{y:.3f}<extra></extra>`,
    }));
  });
  const all=dks.flatMap(k=>((curMod()[k]||{}).metrics||[]).flatMap(x=>['best_fixed_train_aupmc','test_time_aupmc','baseline_auc'].map(m=>x[m]).filter(v=>v!=null)));
  const ymin=all.length?Math.max(Math.min(...all)-.04,.25):.45;
  const ymax=all.length?Math.min(Math.max(...all)+.02,.9):.75;
  Plotly.react('ch-xc',traces,{...th,barmode:'group',height:280,
    margin:{t:14,b:68,l:52,r:12},
    xaxis:{...th.xaxis,tickangle:-25,tickfont:{size:12.5}},
    yaxis:{...th.yaxis,title:{text:'Score',font:{size:12}},range:[ymin,ymax]},
    showlegend:true,legend:{orientation:'h',x:0,y:-0.52,font:{size:12},bgcolor:'rgba(0,0,0,0)'},
  },CFG);
}

function rRankTable(){
  const dks=Object.keys(curMod()||{}); if(!dks.length)return;
  const met='best_fixed_train_aupmc';
  const allM=[...new Set(dks.flatMap(k=>((curMod()[k]||{}).metrics||[]).map(r=>r.model_name)))].sort();
  const ranks={};
  dks.forEach(dk2=>{
    const sorted=[...((curMod()[dk2]||{}).metrics||[])].filter(r=>r[met]!=null).sort((a,b)=>b[met]-a[met]);
    sorted.forEach((r,i)=>{if(!ranks[r.model_name])ranks[r.model_name]={};ranks[r.model_name][dk2]=i+1;});
  });
  allM.forEach(m=>{const v=dks.map(k=>ranks[m]?.[k]||allM.length);ranks[m].avg=v.reduce((a,b)=>a+b,0)/v.length;});
  const sorted=[...allM].sort((a,b)=>(ranks[a]?.avg||99)-(ranks[b]?.avg||99));
  const medal=r=>r===1?'🥇':r===2?'🥈':r===3?'🥉':`#${r}`;
  const ths=['Method',...dks.map(k=>(curMod()[k]||{}).meta?.name||k),'Avg'].map(t=>`<th>${t}</th>`).join('');
  const trs=sorted.map(m=>`<tr><td><span style="color:${MC[m]||'#aaa'};font-weight:700">${MD[m]||m}</span></td>
    ${dks.map(k=>{const r=ranks[m]?.[k];return`<td class="${r===1?'best':''}">${r?medal(r):'–'}</td>`;}).join('')}
    <td style="font-weight:600">${ranks[m]?.avg?.toFixed(1)||'–'}</td></tr>`).join('');
  document.getElementById('tbl-rk').innerHTML=`<table class="mt"><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
}

function rTopGroup(){
  const dks=Object.keys(curMod()||{}); if(!dks.length)return;
  const th=ptBase();
  const allM=[...new Set(dks.flatMap(k=>((curMod()[k]||{}).top_counts||[]).map(r=>r.model_name)))].sort();
  const traces=dks.map((dk2,di)=>{
    const r=(curMod()[dk2]||{}).top_counts||[]; const meta=(curMod()[dk2]||{}).meta||{};
    return{type:'bar',name:meta.name||dk2,x:allM.map(m=>MD[m]||m),
      y:allM.map(m=>{const x=r.find(v=>v.model_name===m);return x?x.top_equivalent_group_fraction:0;}),
      marker:{color:DSC[di%3]},hovertemplate:`<b>%{x}</b><br>${meta.name||dk2}: %{y:.0%}<extra></extra>`};
  });
  Plotly.react('ch-tg',traces,{...th,barmode:'group',height:250,
    margin:{t:14,b:62,l:50,r:12},
    xaxis:{...th.xaxis,tickangle:-28,tickfont:{size:12.5}},
    yaxis:{...th.yaxis,title:{text:'Fraction in top group',font:{size:12}},tickformat:',.0%'},
    showlegend:true,legend:{orientation:'h',x:0,y:-0.52,font:{size:12},bgcolor:'rgba(0,0,0,0)'},
  },CFG);
}

// ── INIT ──────────────────────────────────────────────────────────────────
function init(){
  applyDashboardScale();
  // Build replicate-definition dropdown
  const repMenu=document.getElementById('dd-rep-menu');
  AVAIL_MODES.forEach(({key,label})=>{
    const it=document.createElement('div');
    it.className='dd-item'+(key===S.mode?' on':'');
    it.dataset.v=key; it.textContent=label;
    it.onclick=()=>{switchMode(key);toggleDd('rep');};
    repMenu.appendChild(it);
  });

  // Build degrading-modality dropdown
  const modMenu=document.getElementById('dd-mod-menu');
  AVAIL_MODALITIES.forEach(mod=>{
    const it=document.createElement('div');
    it.className='dd-item'+(mod===S.modality?' on':'');
    it.dataset.v=mod; it.textContent=MODALITY_LABELS[mod]||mod;
    it.onclick=()=>{switchModality(mod);toggleDd('mod');};
    modMenu.appendChild(it);
  });

  initTrajectoryResize();
  initHpResize();
  initTrajectoryWidgetDrag();
  initTrajectoryResizeObserver();
  syncTrajectoryWidgets();
  rebuildCohorts('progressive');  // sets S.ds first
  updateDdLabels();               // now S.ds is populated, ep label resolves correctly
  updateStaticNavVisibility();
  if(S.ds) updateMeta();
  nav('global','progressive');
}

document.addEventListener('DOMContentLoaded',init);

// Hover labels use native Plotly geometry. Do not mutate SVG paths.
</script>
</body>
</html>"""

if __name__ == '__main__':
    main()
