#!/usr/bin/env python3
"""Dataset fingerprinting and initial HP-grid suggestion for M3TRICS.

This script inspects endpoint and modality CSV files and proposes a small,
conservative hyperparameter search space for each requested method. The goal is
not to replace full optimisation, but to provide an nnU-Net-like initial grid
that is adapted to sample size, feature dimensionality, missingness, modality
coverage, and class/event balance.

The suggested grids are capped to ``--max_combinations`` combinations per method
(default: 32). Outputs are JSON by default and can optionally be saved to disk.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BC_DEFAULT_MODELS = ["ZI_LR", "KNN_LR", "ZI_RF", "KNN_RF", "ZI_MLP", "KNN_MLP", "pAM"]
SURV_DEFAULT_MODELS = ["ZI_CoxNet", "KNN_CoxNet", "ZI_RSF", "KNN_RSF", "ZI_MLP", "KNN_MLP", "HealNet"]

MODEL_ALIASES = {
    "zilr": "ZI_LR",
    "zi_lr": "ZI_LR",
    "knnlr": "KNN_LR",
    "knn_lr": "KNN_LR",
    "zirf": "ZI_RF",
    "zi_rf": "ZI_RF",
    "knnrf": "KNN_RF",
    "knn_rf": "KNN_RF",
    "zicoxnet": "ZI_CoxNet",
    "zi_coxnet": "ZI_CoxNet",
    "knncoxnet": "KNN_CoxNet",
    "knn_coxnet": "KNN_CoxNet",
    "zirsf": "ZI_RSF",
    "zi_rsf": "ZI_RSF",
    "knnrsf": "KNN_RSF",
    "knn_rsf": "KNN_RSF",
    "zimlp": "ZI_MLP",
    "zi_mlp": "ZI_MLP",
    "knnmlp": "KNN_MLP",
    "knn_mlp": "KNN_MLP",
    "vaemlp": "VAE_MLP",
    "vae_mlp": "VAE_MLP",
    "pam": "pAM",
    "p-am": "pAM",
    "healnet": "HealNet",
    "smile": "SMILe",
    "smilee": "SMILe",
}

# Lower priority parameters are reduced first when the raw grid exceeds the cap.
REDUCTION_PRIORITY = {
    "LR": ["lr_C", "lr_class_weight"],
    "RF": ["rf_max_depth", "rf_min_samples_leaf", "rf_class_weight", "rf_n_estimators"],
    "CoxNet": ["coxnet_alpha", "coxnet_l1_ratio"],
    "RSF": ["rsf_max_depth", "rsf_min_samples_leaf", "rsf_n_estimators"],
    "MLP": ["learning_rate", "batch_size", "fusion_hidden_dim", "dropout", "fusion_batchnorm"],
    "pAM": ["learning_rate", "batch_size", "pam_dropout", "pam_temperature"],
    "HealNet": ["learning_rate", "batch_size", "healnet_depth", "healnet_num_latents", "healnet_latent_dim"],
    "SMILe": ["learning_rate", "batch_size", "smil_e_latent_dim", "classifier_hidden_dim", "smil_e_alpha", "meta_inner_lr"],
}


@dataclass
class ModalityFingerprint:
    name: str
    path: str
    rows: int
    unique_patients: int
    duplicated_patient_rows: int
    coverage_in_target_patients: float
    n_features: int
    n_numeric_features: int
    n_categorical_features: int
    feature_missing_fraction: float
    median_feature_missing_fraction: float
    high_missing_feature_fraction: float
    constant_feature_fraction: float


@dataclass
class DatasetFingerprint:
    n_target_rows: int
    n_target_patients: int
    task_type: str
    n_modalities: int
    total_feature_dim: int
    feature_dim_to_n_ratio: float
    min_modality_coverage: float
    mean_modality_coverage: float
    mean_feature_missing_fraction: float
    class_positive_rate: Optional[float] = None
    class_minority_rate: Optional[float] = None
    survival_event_rate: Optional[float] = None
    survival_time_median: Optional[float] = None


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _parse_name_value(items: Optional[Sequence[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected name=value format, got: {item}")
        name, value = item.split("=", 1)
        out[name.strip()] = value.strip()
    return out


def _resolve_path(path: str, dataset_dir: Optional[Path]) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    if dataset_dir is not None:
        return (dataset_dir / p).resolve()
    return p.resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _normalise_model_name(model: str) -> str:
    key = str(model).strip().lower().replace(" ", "").replace("-", "_")
    return MODEL_ALIASES.get(key, model.strip())


def _method_family(method: str) -> str:
    m = _normalise_model_name(method)
    if m.endswith("LR"):
        return "LR"
    if m.endswith("RF"):
        return "RF"
    if m.endswith("CoxNet"):
        return "CoxNet"
    if m.endswith("RSF"):
        return "RSF"
    if m in {"ZI_MLP", "KNN_MLP", "VAE_MLP"}:
        return "MLP"
    if m == "pAM":
        return "pAM"
    if m == "HealNet":
        return "HealNet"
    if m == "SMILe":
        return "SMILe"
    return "MLP"


def _parse_models(value: Optional[str], task_type: str) -> List[str]:
    if value:
        models = [_normalise_model_name(x) for x in _split_csv(value)]
    else:
        models = BC_DEFAULT_MODELS if task_type == "binary_classification" else SURV_DEFAULT_MODELS
    return list(dict.fromkeys(models))


def _target_mask(endpoint_df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.task_type == "binary_classification":
        if args.endpoint_col not in endpoint_df.columns:
            raise ValueError(f"Endpoint column '{args.endpoint_col}' not found in endpoint CSV.")
        return endpoint_df[args.endpoint_col].notna()

    missing = [c for c in [args.survival_time_col, args.survival_event_col] if c not in endpoint_df.columns]
    if missing:
        raise ValueError(f"Survival columns not found in endpoint CSV: {missing}")
    return endpoint_df[args.survival_time_col].notna() & endpoint_df[args.survival_event_col].notna()


def _numeric_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
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
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


def _modality_fingerprint(
    name: str,
    path: Path,
    patient_id_col: str,
    target_patients: set,
    drop_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> ModalityFingerprint:
    df = _read_csv(path)
    if patient_id_col not in df.columns:
        raise ValueError(f"Patient ID column '{patient_id_col}' not found in modality '{name}' ({path}).")

    patient_ids = df[patient_id_col].dropna().astype(str)
    unique_patients = int(patient_ids.nunique())
    duplicated_rows = int(max(len(patient_ids) - unique_patients, 0))
    coverage = 0.0 if not target_patients else len(set(patient_ids) & target_patients) / len(target_patients)

    drops = set(drop_cols) | {patient_id_col}
    feature_cols = [c for c in df.columns if c not in drops]
    feat = df[feature_cols] if feature_cols else pd.DataFrame(index=df.index)

    if categorical_cols:
        cat_cols = [c for c in categorical_cols if c in feat.columns]
    else:
        cat_cols = [c for c in feat.columns if not pd.api.types.is_numeric_dtype(feat[c])]
    num_cols = [c for c in feat.columns if c not in set(cat_cols)]

    if len(feature_cols):
        feature_missing = feat.isna().mean(axis=0)
        missing_fraction = float(feat.isna().to_numpy().mean())
        median_missing = float(feature_missing.median())
        high_missing = float((feature_missing > 0.30).mean())
        constant_fraction = float((feat.nunique(dropna=True) <= 1).mean())
    else:
        missing_fraction = 0.0
        median_missing = 0.0
        high_missing = 0.0
        constant_fraction = 0.0

    return ModalityFingerprint(
        name=name,
        path=str(path),
        rows=int(len(df)),
        unique_patients=unique_patients,
        duplicated_patient_rows=duplicated_rows,
        coverage_in_target_patients=float(coverage),
        n_features=int(len(feature_cols)),
        n_numeric_features=int(len(num_cols)),
        n_categorical_features=int(len(cat_cols)),
        feature_missing_fraction=missing_fraction,
        median_feature_missing_fraction=median_missing,
        high_missing_feature_fraction=high_missing,
        constant_feature_fraction=constant_fraction,
    )


def build_fingerprint(args: argparse.Namespace) -> Tuple[DatasetFingerprint, List[ModalityFingerprint], List[str]]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None
    endpoint_path = _resolve_path(args.endpoint_csv, dataset_dir)
    endpoint_df = _read_csv(endpoint_path)

    if args.patient_id_col not in endpoint_df.columns:
        raise ValueError(f"Patient ID column '{args.patient_id_col}' not found in endpoint CSV.")

    valid_mask = _target_mask(endpoint_df, args)
    target_df = endpoint_df.loc[valid_mask].copy()
    target_patients = set(target_df[args.patient_id_col].dropna().astype(str))

    modality_specs = _parse_name_value(args.modality_csv)
    if not modality_specs and args.auto_modalities:
        if dataset_dir is None:
            raise ValueError("--auto_modalities requires --dataset_dir.")
        endpoint_resolved = endpoint_path.resolve()
        for csv_path in sorted(dataset_dir.glob("*.csv")):
            if csv_path.resolve() == endpoint_resolved:
                continue
            modality_specs[csv_path.stem] = str(csv_path)
    if not modality_specs:
        raise ValueError("No modalities provided. Use --modality_csv name=path or --auto_modalities.")

    drop_specs = {k: _split_csv(v) for k, v in _parse_name_value(args.drop_cols).items()}
    cat_specs = {k: _split_csv(v) for k, v in _parse_name_value(args.categorical_cols).items()}

    modalities = []
    warnings = []
    for name, path_str in modality_specs.items():
        path = _resolve_path(path_str, dataset_dir)
        fp = _modality_fingerprint(
            name=name,
            path=path,
            patient_id_col=args.patient_id_col,
            target_patients=target_patients,
            drop_cols=drop_specs.get(name, []),
            categorical_cols=cat_specs.get(name, []),
        )
        modalities.append(fp)
        if fp.coverage_in_target_patients < 0.6:
            warnings.append(f"Modality '{name}' covers only {fp.coverage_in_target_patients:.1%} of target patients.")
        if fp.high_missing_feature_fraction > 0.3:
            warnings.append(f"Modality '{name}' has many high-missingness features ({fp.high_missing_feature_fraction:.1%}).")

    n_patients = int(len(target_patients))
    total_dim = int(sum(m.n_features for m in modalities))
    coverages = [m.coverage_in_target_patients for m in modalities]
    missings = [m.feature_missing_fraction for m in modalities]

    class_positive_rate = None
    class_minority_rate = None
    event_rate = None
    time_median = None
    if args.task_type == "binary_classification":
        y = _numeric_binary(target_df[args.endpoint_col]).dropna()
        if len(y):
            class_positive_rate = float((y == 1).mean())
            class_minority_rate = float(min(class_positive_rate, 1.0 - class_positive_rate))
            if class_minority_rate < 0.2:
                warnings.append(f"Endpoint is imbalanced: minority rate {class_minority_rate:.1%}.")
    else:
        event = _numeric_binary(target_df[args.survival_event_col]).dropna()
        times = pd.to_numeric(target_df[args.survival_time_col], errors="coerce").dropna()
        if len(event):
            event_rate = float((event == 1).mean())
            if event_rate < 0.25:
                warnings.append(f"Low event rate for survival task: {event_rate:.1%}.")
        if len(times):
            time_median = float(times.median())

    fp = DatasetFingerprint(
        n_target_rows=int(len(target_df)),
        n_target_patients=n_patients,
        task_type=args.task_type,
        n_modalities=len(modalities),
        total_feature_dim=total_dim,
        feature_dim_to_n_ratio=float(total_dim / max(n_patients, 1)),
        min_modality_coverage=float(min(coverages)) if coverages else 0.0,
        mean_modality_coverage=float(np.mean(coverages)) if coverages else 0.0,
        mean_feature_missing_fraction=float(np.mean(missings)) if missings else 0.0,
        class_positive_rate=class_positive_rate,
        class_minority_rate=class_minority_rate,
        survival_event_rate=event_rate,
        survival_time_median=time_median,
    )
    return fp, modalities, warnings


def _comma(values: Sequence[Any]) -> str:
    return ",".join(str(v).lower() if isinstance(v, bool) else str(v) for v in values)


def _grid_count(args: Mapping[str, str], ignore: Iterable[str] = ()) -> int:
    ignored = set(ignore)
    count = 1
    for key, value in args.items():
        if key in ignored:
            continue
        n = len(_split_csv(value))
        count *= max(n, 1)
    return int(count)


def _cap_grid(args: MutableMapping[str, str], family: str, max_combinations: int, fixed: Iterable[str] = ()) -> Tuple[Dict[str, str], List[str]]:
    out = dict(args)
    fixed_set = set(fixed)
    notes = []
    priority = REDUCTION_PRIORITY.get(family, list(out.keys()))
    while _grid_count(out, ignore=fixed_set) > max_combinations:
        reduced = False
        for key in priority:
            if key in fixed_set or key not in out:
                continue
            vals = _split_csv(out[key])
            if len(vals) > 1:
                # Keep central/default-ish values first for conservative initial search.
                keep_idx = len(vals) // 2
                if key in {"learning_rate", "lr_C", "coxnet_alpha"}:
                    keep_idx = min(1, len(vals) - 1)
                out[key] = vals[keep_idx]
                notes.append(f"Reduced {key} to {out[key]} to keep grid <= {max_combinations} combinations.")
                reduced = True
                break
        if not reduced:
            break
    return out, notes


def _base_context(fp: DatasetFingerprint) -> Dict[str, bool]:
    n = fp.n_target_patients
    p_over_n = fp.feature_dim_to_n_ratio
    return {
        "tiny_n": n < 80,
        "small_n": n < 160,
        "medium_n": 160 <= n < 500,
        "large_n": n >= 500,
        "high_dim": p_over_n > 10,
        "moderate_dim": 3 < p_over_n <= 10,
        "low_dim": p_over_n <= 3,
        "imbalanced": (fp.class_minority_rate is not None and fp.class_minority_rate < 0.35)
        or (fp.survival_event_rate is not None and fp.survival_event_rate < 0.35),
        "missing_or_sparse": fp.mean_feature_missing_fraction > 0.05 or fp.min_modality_coverage < 0.85,
    }


def suggest_grid_for_method(method: str, fp: DatasetFingerprint, max_combinations: int) -> Dict[str, Any]:
    method = _normalise_model_name(method)
    family = _method_family(method)
    ctx = _base_context(fp)
    rationale: List[str] = []

    if ctx["tiny_n"]:
        batch_sizes = [8]
        lrs = ["1e-4", "5e-5"]
        hidden_dims = [16, 32]
        dropout = [0.2, 0.3]
        rationale.append("Tiny cohort: conservative batch size and compact neural widths.")
    elif ctx["small_n"]:
        batch_sizes = [8, 16]
        lrs = ["1e-4", "5e-5"]
        hidden_dims = [16, 32]
        dropout = [0.1, 0.2]
        rationale.append("Small cohort: compact grid to reduce split-driven overfitting.")
    elif ctx["medium_n"]:
        batch_sizes = [16, 32]
        lrs = ["1e-4", "5e-5", "1e-5"]
        hidden_dims = [32, 64]
        dropout = [0.1, 0.2]
        rationale.append("Medium cohort: moderate width and learning-rate search.")
    else:
        batch_sizes = [32, 64]
        lrs = ["1e-4", "5e-5", "1e-5"]
        hidden_dims = [64, 128]
        dropout = [0.1, 0.2]
        rationale.append("Larger cohort: allows wider representations and larger batches.")

    if ctx["high_dim"]:
        hidden_dims = [min(hidden_dims), max(min(hidden_dims) * 2, min(hidden_dims))]
        dropout = sorted(set(dropout + [0.3]))[-2:]
        rationale.append("High feature-dimension-to-sample ratio: favour compact models and stronger dropout.")

    class_weight_values = ["none", "balanced"] if ctx["imbalanced"] else ["none"]
    if ctx["imbalanced"]:
        rationale.append("Imbalanced endpoint/event distribution: include class-balanced sklearn baselines.")

    if family == "LR":
        fixed_args = {
            "epochs": "1",
            "early_stopping_patience": "1",
            "batch_size": "64",
            "learning_rate": "1.0",
            "weight_decay": "0.0",
            "imputation_method": "knn" if method.startswith("KNN") else "zero",
        }
        args = {
            "lr_C": "0.01,0.1,1.0" if ctx["high_dim"] else "0.1,1.0,10.0",
            "lr_penalty": "l2",
            "lr_solver": "lbfgs",
            "lr_class_weight": _comma(class_weight_values),
            "lr_max_iter": "1000",
        }
    elif family == "RF":
        fixed_args = {
            "epochs": "1",
            "early_stopping_patience": "1",
            "batch_size": "64",
            "learning_rate": "1.0",
            "weight_decay": "0.0",
            "imputation_method": "knn" if method.startswith("KNN") else "zero",
        }
        args = {
            "rf_n_estimators": "200,500" if not ctx["tiny_n"] else "200",
            "rf_max_depth": "3,5,none" if ctx["small_n"] or ctx["high_dim"] else "5,10,none",
            "rf_min_samples_split": "2",
            "rf_min_samples_leaf": "1,3" if not ctx["tiny_n"] else "3",
            "rf_max_features": "sqrt",
            "rf_class_weight": _comma(class_weight_values),
            "rf_n_jobs": "-1",
        }
    elif family == "CoxNet":
        fixed_args = {
            "epochs": "1",
            "early_stopping_patience": "1",
            "batch_size": "64",
            "learning_rate": "1.0",
            "weight_decay": "0.0",
            "imputation_method": "knn" if method.startswith("KNN") else "zero",
        }
        args = {
            "coxnet_alpha": "0.01,0.1,1.0" if ctx["high_dim"] else "0.001,0.01,0.1",
            "coxnet_l1_ratio": "0.1,0.5" if ctx["high_dim"] else "0.1,0.5,0.9",
            "coxnet_max_iter": "100000",
            "coxnet_tol": "1e-7",
        }
    elif family == "RSF":
        fixed_args = {
            "epochs": "1",
            "early_stopping_patience": "1",
            "batch_size": "64",
            "learning_rate": "1.0",
            "weight_decay": "0.0",
            "imputation_method": "knn" if method.startswith("KNN") else "zero",
        }
        args = {
            "rsf_n_estimators": "100,300" if not ctx["tiny_n"] else "100",
            "rsf_max_depth": "3,5" if ctx["small_n"] or ctx["high_dim"] else "5,none",
            "rsf_min_samples_split": "6",
            "rsf_min_samples_leaf": "3,5",
            "rsf_max_features": "sqrt",
            "rsf_n_jobs": "-1",
        }
    elif family == "pAM":
        fixed_args = {}
        args = {
            "epochs": "80",
            "early_stopping_patience": "20",
            "batch_size": _comma(batch_sizes),
            "learning_rate": _comma(lrs),
            "weight_decay": "1e-4",
            "pam_dropout": _comma(dropout),
            "pam_temperature": "1.0,2.0",
        }
    elif family == "HealNet":
        latents = [8, 16] if ctx["small_n"] or ctx["high_dim"] else [16, 32]
        fixed_args = {}
        args = {
            "epochs": "100",
            "early_stopping_patience": "20",
            "batch_size": _comma(batch_sizes[:2]),
            "learning_rate": _comma(lrs[:2]),
            "weight_decay": "1e-4",
            "healnet_depth": "1,2" if not ctx["tiny_n"] else "1",
            "healnet_num_freq_bands": "2",
            "healnet_num_latents": _comma(latents),
            "healnet_latent_dim": _comma(latents),
            "healnet_cross_heads": "1",
            "healnet_latent_heads": "2",
            "healnet_cross_dim_head": "8",
            "healnet_latent_dim_head": "8",
            "healnet_attn_dropout": "0.2",
            "healnet_ff_dropout": "0.2",
            "healnet_self_per_cross_attn": "0",
            "paired_hp_groups": "healnet_num_latents:healnet_latent_dim",
        }
    elif family == "SMILe":
        latent = [8, 16] if ctx["small_n"] or ctx["high_dim"] else [16, 32]
        fixed_args = {}
        args = {
            "epochs": "80",
            "early_stopping_patience": "20",
            "batch_size": _comma(batch_sizes[:2]),
            "learning_rate": "5e-5,1e-5",
            "weight_decay": "1e-4",
            "smil_e_latent_dim": _comma(latent),
            "smil_e_num_priors": "2",
            "smil_e_num_heads": "1",
            "smil_e_dropout": "0.2",
            "smil_e_alpha": "1e-3,1e-2",
            "smil_e_beta": "1e-2",
            "meta_inner_lr": "5e-4,5e-5",
            "meta_val_fraction": "0.25",
            "classifier_hidden_dim": _comma(hidden_dims[:2]),
            "paired_hp_groups": "learning_rate:meta_inner_lr",
        }
    else:  # MLP family
        fixed_args = {}
        args = {
            "epochs": "80",
            "early_stopping_patience": "20",
            "batch_size": _comma(batch_sizes),
            "learning_rate": _comma(lrs),
            "weight_decay": "1e-4",
            "fusion_hidden_dim": _comma(hidden_dims),
            "fusion_hidden_layers": "1",
            "fusion_batchnorm": "false,true" if not ctx["tiny_n"] and not ctx["high_dim"] else "false",
            "modality_hidden_layers": "1",
            "dropout": _comma(dropout),
            "imputation_method": "vae" if method == "VAE_MLP" else ("knn" if method.startswith("KNN") else "zero"),
        }

    fixed = {"epochs", "early_stopping_patience", "imputation_method", "paired_hp_groups"}
    capped, cap_notes = _cap_grid(args, family, max_combinations, fixed=fixed)
    count = _grid_count(capped, ignore=fixed)

    return {
        "family": family,
        "combination_count": count,
        "fixed_args": fixed_args,
        "args": capped,
        "rationale": rationale + cap_notes,
    }


def build_recommendations(fp: DatasetFingerprint, modalities: Sequence[ModalityFingerprint]) -> List[str]:
    recs = []
    if fp.n_target_patients < 100:
        recs.append("Prioritise simple baselines and compact neural grids; nested-CV variance may dominate small differences.")
    if fp.feature_dim_to_n_ratio > 10:
        recs.append("Feature dimensionality is high relative to N; avoid large hidden dimensions in the first search.")
    if fp.min_modality_coverage < 0.8:
        recs.append("Natural modality incompleteness is substantial; include KNN/imputation and missing-aware architectures.")
    if fp.class_minority_rate is not None and fp.class_minority_rate < 0.25:
        recs.append("Binary endpoint is imbalanced; include balanced LR/RF variants and inspect AUCPR, not only AUC.")
    if fp.survival_event_rate is not None and fp.survival_event_rate < 0.25:
        recs.append("Survival event rate is low; keep survival grids conservative and inspect C-index uncertainty.")
    if any(m.duplicated_patient_rows > 0 for m in modalities):
        recs.append("At least one modality has duplicated patient rows; define an aggregation method explicitly.")
    if not recs:
        recs.append("Dataset fingerprint does not flag major constraints; the default compact grid is appropriate.")
    return recs


def shell_snippet(suggested_grids: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "# M3TRICS fingerprint suggestions",
        "# Copy the block for one method into a method-specific hyperparams config or pass args manually.",
    ]
    for method, spec in suggested_grids.items():
        lines.append("")
        lines.append(f"# {method} ({spec['combination_count']} combinations)")
        for key, value in spec["args"].items():
            env_key = key.upper()
            lines.append(f'{env_key}="{value}"')
    return "\n".join(lines) + "\n"


def build_output(args: argparse.Namespace) -> Dict[str, Any]:
    fp, modalities, warnings = build_fingerprint(args)
    models = _parse_models(args.run_models, args.task_type)
    grids = {m: suggest_grid_for_method(m, fp, args.max_combinations) for m in models}
    return {
        "dataset_fingerprint": asdict(fp),
        "modality_fingerprints": [asdict(m) for m in modalities],
        "warnings": warnings,
        "recommendations": build_recommendations(fp, modalities),
        "max_combinations_per_method": int(args.max_combinations),
        "suggested_grids": grids,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute an M3TRICS dataset fingerprint and suggest an initial HP search space."
    )
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory containing dataset CSV files.")
    parser.add_argument("--endpoint_csv", type=str, required=True, help="Endpoint CSV path or filename relative to --dataset_dir.")
    parser.add_argument("--patient_id_col", type=str, required=True, help="Shared patient identifier column.")
    parser.add_argument("--endpoint_col", type=str, default=None, help="Binary endpoint column for binary_classification.")
    parser.add_argument("--task_type", type=str, default="binary_classification", choices=["binary_classification", "survival"])
    parser.add_argument("--survival_time_col", type=str, default=None)
    parser.add_argument("--survival_event_col", type=str, default=None)
    parser.add_argument("--modality_csv", action="append", default=[], help="Repeated name=csv_path modality specification.")
    parser.add_argument("--drop_cols", action="append", default=[], help="Repeated modality=col1,col2 drop-column specification.")
    parser.add_argument("--categorical_cols", action="append", default=[], help="Repeated modality=col1,col2 categorical-column specification.")
    parser.add_argument("--auto_modalities", action="store_true", help="Use all CSVs in --dataset_dir except endpoint CSV as modalities.")
    parser.add_argument("--run_models", type=str, default=None, help="Comma-separated M3TRICS methods to suggest grids for.")
    parser.add_argument("--max_combinations", type=int, default=32, help="Maximum HP combinations per method.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save JSON output.")
    parser.add_argument("--output_shell", type=str, default=None, help="Optional path to save a shell-style snippet.")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON instead of pretty JSON.")

    args = parser.parse_args(argv)
    if args.task_type == "binary_classification" and not args.endpoint_col:
        parser.error("--endpoint_col is required for binary_classification.")
    if args.task_type == "survival" and (not args.survival_time_col or not args.survival_event_col):
        parser.error("--survival_time_col and --survival_event_col are required for survival.")
    if args.max_combinations < 1:
        parser.error("--max_combinations must be >= 1.")

    output = build_output(args)
    json_text = json.dumps(output, indent=None if args.compact else 2, sort_keys=False)
    print(json_text)

    if args.output_json:
        Path(args.output_json).expanduser().write_text(json_text + "\n")
    if args.output_shell:
        Path(args.output_shell).expanduser().write_text(shell_snippet(output["suggested_grids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
