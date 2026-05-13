import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from utils import (
    SURVIVAL_CENSORSHIP_COL,
    SURVIVAL_Y_DISC_COL,
    add_survival_target_columns,
    normalize_task_type,
)


def filter_by_patients(df, patient_ids, id_col="patient"):
    return df[df[id_col].isin(patient_ids)].copy()


def _normalize_method(value, valid_values, arg_name):
    value_l = str(value).strip().lower()
    if value_l not in valid_values:
        valid_text = ", ".join(sorted(valid_values))
        raise ValueError(f"Invalid {arg_name}='{value}'. Expected one of: {valid_text}.")
    return value_l


def _ensure_unambiguous_id_column(df, id_col):
    """Return a copy where id_col is available as a plain column, not also as an index level."""
    work_df = df.copy()
    index_names = [
        name for name in getattr(work_df.index, "names", [work_df.index.name])
        if name is not None
    ]

    if id_col in work_df.columns and id_col in index_names:
        return work_df.reset_index(drop=True)

    if id_col not in work_df.columns and id_col in index_names:
        return work_df.reset_index()

    return work_df


def collapse_patient_rows(df, id_col, strategy="mean"):
    """Collapse duplicated patient rows into one feature row per patient."""
    work_df = _ensure_unambiguous_id_column(df, id_col)
    if id_col not in work_df.columns:
        raise ValueError(f"Dataframe does not contain the id column '{id_col}'.")

    strategy_l = str(strategy).strip().lower()
    if strategy_l not in {"mean", "max", "keep"}:
        raise ValueError("strategy must be one of: mean, max, keep")

    if strategy_l == "keep":
        return work_df

    feature_cols = [c for c in work_df.columns if c != id_col]
    numeric_features = work_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    dense_df = pd.concat([work_df[[id_col]], numeric_features], axis=1).copy()

    if strategy_l == "mean":
        return dense_df.groupby(id_col, as_index=False, sort=False)[feature_cols].mean()
    return dense_df.groupby(id_col, as_index=False, sort=False)[feature_cols].max()


def mode_of_series(series):
    non_null = series.dropna()
    if non_null.empty:
        raise ValueError("Cannot compute mode of an all-missing column.")
    mode_values = non_null.mode(dropna=True)
    if mode_values.empty:
        return non_null.iloc[0]
    return mode_values.iloc[0]


def validate_numeric_columns(df, numeric_cols, modality_name):
    for col in numeric_cols:
        original = df[col]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_mask = original.notna() & converted.isna()
        if invalid_mask.any():
            examples = original.loc[invalid_mask].astype(str).head(5).tolist()
            raise ValueError(
                f"Modality '{modality_name}' column '{col}' is not numeric. "
                f"Examples: {examples}. Mark it as categorical or drop it."
            )
        df[col] = converted.astype(np.float32)


def _build_distance_matrix(df, numeric_cols, categorical_cols):
    blocks = []

    if numeric_cols:
        numeric_df = df[numeric_cols].astype(np.float32)
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(numeric_df)
        blocks.append(
            pd.DataFrame(
                scaled_numeric,
                index=df.index,
                columns=[f"num::{col}" for col in numeric_cols],
            )
        )

    if categorical_cols:
        categorical_df = df[categorical_cols].astype(str)
        one_hot = pd.get_dummies(
            categorical_df,
            columns=categorical_cols,
            prefix=[f"cat::{col}" for col in categorical_cols],
            dtype=np.float32,
        )
        blocks.append(one_hot.astype(np.float32))

    if not blocks:
        raise ValueError("Cannot build KNN distance matrix with zero predictor columns.")

    return pd.concat(blocks, axis=1)


def _knn_mode_impute(df, feature_cols, categorical_cols, numeric_cols, k):
    work_df = df.copy()
    if not categorical_cols:
        return work_df

    provisional_df = work_df.copy()
    for col in categorical_cols:
        fill_value = mode_of_series(provisional_df[col])
        filled = provisional_df[col].fillna(fill_value)
        provisional_df[col] = filled.infer_objects(copy=False)

    for target_col in categorical_cols:
        missing_mask = work_df[target_col].isna()
        if not missing_mask.any():
            continue

        observed_mask = ~work_df[target_col].isna()
        if observed_mask.sum() == 0:
            raise ValueError(
                f"Cannot KNN-impute modality column '{target_col}' because it is entirely missing."
            )

        predictor_cols = [col for col in feature_cols if col != target_col]
        predictor_numeric_cols = [col for col in numeric_cols if col != target_col]
        predictor_categorical_cols = [col for col in categorical_cols if col != target_col]
        if not predictor_cols:
            fill_value = mode_of_series(work_df.loc[observed_mask, target_col])
            work_df.loc[missing_mask, target_col] = fill_value
            provisional_df.loc[missing_mask, target_col] = fill_value
            continue

        distance_df = _build_distance_matrix(
            provisional_df[predictor_cols],
            numeric_cols=predictor_numeric_cols,
            categorical_cols=predictor_categorical_cols,
        )
        observed_features = distance_df.loc[observed_mask]
        missing_features = distance_df.loc[missing_mask]
        n_neighbors = min(int(k), int(observed_features.shape[0]))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(observed_features.to_numpy(dtype=np.float32, copy=False))
        neighbor_indices = knn.kneighbors(
            missing_features.to_numpy(dtype=np.float32, copy=False),
            return_distance=False,
        )

        observed_values = work_df.loc[observed_mask, target_col].reset_index(drop=True)
        imputed_values = []
        for row_indices in neighbor_indices:
            neighbor_values = observed_values.iloc[row_indices]
            imputed_values.append(mode_of_series(neighbor_values))

        work_df.loc[missing_mask, target_col] = imputed_values
        provisional_df.loc[missing_mask, target_col] = imputed_values

    return work_df


def impute_modality_df(
    df,
    id_col,
    categorical_cols,
    numeric_imputation,
    categorical_imputation,
    knn_neighbors,
    modality_name,
):
    work_df = df.copy()
    feature_cols = [col for col in work_df.columns if col != id_col]
    categorical_cols = list(categorical_cols)
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    numeric_missing_cols = [col for col in numeric_cols if work_df[col].isna().any()]
    categorical_missing_cols = [col for col in categorical_cols if work_df[col].isna().any()]

    validate_numeric_columns(work_df, numeric_cols, modality_name=modality_name)

    if numeric_missing_cols:
        if numeric_imputation == "mean":
            for col in numeric_missing_cols:
                work_df[col] = work_df[col].fillna(float(work_df[col].mean()))
        elif numeric_imputation == "median":
            for col in numeric_missing_cols:
                work_df[col] = work_df[col].fillna(float(work_df[col].median()))
        elif numeric_imputation == "knn_mean":
            numeric_df = work_df[numeric_cols].copy()
            imputer = KNNImputer(n_neighbors=int(knn_neighbors))
            imputed_numeric = imputer.fit_transform(numeric_df)
            work_df[numeric_cols] = pd.DataFrame(
                imputed_numeric,
                columns=numeric_cols,
                index=work_df.index,
            ).astype(np.float32)
        else:
            raise ValueError(f"Unsupported numeric imputation method: {numeric_imputation}")

    if categorical_missing_cols:
        if categorical_imputation == "column_mode":
            for col in categorical_missing_cols:
                filled = work_df[col].fillna(mode_of_series(work_df[col]))
                work_df[col] = filled.infer_objects(copy=False)
        elif categorical_imputation == "knn_mode":
            work_df = _knn_mode_impute(
                work_df,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                k=int(knn_neighbors),
            )
        else:
            raise ValueError(f"Unsupported categorical imputation method: {categorical_imputation}")

    return work_df


def summarize_missing_values(modality_frames, modality_configs, id_col):
    rows = []
    total_missing = 0
    for modality_name, df in modality_frames.items():
        categorical_cols = modality_configs[modality_name]["categorical_cols"]
        feature_cols = [col for col in df.columns if col != id_col]
        numeric_cols = [col for col in feature_cols if col not in categorical_cols]
        missing_counts = df[feature_cols].isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        modality_missing = int(missing_counts.sum())
        total_missing += modality_missing
        rows.append(
            {
                "modality": modality_name,
                "feature_columns": len(feature_cols),
                "categorical_columns": len(categorical_cols),
                "numeric_columns": len(numeric_cols),
                "total_missing_cells": modality_missing,
                "columns_with_missing": int((missing_counts > 0).sum()),
                "missing_by_column": {col: int(count) for col, count in missing_counts.items()},
            }
        )
    return rows, total_missing


def summarize_duplicate_patient_rows(modality_frames, modality_configs, id_col):
    rows = []
    for modality_name, df in modality_frames.items():
        counts = df[id_col].value_counts()
        duplicated = counts[counts > 1]
        rows.append(
            {
                "modality": modality_name,
                "aggregation_method": modality_configs[modality_name]["aggregation_method"],
                "duplicated_patient_count": int(len(duplicated)),
                "max_rows_per_patient": int(duplicated.max()) if not duplicated.empty else 1,
                "example_patient_ids": duplicated.index.tolist()[:10],
            }
        )
    return rows


def validate_imputation_requirements(modality_frames, modality_configs, id_col):
    errors = []
    for modality_name, df in modality_frames.items():
        feature_cols = [col for col in df.columns if col != id_col]
        missing_cols = [col for col in feature_cols if df[col].isna().any()]
        if not missing_cols:
            continue

        categorical_cols = set(modality_configs[modality_name]["categorical_cols"])
        categorical_missing_cols = [col for col in missing_cols if col in categorical_cols]
        numeric_missing_cols = [col for col in missing_cols if col not in categorical_cols]

        numeric_imputation = modality_configs[modality_name]["numeric_imputation"]
        categorical_imputation = modality_configs[modality_name]["categorical_imputation"]
        knn_neighbors = modality_configs[modality_name]["knn_neighbors"]

        if numeric_missing_cols and numeric_imputation is None:
            errors.append(
                f"Imputation method not defined for modality '{modality_name}' with missing numeric columns: "
                f"{', '.join(numeric_missing_cols)}"
            )
        if categorical_missing_cols and categorical_imputation is None:
            errors.append(
                f"Imputation method not defined for modality '{modality_name}' with missing categorical columns: "
                f"{', '.join(categorical_missing_cols)}"
            )

        if numeric_missing_cols and numeric_imputation == "knn_mean" and knn_neighbors is None:
            errors.append(
                f"KNN neighbors not defined for modality '{modality_name}' using numeric knn_mean imputation."
            )
        if categorical_missing_cols and categorical_imputation == "knn_mode" and knn_neighbors is None:
            errors.append(
                f"KNN neighbors not defined for modality '{modality_name}' using categorical knn_mode imputation."
            )

    if errors:
        raise ValueError("\n".join(errors))


def load_endpoint_df(
    path,
    patient_id_col,
    endpoint_col=None,
    task_type="binary_classification",
    survival_time_col=None,
    survival_event_col=None,
    survival_n_bins=4,
):
    endpoint_df = pd.read_csv(path)
    if patient_id_col not in endpoint_df.columns:
        raise ValueError(
            f"Required column '{patient_id_col}' not found in endpoint CSV '{path}'."
        )
    if endpoint_df[patient_id_col].isna().any():
        raise ValueError(f"Endpoint CSV '{path}' has missing values in '{patient_id_col}'.")

    task_type_l = normalize_task_type(task_type)
    if task_type_l == "survival":
        if not survival_time_col or not survival_event_col:
            raise ValueError(
                "Survival mode requires both survival_time_col and survival_event_col."
            )
        endpoint_df, _ = add_survival_target_columns(
            endpoint_df=endpoint_df,
            patient_id_col=patient_id_col,
            time_col=survival_time_col,
            event_col=survival_event_col,
            n_bins=int(survival_n_bins),
            y_disc_col=SURVIVAL_Y_DISC_COL,
            censorship_col=SURVIVAL_CENSORSHIP_COL,
        )
    else:
        if not endpoint_col:
            raise ValueError("Classification mode requires endpoint_col.")
        if endpoint_col not in endpoint_df.columns:
            raise ValueError(
                f"Required column '{endpoint_col}' not found in endpoint CSV '{path}'."
            )

        raw_labels = endpoint_df[endpoint_col]
        coerced_labels = pd.to_numeric(raw_labels, errors="coerce")
        invalid_label_mask = coerced_labels.isna()
        if invalid_label_mask.any():
            invalid_examples = (
                endpoint_df.loc[invalid_label_mask, [patient_id_col, endpoint_col]]
                .head(10)
                .to_dict("records")
            )
            n_invalid = int(invalid_label_mask.sum())
            print(
                f"[endpoints] Dropping {n_invalid} rows with non-numeric or missing labels in "
                f"'{endpoint_col}'. Examples: {invalid_examples}"
            )
            endpoint_df = endpoint_df.loc[~invalid_label_mask].copy()
            coerced_labels = coerced_labels.loc[~invalid_label_mask].copy()

        if endpoint_df.empty:
            raise ValueError(
                f"Endpoint CSV '{path}' has no valid rows left after filtering invalid labels "
                f"in '{endpoint_col}'."
            )

        endpoint_df[endpoint_col] = coerced_labels.astype(np.float32)

    duplicated = endpoint_df[patient_id_col].duplicated(keep=False)
    if duplicated.any():
        preview = endpoint_df.loc[duplicated, patient_id_col].astype(str).head(10).tolist()
        raise ValueError(
            f"Endpoint CSV '{path}' has duplicated patient ids in '{patient_id_col}'. "
            f"Examples: {preview}"
        )
    return endpoint_df


def find_label_column(inst_df, endpoint):
    preferred = f"{endpoint}_label"
    if preferred in inst_df.columns:
        return preferred
    if endpoint in inst_df.columns:
        return endpoint
    raise ValueError(f"Label column not found. Tried '{preferred}' and '{endpoint}'.")


def load_configured_modality_frames(
    modality_paths,
    patient_id_col,
    endpoint_df,
    drop_cols_map=None,
    categorical_cols_map=None,
    aggregation_map=None,
    numeric_imputation_map=None,
    categorical_imputation_map=None,
    knn_neighbors_map=None,
):
    if not modality_paths:
        raise ValueError("At least one modality CSV must be provided.")

    drop_cols_map = drop_cols_map or OrderedDict()
    categorical_cols_map = categorical_cols_map or OrderedDict()
    aggregation_map = aggregation_map or OrderedDict()
    numeric_imputation_map = numeric_imputation_map or OrderedDict()
    categorical_imputation_map = categorical_imputation_map or OrderedDict()
    knn_neighbors_map = knn_neighbors_map or OrderedDict()

    modality_frames = OrderedDict()
    modality_configs = OrderedDict()
    endpoint_patient_ids = endpoint_df[patient_id_col].tolist()

    for modality_name, csv_path in modality_paths.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Modality CSV for '{modality_name}' not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if patient_id_col not in df.columns:
            raise ValueError(
                f"Modality '{modality_name}' does not contain patient id column '{patient_id_col}'."
            )

        drop_cols = drop_cols_map.get(modality_name, [])
        missing_drop_cols = [col for col in drop_cols if col not in df.columns]
        if missing_drop_cols:
            raise ValueError(
                f"Modality '{modality_name}' drop columns not found: {missing_drop_cols}"
            )
        if drop_cols:
            df = df.drop(columns=drop_cols)

        df = filter_by_patients(df, endpoint_patient_ids, id_col=patient_id_col)

        categorical_cols = categorical_cols_map.get(modality_name, [])
        missing_categorical_cols = [col for col in categorical_cols if col not in df.columns]
        if missing_categorical_cols:
            raise ValueError(
                f"Modality '{modality_name}' categorical columns not found: {missing_categorical_cols}"
            )

        aggregation_method = aggregation_map.get(modality_name)
        if aggregation_method is not None:
            aggregation_method = _normalize_method(
                aggregation_method,
                {"mean", "attention"},
                arg_name="aggregation_method",
            )

        numeric_imputation = numeric_imputation_map.get(modality_name)
        if numeric_imputation is not None:
            numeric_imputation = _normalize_method(
                numeric_imputation,
                {"mean", "median", "knn_mean"},
                "numeric_imputation",
            )
        categorical_imputation = categorical_imputation_map.get(modality_name)
        if categorical_imputation is not None:
            categorical_imputation = _normalize_method(
                categorical_imputation,
                {"column_mode", "knn_mode"},
                "categorical_imputation",
            )
        raw_knn_neighbors = knn_neighbors_map.get(modality_name)
        if raw_knn_neighbors is None:
            knn_neighbors = None
        else:
            try:
                knn_neighbors = int(raw_knn_neighbors)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid knn_neighbors for modality '{modality_name}': {raw_knn_neighbors!r}"
                ) from exc
            if knn_neighbors < 1:
                raise ValueError(f"knn_neighbors for modality '{modality_name}' must be >= 1")

        feature_cols = [col for col in df.columns if col != patient_id_col]
        numeric_cols = [col for col in feature_cols if col not in categorical_cols]
        validate_numeric_columns(df, numeric_cols, modality_name=modality_name)

        modality_frames[modality_name] = df
        modality_configs[modality_name] = {
            "csv_path": csv_path,
            "drop_cols": drop_cols,
            "categorical_cols": categorical_cols,
            "aggregation_method": aggregation_method,
            "numeric_imputation": numeric_imputation,
            "categorical_imputation": categorical_imputation,
            "knn_neighbors": knn_neighbors,
        }

    return modality_frames, modality_configs


def align_complete_multimodal_cohort(modality_frames, endpoint_df, patient_id_col):
    if endpoint_df.empty:
        raise ValueError("Endpoint dataframe is empty before cohort alignment.")
    if not modality_frames:
        raise ValueError("At least one modality dataframe is required for cohort alignment.")

    endpoint_ids = endpoint_df[patient_id_col].astype(str)
    cohort_ids = set(endpoint_ids.tolist())
    summary_rows = []

    for modality_name, df in modality_frames.items():
        modality_ids = set(df[patient_id_col].astype(str).tolist())
        summary_rows.append(
            {
                "modality": modality_name,
                "patients_before": int(len(modality_ids)),
                "missing_vs_endpoints": int(len(cohort_ids - modality_ids)),
            }
        )
        cohort_ids &= modality_ids

    if not cohort_ids:
        raise ValueError(
            "The intersection between endpoints and all configured modalities is empty."
        )

    endpoint_order = endpoint_ids.drop_duplicates().tolist()
    ordered_cohort_ids = [pid for pid in endpoint_order if pid in cohort_ids]

    aligned_endpoint_df = filter_by_patients(
        endpoint_df.assign(**{patient_id_col: endpoint_ids}),
        ordered_cohort_ids,
        id_col=patient_id_col,
    )
    aligned_modality_frames = OrderedDict()
    for modality_name, df in modality_frames.items():
        work_df = df.copy()
        work_df[patient_id_col] = work_df[patient_id_col].astype(str)
        aligned_modality_frames[modality_name] = filter_by_patients(
            work_df,
            ordered_cohort_ids,
            id_col=patient_id_col,
        )

    cohort_summary = {
        "endpoint_patients_before": int(endpoint_df[patient_id_col].nunique()),
        "endpoint_patients_after": int(aligned_endpoint_df[patient_id_col].nunique()),
        "dropped_from_endpoints": int(endpoint_df[patient_id_col].nunique() - aligned_endpoint_df[patient_id_col].nunique()),
        "cohort_summary_by_modality": summary_rows,
    }
    return aligned_modality_frames, aligned_endpoint_df, cohort_summary


def save_processed_outputs(
    output_dir,
    endpoint_df,
    patient_id_col,
    endpoint_col,
    modality_frames,
    summary_payload,
    endpoint_columns=None,
):
    os.makedirs(output_dir, exist_ok=True)
    if endpoint_columns is None:
        endpoint_columns = [patient_id_col, endpoint_col]
    missing_endpoint_cols = [col for col in endpoint_columns if col not in endpoint_df.columns]
    if missing_endpoint_cols:
        raise ValueError(
            f"Cannot save processed endpoints: missing columns {missing_endpoint_cols}."
        )
    endpoint_out = endpoint_df[list(endpoint_columns)].copy()
    endpoint_out.to_csv(os.path.join(output_dir, "endpoints_selected.csv"), index=False)

    modalities_dir = os.path.join(output_dir, "modalities")
    os.makedirs(modalities_dir, exist_ok=True)
    for modality_name, df in modality_frames.items():
        df.to_csv(os.path.join(modalities_dir, f"{modality_name}.csv"), index=False)

    with open(os.path.join(output_dir, "preprocessing_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)


def validate_and_prepare_modality_rows(dfs, id_col, radio_aggregation_method="mean"):
    """Validate modality row multiplicity and prepare radiology according to the chosen aggregation."""
    radio_aggregation_l = str(radio_aggregation_method).strip().lower()
    if radio_aggregation_l not in {"mean", "attention"}:
        raise ValueError("radio_aggregation_method must be one of: mean, attention")

    cleaned = {}
    for mod_name, df in dfs.items():
        if id_col not in df.columns:
            raise ValueError(f"Modality '{mod_name}' does not contain the id column '{id_col}'.")

        work_df = df.copy()
        patient_counts = work_df[id_col].value_counts()
        duplicated = patient_counts[patient_counts > 1]
        if duplicated.empty:
            cleaned[mod_name] = work_df
            continue

        dup_ids = duplicated.index.tolist()
        preview = dup_ids[:25]
        preview_txt = ", ".join(str(x) for x in preview)
        suffix = " ..." if len(dup_ids) > 25 else ""

        if mod_name != "radio":
            print(
                f"[{mod_name}] Duplicated patients detected: {len(dup_ids)}. "
                f"This modality must already be one-row-per-patient. Patient IDs: {preview_txt}{suffix}"
            )
            raise ValueError(
                f"Duplicated patients detected in modality '{mod_name}'. "
                "Only radiology is allowed to have duplicated rows before aggregation."
            )

        if radio_aggregation_l == "mean":
            print(
                f"[radio] Duplicated patients detected: {len(dup_ids)}. "
                f"Collapsing by mean. Patient IDs: {preview_txt}{suffix}"
            )
            cleaned[mod_name] = collapse_patient_rows(work_df, id_col=id_col, strategy="mean")
        else:
            print(
                f"[radio] Duplicated patients detected: {len(dup_ids)}. "
                f"Keeping all rows for downstream attention pooling. Patient IDs: {preview_txt}{suffix}"
            )
            cleaned[mod_name] = work_df

    return cleaned


def _resolve_processed_bundle_dir(dataset_dir):
    candidate_dirs = [
        dataset_dir,
        os.path.join(dataset_dir, "processed_data"),
    ]
    for candidate in candidate_dirs:
        endpoints_path = os.path.join(candidate, "endpoints_selected.csv")
        modalities_dir = os.path.join(candidate, "modalities")
        if os.path.isfile(endpoints_path) and os.path.isdir(modalities_dir):
            return candidate
    return None


def _infer_patient_id_col_from_processed_bundle(endpoints_df, summary_path=None):
    if summary_path and os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary_payload = json.load(handle)
        patient_id_col = summary_payload.get("patient_id_col")
        if patient_id_col and patient_id_col in endpoints_df.columns:
            return patient_id_col

    if "patient" in endpoints_df.columns:
        return "patient"
    if len(endpoints_df.columns) == 2:
        return endpoints_df.columns[0]
    raise ValueError(
        "Could not infer patient_id_col from processed endpoints CSV. "
        "Ensure preprocessing_summary.json contains patient_id_col."
    )


def _load_processed_dataset_bundle(
    dataset_dir,
    endpoint,
    task_type="binary_classification",
    survival_time_col=None,
    survival_event_col=None,
    survival_n_bins=4,
):
    processed_root = _resolve_processed_bundle_dir(dataset_dir)
    if processed_root is None:
        return None

    endpoints_path = os.path.join(processed_root, "endpoints_selected.csv")
    summary_path = os.path.join(processed_root, "preprocessing_summary.json")
    modalities_dir = os.path.join(processed_root, "modalities")

    inst_df = pd.read_csv(endpoints_path)
    patient_id_col = _infer_patient_id_col_from_processed_bundle(inst_df, summary_path=summary_path)
    task_type_l = normalize_task_type(task_type)
    if task_type_l == "survival":
        if not survival_time_col or not survival_event_col:
            raise ValueError(
                "Survival mode requires both survival_time_col and survival_event_col."
            )
        inst_df, _ = add_survival_target_columns(
            endpoint_df=inst_df,
            patient_id_col=patient_id_col,
            time_col=survival_time_col,
            event_col=survival_event_col,
            n_bins=int(survival_n_bins),
            y_disc_col=SURVIVAL_Y_DISC_COL,
            censorship_col=SURVIVAL_CENSORSHIP_COL,
        )
        label_col = SURVIVAL_Y_DISC_COL
        keep_cols = [
            patient_id_col,
            survival_time_col,
            survival_event_col,
            SURVIVAL_CENSORSHIP_COL,
            SURVIVAL_Y_DISC_COL,
        ]
        inst_df = inst_df[keep_cols].copy()
    else:
        label_col = find_label_column(inst_df, endpoint)
        inst_df = inst_df[[patient_id_col, label_col]].copy()

    modality_files = sorted(
        filename for filename in os.listdir(modalities_dir) if filename.lower().endswith(".csv")
    )
    if not modality_files:
        raise ValueError(f"No modality CSVs found in processed bundle: '{modalities_dir}'.")

    dfs = OrderedDict()
    for filename in modality_files:
        modality_name = os.path.splitext(filename)[0]
        df = pd.read_csv(os.path.join(modalities_dir, filename))
        if patient_id_col not in df.columns:
            raise ValueError(
                f"Processed modality '{modality_name}' does not contain patient id column '{patient_id_col}'."
            )
        dfs[modality_name] = df.set_index(patient_id_col, drop=False)

    inst_df = inst_df.set_index(patient_id_col, drop=False)
    task_config = {
        "task_type": task_type_l,
        "label_col": label_col,
    }
    if task_type_l == "survival":
        task_config.update(
            {
                "survival_time_col": survival_time_col,
                "survival_event_col": survival_event_col,
                "survival_censorship_col": SURVIVAL_CENSORSHIP_COL,
                "survival_y_disc_col": SURVIVAL_Y_DISC_COL,
                "survival_n_bins": int(inst_df[SURVIVAL_Y_DISC_COL].nunique()),
            }
        )
    return inst_df, dfs, label_col, patient_id_col, task_config


def _resolve_csv_path(dataset_dir, filename, required=False):
    candidate = os.path.join(dataset_dir, filename)
    if os.path.exists(candidate):
        return candidate
    if required:
        raise FileNotFoundError(f"Required file not found: '{candidate}'")
    return None


def _load_mimm_pathology_df(patho_path, inst_df, id_col):
    path_df = pd.read_csv(patho_path)
    path_df = path_df.rename(columns=lambda x: x.replace("embedding_", "patho_"))
    path_df = pd.merge(path_df, inst_df[[id_col]], on=id_col, how="inner")
    keep = [id_col] + [c for c in path_df.columns if c.startswith("patho_")]
    return path_df[keep]


def _load_mimm_prefixed_df(csv_path, inst_df, prefix, id_col):
    df = pd.read_csv(csv_path)
    df = df.rename(columns=lambda x: f"{prefix}_{x}" if x != id_col else x)
    return pd.merge(df, inst_df[[id_col]], on=id_col, how="inner")


def _load_mimm_radio_df(radio_path, inst_df, id_col):
    rad_df = pd.read_csv(radio_path).rename(columns=lambda x: x.replace("pred_", "radio_"))
    rad_df = rad_df.drop(columns=["image_path", "lesion_tag"], errors="ignore")
    rad_df = pd.merge(rad_df, inst_df[[id_col]], on=id_col, how="inner")
    keep = [id_col] + [c for c in rad_df.columns if c.startswith("radio_")]
    return rad_df[keep]


def _load_legacy_mimm_raw_bundle(dataset_dir, endpoint, task_type="binary_classification"):
    task_type_l = normalize_task_type(task_type)
    if task_type_l != "binary_classification":
        raise NotImplementedError(
            "Legacy raw MIMM loading is only supported for binary classification. "
            "Use a generic processed bundle for survival runs."
        )
    id_col = "patient"
    inst_path = _resolve_csv_path(dataset_dir, "patients_mimm.csv", required=True)
    patho_path = _resolve_csv_path(dataset_dir, "pathology_mimm.csv", required=False)
    radio_path = _resolve_csv_path(dataset_dir, "radiology_mimm.csv", required=False)
    clin_path = _resolve_csv_path(dataset_dir, "clinical_mimm.csv", required=False)
    blood_path = _resolve_csv_path(dataset_dir, "blood_mimm.csv", required=False)
    radio_report_path = _resolve_csv_path(dataset_dir, "radioreports_mimm.csv", required=False)

    inst_df = pd.read_csv(inst_path)
    if id_col not in inst_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in the labels CSV.")

    label_col = find_label_column(inst_df, endpoint)
    inst_df = inst_df[[id_col, label_col]].copy()

    dfs = OrderedDict()
    if patho_path:
        dfs["path"] = _load_mimm_pathology_df(patho_path, inst_df, id_col)
    if radio_path:
        dfs["radio"] = _load_mimm_radio_df(radio_path, inst_df, id_col)
    if clin_path:
        dfs["clin"] = _load_mimm_prefixed_df(clin_path, inst_df, "clin", id_col)
    if blood_path:
        dfs["blood"] = _load_mimm_prefixed_df(blood_path, inst_df, "blood", id_col)
    if radio_report_path:
        dfs["radio_report"] = _load_mimm_prefixed_df(radio_report_path, inst_df, "radio_report", id_col)

    if not dfs:
        raise ValueError("No modality CSV found. Provide --dataset_dir with MIMM modality files.")

    dfs = validate_and_prepare_modality_rows(dfs, id_col, radio_aggregation_method="attention")
    for mod in list(dfs.keys()):
        dfs[mod] = dfs[mod].set_index(id_col, drop=False)

    inst_df = inst_df.set_index(id_col, drop=False)
    task_config = {
        "task_type": "binary_classification",
        "label_col": label_col,
    }
    return inst_df, dfs, label_col, id_col, task_config


def load_or_preprocess_dataset(args):
    """Load a generic processed bundle or a legacy raw MIMM dataset."""
    dataset_dir = getattr(args, "dataset_dir", None)
    if not dataset_dir:
        raise ValueError("Dataset loading requires --dataset_dir.")

    task_type = normalize_task_type(getattr(args, "task_type", "binary_classification"))
    processed_bundle = _load_processed_dataset_bundle(
        dataset_dir,
        endpoint=args.endpoint,
        task_type=task_type,
        survival_time_col=getattr(args, "survival_time_col", None),
        survival_event_col=getattr(args, "survival_event_col", None),
        survival_n_bins=int(getattr(args, "survival_n_bins", 4)),
    )
    if processed_bundle is not None:
        return processed_bundle

    dataset_name = str(args.dataset).strip().lower()
    if dataset_name == "mimm":
        return _load_legacy_mimm_raw_bundle(
            dataset_dir=dataset_dir,
            endpoint=args.endpoint,
            task_type=task_type,
        )

    raise NotImplementedError(
        f"Dataset '{args.dataset}' is not available as a generic processed bundle and has no legacy raw loader. "
        "Run m3trics first and pass --dataset_dir pointing to the processed_data directory (or its parent results root)."
    )
