import importlib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

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
        provisional_df[col] = provisional_df[col].fillna(fill_value)

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

    validate_numeric_columns(work_df, numeric_cols, modality_name=modality_name)

    if numeric_cols:
        if numeric_imputation == "mean":
            for col in numeric_cols:
                if work_df[col].isna().any():
                    work_df[col] = work_df[col].fillna(float(work_df[col].mean()))
        elif numeric_imputation == "median":
            for col in numeric_cols:
                if work_df[col].isna().any():
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

    if categorical_cols:
        if categorical_imputation == "column_mode":
            for col in categorical_cols:
                if work_df[col].isna().any():
                    work_df[col] = work_df[col].fillna(mode_of_series(work_df[col]))
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


def aggregate_modality_df(df, id_col, method):
    method_l = str(method).strip().lower()
    if method_l in {"mean", "max", "keep"}:
        return collapse_patient_rows(df, id_col=id_col, strategy=method_l)
    if method_l == "first":
        return df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    raise ValueError("Aggregation method must be one of: mean, max, first, keep")


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

def load_or_preprocess_dataset(args):
    """Load dataset-specific preprocessed bundle."""
    dataset_name = str(args.dataset).strip().lower()
    module_name = f"dataset.{dataset_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only fallback this specific missing module to a clear user-facing error.
        if exc.name == module_name:
            raise NotImplementedError(
                f"Preprocessing for dataset '{args.dataset}' is not implemented yet. "
                "Add dataset/<dataset_name>.py with load_preprocessed_dataset(args)."
            ) from exc
        raise

    loader_fn = getattr(module, "load_preprocessed_dataset", None)
    if not callable(loader_fn):
        raise ValueError(
            f"Dataset module '{module_name}' does not expose "
            "load_preprocessed_dataset(args)."
        )
    bundle = loader_fn(args)
    if not isinstance(bundle, tuple) or len(bundle) != 4:
        raise ValueError(
            f"Dataset module '{module_name}' must return "
            "(inst_df, dfs, label_col, patient_id_col)."
        )
    return bundle
