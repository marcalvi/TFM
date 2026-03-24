import importlib

import pandas as pd


def collapse_patient_rows(df, id_col, strategy="mean"):
    """Collapse duplicated patient rows into one feature row per patient."""
    if id_col not in df.columns:
        raise ValueError(f"Dataframe does not contain the id column '{id_col}'.")

    strategy_l = str(strategy).strip().lower()
    if strategy_l not in {"mean", "max", "keep"}:
        raise ValueError("strategy must be one of: mean, max, keep")

    work_df = df.copy()
    if strategy_l == "keep":
        return work_df

    feature_cols = [c for c in work_df.columns if c != id_col]
    numeric_features = work_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    dense_df = pd.concat([work_df[[id_col]], numeric_features], axis=1).copy()

    if strategy_l == "mean":
        return dense_df.groupby(id_col, as_index=False, sort=False)[feature_cols].mean()
    return dense_df.groupby(id_col, as_index=False, sort=False)[feature_cols].max()


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
