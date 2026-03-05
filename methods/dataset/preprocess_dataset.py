import importlib

import pandas as pd

def check_and_collapse_modality_rows(dfs, id_col):
    """Ensure one row per patient in each modality dataframe."""
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
        print(
            f"[{mod_name}] Duplicated patients detected: {len(dup_ids)}. "
            f"Collapsing by mean. Patient IDs: {preview_txt}{suffix}"
        )

        feature_cols = [c for c in work_df.columns if c != id_col]
        numeric_features = work_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        dense_df = pd.concat([work_df[[id_col]], numeric_features], axis=1).copy()
        collapsed = dense_df.groupby(id_col, as_index=False)[feature_cols].mean()
        cleaned[mod_name] = collapsed

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
    if not isinstance(bundle, tuple) or len(bundle) != 3:
        raise ValueError(
            f"Dataset module '{module_name}' must return "
            "(inst_df, dfs, label_col)."
        )
    return bundle
