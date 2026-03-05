import os

import pandas as pd

from .preprocess_dataset import check_and_collapse_modality_rows


def load_preprocessed_dataset(args):
    """MIMM-specific preprocessing pipeline."""

    def _find_label_column(inst_df, endpoint):
        preferred = f"{endpoint}_label"
        if preferred in inst_df.columns:
            return preferred
        if endpoint in inst_df.columns:
            return endpoint
        raise ValueError(f"Label column not found. Tried '{preferred}' and '{endpoint}'.")

    def _load_patho_df(patho_path, inst_df, id_col):
        path_df = pd.read_csv(patho_path)
        path_df = path_df.rename(columns=lambda x: x.replace("embedding_", "patho_"))
        path_df = pd.merge(path_df, inst_df[[id_col]], on=id_col, how="inner")
        keep = [id_col] + [c for c in path_df.columns if c.startswith("patho_")]
        return path_df[keep]

    def _load_prefixed_df(csv_path, inst_df, prefix, id_col):
        df = pd.read_csv(csv_path)
        df = df.rename(columns=lambda x: f"{prefix}_{x}" if x != id_col else x)
        return pd.merge(df, inst_df[[id_col]], on=id_col, how="inner")

    def _load_radio_df(radio_path, inst_df, id_col):
        rad_df = pd.read_csv(radio_path).rename(columns=lambda x: x.replace("pred_", "radio_"))
        rad_df = rad_df.drop(columns=["image_path", "lesion_tag"], errors="ignore")
        rad_df = pd.merge(rad_df, inst_df[[id_col]], on=id_col, how="inner")
        keep = [id_col] + [c for c in rad_df.columns if c.startswith("radio_")]
        return rad_df[keep]

    def _resolve_csv_path(dataset_dir, filename, required=False):
        candidate = os.path.join(dataset_dir, filename)
        if os.path.exists(candidate):
            return candidate
        if required:
            raise FileNotFoundError(
                f"Required file not found: '{candidate}'"
            )
        return None

    if getattr(args, "patient_ids_col", "patient") != "patient":
        raise ValueError(
            "MIMM preprocessing expects patient ID column 'patient'. "
            "Do not override --patient_ids_col for MIMM."
        )
    id_col = "patient"
    dataset_dir = getattr(args, "dataset_dir", None)
    if not dataset_dir:
        raise ValueError(
            "MIMM preprocessing requires --dataset_dir with all dataset CSV files."
        )

    inst_path = _resolve_csv_path(
        dataset_dir,
        "patients_MIMM.csv",
        required=True,
    )
    patho_path = _resolve_csv_path(
        dataset_dir,
        "pathology_MIMM.csv",
        required=False,
    )
    radio_path = _resolve_csv_path(
        dataset_dir,
        "radiology_MIMM.csv",
        required=False,
    )
    clin_path = _resolve_csv_path(
        dataset_dir,
        "clinical_MIMM.csv",
        required=False,
    )
    blood_path = _resolve_csv_path(
        dataset_dir,
        "blood_MIMM.csv",
        required=False,
    )
    radio_report_path = _resolve_csv_path(
        dataset_dir,
        "radioreports_MIMM.csv",
        required=False,
    )

    inst_df = pd.read_csv(inst_path)
    if id_col not in inst_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in the labels CSV.")

    label_col = _find_label_column(inst_df, args.endpoint)
    inst_df = inst_df[[id_col, label_col]].copy()

    dfs = {}
    if patho_path:
        dfs["path"] = _load_patho_df(patho_path, inst_df, id_col)
    if radio_path:
        dfs["radio"] = _load_radio_df(radio_path, inst_df, id_col)
    if clin_path:
        dfs["clin"] = _load_prefixed_df(clin_path, inst_df, "clin", id_col)
    if blood_path:
        dfs["blood"] = _load_prefixed_df(blood_path, inst_df, "blood", id_col)
    if radio_report_path:
        dfs["radio_report"] = _load_prefixed_df(radio_report_path, inst_df, "radio_report", id_col)

    if not dfs:
        raise ValueError(
            "No modality CSV found. Provide --dataset_dir with MIMM modality files."
        )

    dfs = check_and_collapse_modality_rows(dfs, id_col)

    for mod in list(dfs.keys()):
        dfs[mod] = dfs[mod].set_index(id_col, drop=False)

    inst_df = inst_df.set_index(id_col, drop=False)
    return inst_df, dfs, label_col
