import argparse
import json
import os
import sys
from collections import OrderedDict
import pandas as pd
from dataset.preprocess_dataset import (
    aggregate_modality_df,
    impute_modality_df,
    summarize_duplicate_patient_rows,
    summarize_missing_values,
    validate_numeric_columns,
)

def _parse_key_value_arg(raw_value, arg_name):
    if "=" not in raw_value:
        raise ValueError(f"{arg_name} expects NAME=VALUE, got '{raw_value}'.")
    key, value = raw_value.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"{arg_name} expects a non-empty NAME in '{raw_value}'.")
    return key, value

def _parse_keyed_str_map(values, arg_name):
    mapping = OrderedDict()
    for raw_value in values or []:
        key, value = _parse_key_value_arg(raw_value, arg_name)
        mapping[key] = value
    return mapping


def _parse_keyed_list_map(values, arg_name):
    mapping = OrderedDict()
    for raw_value in values or []:
        key, value = _parse_key_value_arg(raw_value, arg_name)
        if value == "":
            mapping[key] = []
        else:
            mapping[key] = [item.strip() for item in value.split(",") if item.strip()]
    return mapping

def _normalize_method(value, valid_values, arg_name):
    value_l = str(value).strip().lower()
    if value_l not in valid_values:
        valid_text = ", ".join(sorted(valid_values))
        raise ValueError(f"Invalid {arg_name}='{value}'. Expected one of: {valid_text}.")
    return value_l

def _prompt_ok_or_stop(message):
    print(message)
    while True:
        answer = input("Type 'ok' to continue or 'stop' to abort: ").strip().lower()
        if answer == "ok":
            return
        if answer == "stop":
            raise SystemExit("Stopped by user.")
        print("Invalid response. Use 'ok' or 'stop'.")

def _print_missingness_report(summary_rows, numeric_imputation, categorical_imputation, knn_neighbors):
    print("\n=== Missing Values Report ===")
    print(f"Numeric imputation plan: {numeric_imputation}")
    print(f"Categorical imputation plan: {categorical_imputation}")
    if "knn" in numeric_imputation or "knn" in categorical_imputation:
        print(f"KNN neighbors: {int(knn_neighbors)}")
    for row in summary_rows:
        print(
            f"[{row['modality']}] total_missing_cells={row['total_missing_cells']} | "
            f"columns_with_missing={row['columns_with_missing']}"
        )
        for col_name, miss_count in row["missing_by_column"].items():
            print(f"  - {col_name}: {miss_count}")

def _print_duplicate_report(summary_rows):
    print("\n=== Duplicate Patient-ID Report ===")
    for row in summary_rows:
        aggregation_method = row["aggregation_method"] if row["aggregation_method"] is not None else "none"
        print(
            f"[{row['modality']}] duplicated_patient_count={row['duplicated_patient_count']} | "
            f"aggregation_method={aggregation_method} | "
            f"max_rows_per_patient={row['max_rows_per_patient']}"
        )
        if row["example_patient_ids"]:
            preview = ", ".join(str(x) for x in row["example_patient_ids"])
            print(f"  - example duplicated IDs: {preview}")


def _load_endpoint_df(path, patient_id_col, endpoint_col):
    endpoint_df = pd.read_csv(path)
    for required_col in [patient_id_col, endpoint_col]:
        if required_col not in endpoint_df.columns:
            raise ValueError(
                f"Required column '{required_col}' not found in endpoint CSV '{path}'."
            )
    if endpoint_df[patient_id_col].isna().any():
        raise ValueError(f"Endpoint CSV '{path}' has missing values in '{patient_id_col}'.")
    if endpoint_df[endpoint_col].isna().any():
        raise ValueError(f"Endpoint CSV '{path}' has missing values in '{endpoint_col}'.")
    duplicated = endpoint_df[patient_id_col].duplicated(keep=False)
    if duplicated.any():
        preview = endpoint_df.loc[duplicated, patient_id_col].astype(str).head(10).tolist()
        raise ValueError(
            f"Endpoint CSV '{path}' has duplicated patient ids in '{patient_id_col}'. "
            f"Examples: {preview}"
        )
    return endpoint_df

def _load_modality_frames(args):
    modality_paths = _parse_keyed_str_map(args.modality_csv, "--modality_csv")
    if not modality_paths:
        raise ValueError("At least one --modality_csv NAME=PATH argument is required.")

    drop_cols_map = _parse_keyed_list_map(args.drop_cols, "--drop_cols")
    categorical_cols_map = _parse_keyed_list_map(args.categorical_cols, "--categorical_cols")
    aggregation_map = _parse_keyed_str_map(args.aggregation_method, "--aggregation_method")

    modality_frames = OrderedDict()
    modality_configs = OrderedDict()

    for modality_name, csv_path in modality_paths.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Modality CSV for '{modality_name}' not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if args.patient_id_col not in df.columns:
            raise ValueError(
                f"Modality '{modality_name}' does not contain patient id column '{args.patient_id_col}'."
            )

        drop_cols = drop_cols_map.get(modality_name, [])
        missing_drop_cols = [col for col in drop_cols if col not in df.columns]
        if missing_drop_cols:
            raise ValueError(
                f"Modality '{modality_name}' drop columns not found: {missing_drop_cols}"
            )
        if drop_cols:
            df = df.drop(columns=drop_cols)

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
                {"mean", "max", "first", "keep"},
                arg_name="aggregation_method",
            )

        feature_cols = [col for col in df.columns if col != args.patient_id_col]
        numeric_cols = [col for col in feature_cols if col not in categorical_cols]
        validate_numeric_columns(df, numeric_cols, modality_name=modality_name)

        modality_frames[modality_name] = df
        modality_configs[modality_name] = {
            "csv_path": csv_path,
            "drop_cols": drop_cols,
            "categorical_cols": categorical_cols,
            "aggregation_method": aggregation_method,
        }

    return modality_frames, modality_configs


def _save_outputs(output_dir, endpoint_df, patient_id_col, endpoint_col, modality_frames, summary_payload):
    os.makedirs(output_dir, exist_ok=True)
    endpoint_out = endpoint_df[[patient_id_col, endpoint_col]].copy()
    endpoint_out.to_csv(os.path.join(output_dir, "endpoints_selected.csv"), index=False)

    modalities_dir = os.path.join(output_dir, "modalities")
    os.makedirs(modalities_dir, exist_ok=True)
    for modality_name, df in modality_frames.items():
        df.to_csv(os.path.join(modalities_dir, f"{modality_name}.csv"), index=False)

    with open(os.path.join(output_dir, "preflight_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--results_root", required=True, type=str)
    parser.add_argument("--endpoint_csv", required=True, type=str)
    parser.add_argument("--patient_id_col", required=True, type=str)
    parser.add_argument("--endpoint_col", required=True, type=str)
    parser.add_argument("--numeric_imputation", required=True, type=str)
    parser.add_argument("--categorical_imputation", required=True, type=str)
    parser.add_argument("--knn_neighbors", type=int, default=5)
    parser.add_argument("--modality_csv", action="append", default=[])
    parser.add_argument("--categorical_cols", action="append", default=[])
    parser.add_argument("--drop_cols", action="append", default=[])
    parser.add_argument("--aggregation_method", action="append", default=[])
    return parser.parse_args()


def main():
    args = get_args()
    output_dir = os.path.join(args.results_root, "processed_data")
    numeric_imputation = _normalize_method(
        args.numeric_imputation,
        {"mean", "median", "knn_mean"},
        "numeric_imputation",
    )
    categorical_imputation = _normalize_method(
        args.categorical_imputation,
        {"column_mode", "knn_mode"},
        "categorical_imputation",
    )
    if int(args.knn_neighbors) < 1:
        raise ValueError("--knn_neighbors must be >= 1")

    endpoint_df = _load_endpoint_df(
        path=args.endpoint_csv,
        patient_id_col=args.patient_id_col,
        endpoint_col=args.endpoint_col,
    )
    modality_frames, modality_configs = _load_modality_frames(args)

    missing_summary, total_missing = summarize_missing_values(
        modality_frames,
        modality_configs,
        id_col=args.patient_id_col,
    )
    _print_missingness_report(
        missing_summary,
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
        knn_neighbors=args.knn_neighbors,
    )
    if total_missing > 0:
        _prompt_ok_or_stop(
            "Missing values were found. The configured imputation plan will now be applied."
        )
        imputed_frames = OrderedDict()
        for modality_name, df in modality_frames.items():
            imputed_frames[modality_name] = impute_modality_df(
                df=df,
                id_col=args.patient_id_col,
                categorical_cols=modality_configs[modality_name]["categorical_cols"],
                numeric_imputation=numeric_imputation,
                categorical_imputation=categorical_imputation,
                knn_neighbors=args.knn_neighbors,
                modality_name=modality_name,
            )
        modality_frames = imputed_frames
        print("Missing-value imputation completed.")
    else:
        print("No missing values found in modality dataframes.")

    duplicate_summary = summarize_duplicate_patient_rows(
        modality_frames,
        modality_configs,
        id_col=args.patient_id_col,
    )
    _print_duplicate_report(duplicate_summary)
    duplicated_modalities = [
        row for row in duplicate_summary if int(row["duplicated_patient_count"]) > 0
    ]
    if duplicated_modalities:
        missing_aggregation_modalities = [
            row["modality"]
            for row in duplicated_modalities
            if row["aggregation_method"] is None
        ]
        if missing_aggregation_modalities:
            raise ValueError(
                "Duplicated patient ids were found in modalities without an aggregation method: "
                f"{', '.join(missing_aggregation_modalities)}. "
                "Define --aggregation_method for those modalities before continuing."
            )
        _prompt_ok_or_stop(
            "Duplicated patient ids were found. The configured aggregation methods will now be applied."
        )
        aggregated_frames = OrderedDict()
        for modality_name, df in modality_frames.items():
            aggregation_method = modality_configs[modality_name]["aggregation_method"]
            if aggregation_method is None:
                aggregated_frames[modality_name] = df
            else:
                aggregated_frames[modality_name] = aggregate_modality_df(
                    df=df,
                    id_col=args.patient_id_col,
                    method=aggregation_method,
                )
        modality_frames = aggregated_frames
        print("Duplicate-row aggregation completed.")
    else:
        print("No duplicated patient ids found in modality dataframes.")

    final_shapes = {
        modality_name: {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        }
        for modality_name, df in modality_frames.items()
    }
    summary_payload = {
        "dataset": args.dataset,
        "results_root": args.results_root,
        "endpoint_csv": args.endpoint_csv,
        "endpoint_col": args.endpoint_col,
        "patient_id_col": args.patient_id_col,
        "numeric_imputation": numeric_imputation,
        "categorical_imputation": categorical_imputation,
        "knn_neighbors": int(args.knn_neighbors),
        "missing_summary": missing_summary,
        "duplicate_summary": duplicate_summary,
        "final_shapes": final_shapes,
    }
    _save_outputs(
        output_dir=output_dir,
        endpoint_df=endpoint_df,
        patient_id_col=args.patient_id_col,
        endpoint_col=args.endpoint_col,
        modality_frames=modality_frames,
        summary_payload=summary_payload,
    )

    print("\n=== Preflight Complete ===")
    print(f"Processed endpoint CSV saved to: {os.path.join(output_dir, 'endpoints_selected.csv')}")
    print(f"Processed modality CSVs saved to: {os.path.join(output_dir, 'modalities')}")
    print("Model execution is intentionally not launched yet. This template stops after preflight.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[m3trics] ERROR: {exc}", file=sys.stderr)
        raise
