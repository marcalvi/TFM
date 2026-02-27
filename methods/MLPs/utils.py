import numpy as np
import torch
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score

# Function to select device (CUDA, MPS, or CPU)
def select_device():
    # Priority: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Function to filter dataframes by patient IDs
def filter_by_patients(df, patient_ids):
    return df[df["patient"].isin(patient_ids)].copy()

# Generic parser for scalar or comma-separated list with target value type
def parse_value_or_list(raw_value, dtype, to_lower=None):
    items = [x.strip() for x in str(raw_value).split(",") if x.strip()]
    if not items:
        raise ValueError(f"Empty value received for parameter '{raw_value}'.")
    casted = [dtype(x) for x in items]

    if dtype is str:
        use_lower = True if to_lower is None else bool(to_lower)
        if use_lower:
            casted = [x.lower() for x in casted]

    return casted

# Format hp_name for runs' names
def _format_hp_name(cfg, missing_pct, missing_scope, missing_location):
    lr_str = f"{cfg['learning_rate']:.0e}"
    bs_str = str(cfg["batch_size"])
    fusion_str = str(cfg["fusion_hidden_dim"])
    modality_layers_str = str(cfg["modality_hidden_layers"])
    fusion_layers_str = str(cfg["fusion_hidden_layers"])
    dropout_str = str(cfg["dropout"]).replace(".", "p")
    return (
        f"lr{lr_str}_"
        f"bs{bs_str}_"
        f"modL{modality_layers_str}_"
        f"fusion{fusion_str}_"
        f"fusionL{fusion_layers_str}_"
        f"drop{dropout_str}_"
        f"miss{missing_pct}_"
        f"scope{missing_scope}_"
        f"loc{missing_location}"
    )

# Function to build hyperparameter grid from args (supports scalar and comma-separated list for each HP argument)
def build_hyperparameter_grid(args, missing_prob, missing_scope, missing_location):
    # missingness config
    missing_pct = f"{float(missing_prob) * 100:g}"
    missing_scope = str(missing_scope).strip().lower()
    missing_location = str(missing_location).strip().lower()

    # hp configs
    batch_sizes = parse_value_or_list(args.batch_size, int)
    learning_rates = parse_value_or_list(args.learning_rate, float)
    modality_hidden_layers = parse_value_or_list(args.modality_hidden_layers, int)
    fusion_hidden_dims = parse_value_or_list(args.fusion_hidden_dim, int)
    fusion_hidden_layers = parse_value_or_list(args.fusion_hidden_layers, int)
    dropouts = parse_value_or_list(args.dropout, float)

    hp_configs = []
    seen = set()
    for bs, lr, mod_layers, fusion_dim, fusion_layers, dropout in product(
        batch_sizes,
        learning_rates,
        modality_hidden_layers,
        fusion_hidden_dims,
        fusion_hidden_layers,
        dropouts,
    ):
        cfg = {
            "batch_size": int(bs),
            "learning_rate": float(lr),
            "modality_hidden_layers": int(mod_layers),
            "fusion_hidden_dim": int(fusion_dim),
            "fusion_hidden_layers": int(fusion_layers),
            "dropout": float(dropout),
        }
        key = (
            cfg["batch_size"],
            cfg["learning_rate"],
            cfg["modality_hidden_layers"],
            cfg["fusion_hidden_dim"],
            cfg["fusion_hidden_layers"],
            cfg["dropout"],
        )
        if key in seen:
            continue
        seen.add(key)
        cfg["name"] = _format_hp_name(cfg, missing_pct, missing_scope, missing_location)
        hp_configs.append(cfg)

    if not hp_configs:
        raise ValueError("No hyperparameter combinations were generated.")

    return hp_configs

# Function to scale features of each modality using only traiing data
def fit_and_transform_modalities(dfs_train_raw, dfs_eval_raw):
    dfs_train_scaled = {}
    dfs_eval_scaled = {}
    scalers = {}

    for name in dfs_train_raw.keys():
        df_tr = dfs_train_raw[name].copy()
        df_ev = dfs_eval_raw[name].copy()
        feats = [c for c in df_tr.columns if c != "patient"]

        scaler = StandardScaler()
        if len(df_tr) > 0 and feats:
            tr_values = df_tr[feats].to_numpy(dtype=np.float32, copy=True)
            df_tr[feats] = scaler.fit_transform(tr_values).astype(np.float32)

            if len(df_ev) > 0:
                ev_values = df_ev[feats].to_numpy(dtype=np.float32, copy=True)
                df_ev[feats] = scaler.transform(ev_values).astype(np.float32)

        dfs_train_scaled[name] = df_tr
        dfs_eval_scaled[name] = df_ev
        scalers[name] = scaler

    return dfs_train_scaled, dfs_eval_scaled, scalers

def safe_binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_prob_safe = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        aucpr = average_precision_score(y_true, y_prob)
    else:
        auc = 0.5
        aucpr = float(y_true.mean())

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "AUC": auc,
        "AUCPR": aucpr,
        "ACC": accuracy_score(y_true, y_pred),
        "SEN": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "SP": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "MCC": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
        "LOGLOSS": float(-(y_true * np.log(y_prob_safe) + (1 - y_true) * np.log(1 - y_prob_safe)).mean()),
        "Predicted_Probas": y_prob.tolist(),
        "Binary_Preds": y_pred.tolist(),
        "True_Labels": y_true.tolist(),
    }
