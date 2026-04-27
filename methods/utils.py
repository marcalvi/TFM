import numpy as np
import torch
import pandas as pd
import random
from itertools import product
from models import MultimodalMLP, PAM, PAMDiPAM, MLPDiPAM, SMILE, HealNetBinaryWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

# ------------------------ 0. CONFIGURATION --------------------------

def normalize_model_name(model_name):
    """Map CLI model names/aliases to a canonical lowercase identifier."""
    name = str(model_name).strip().lower()
    compact = name.replace("_", "").replace("-", "")
    if compact == "mlp":
        return "mlp"
    if compact == "pam":
        return "pam"
    if compact == "pamdipam":
        return "pam_dipam"
    if compact == "mlpdipam":
        return "mlp_dipam"
    if compact in {"smile", "smilee", "smilextended"}:
        return "smil_e"
    if compact in {"metasmile", "metasmilee", "metasmilextended"}:
        return "smil_e"
    if compact == "healnet":
        return "healnet"
    return name

# Function to select device (CUDA, MPS, or CPU)
def select_device():
    # Priority: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Function to set global seed for reproducibility across random, numpy, and torch (including CUDA)
def set_global_seed(seed, deterministic=True):
    """Seed python/numpy/torch RNGs for reproducible runs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# ------------------------ 1. PREPROCESSING --------------------------

# Function to scale features of each modality using only traiing data
def fit_and_transform_modalities(dfs_train_raw, dfs_eval_raw, id_col="patient"):
    dfs_train_scaled = {}
    dfs_eval_scaled = {}
    scalers = {}

    for name in dfs_train_raw.keys():
        df_tr = dfs_train_raw[name].copy()
        df_ev = dfs_eval_raw[name].copy()
        feats = [c for c in df_tr.columns if c != id_col]

        scaler = StandardScaler()
        if len(df_tr) > 0 and feats:
            tr_values = df_tr[feats].to_numpy(dtype=np.float32, copy=True)
            tr_transformed = scaler.fit_transform(tr_values).astype(np.float32)
            tr_feat_df = pd.DataFrame(tr_transformed, columns=feats, index=df_tr.index)
            base_cols_tr = [c for c in df_tr.columns if c not in feats]
            df_tr = pd.concat([df_tr[base_cols_tr], tr_feat_df], axis=1)

            if len(df_ev) > 0:
                ev_values = df_ev[feats].to_numpy(dtype=np.float32, copy=True)
                ev_transformed = scaler.transform(ev_values).astype(np.float32)
                ev_feat_df = pd.DataFrame(ev_transformed, columns=feats, index=df_ev.index)
                base_cols_ev = [c for c in df_ev.columns if c not in feats]
                df_ev = pd.concat([df_ev[base_cols_ev], ev_feat_df], axis=1)

        dfs_train_scaled[name] = df_tr
        dfs_eval_scaled[name] = df_ev
        scalers[name] = scaler

    return dfs_train_scaled, dfs_eval_scaled, scalers

# ------------------------ 2. HYPERPARAMETERS ------------------------

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


def parse_bool_value_or_list(raw_value):
    items = [x.strip() for x in str(raw_value).split(",") if x.strip()]
    if not items:
        raise ValueError(f"Empty value received for parameter '{raw_value}'.")
    out = []
    for item in items:
        item_l = item.lower()
        if item_l in {"1", "true", "yes", "y"}:
            out.append(True)
        elif item_l in {"0", "false", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(
                f"Invalid boolean value '{item}'. Use one of: true, false, 1, 0, yes, no."
            )
    return out


def parse_paired_hp_groups(raw_value):
    groups = []
    for raw_group in [chunk.strip() for chunk in str(raw_value or "").split(";") if chunk.strip()]:
        group = tuple(item.strip() for item in raw_group.split(",") if item.strip())
        if len(group) < 2:
            raise ValueError(
                f"Invalid paired hyperparameter group '{raw_group}'. Use at least two names separated by commas."
            )
        groups.append(group)
    return groups


def _format_training_control_suffix(cfg):
    wd_str = f"{float(cfg['weight_decay']):.0e}"
    return f"_wd{wd_str}"


# Format hp_name for runs' names
def _format_hp_name(cfg, train_missing_pct, missing_location, model_name):
    model_name = normalize_model_name(model_name)
    lr_str = f"{cfg['learning_rate']:.0e}"
    bs_str = str(cfg["batch_size"])
    training_suffix = _format_training_control_suffix(cfg)
    if model_name in {"pam"}:
        dropout_str = str(cfg["pam_dropout"]).replace(".", "p")
        temp_str = str(cfg["pam_temperature"]).replace(".", "p")
        return (
            f"lr{lr_str}_"
            f"bs{bs_str}_"
            f"drop{dropout_str}_"
            f"temp{temp_str}_"
            f"{training_suffix}"
            f"trmiss{train_missing_pct}_"
            f"trloc{missing_location}"
        )
    if model_name in {"pam_dipam", "mlp_dipam"}:
        dropout_str = str(cfg["pam_dropout"]).replace(".", "p")
        temp_str = str(cfg["pam_temperature"]).replace(".", "p")
        alpha_str = str(cfg["distill_alpha"]).replace(".", "p")
        beta_str = str(cfg["distill_beta"]).replace(".", "p")
        return (
            f"lr{lr_str}_"
            f"bs{bs_str}_"
            f"drop{dropout_str}_"
            f"temp{temp_str}_"
            f"a{alpha_str}_"
            f"b{beta_str}_"
            f"{training_suffix}"
            f"trmiss{train_missing_pct}_"
            f"trloc{missing_location}"
        )
    if model_name in {"smil_e"}:
        latent_str = str(cfg["smil_e_latent_dim"])
        priors_str = str(cfg["smil_e_num_priors"])
        heads_str = str(cfg["smil_e_num_heads"])
        dropout_str = str(cfg["smil_e_dropout"]).replace(".", "p")
        cls_hidden_str = str(cfg["classifier_hidden_dim"])
        meta_learning = bool(cfg.get("meta_learning", False))
        meta_flag = "1" if meta_learning else "0"
        name = (
            f"lr{lr_str}_"
            f"bs{bs_str}_"
            f"lat{latent_str}_"
            f"pri{priors_str}_"
            f"heads{heads_str}_"
            f"drop{dropout_str}_"
            f"clf{cls_hidden_str}_"
            f"meta{meta_flag}_"
            f"{training_suffix}"
            f"trmiss{train_missing_pct}_"
            f"trloc{missing_location}"
        )
        if meta_learning:
            alpha_str = str(cfg["smil_e_alpha"]).replace(".", "p")
            beta_str = str(cfg["smil_e_beta"]).replace(".", "p")
            meta_lr_str = f"{float(cfg['meta_inner_lr']):.0e}"
            meta_val_str = str(cfg["meta_val_fraction"]).replace(".", "p")
            name += (
                f"_alpha{alpha_str}_"
                f"beta{beta_str}_"
                f"milr{meta_lr_str}_"
                f"mvf{meta_val_str}"
            )
        return name
    if model_name in {"healnet"}:
        depth_str = str(cfg["healnet_depth"])
        nfb_str = str(cfg["healnet_num_freq_bands"])
        lat_str = str(cfg["healnet_num_latents"])
        ldim_str = str(cfg["healnet_latent_dim"])
        xh_str = str(cfg["healnet_cross_heads"])
        lh_str = str(cfg["healnet_latent_heads"])
        cdh_str = str(cfg["healnet_cross_dim_head"])
        ldh_str = str(cfg["healnet_latent_dim_head"])
        adrop_str = str(cfg["healnet_attn_dropout"]).replace(".", "p")
        fdrop_str = str(cfg["healnet_ff_dropout"]).replace(".", "p")
        selfx_str = str(cfg["healnet_self_per_cross_attn"])
        return (
            f"lr{lr_str}_"
            f"bs{bs_str}_"
            f"depth{depth_str}_"
            f"nfb{nfb_str}_"
            f"lat{lat_str}_"
            f"ldim{ldim_str}_"
            f"xh{xh_str}_"
            f"lh{lh_str}_"
            f"cdh{cdh_str}_"
            f"ldh{ldh_str}_"
            f"adrop{adrop_str}_"
            f"fdrop{fdrop_str}_"
            f"selfx{selfx_str}_"
            f"{training_suffix}"
            f"trmiss{train_missing_pct}_"
            f"trloc{missing_location}"
        )

    fusion_str = str(cfg["fusion_hidden_dim"])
    modality_layers_str = str(cfg["modality_hidden_layers"])
    fusion_layers_str = str(cfg["fusion_hidden_layers"])
    fusion_bn_str = "1" if bool(cfg.get("fusion_batchnorm", False)) else "0"
    dropout_str = str(cfg["dropout"]).replace(".", "p")
    return (
        f"lr{lr_str}_"
        f"bs{bs_str}_"
        f"modL{modality_layers_str}_"
        f"fusion{fusion_str}_"
        f"fusionL{fusion_layers_str}_"
        f"fusionBN{fusion_bn_str}_"
        f"drop{dropout_str}_"
        f"{training_suffix}"
        f"trmiss{train_missing_pct}_"
        f"trloc{missing_location}"
    )

# Function to build hyperparameter grid from args (supports scalar and comma-separated list for each HP argument)
def build_hyperparameter_grid(args, train_missing_prop, missing_location):
    # Train missingness config (used in run naming for reproducibility).
    train_missing_pct = f"{float(train_missing_prop) * 100:g}"
    missing_location = str(missing_location).strip().lower()
    model_name = normalize_model_name(args.model)

    # hp configs
    batch_sizes = parse_value_or_list(args.batch_size, int)
    learning_rates = parse_value_or_list(args.learning_rate, float)
    weight_decays = parse_value_or_list(args.weight_decay, float)

    hp_configs = []
    seen = set()

    if model_name in {"pam"}:
        pam_dropouts = parse_value_or_list(args.pam_dropout, float)
        temperatures = parse_value_or_list(args.pam_temperature, float)

        for bs, lr, weight_decay, dropout, temp in product(
            batch_sizes,
            learning_rates,
            weight_decays,
            pam_dropouts,
            temperatures,
        ):
            cfg = {
                "batch_size": int(bs),
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "pam_dropout": float(dropout),
                "pam_temperature": float(temp),
            }
            key = (
                cfg["batch_size"],
                cfg["learning_rate"],
                cfg["weight_decay"],
                cfg["pam_dropout"],
                cfg["pam_temperature"],
            )
            if key in seen:
                continue
            seen.add(key)
            cfg["name"] = _format_hp_name(
                cfg, train_missing_pct, missing_location, model_name=model_name
            )
            hp_configs.append(cfg)
    if model_name in {"pam_dipam", "mlp_dipam"}:
        pam_dropouts = parse_value_or_list(args.pam_dropout, float)
        temperatures = parse_value_or_list(args.pam_temperature, float)
        distill_alphas = parse_value_or_list(args.distill_alpha, float)
        distill_betas = parse_value_or_list(args.distill_beta, float)

        for bs, lr, weight_decay, dropout, temp, alpha, beta in product(
            batch_sizes,
            learning_rates,
            weight_decays,
            pam_dropouts,
            temperatures,
            distill_alphas,
            distill_betas,
        ):
            cfg = {
                "batch_size": int(bs),
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "pam_dropout": float(dropout),
                "pam_temperature": float(temp),
                "distill_alpha": float(alpha),
                "distill_beta": float(beta),
            }
            key = (
                cfg["batch_size"],
                cfg["learning_rate"],
                cfg["weight_decay"],
                cfg["pam_dropout"],
                cfg["pam_temperature"],
                cfg["distill_alpha"],
                cfg["distill_beta"],
            )
            if key in seen:
                continue
            seen.add(key)
            cfg["name"] = _format_hp_name(
                cfg, train_missing_pct, missing_location, model_name=model_name
            )
            hp_configs.append(cfg)
    if model_name in {"mlp"}:
        modality_hidden_layers = parse_value_or_list(args.modality_hidden_layers, int)
        fusion_hidden_dims = parse_value_or_list(args.fusion_hidden_dim, int)
        fusion_hidden_layers = parse_value_or_list(args.fusion_hidden_layers, int)
        fusion_batchnorm_values = parse_bool_value_or_list(args.fusion_batchnorm)
        dropouts = parse_value_or_list(args.dropout, float)

        for bs, lr, weight_decay, mod_layers, fusion_dim, fusion_layers, fusion_batchnorm, dropout in product(
            batch_sizes,
            learning_rates,
            weight_decays,
            modality_hidden_layers,
            fusion_hidden_dims,
            fusion_hidden_layers,
            fusion_batchnorm_values,
            dropouts,
        ):
            cfg = {
                "batch_size": int(bs),
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "modality_hidden_layers": int(mod_layers),
                "fusion_hidden_dim": int(fusion_dim),
                "fusion_hidden_layers": int(fusion_layers),
                "fusion_batchnorm": bool(fusion_batchnorm),
                "dropout": float(dropout),
            }
            key = (
                cfg["batch_size"],
                cfg["learning_rate"],
                cfg["weight_decay"],
                cfg["modality_hidden_layers"],
                cfg["fusion_hidden_dim"],
                cfg["fusion_hidden_layers"],
                cfg["fusion_batchnorm"],
                cfg["dropout"],
            )
            if key in seen:
                continue
            seen.add(key)
            cfg["name"] = _format_hp_name(
                cfg, train_missing_pct, missing_location, model_name=model_name
            )
            hp_configs.append(cfg)
    if model_name in {"smil_e"}:
        smil_e_latent_dims = parse_value_or_list(args.smil_e_latent_dim, int)
        smil_e_num_priors = parse_value_or_list(args.smil_e_num_priors, int)
        smil_e_num_heads = parse_value_or_list(args.smil_e_num_heads, int)
        smil_e_dropouts = parse_value_or_list(args.smil_e_dropout, float)
        classifier_hidden_dims = parse_value_or_list(args.classifier_hidden_dim, int)
        meta_learning = bool(getattr(args, "meta_learning", False))
        smil_e_alphas = parse_value_or_list(args.smil_e_alpha, float)
        smil_e_betas = parse_value_or_list(args.smil_e_beta, float)
        alpha_beta_options = (
            list(product(smil_e_alphas, smil_e_betas))
            if meta_learning
            else [(float(smil_e_alphas[0]), float(smil_e_betas[0]))]
        )

        for bs, lr, weight_decay, latent_dim, num_priors, num_heads, dropout, classifier_hidden_dim, alpha_beta in product(
            batch_sizes,
            learning_rates,
            weight_decays,
            smil_e_latent_dims,
            smil_e_num_priors,
            smil_e_num_heads,
            smil_e_dropouts,
            classifier_hidden_dims,
            alpha_beta_options,
        ):
            if int(latent_dim) % int(num_heads) != 0:
                continue
            alpha, beta = alpha_beta
            cfg = {
                "batch_size": int(bs),
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "smil_e_latent_dim": int(latent_dim),
                "smil_e_num_priors": int(num_priors),
                "smil_e_num_heads": int(num_heads),
                "smil_e_dropout": float(dropout),
                "classifier_hidden_dim": int(classifier_hidden_dim),
                "smil_e_alpha": float(alpha),
                "smil_e_beta": float(beta),
                "meta_learning": bool(meta_learning),
                "meta_inner_lr": float(args.meta_inner_lr),
                "meta_val_fraction": float(args.meta_val_fraction),
            }
            key = (
                cfg["batch_size"],
                cfg["learning_rate"],
                cfg["weight_decay"],
                cfg["smil_e_latent_dim"],
                cfg["smil_e_num_priors"],
                cfg["smil_e_num_heads"],
                cfg["smil_e_dropout"],
                cfg["classifier_hidden_dim"],
                cfg["meta_learning"],
                cfg["smil_e_alpha"] if cfg["meta_learning"] else None,
                cfg["smil_e_beta"] if cfg["meta_learning"] else None,
                cfg["meta_inner_lr"] if cfg["meta_learning"] else None,
                cfg["meta_val_fraction"] if cfg["meta_learning"] else None,
            )
            if key in seen:
                continue
            seen.add(key)
            cfg["name"] = _format_hp_name(
                cfg, train_missing_pct, missing_location, model_name=model_name
            )
            hp_configs.append(cfg)
    if model_name in {"healnet"}:
        healnet_depths = parse_value_or_list(args.healnet_depth, int)
        healnet_num_freq_bands = parse_value_or_list(args.healnet_num_freq_bands, int)
        healnet_num_latents = parse_value_or_list(args.healnet_num_latents, int)
        healnet_latent_dims = parse_value_or_list(args.healnet_latent_dim, int)
        healnet_cross_heads = parse_value_or_list(args.healnet_cross_heads, int)
        healnet_latent_heads = parse_value_or_list(args.healnet_latent_heads, int)
        healnet_cross_dim_head = parse_value_or_list(args.healnet_cross_dim_head, int)
        healnet_latent_dim_head = parse_value_or_list(args.healnet_latent_dim_head, int)
        healnet_attn_dropout = parse_value_or_list(args.healnet_attn_dropout, float)
        healnet_ff_dropout = parse_value_or_list(args.healnet_ff_dropout, float)
        healnet_self_per_cross_attn = parse_value_or_list(args.healnet_self_per_cross_attn, int)
        paired_groups = {tuple(group) for group in parse_paired_hp_groups(getattr(args, "paired_hp_groups", ""))}

        latent_bottleneck_options = [
            (int(num_latents), int(latent_dim))
            for num_latents, latent_dim in (
                zip(healnet_num_latents, healnet_latent_dims)
                if ("healnet_num_latents", "healnet_latent_dim") in paired_groups
                else product(healnet_num_latents, healnet_latent_dims)
            )
        ]
        attention_head_dim_options = [
            (int(cross_dim_head), int(latent_dim_head))
            for cross_dim_head, latent_dim_head in (
                zip(healnet_cross_dim_head, healnet_latent_dim_head)
                if ("healnet_cross_dim_head", "healnet_latent_dim_head") in paired_groups
                else product(healnet_cross_dim_head, healnet_latent_dim_head)
            )
        ]

        if ("healnet_num_latents", "healnet_latent_dim") in paired_groups and len(healnet_num_latents) != len(healnet_latent_dims):
            raise ValueError(
                "Paired hyperparameters 'healnet_num_latents' and 'healnet_latent_dim' must define the same number of values."
            )
        if ("healnet_cross_dim_head", "healnet_latent_dim_head") in paired_groups and len(healnet_cross_dim_head) != len(healnet_latent_dim_head):
            raise ValueError(
                "Paired hyperparameters 'healnet_cross_dim_head' and 'healnet_latent_dim_head' must define the same number of values."
            )

        for (
            bs,
            lr,
            depth,
            num_freq_bands,
            weight_decay,
            latent_bottleneck_pair,
            cross_heads,
            latent_heads,
            attention_head_dim_pair,
            attn_dropout,
            ff_dropout,
            self_per_cross_attn,
        ) in product(
            batch_sizes,
            learning_rates,
            healnet_depths,
            healnet_num_freq_bands,
            weight_decays,
            latent_bottleneck_options,
            healnet_cross_heads,
            healnet_latent_heads,
            attention_head_dim_options,
            healnet_attn_dropout,
            healnet_ff_dropout,
            healnet_self_per_cross_attn,
        ):
            num_latents, latent_dim = latent_bottleneck_pair
            cross_dim_head, latent_dim_head = attention_head_dim_pair
            cfg = {
                "batch_size": int(bs),
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "healnet_depth": int(depth),
                "healnet_num_freq_bands": int(num_freq_bands),
                "healnet_num_latents": int(num_latents),
                "healnet_latent_dim": int(latent_dim),
                "healnet_cross_heads": int(cross_heads),
                "healnet_latent_heads": int(latent_heads),
                "healnet_cross_dim_head": int(cross_dim_head),
                "healnet_latent_dim_head": int(latent_dim_head),
                "healnet_attn_dropout": float(attn_dropout),
                "healnet_ff_dropout": float(ff_dropout),
                "healnet_self_per_cross_attn": int(self_per_cross_attn),
            }
            key = (
                cfg["batch_size"],
                cfg["learning_rate"],
                cfg["weight_decay"],
                cfg["healnet_depth"],
                cfg["healnet_num_freq_bands"],
                cfg["healnet_num_latents"],
                cfg["healnet_latent_dim"],
                cfg["healnet_cross_heads"],
                cfg["healnet_latent_heads"],
                cfg["healnet_cross_dim_head"],
                cfg["healnet_latent_dim_head"],
                cfg["healnet_attn_dropout"],
                cfg["healnet_ff_dropout"],
                cfg["healnet_self_per_cross_attn"],
            )
            if key in seen:
                continue
            seen.add(key)
            cfg["name"] = _format_hp_name(
                cfg, train_missing_pct, missing_location, model_name=model_name
            )
            hp_configs.append(cfg)

    if not hp_configs:
        raise ValueError("No hyperparameter combinations were generated.")

    return hp_configs

# ------------------------ 4. MODEL TRAINING --------------------------

# Helper function to build model based on name and input dimensions
def build_model(model_name, input_dims, model_kwargs):
    model_name = normalize_model_name(model_name)
    if model_name in {"mlp"}:
        return MultimodalMLP(input_dims, **model_kwargs)
    if model_name in {"pam"}:
        return PAM(input_dims, **model_kwargs)
    if model_name in {"pam_dipam"}:
        return PAMDiPAM(input_dims, **model_kwargs)
    if model_name in {"mlp_dipam"}:
        return MLPDiPAM(input_dims, **model_kwargs)
    if model_name in {"smil_e"}:
        init_kwargs = dict(model_kwargs or {})
        for key in {"meta_learning", "meta_inner_lr", "meta_val_fraction", "meta_inner_steps"}:
            init_kwargs.pop(key, None)
        return SMILE(input_dims, **init_kwargs)
    if model_name in {"healnet"}:
        return HealNetBinaryWrapper(input_dims, **model_kwargs)
    raise ValueError(
        f"Unsupported model '{model_name}'. Supported: mlp, pam, pam_dipam, mlp_dipam, smile, healnet"
    )

# -------------------------- 5. EVALUATION ----------------------------

# Metric calculation for binary classification that handles edge cases gracefully
def safe_binary_metrics(y_true, y_prob, include_raw=False):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            f"safe_binary_metrics expects same length for y_true and y_prob, got "
            f"{y_true.shape[0]} and {y_prob.shape[0]}."
        )
    if y_true.shape[0] == 0:
        raise ValueError("safe_binary_metrics received empty arrays.")

    y_prob_safe = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = y_prob >= 0.5
    y_true_pos = y_true == 1

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
        aucpr = float(average_precision_score(y_true, y_prob))
    else:
        auc = 0.5
        aucpr = float(y_true.mean())

    tp = int(np.sum(y_true_pos & y_pred))
    tn = int(np.sum(~y_true_pos & ~y_pred))
    fp = int(np.sum(~y_true_pos & y_pred))
    fn = int(np.sum(y_true_pos & ~y_pred))

    total = tp + tn + fp + fn
    acc = float((tp + tn) / total) if total > 0 else 0.0
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    sp = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    mcc_den = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float(((tp * tn) - (fp * fn)) / mcc_den) if mcc_den > 0 else 0.0

    metrics = {
        "AUC": auc,
        "AUCPR": aucpr,
        "ACC": acc,
        "SEN": sen,
        "SP": sp,
        "MCC": mcc,
        "LOGLOSS": float(-(y_true * np.log(y_prob_safe) + (1 - y_true) * np.log(1 - y_prob_safe)).mean()),
    }
    if include_raw:
        metrics["Predicted_Probas"] = y_prob.tolist()
        metrics["Binary_Preds"] = y_pred.astype(np.int8).tolist()
        metrics["True_Labels"] = y_true.tolist()
    return metrics
