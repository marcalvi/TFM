import numpy as np
import torch
import pandas as pd
import random
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

# ------------------------ 0. CONFIGURATION --------------------------

SURVIVAL_Y_DISC_COL = "y_disc"
SURVIVAL_CENSORSHIP_COL = "censorship"


def normalize_model_name(model_name):
    """Map CLI model names/aliases to a canonical lowercase identifier."""
    name = str(model_name).strip().lower()
    compact = name.replace("_", "").replace("-", "")
    if compact in {"lr", "logisticregression", "logreg"}:
        return "lr"
    if compact in {"rf", "randomforest", "randomforestclassifier", "randomforestbaseline"}:
        return "rf"
    if compact in {"coxnet", "coxnetbaseline"}:
        return "coxnet"
    if compact in {"rsf", "randomsurvivalforest", "randomsurvivalforestbaseline"}:
        return "rsf"
    if compact == "mlp":
        return "mlp"
    if compact == "pam":
        return "pam"
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


def normalize_task_type(task_type):
    task_type_l = str(task_type).strip().lower()
    if task_type_l == "classification":
        return "binary_classification"
    if task_type_l in {"binary_classification", "survival"}:
        return task_type_l
    return task_type_l

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


def _append_distillation_suffix(name, cfg):
    if not bool(cfg.get("knowledge_distillation", False)):
        return name
    alpha_str = str(cfg.get("distill_alpha", 1.0)).replace(".", "p")
    beta_str = str(cfg.get("distill_beta", 0.3)).replace(".", "p")
    return f"{name}_kd_a{alpha_str}_b{beta_str}"


# Format hp_name for runs' names
def _format_hp_name(cfg, train_missing_pct, degrading_modality, model_name):
    model_name = normalize_model_name(model_name)
    lr_str = f"{cfg['learning_rate']:.0e}"
    bs_str = str(cfg["batch_size"])
    training_suffix = _format_training_control_suffix(cfg)
    if model_name in {"lr"}:
        c_str = f"{float(cfg['lr_C']):g}".replace(".", "p")
        penalty_str = str(cfg["lr_penalty"]).replace("_", "")
        solver_str = str(cfg["lr_solver"]).replace("_", "")
        cw_str = str(cfg["lr_class_weight"]).replace("_", "")
        return (
            f"C{c_str}_"
            f"pen{penalty_str}_"
            f"solver{solver_str}_"
            f"cw{cw_str}_"
            f"maxit{int(cfg['lr_max_iter'])}_"
            f"trmiss{train_missing_pct}_"
            f"degmod{degrading_modality}"
        )
    if model_name in {"rf"}:
        depth_str = str(cfg["rf_max_depth"]).replace("_", "").replace(".", "p")
        features_str = str(cfg["rf_max_features"]).replace("_", "").replace(".", "p")
        cw_str = str(cfg["rf_class_weight"]).replace("_", "")
        return (
            f"trees{int(cfg['rf_n_estimators'])}_"
            f"depth{depth_str}_"
            f"split{int(cfg['rf_min_samples_split'])}_"
            f"leaf{int(cfg['rf_min_samples_leaf'])}_"
            f"feat{features_str}_"
            f"cw{cw_str}_"
            f"trmiss{train_missing_pct}_"
            f"degmod{degrading_modality}"
        )
    if model_name in {"coxnet"}:
        alpha_str = f"{float(cfg['coxnet_alpha']):g}".replace(".", "p")
        l1_str = f"{float(cfg['coxnet_l1_ratio']):g}".replace(".", "p")
        tol_str = f"{float(cfg['coxnet_tol']):.0e}"
        return (
            f"alpha{alpha_str}_"
            f"l1{l1_str}_"
            f"maxit{int(cfg['coxnet_max_iter'])}_"
            f"tol{tol_str}_"
            f"trmiss{train_missing_pct}_"
            f"degmod{degrading_modality}"
        )
    if model_name in {"rsf"}:
        depth_str = str(cfg["rsf_max_depth"]).replace("_", "").replace(".", "p")
        features_str = str(cfg["rsf_max_features"]).replace("_", "").replace(".", "p")
        return (
            f"trees{int(cfg['rsf_n_estimators'])}_"
            f"depth{depth_str}_"
            f"split{int(cfg['rsf_min_samples_split'])}_"
            f"leaf{int(cfg['rsf_min_samples_leaf'])}_"
            f"feat{features_str}_"
            f"trmiss{train_missing_pct}_"
            f"degmod{degrading_modality}"
        )
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
            f"degmod{degrading_modality}"
        )
    if model_name in {"smil_e"}:
        latent_str = str(cfg["smil_e_latent_dim"])
        priors_str = str(cfg["smil_e_num_priors"])
        heads_str = str(cfg["smil_e_num_heads"])
        dropout_str = str(cfg["smil_e_dropout"]).replace(".", "p")
        cls_hidden_str = str(cfg["classifier_hidden_dim"])
        alpha_str = str(cfg["smil_e_alpha"]).replace(".", "p")
        beta_str = str(cfg["smil_e_beta"]).replace(".", "p")
        meta_lr_str = f"{float(cfg['meta_inner_lr']):.0e}"
        meta_val_str = str(cfg["meta_val_fraction"]).replace(".", "p")
        name = (
            f"lr{lr_str}_"
            f"bs{bs_str}_"
            f"lat{latent_str}_"
            f"pri{priors_str}_"
            f"heads{heads_str}_"
            f"drop{dropout_str}_"
            f"clf{cls_hidden_str}_"
            f"alpha{alpha_str}_"
            f"beta{beta_str}_"
            f"milr{meta_lr_str}_"
            f"mvf{meta_val_str}_"
            f"{training_suffix}"
            f"trmiss{train_missing_pct}_"
            f"degmod{degrading_modality}"
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
            f"degmod{degrading_modality}"
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
        f"degmod{degrading_modality}"
    )

# Function to build hyperparameter grid from args (supports scalar and comma-separated list for each HP argument)
def build_hyperparameter_grid(args, train_missing_prop, degrading_modality):
    # Train missingness config (used in run naming for reproducibility).
    train_missing_pct = f"{float(train_missing_prop) * 100:g}"
    degrading_modality = str(degrading_modality).strip().lower()
    model_name = normalize_model_name(args.model)

    # hp configs
    batch_sizes = parse_value_or_list(args.batch_size, int)
    learning_rates = parse_value_or_list(args.learning_rate, float)
    weight_decays = parse_value_or_list(args.weight_decay, float)

    hp_configs = []
    seen = set()

    if model_name in {"lr"}:
        lr_C_values = parse_value_or_list(args.lr_C, float)
        lr_penalties = parse_value_or_list(args.lr_penalty, str, to_lower=True)
        lr_solvers = parse_value_or_list(args.lr_solver, str, to_lower=True)
        lr_class_weights = parse_value_or_list(args.lr_class_weight, str, to_lower=True)
        lr_max_iters = parse_value_or_list(args.lr_max_iter, int)

        for bs, lr, weight_decay, lr_C, lr_penalty, lr_solver, lr_class_weight, lr_max_iter in product(
            batch_sizes,
            learning_rates,
            weight_decays,
            lr_C_values,
            lr_penalties,
            lr_solvers,
            lr_class_weights,
            lr_max_iters,
        ):
            cfg = {
                "batch_size": int(bs),
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "lr_C": float(lr_C),
                "lr_penalty": str(lr_penalty),
                "lr_solver": str(lr_solver),
                "lr_class_weight": str(lr_class_weight),
                "lr_max_iter": int(lr_max_iter),
            }
            key = (
                cfg["batch_size"],
                cfg["learning_rate"],
                cfg["weight_decay"],
                cfg["lr_C"],
                cfg["lr_penalty"],
                cfg["lr_solver"],
                cfg["lr_class_weight"],
                cfg["lr_max_iter"],
            )
            if key in seen:
                continue
            seen.add(key)
            cfg["name"] = _format_hp_name(
                cfg, train_missing_pct, degrading_modality, model_name=model_name
            )
            hp_configs.append(cfg)

    if model_name in {"rf"}:
        rf_n_estimators = parse_value_or_list(args.rf_n_estimators, int)
        rf_max_depths = parse_value_or_list(args.rf_max_depth, str, to_lower=True)
        rf_min_samples_splits = parse_value_or_list(args.rf_min_samples_split, int)
        rf_min_samples_leafs = parse_value_or_list(args.rf_min_samples_leaf, int)
        rf_max_features = parse_value_or_list(args.rf_max_features, str, to_lower=True)
        rf_class_weights = parse_value_or_list(args.rf_class_weight, str, to_lower=True)
        rf_n_jobs = parse_value_or_list(args.rf_n_jobs, int)

        for bs, lr, weight_decay, n_estimators, max_depth, min_split, min_leaf, max_features, class_weight, n_jobs in product(
            batch_sizes,
            learning_rates,
            weight_decays,
            rf_n_estimators,
            rf_max_depths,
            rf_min_samples_splits,
            rf_min_samples_leafs,
            rf_max_features,
            rf_class_weights,
            rf_n_jobs,
        ):
            cfg = {
                "batch_size": int(bs),
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "rf_n_estimators": int(n_estimators),
                "rf_max_depth": str(max_depth),
                "rf_min_samples_split": int(min_split),
                "rf_min_samples_leaf": int(min_leaf),
                "rf_max_features": str(max_features),
                "rf_class_weight": str(class_weight),
                "rf_n_jobs": int(n_jobs),
            }
            key = (
                cfg["batch_size"],
                cfg["learning_rate"],
                cfg["weight_decay"],
                cfg["rf_n_estimators"],
                cfg["rf_max_depth"],
                cfg["rf_min_samples_split"],
                cfg["rf_min_samples_leaf"],
                cfg["rf_max_features"],
                cfg["rf_class_weight"],
                cfg["rf_n_jobs"],
            )
            if key in seen:
                continue
            seen.add(key)
            cfg["name"] = _format_hp_name(
                cfg, train_missing_pct, degrading_modality, model_name=model_name
            )
            hp_configs.append(cfg)

    if model_name in {"coxnet"}:
        coxnet_alphas = parse_value_or_list(args.coxnet_alpha, float)
        coxnet_l1_ratios = parse_value_or_list(args.coxnet_l1_ratio, float)
        coxnet_max_iters = parse_value_or_list(args.coxnet_max_iter, int)
        coxnet_tols = parse_value_or_list(args.coxnet_tol, float)

        for bs, lr, weight_decay, alpha, l1_ratio, max_iter, tol in product(
            batch_sizes,
            learning_rates,
            weight_decays,
            coxnet_alphas,
            coxnet_l1_ratios,
            coxnet_max_iters,
            coxnet_tols,
        ):
            cfg = {
                "batch_size": int(bs),
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "coxnet_alpha": float(alpha),
                "coxnet_l1_ratio": float(l1_ratio),
                "coxnet_max_iter": int(max_iter),
                "coxnet_tol": float(tol),
            }
            key = (
                cfg["batch_size"],
                cfg["learning_rate"],
                cfg["weight_decay"],
                cfg["coxnet_alpha"],
                cfg["coxnet_l1_ratio"],
                cfg["coxnet_max_iter"],
                cfg["coxnet_tol"],
            )
            if key in seen:
                continue
            seen.add(key)
            cfg["name"] = _format_hp_name(
                cfg, train_missing_pct, degrading_modality, model_name=model_name
            )
            hp_configs.append(cfg)

    if model_name in {"rsf"}:
        rsf_n_estimators = parse_value_or_list(args.rsf_n_estimators, int)
        rsf_max_depths = parse_value_or_list(args.rsf_max_depth, str, to_lower=True)
        rsf_min_samples_splits = parse_value_or_list(args.rsf_min_samples_split, int)
        rsf_min_samples_leafs = parse_value_or_list(args.rsf_min_samples_leaf, int)
        rsf_max_features = parse_value_or_list(args.rsf_max_features, str, to_lower=True)
        rsf_n_jobs = parse_value_or_list(args.rsf_n_jobs, int)

        for bs, lr, weight_decay, n_estimators, max_depth, min_split, min_leaf, max_features, n_jobs in product(
            batch_sizes,
            learning_rates,
            weight_decays,
            rsf_n_estimators,
            rsf_max_depths,
            rsf_min_samples_splits,
            rsf_min_samples_leafs,
            rsf_max_features,
            rsf_n_jobs,
        ):
            cfg = {
                "batch_size": int(bs),
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "rsf_n_estimators": int(n_estimators),
                "rsf_max_depth": str(max_depth),
                "rsf_min_samples_split": int(min_split),
                "rsf_min_samples_leaf": int(min_leaf),
                "rsf_max_features": str(max_features),
                "rsf_n_jobs": int(n_jobs),
            }
            key = (
                cfg["batch_size"],
                cfg["learning_rate"],
                cfg["weight_decay"],
                cfg["rsf_n_estimators"],
                cfg["rsf_max_depth"],
                cfg["rsf_min_samples_split"],
                cfg["rsf_min_samples_leaf"],
                cfg["rsf_max_features"],
                cfg["rsf_n_jobs"],
            )
            if key in seen:
                continue
            seen.add(key)
            cfg["name"] = _format_hp_name(
                cfg, train_missing_pct, degrading_modality, model_name=model_name
            )
            hp_configs.append(cfg)

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
                cfg, train_missing_pct, degrading_modality, model_name=model_name
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
                cfg, train_missing_pct, degrading_modality, model_name=model_name
            )
            hp_configs.append(cfg)
    if model_name in {"smil_e"}:
        smil_e_latent_dims = parse_value_or_list(args.smil_e_latent_dim, int)
        smil_e_num_priors = parse_value_or_list(args.smil_e_num_priors, int)
        smil_e_num_heads = parse_value_or_list(args.smil_e_num_heads, int)
        smil_e_dropouts = parse_value_or_list(args.smil_e_dropout, float)
        classifier_hidden_dims = parse_value_or_list(args.classifier_hidden_dim, int)
        smil_e_alphas = parse_value_or_list(args.smil_e_alpha, float)
        smil_e_betas = parse_value_or_list(args.smil_e_beta, float)
        meta_inner_lrs = parse_value_or_list(args.meta_inner_lr, float)
        meta_val_fractions = parse_value_or_list(args.meta_val_fraction, float)
        paired_groups = {tuple(group) for group in parse_paired_hp_groups(getattr(args, "paired_hp_groups", ""))}

        if ("learning_rate", "meta_inner_lr") in paired_groups:
            if len(learning_rates) != len(meta_inner_lrs):
                raise ValueError(
                    "Paired hyperparameters 'learning_rate' and 'meta_inner_lr' must define the same number of values."
                )
            lr_meta_options = [
                (float(learning_rate), float(meta_inner_lr))
                for learning_rate, meta_inner_lr in zip(learning_rates, meta_inner_lrs)
            ]
        else:
            lr_meta_options = [
                (float(learning_rate), float(meta_inner_lr))
                for learning_rate, meta_inner_lr in product(learning_rates, meta_inner_lrs)
            ]

        for bs, lr_meta_pair, weight_decay, latent_dim, num_priors, num_heads, dropout, classifier_hidden_dim, alpha, beta, meta_val_fraction in product(
            batch_sizes,
            lr_meta_options,
            weight_decays,
            smil_e_latent_dims,
            smil_e_num_priors,
            smil_e_num_heads,
            smil_e_dropouts,
            classifier_hidden_dims,
            smil_e_alphas,
            smil_e_betas,
            meta_val_fractions,
        ):
            lr, meta_inner_lr = lr_meta_pair
            if int(latent_dim) % int(num_heads) != 0:
                continue
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
                "meta_inner_lr": float(meta_inner_lr),
                "meta_val_fraction": float(meta_val_fraction),
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
                cfg["smil_e_alpha"],
                cfg["smil_e_beta"],
                cfg["meta_inner_lr"],
                cfg["meta_val_fraction"],
            )
            if key in seen:
                continue
            seen.add(key)
            cfg["name"] = _format_hp_name(
                cfg, train_missing_pct, degrading_modality, model_name=model_name
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
                cfg, train_missing_pct, degrading_modality, model_name=model_name
            )
            hp_configs.append(cfg)

    if not hp_configs:
        raise ValueError("No hyperparameter combinations were generated.")

    if bool(getattr(args, "knowledge_distillation", False)):
        if model_name in {"lr", "rf", "coxnet", "rsf"}:
            raise ValueError("Knowledge distillation is available only for torch models, not sklearn baselines.")
        if model_name in {"smil_e"}:
            raise ValueError("Knowledge distillation is not enabled for SMILe because it uses a dedicated meta-learning training loop.")
        distill_alphas = parse_value_or_list(args.distill_alpha, float)
        distill_betas = parse_value_or_list(args.distill_beta, float)
        expanded_configs = []
        for base_cfg in hp_configs:
            for alpha, beta in product(distill_alphas, distill_betas):
                cfg = dict(base_cfg)
                cfg["knowledge_distillation"] = True
                cfg["distill_alpha"] = float(alpha)
                cfg["distill_beta"] = float(beta)
                cfg["name"] = _append_distillation_suffix(str(base_cfg["name"]), cfg)
                expanded_configs.append(cfg)
        hp_configs = expanded_configs

    return hp_configs

# ------------------------ 3. SURVIVAL UTILS --------------------------

def _normalize_binary_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isna().all():
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
        "censor": 0.0,
    }
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.map(mapping)


def add_survival_target_columns(
    endpoint_df,
    patient_id_col,
    time_col,
    event_col,
    n_bins=4,
    y_disc_col=SURVIVAL_Y_DISC_COL,
    censorship_col=SURVIVAL_CENSORSHIP_COL,
):
    work_df = endpoint_df.copy()
    required_cols = [patient_id_col, time_col, event_col]
    for required_col in required_cols:
        if required_col not in work_df.columns:
            raise ValueError(
                f"Required survival column '{required_col}' not found in endpoint CSV."
            )

    event_time = pd.to_numeric(work_df[time_col], errors="coerce")
    event_observed = _normalize_binary_series(work_df[event_col])
    invalid_mask = event_time.isna() | event_observed.isna()
    if invalid_mask.any():
        invalid_examples = (
            work_df.loc[invalid_mask, required_cols]
            .head(10)
            .to_dict("records")
        )
        n_invalid = int(invalid_mask.sum())
        print(
            f"[survival] Dropping {n_invalid} rows with invalid time/event values. "
            f"Examples: {invalid_examples}"
        )
        work_df = work_df.loc[~invalid_mask].copy()
        event_time = event_time.loc[~invalid_mask].copy()
        event_observed = event_observed.loc[~invalid_mask].copy()

    if work_df.empty:
        raise ValueError("No valid endpoint rows left after survival target filtering.")

    event_observed = event_observed.astype(np.int64)
    invalid_binary_mask = ~event_observed.isin([0, 1])
    if invalid_binary_mask.any():
        invalid_examples = (
            work_df.loc[invalid_binary_mask, [patient_id_col, event_col]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"Survival event column '{event_col}' must contain binary values. "
            f"Examples of invalid rows: {invalid_examples}"
        )

    work_df[time_col] = event_time.astype(np.float32)
    work_df[event_col] = event_observed.astype(np.int64)
    work_df[censorship_col] = (1 - work_df[event_col].astype(np.int64)).astype(np.int64)

    times = work_df[time_col].astype(np.float64)
    if times.nunique() <= 1:
        work_df[y_disc_col] = 0
        bin_edges = np.array([times.min() - 1e-6, times.max() + 1e-6], dtype=np.float64)
        return work_df, bin_edges

    requested_bins = max(int(n_bins), 2)
    try:
        y_disc, bin_edges = pd.qcut(
            times,
            q=requested_bins,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
    except ValueError:
        min_t = float(times.min()) - 1e-6
        max_t = float(times.max()) + 1e-6
        bin_edges = np.linspace(min_t, max_t, num=requested_bins + 1, dtype=np.float64)
        y_disc = pd.cut(
            times,
            bins=bin_edges,
            labels=False,
            include_lowest=True,
            right=False,
        )

    y_disc = pd.Series(y_disc, index=work_df.index).fillna(0).astype(np.int64)
    work_df[y_disc_col] = y_disc
    return work_df, np.asarray(bin_edges, dtype=np.float64)


def survival_logits_to_outputs(logits):
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1)
    return hazards, survival, risk


def nll_survival_loss(hazards, survival, y_disc, censorship, eps=1e-7):
    batch_size = len(y_disc)
    y_disc = y_disc.view(batch_size, 1).long()
    censorship = censorship.view(batch_size, 1).float()
    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1)
    survival_padded = torch.cat([torch.ones_like(censorship), survival], dim=1)

    uncensored = -(
        (1 - censorship)
        * (
            torch.log(torch.gather(survival_padded, 1, y_disc).clamp(min=eps))
            + torch.log(torch.gather(hazards, 1, y_disc).clamp(min=eps))
        )
    )
    censored = -(
        censorship
        * torch.log(torch.gather(survival_padded, 1, y_disc + 1).clamp(min=eps))
    )
    return (uncensored + censored).mean()


def ce_survival_loss(hazards, survival, y_disc, censorship, alpha=0.4, eps=1e-7):
    batch_size = len(y_disc)
    y_disc = y_disc.view(batch_size, 1).long()
    censorship = censorship.view(batch_size, 1).float()
    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1)
    survival_padded = torch.cat([torch.ones_like(censorship), survival], dim=1)
    reg = -(
        (1 - censorship)
        * (
            torch.log(torch.gather(survival_padded, 1, y_disc).clamp(min=eps))
            + torch.log(torch.gather(hazards, 1, y_disc).clamp(min=eps))
        )
    )
    ce_term = -(
        censorship * torch.log(torch.gather(survival, 1, y_disc).clamp(min=eps))
        + (1 - censorship) * torch.log(
            1 - torch.gather(survival, 1, y_disc).clamp(min=eps)
        )
    )
    return ((1 - float(alpha)) * ce_term + float(alpha) * reg).mean()


def cox_survival_loss(hazards, survival, censorship):
    current_batch_len = len(survival)
    risk_mat = np.zeros([current_batch_len, current_batch_len], dtype=np.float32)
    survival_np = survival.detach().cpu().numpy().reshape(-1)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            risk_mat[i, j] = float(survival_np[j] >= survival_np[i])

    risk_mat = torch.FloatTensor(risk_mat).to(hazards.device)
    theta = hazards.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * risk_mat, dim=1)))
        * (1 - censorship.float())
    )
    return loss_cox


def compute_survival_loss_from_logits(logits, y_disc, censorship, loss_name="nll"):
    hazards, survival, _ = survival_logits_to_outputs(logits)
    loss_name_l = str(loss_name).strip().lower()
    if loss_name_l == "nll":
        return nll_survival_loss(hazards, survival, y_disc, censorship)
    if loss_name_l == "ce_survival":
        return ce_survival_loss(hazards, survival, y_disc, censorship)
    if loss_name_l == "cox":
        return cox_survival_loss(hazards, survival, censorship)
    raise ValueError(
        f"Unsupported survival loss '{loss_name}'. Valid values: nll, ce_survival, cox."
    )


def concordance_index_censored(event_observed, event_times, risk_scores):
    event_observed = np.asarray(event_observed, dtype=bool).reshape(-1)
    event_times = np.asarray(event_times, dtype=np.float64).reshape(-1)
    risk_scores = np.asarray(risk_scores, dtype=np.float64).reshape(-1)
    if not (len(event_observed) == len(event_times) == len(risk_scores)):
        raise ValueError("concordance_index_censored expects equal-length arrays.")

    concordant = 0.0
    permissible = 0.0

    n = len(event_times)
    for i in range(n):
        for j in range(i + 1, n):
            comparable = False
            sign = 0.0
            if event_observed[i] and event_times[i] < event_times[j]:
                comparable = True
                sign = np.sign(risk_scores[i] - risk_scores[j])
            elif event_observed[j] and event_times[j] < event_times[i]:
                comparable = True
                sign = np.sign(risk_scores[j] - risk_scores[i])

            if not comparable:
                continue

            permissible += 1.0
            if sign > 0:
                concordant += 1.0
            elif sign == 0:
                concordant += 0.5

    if permissible == 0.0:
        return 0.5
    return float(concordant / permissible)


def safe_survival_metrics(
    event_times,
    event_observed,
    censorship,
    y_disc,
    logits,
    loss_name="nll",
    include_raw=False,
):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(
            f"safe_survival_metrics expects logits with shape [N, n_bins], got {tuple(logits.shape)}."
        )

    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    y_disc_t = torch.as_tensor(np.asarray(y_disc, dtype=np.int64), dtype=torch.long)
    censorship_t = torch.as_tensor(np.asarray(censorship, dtype=np.float32), dtype=torch.float32)
    hazards_t, survival_t, risk_t = survival_logits_to_outputs(logits_t)
    loss_t = compute_survival_loss_from_logits(
        logits_t,
        y_disc=y_disc_t,
        censorship=censorship_t,
        loss_name=loss_name,
    )
    cindex = concordance_index_censored(
        event_observed=np.asarray(event_observed, dtype=bool),
        event_times=np.asarray(event_times, dtype=np.float64),
        risk_scores=risk_t.detach().cpu().numpy(),
    )
    metrics = {
        "CINDEX": float(cindex),
        "LOSS": float(loss_t.detach().cpu().item()),
    }
    if include_raw:
        metrics["Hazards"] = hazards_t.detach().cpu().numpy().tolist()
        metrics["Survival"] = survival_t.detach().cpu().numpy().tolist()
        metrics["Risk"] = risk_t.detach().cpu().numpy().tolist()
        metrics["Logits"] = logits_t.detach().cpu().numpy().tolist()
    return metrics


# ------------------------ 4. MODEL TRAINING --------------------------

# Helper function to build model based on name and input dimensions
def build_model(model_name, input_dims, model_kwargs):
    from models import MultimodalMLP, PAM, SMILE, HealNetBinaryWrapper

    model_name = normalize_model_name(model_name)
    if model_name in {"lr"}:
        raise ValueError(
            "Logistic regression is a sklearn baseline handled directly in scripts/train_ncv.py."
        )
    if model_name in {"rf"}:
        raise ValueError(
            "Random forest classification is a sklearn baseline handled directly in scripts/train_ncv.py."
        )
    if model_name in {"coxnet", "rsf"}:
        raise ValueError(
            "CoxNet and Random Survival Forest are sksurv baselines handled directly in scripts/train_ncv.py."
        )
    if model_name in {"mlp"}:
        return MultimodalMLP(input_dims, **model_kwargs)
    if model_name in {"pam"}:
        return PAM(input_dims, **model_kwargs)
    if model_name in {"smil_e"}:
        init_kwargs = dict(model_kwargs or {})
        for key in {"meta_inner_lr", "meta_val_fraction", "meta_inner_steps"}:
            init_kwargs.pop(key, None)
        return SMILE(input_dims, **init_kwargs)
    if model_name in {"healnet"}:
        return HealNetBinaryWrapper(input_dims, **model_kwargs)
    raise ValueError(
        f"Unsupported model '{model_name}'. Supported: lr, coxnet, rsf, mlp, pam, smile, healnet"
    )


def primary_metric_name(task_config):
    task_type = normalize_task_type((task_config or {}).get("task_type", "binary_classification"))
    if task_type == "survival":
        return "CINDEX"
    return "AUC"


def primary_loss_name(task_config):
    task_type = normalize_task_type((task_config or {}).get("task_type", "binary_classification"))
    if task_type == "survival":
        return "LOSS"
    return "LOGLOSS"

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


def safe_task_metrics(task_config, **kwargs):
    task_type = normalize_task_type((task_config or {}).get("task_type", "binary_classification"))
    if task_type == "survival":
        return safe_survival_metrics(
            event_times=kwargs["event_times"],
            event_observed=kwargs["event_observed"],
            censorship=kwargs["censorship"],
            y_disc=kwargs["y_disc"],
            logits=kwargs["logits"],
            loss_name=str((task_config or {}).get("survival_loss", "nll")).strip().lower(),
            include_raw=bool(kwargs.get("include_raw", False)),
        )
    return safe_binary_metrics(
        y_true=kwargs["y_true"],
        y_prob=kwargs["y_prob"],
        include_raw=bool(kwargs.get("include_raw", False)),
    )
