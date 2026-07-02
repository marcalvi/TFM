from .healnet import MODEL_CONFIG as HEALNET_CONFIG
from .am import MODEL_CONFIG as AM_CONFIG
from .knn_coxnet import MODEL_CONFIG as KNN_COXNET_CONFIG
from .knn_lr import MODEL_CONFIG as KNN_LR_CONFIG
from .knn_mlp import MODEL_CONFIG as KNN_MLP_CONFIG
from .knn_mmlp import MODEL_CONFIG as KNN_MMLP_CONFIG
from .knn_rf import MODEL_CONFIG as KNN_RF_CONFIG
from .knn_rsf import MODEL_CONFIG as KNN_RSF_CONFIG
from .pam import MODEL_CONFIG as PAM_CONFIG
from .knn_pmmlp import MODEL_CONFIG as KNN_PMMLP_CONFIG
from .smile import MODEL_CONFIG as SMILE_CONFIG
from .vae_coxnet import MODEL_CONFIG as VAE_COXNET_CONFIG
from .vae_lr import MODEL_CONFIG as VAE_LR_CONFIG
from .vae_mlp import MODEL_CONFIG as VAE_MLP_CONFIG
from .vae_mmlp import MODEL_CONFIG as VAE_MMLP_CONFIG
from .vae_pmmlp import MODEL_CONFIG as VAE_PMMLP_CONFIG
from .vae_rf import MODEL_CONFIG as VAE_RF_CONFIG
from .vae_rsf import MODEL_CONFIG as VAE_RSF_CONFIG
from .zi_coxnet import MODEL_CONFIG as ZI_COXNET_CONFIG
from .zi_lr import MODEL_CONFIG as ZI_LR_CONFIG
from .zi_mlp import MODEL_CONFIG as ZI_MLP_CONFIG
from .zi_mmlp import MODEL_CONFIG as ZI_MMLP_CONFIG
from .zi_pmmlp import MODEL_CONFIG as ZI_PMMLP_CONFIG
from .zi_rf import MODEL_CONFIG as ZI_RF_CONFIG
from .zi_rsf import MODEL_CONFIG as ZI_RSF_CONFIG

MODEL_CONFIGS = {
    "zi_lr": ZI_LR_CONFIG,
    "knn_lr": KNN_LR_CONFIG,
    "vae_lr": VAE_LR_CONFIG,
    "zi_rf": ZI_RF_CONFIG,
    "knn_rf": KNN_RF_CONFIG,
    "vae_rf": VAE_RF_CONFIG,
    "zi_coxnet": ZI_COXNET_CONFIG,
    "knn_coxnet": KNN_COXNET_CONFIG,
    "vae_coxnet": VAE_COXNET_CONFIG,
    "zi_rsf": ZI_RSF_CONFIG,
    "knn_rsf": KNN_RSF_CONFIG,
    "vae_rsf": VAE_RSF_CONFIG,
    "zi_mlp": ZI_MLP_CONFIG,
    "knn_mlp": KNN_MLP_CONFIG,
    "zi_mmlp": ZI_MMLP_CONFIG,
    "knn_mmlp": KNN_MMLP_CONFIG,
    "zi_pmmlp": ZI_PMMLP_CONFIG,
    "knn_pmmlp": KNN_PMMLP_CONFIG,
    "vae_mlp": VAE_MLP_CONFIG,
    "vae_mmlp": VAE_MMLP_CONFIG,
    "vae_pmmlp": VAE_PMMLP_CONFIG,
    "am": AM_CONFIG,
    "pam": PAM_CONFIG,
    "healnet": HEALNET_CONFIG,
    "smile": SMILE_CONFIG,
}

MODEL_NAME_ALIASES = {
    "zi_mlp": "zi_mlp",
    "zimlp": "zi_mlp",
    "zi_mmlp": "zi_mmlp",
    "zimmlp": "zi_mmlp",
    "zi_pmmlp": "zi_pmmlp",
    "zipmmlp": "zi_pmmlp",
    "zi_p_mmlp": "zi_pmmlp",
    "zi_lr": "zi_lr",
    "zilr": "zi_lr",
    "zi_rf": "zi_rf",
    "zirf": "zi_rf",
    "vae_rf": "vae_rf",
    "vaerf": "vae_rf",
    "zi_coxnet": "zi_coxnet",
    "zicoxnet": "zi_coxnet",
    "vae_coxnet": "vae_coxnet",
    "vaecoxnet": "vae_coxnet",
    "knn_mlp": "knn_mlp",
    "knnmlp": "knn_mlp",
    "knn_mmlp": "knn_mmlp",
    "knnmmlp": "knn_mmlp",
    "knn_pmmlp": "knn_pmmlp",
    "knnpmmlp": "knn_pmmlp",
    "knn_p_mmlp": "knn_pmmlp",
    "knn_lr": "knn_lr",
    "knnlr": "knn_lr",
    "knn_rf": "knn_rf",
    "knnrf": "knn_rf",
    "knn_coxnet": "knn_coxnet",
    "knncoxnet": "knn_coxnet",
    "zi_rsf": "zi_rsf",
    "zirsf": "zi_rsf",
    "knn_rsf": "knn_rsf",
    "knnrsf": "knn_rsf",
    "vae_rsf": "vae_rsf",
    "vaersf": "vae_rsf",
    "vae_mlp": "vae_mlp",
    "vaemlp": "vae_mlp",
    "vae_mmlp": "vae_mmlp",
    "vaemmlp": "vae_mmlp",
    "vae_pmmlp": "vae_pmmlp",
    "vaepmmlp": "vae_pmmlp",
    "vae_p_mmlp": "vae_pmmlp",
    "vae_lr": "vae_lr",
    "vaelr": "vae_lr",
    "pam": "pam",
    "p-am": "pam",
    "am": "am",
    "healnet": "healnet",
    "smile": "smile",
    "smilee": "smile",
    "meta_smile": "smile",
    "metasmile": "smile",
    "metasmilee": "smile",
}


def list_available_model_configs():
    return [str(cfg.get("display_name", key)) for key, cfg in MODEL_CONFIGS.items()]


def get_model_config(model_name):
    key = str(model_name).strip().lower().replace(" ", "")
    canonical_key = MODEL_NAME_ALIASES.get(key)
    if canonical_key is None:
        valid = ", ".join(list_available_model_configs())
        raise ValueError(f"Unknown model config '{model_name}'. Available models: {valid}")
    return MODEL_CONFIGS[canonical_key]
