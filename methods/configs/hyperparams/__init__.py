from .di_pam import MODEL_CONFIG as DI_PAM_CONFIG
from .healnet import MODEL_CONFIG as HEALNET_CONFIG
from .knn_mlp import MODEL_CONFIG as KNN_MLP_CONFIG
from .pam import MODEL_CONFIG as PAM_CONFIG
from .smile import MODEL_CONFIG as SMILE_CONFIG
from .vae_mlp import MODEL_CONFIG as VAE_MLP_CONFIG
from .zi_mlp import MODEL_CONFIG as ZI_MLP_CONFIG

MODEL_CONFIGS = {
    "zi_mlp": ZI_MLP_CONFIG,
    "knn_mlp": KNN_MLP_CONFIG,
    "vae_mlp": VAE_MLP_CONFIG,
    "pam": PAM_CONFIG,
    "di_pam": DI_PAM_CONFIG,
    "healnet": HEALNET_CONFIG,
    "smile": SMILE_CONFIG,
}

MODEL_NAME_ALIASES = {
    "zi_mlp": "zi_mlp",
    "zimlp": "zi_mlp",
    "knn_mlp": "knn_mlp",
    "knnmlp": "knn_mlp",
    "vae_mlp": "vae_mlp",
    "vaemlp": "vae_mlp",
    "pam": "pam",
    "p-am": "pam",
    "di_pam": "di_pam",
    "dipam": "di_pam",
    "di-pam": "di_pam",
    "healnet": "healnet",
    "smile": "smile",
    "smilee": "smile",
}


def list_available_model_configs():
    return list(MODEL_CONFIGS.keys())


def get_model_config(model_name):
    key = str(model_name).strip().lower().replace(" ", "")
    canonical_key = MODEL_NAME_ALIASES.get(key)
    if canonical_key is None:
        valid = ", ".join(list_available_model_configs())
        raise ValueError(f"Unknown model config '{model_name}'. Available configs: {valid}")
    return MODEL_CONFIGS[canonical_key]
