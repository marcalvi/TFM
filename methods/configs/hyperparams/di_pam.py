<<<<<<< HEAD
MODEL_CONFIG = {
    "display_name": "Di-pAM",
    "model": "Di-pAM",
    "fixed_args": {
        "epochs": 80,
        "early_stopping_patience": 20,
        "lr_patience": 5,
    },
    "hp_grid_args": {
        "batch_size": "16,32",
        "learning_rate": "1e-5,5e-5,1e-4",
        "weight_decay": "1e-4",
        "pam_dropout": "0.2,0.4",
        "pam_temperature": "1.0,2.0",
        "distill_alpha": "1.0,2.0",
        "distill_beta": "0.1,0.3",
    },
}
=======
from .mlp_di_pam import MODEL_CONFIG
>>>>>>> 258c07901894698d7aceb4b6454a44203554deee
