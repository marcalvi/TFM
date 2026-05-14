MODEL_CONFIG = {
    "display_name": "pAM",
    "model": "pAM",
    "fixed_args": {
        "epochs": 80,
        "early_stopping_patience": 20,
        "weight_decay": "1e-4",
    },
    "hp_grid_args": {
        "batch_size": "32",
        "learning_rate": "1e-4,5e-5,1e-5",
        "pam_dropout": "0.2,0.4",
        "pam_temperature": "1.0,2.0",
    },
}
