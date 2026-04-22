MODEL_CONFIG = {
    "display_name": "PAMDiPAM",
    "model": "PAMDiPAM",
    "fixed_args": {
        "epochs": 80,
        "early_stopping_patience": 20,
        "lr_patience": 5,
        "weight_decay": "1e-4",
    },
    "hp_grid_args": {
        "batch_size": "16,32",
        "learning_rate": "1e-5,1e-4",
        "pam_dropout": "0.2,0.4",
        "pam_temperature": "1.0,2.0",
        "distill_alpha": "1.0,2.0",
        "distill_beta": "0.1,0.3",
    },
}
