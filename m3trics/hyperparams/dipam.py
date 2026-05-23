MODEL_CONFIG = {
    "display_name": "Di-PAM",
    "model": "Di-PAM",
    "args": {
        "epochs": 80,
        "early_stopping_patience": 20,
        "batch_size": "8,16",
        "learning_rate": "1e-4,5e-5,1e-5",
        "weight_decay": "1e-4",
        "pam_dropout": "0.2,0.4",
        "pam_temperature": "1.0,2.0",
        "distill_alpha": "0.25",
        "distill_beta": "0.05",
    },
}
