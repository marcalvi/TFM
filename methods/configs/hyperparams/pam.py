MODEL_CONFIG = {
    "display_name": "pAM",
    "model": "pAM",
    "epochs": 80,
    "main_args": {
        "batch_size": "16,32",
        "learning_rate": "1e-5,5e-5,1e-4",
        "pam_dropout": "0.2,0.4",
        "pam_temperature": "1.0,2.0",
    },
}
