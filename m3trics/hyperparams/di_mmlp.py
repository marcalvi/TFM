MODEL_CONFIG = {
    "display_name": "Di-MMLP",
    "model": "DiMMLP",
    "args": {
        "epochs": 80,
        "early_stopping_patience": 20,
        "batch_size": "8,16",
        "learning_rate": "1e-4,5e-5,1e-5",
        "weight_decay": "1e-4",
        "fusion_hidden_dim": "32,64",
        "fusion_hidden_layers": "1",
        "fusion_batchnorm": "false",
        "modality_hidden_layers": "1",
        "dropout": "0.1,0.2",
        "distill_alpha": "0.25",
        "distill_beta": "0.05",
        "imputation_method": "zero",
    },
}
