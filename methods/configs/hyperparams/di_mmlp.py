MODEL_CONFIG = {
    "display_name": "Di-MMLP",
    "model": "DiMMLP",
    "fixed_args": {
        "epochs": 80,
        "early_stopping_patience": 20,
        "weight_decay": "1e-4",
        "fusion_hidden_layers": "1",
        "fusion_batchnorm": "false",
        "modality_hidden_layers": "1",
        "imputation_method": "zero",
    },
    "hp_grid_args": {
        "batch_size": "32",
        "learning_rate": "1e-4,5e-5,1e-5",
        "fusion_hidden_dim": "32,64",
        "dropout": "0.2,0.1",
        "distill_alpha": "1.0,2.0",
        "distill_beta": "0.1,0.3",
    },
}
