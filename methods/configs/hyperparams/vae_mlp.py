MODEL_CONFIG = {
    "display_name": "VAE_MLP",
    "model": "MLP",
    "epochs": 80,
    "main_args": {
        "batch_size": "16,32",
        "learning_rate": "1e-5,5e-5,1e-4",
        "fusion_hidden_dim": "32,64",
        "fusion_hidden_layers": "1",
        "fusion_batchnorm": "false",
        "modality_hidden_layers": "1",
        "dropout": "0.2,0.1",
        "imputation_method": "vae",
    },
}
