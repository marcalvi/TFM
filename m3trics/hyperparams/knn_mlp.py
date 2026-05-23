MODEL_CONFIG = {
    "display_name": "KNN_MLP",
    "model": "MLP",
    "args": {
        "epochs": 80,
        "early_stopping_patience": 20,
        "batch_size": "8,16",
        "learning_rate": "1e-4,5e-5,1e-5",
        "weight_decay": "1e-4",
        "fusion_hidden_dim": "16,32",
        "fusion_hidden_layers": "1",
        "fusion_batchnorm": "false",
        "modality_hidden_layers": "1",
        "dropout": "0.1,0.2",
        "imputation_method": "knn",
    },
}
