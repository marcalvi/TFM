MODEL_CONFIG = {
    "display_name": "KNN_CoxNet",
    "model": "CoxNet",
    "args": {
        "epochs": 1,
        "early_stopping_patience": 1,
        "batch_size": "64",
        "learning_rate": "1.0",
        "weight_decay": "0.0",
        "coxnet_alpha": "0.01,0.1,1.0",
        "coxnet_l1_ratio": "0.1,0.5",
        "coxnet_max_iter": "100000",
        "coxnet_tol": "1e-7",
        "imputation_method": "knn",
    },
}
