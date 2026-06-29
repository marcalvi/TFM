MODEL_CONFIG = {
    "display_name": "ZI_RSF",
    "model": "RSF",
    "args": {
        "epochs": 1,
        "early_stopping_patience": 1,
        "batch_size": "64",
        "learning_rate": "1.0",
        "weight_decay": "0.0",
        "rsf_n_estimators": "100",
        "rsf_max_depth": "none,5",
        "rsf_min_samples_split": "6",
        "rsf_min_samples_leaf": "3,5",
        "rsf_max_features": "sqrt",
        "rsf_n_jobs": "-1",
        "imputation_method": "zero",
    },
}
