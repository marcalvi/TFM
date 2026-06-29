MODEL_CONFIG = {
    "display_name": "ZI_RF",
    "model": "RF",
    "args": {
        "epochs": 1,
        "early_stopping_patience": 1,
        "batch_size": "64",
        "learning_rate": "1.0",
        "weight_decay": "0.0",
        "rf_n_estimators": "200",
        "rf_max_depth": "none,5,10",
        "rf_min_samples_split": "2",
        "rf_min_samples_leaf": "1,3",
        "rf_max_features": "sqrt",
        "rf_class_weight": "none,balanced",
        "rf_n_jobs": "-1",
        "imputation_method": "zero",
    },
}
