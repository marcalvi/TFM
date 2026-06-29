MODEL_CONFIG = {
    "display_name": "KNN_LR",
    "model": "LR",
    "args": {
        "epochs": 1,
        "early_stopping_patience": 1,
        "batch_size": "64",
        "learning_rate": "1.0",
        "weight_decay": "0.0",
        "lr_C": "0.01,0.1,1.0,10.0",
        "lr_penalty": "l2",
        "lr_solver": "lbfgs",
        "lr_class_weight": "none,balanced",
        "lr_max_iter": "1000",
        "imputation_method": "knn",
    },
}
