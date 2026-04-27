MODEL_CONFIG = {
    "display_name": "SMILe",
    "model": "SMILe",
    "fixed_args": {
        "epochs": 40,
        "early_stopping_patience": 20,
        "lr_patience": 5,
        "weight_decay": "1e-4",
        "smil_e_num_heads": "1",
        "smil_e_dropout": "0.2",
        "meta_learning": False,
        "meta_inner_lr": 1e-3,
        "meta_val_fraction": 0.25,
        "smil_e_alpha": "1e-2",
        "smil_e_beta": "1e-2",
    },
    "hp_grid_args": {
        "batch_size": "16,32",
        "learning_rate": "1e-5",
        "smil_e_latent_dim": "8,16",
        "smil_e_num_priors": "2,4",
        "classifier_hidden_dim": "32,64",
    },
}
