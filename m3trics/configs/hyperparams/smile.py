MODEL_CONFIG = {
    "display_name": "SMILe",
    "model": "SMILe",
    "fixed_args": {
        "epochs": 80,
        "early_stopping_patience": 20,
        "weight_decay": "1e-4",
        "smil_e_num_heads": "1",
        "smil_e_dropout": "0.2",
        "meta_val_fraction": "0.25",
        "smil_e_beta": "1e-2",
    },
    "hp_grid_args": {
        "batch_size": "32",
        "learning_rate": "1e-4,5e-5,1e-5",
        "smil_e_latent_dim": "8,16",
        "smil_e_num_priors": "2,4",
        "smil_e_alpha": "1e-3,1e-2",
        "meta_inner_lr": "5e-5,5e-4",
        "classifier_hidden_dim": "16,32",
    },
}
