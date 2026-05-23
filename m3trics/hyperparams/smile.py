MODEL_CONFIG = {
    "display_name": "SMILe",
    "model": "SMILe",
    "args": {
        "epochs": 80,
        "early_stopping_patience": 20,
        "batch_size": "8,16",
        "learning_rate": "5e-5,1e-5",
        "weight_decay": "1e-4",
        "smil_e_latent_dim": "8,16",
        "smil_e_num_priors": "2",
        "smil_e_num_heads": "1",
        "smil_e_dropout": "0.2",
        "smil_e_alpha": "1e-3,1e-2",
        "smil_e_beta": "1e-2",
        "meta_inner_lr": "5e-4,5e-5",
        "meta_val_fraction": "0.25",
        "classifier_hidden_dim": "16,32",
    },
    "paired_args": [
        ("learning_rate", "meta_inner_lr"),
    ],
}
