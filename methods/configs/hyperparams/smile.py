MODEL_CONFIG = {
    "display_name": "SMILe",
    "model": "SMILe",
    "epochs": 80,
    "main_args": {
        "batch_size": "16,32",
        "learning_rate": "5e-5,1e-4",
        "smil_e_latent_dim": "32,64",
        "smil_e_num_priors": "32,64",
        "smil_e_num_heads": "4",
        "smil_e_dropout": "0.1,0.2",
        "smil_e_alpha": "1e-2",
        "smil_e_beta": "1e-2",
    },
}
