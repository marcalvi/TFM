LATENT_BOTTLENECK_DIMS = "8,16"
ATTENTION_HEAD_DIMS = "8,16"

MODEL_CONFIG = {
    "display_name": "HealNet",
    "model": "HealNet",
    "fixed_args": {
        "epochs": 80,
        "early_stopping_patience": 20,
        "lr_patience": 5,
    },
    "hp_grid_args": {
        "batch_size": "8,16,32",
        "learning_rate": "5e-6,1e-5",
        "weight_decay": "1e-4",
        "healnet_depth": "1,2",
        "healnet_num_freq_bands": "2",
        "healnet_num_latents": LATENT_BOTTLENECK_DIMS,
        "healnet_latent_dim": LATENT_BOTTLENECK_DIMS,
        "healnet_cross_heads": "1",
        "healnet_latent_heads": "2",
        "healnet_cross_dim_head": ATTENTION_HEAD_DIMS,
        "healnet_latent_dim_head": ATTENTION_HEAD_DIMS,
        "healnet_attn_dropout": "0.2",
        "healnet_ff_dropout": "0.2",
        "healnet_self_per_cross_attn": "1",
    },
    "paired_hp_grid_args": [
        ("healnet_num_latents", "healnet_latent_dim"),
        ("healnet_cross_dim_head", "healnet_latent_dim_head"),
    ],
}
