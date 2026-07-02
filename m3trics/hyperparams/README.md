# Hyperparameter Configs

This folder contains the default hyperparameter search spaces used by M3TRICS when `FINGERPRINT="false"`.

Each file exposes one `MODEL_CONFIG` dictionary. The launcher selects these configs through `RUN_MODELS`, and `scripts/m3trics.py` expands comma-separated values into the nested-CV hyperparameter grid.

## Standard Config Format

```python
MODEL_CONFIG = {
    "display_name": "ZI_MLP",
    "model": "MultimodalMLP",
    "args": {
        "epochs": 100,
        "batch_size": "8,16",
        "learning_rate": "1e-4,5e-4",
        "weight_decay": "1e-4",
        "dropout": "0.1,0.2",
    },
}
```

Rules:

- scalar values are treated as fixed arguments,
- comma-separated strings are expanded as hyperparameter candidates,
- independent comma-separated arguments are combined as a Cartesian product,
- `paired_args` should be used only when two arguments must vary together.
- for sklearn/sksurv baselines, `pca_n_components` can be `none`, an integer number of components, or a float in `(0, 1)` interpreted as the target explained variance. PCA is fitted inside each nested-CV training split.

Example:

```python
MODEL_CONFIG = {
    "display_name": "HealNet",
    "model": "HealNet",
    "args": {
        "healnet_num_latents": "8,16",
        "healnet_latent_dim": "32,64",
    },
    "paired_args": [("healnet_num_latents", "healnet_latent_dim")],
}
```

In this case, M3TRICS evaluates `(8, 32)` and `(16, 64)`, not the full Cartesian product.

## Fingerprint Mode

If a launcher sets:

```bash
FINGERPRINT="true"
FINGERPRINT_MAX_COMBINATIONS="32"
```

the default configs in this folder are not used for the selected methods. Instead, the launcher first runs:

```bash
python scripts/fingerprint.py ...
```

and saves:

```text
results/<DATASET>_<ENDPOINT_COL>/fingerprint/
├── fingerprint.json
└── fingerprint_hp_suggestions.sh
```

Then `scripts/m3trics.py` receives:

```bash
--fingerprint_hp_json results/<DATASET>_<ENDPOINT_COL>/fingerprint/fingerprint.json
```

and replaces the selected method config with the grid stored in `fingerprint.json`.

This replacement is strict:

- `fixed_args`, `hp_grid_args`, and `args` from `hyperparams/*.py` are discarded for that run,
- the fingerprint grid becomes the only hyperparameter source for the selected method,
- if a selected method is missing from `fingerprint.json`, training raises an error.

This avoids silently mixing default grids with fingerprint-generated grids.

## Knowledge Distillation

Knowledge distillation is configured from the launcher with:

```bash
DISTILL_MODELS="ZI_MLP,pAM,HealNet"
DISTILL_ALPHA="0.25,0.5"
DISTILL_BETA="0.05,0.1"
```

M3TRICS automatically creates `<method>_KD` variants for the listed methods. The base method grid comes either from this folder or from fingerprint mode. The distillation-specific weights `DISTILL_ALPHA` and `DISTILL_BETA` are added as additional hyperparameters from the launcher.

If `FINGERPRINT="true"`, a KD method uses the same fingerprint grid as its base method. For example, `ZI_MLP_KD` uses the suggested grid for `ZI_MLP`.

## Adding A New Config

To add a new method config:

1. Create `hyperparams/my_method.py`.
2. Define `MODEL_CONFIG`.
3. Register it in `hyperparams/__init__.py`.
4. Make sure the `model` alias is supported by `scripts/utils.py::normalize_model_name` and `scripts/utils.py::build_model`, or by the sklearn/sksurv baseline logic.
5. If the method should work with `FINGERPRINT="true"`, also add support in `scripts/fingerprint.py`.

See `models/README.md` for the model-side interface and fingerprint requirements for new methods.

## Per-Modality Feature Reduction

M3TRICS supports launcher-level per-modality feature reduction:

- Configure it in the launcher with `*_FEATURE_REDUCTION="pca"` and `*_PCA_NUM_COMPONENTS`.
- It is fitted inside each CV training split and applied to every method, including deep-learning methods and sklearn/sksurv baselines.
- PCA is therefore no longer encoded as a separate method name. Use `ZI_LR`, `KNN_LR`, `ZI_RF`, `KNN_RF`, `ZI_CoxNet`, `KNN_CoxNet`, `ZI_RSF`, and `KNN_RSF` when comparing methods under the same reduced feature representation.

Method-specific hyperparameter configs should not add a second PCA step unless a new method explicitly requires it.
