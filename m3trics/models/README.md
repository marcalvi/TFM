# Adding New M3TRICS Models

M3TRICS models must follow a small interface so that nested cross-validation, missingness simulation, distillation, prediction saving, and analysis can work without method-specific code.

## Constructor Pattern

A standard model should accept:

```python
ModelClass(
    input_dims: list[int],
    output_dim: int = 1,
    **model_specific_kwargs,
)
```

- `input_dims`: one feature dimension per modality, in the same order used by the dataset loader.
- `output_dim`: prediction head size.
- binary classification: use `output_dim=1`.
- discrete-time survival: use `output_dim=survival_n_bins`, usually `4` unless configured otherwise.

Model-specific options can be added as keyword arguments and exposed in the corresponding hyperparameter config.

## Forward Signature

Use this forward signature whenever possible:

```python
def forward(self, Xs, present_mask=None, return_aux=False, kd=False):
    ...
```

- `Xs`: list of tensors, one tensor per modality.
- `present_mask`: optional tensor with shape `[batch, n_modalities]`, where 1 means the modality is available and 0 means missing.
- `return_aux`: if `False`, return logits only. If `True`, return `(logits, aux_dict)` for diagnostics or method-specific outputs.
- `kd`: if `True`, return `(logits, aux_dict)` with an inner representation suitable for knowledge distillation. This is the preferred flag used by the generic KD training loop.

Expected logits:

- binary classification: shape `[batch, 1]` or `[batch]`.
- survival: shape `[batch, survival_n_bins]`.

## Missing-Modality Masks

Architecture-based methods should use `present_mask` directly instead of inferring missingness from zeros. Imputation-based methods may receive already-filled modality tensors, but still receive `present_mask` for methods that need it.

Do not assume that all modalities are present. The same model may be evaluated under complete train/complete test, missing train/complete test, complete train/missing test, or missing train/missing test conditions.

## Survival Heads

For Marta-style discrete-time survival, the model output dimension must equal `survival_n_bins`.

M3TRICS converts survival logits as follows:

```python
hazards = sigmoid(logits)
survival = cumprod(1 - hazards, dim=1)
risk = -sum(survival, dim=1)
```

Training losses currently supported by the framework are:

- `nll`
- `ce_survival`
- `cox`

The event convention is:

- `event_observed = 1`: event occurred.
- `censorship = 1 - event_observed`.

C-index analysis uses the saved `risk` scores.

## Registering A New Model

A new deep model usually needs changes in five places.

1. Add the implementation under `models/`, for example `models/my_model.py`.
2. Export it in `models/__init__.py`.
3. Add an alias in `scripts/utils.py::normalize_model_name`.
4. Add a construction branch in `scripts/utils.py::build_model`.
5. Add a hyperparameter config under `hyperparams/` and register it in `hyperparams/__init__.py`.

The launcher can then include the new method through `RUN_MODELS`, using the key registered in `hyperparams/__init__.py`.

## Hyperparameter Configs

Each config file should expose `MODEL_CONFIG`:

```python
MODEL_CONFIG = {
    "display_name": "MyModel",
    "model": "MyModel",
    "args": {
        "epochs": 100,
        "batch_size": "16,32",
        "learning_rate": "1e-4,5e-4",
        "weight_decay": "0.0,1e-4",
        "my_parameter": "2,4",
    },
}
```

Comma-separated values are expanded into the hyperparameter grid. Use `paired_args` only when parameters must vary together instead of as a Cartesian product.

## Fingerprint Support For New Methods

The normal source of method hyperparameters is `hyperparams/*.py`. However, if a launcher sets:

```bash
FINGERPRINT="true"
```

M3TRICS first runs `scripts/fingerprint.py`, writes `fingerprint.json`, and then uses that JSON to override the selected method configs. In this mode, the suggested grid replaces `fixed_args`, `hp_grid_args`, and `args` from `hyperparams/*.py` before training.

For existing methods this is automatic. For a new method, add fingerprint support if you want it to work with `FINGERPRINT=true`:

1. Add a model alias in `scripts/fingerprint.py::MODEL_ALIASES`.
2. Map the method to a functional family in `scripts/fingerprint.py::_method_family`.
3. Add a branch in `scripts/fingerprint.py::suggest_grid_for_method`.
4. Include only arguments that are accepted by `scripts/m3trics.py::build_training_arg_parser`.
5. Use comma-separated strings for searchable HPs and scalar strings for fixed HPs.
6. Use `paired_hp_groups` in the fingerprint output when two arguments must vary together.

Example fingerprint grid entry:

```python
args = {
    "epochs": "80",
    "early_stopping_patience": "20",
    "batch_size": "8,16",
    "learning_rate": "1e-4,5e-5",
    "weight_decay": "1e-4",
    "my_hidden_dim": "16,32",
    "my_dropout": "0.1,0.2",
}
```

If the user selects a method that is not present in `fingerprint.json` while `FINGERPRINT=true`, M3TRICS raises an error instead of silently falling back to the default hyperparameter config. This keeps fingerprint runs reproducible and avoids mixing two different grid sources.

## Auxiliary Outputs And Distillation

Knowledge distillation is configured from the launcher with `DISTILL_MODELS`, not by adding a separate method class. If a supported torch method appears in `DISTILL_MODELS`, M3TRICS trains the base method normally and also trains a `DI-<method>` variant. The KD run first pretrains a teacher, freezes it, and then trains a student under the configured modality-availability conditions. In static-cohort mode, this is the observed cohort as-is; in progressive missingness mode, this includes the configured synthetic missingness.

The student loss combines the task loss with two optional distillation terms controlled by `DISTILL_ALPHA` and `DISTILL_BETA`. These values can be comma-separated lists and are expanded into the hyperparameter grid.

If your model can expose a useful representation for distillation, return an auxiliary dictionary when `kd=True` using one of these keys: `hidden_feature`, `fusion_feature`, `repr`, `features`, `embedding`, `latent`, `pooled`, or `bottleneck`. If no auxiliary representation is returned, M3TRICS falls back to logits for the representation term.

```python
return logits, {
    "fusion_feature": fused_representation,
}
```

The training loop must explicitly know how to consume any additional custom auxiliary keys. For standard classifiers, returning logits only is valid.

Current generic KD representations are:

- `MultimodalMLP`: concatenated fusion vector after modality encoders.
- `pAM`: Hadamard product between attention weights and modality-wise prediction scores.
- `HealNet`: latent bottleneck matrix, with pooled latent representation also exposed.

## Non-PyTorch Baselines

The logistic-regression, Random Forest, CoxNet, and Random Survival Forest baselines are handled directly in `scripts/train_ncv.py` because they are sklearn/sksurv estimators, not `torch.nn.Module` objects. If launcher-level per-modality PCA is enabled, it is fitted only on the corresponding training split and applied to every method before these baselines receive concatenated features.

- `ZI_LR`, `KNN_LR`, and `VAE_LR` are binary-classification-only baselines.
- `ZI_RF`, `KNN_RF`, and `VAE_RF` are binary-classification-only baselines.
- `ZI_CoxNet`, `KNN_CoxNet`, `VAE_CoxNet`, `ZI_RSF`, `KNN_RSF`, and `VAE_RSF` are survival-only baselines.
- `RF` means `sklearn.ensemble.RandomForestClassifier`.
- `RSF` means Random Survival Forest, not a binary-classification random forest.

New sklearn-style baselines should follow the same pattern: fit on the nested-CV training split, predict on validation/test splits, and write the same prediction columns as neural models. Survival baselines that predict risk directly should write `inner_model_<k>_risk` plus `event_time`, `event_observed`, `censorship`, and `y_disc`; C-index analysis uses the risk column directly.

VAE-based imputation is treated as a split-level imputer. When several `VAE_*` methods are run in the same M3TRICS execution, the VAE imputer is trained once for each matching split, missingness pattern, and VAE-imputer configuration, then reused by the downstream `VAE_LR`, `VAE_RF`, `VAE_CoxNet`, `VAE_RSF`, `VAE_MMLP`, and `VAE_pMMLP` models.

## Per-Modality Encoders And Feature Reduction

`MultimodalMLP` and `PAM` support `pm_encoders`:

- `pm_encoders=true`: each modality is first projected by a learned per-modality encoder before fusion (`pMMLP` / `pAM`).
- `pm_encoders=false`: the model uses the preprocessed modality features directly (`MMLP` / `AM`).

Launcher-level per-modality PCA is independent from these learned encoders. PCA is fitted inside each CV training split and can be used to compare explicit feature reduction against learned per-modality projections in small clinical cohorts.
