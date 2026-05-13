# Changelog — Adaptations for `methods`

All modifications made on top of the vendored `healnet-adoption` copy are
documented here so the repo remains auditable.

```
Version: 0.1.0-methods
Date: 2026-03-16
Based on: konst-int-i/healnet
Reference fork reviewed: Marta Buetas-Arcas (`healnet-marta`)
```

## Summary
- Ported sample-level missing-modality handling into the vendored `HealNet`
  model so mixed batches can update only the samples that actually have a
  modality.
- Added a unit test that exercises mixed per-sample missingness in one batch.
- Integrated a hybrid batching policy in the benchmark wrapper:
  homogeneous full batches first, residual mixed batches second.

## Detailed changes

### 1) `healnet/models/healnet.py`
- Extended `HealNet.forward(...)` with two optional arguments:
  - `masks`: per-modality token masks
  - `present`: per-modality sample-level availability vectors
- Preserved the original whole-batch missing behavior for modalities passed as
  `None`.
- Added bucketed processing for mixed batches:
  - for each modality, compute the subset of samples where the modality is
    present
  - run cross-attention only on that subset
  - scatter the updated latent states back into the full batch
- Added shape checks for `present` and modality masks to fail fast on malformed
  inputs.
- Replaced the old `self_attn, self_ff = layer[-1]` assumption with iteration
  over the full latent self-attention block list. This keeps the repo correct
  when `self_per_cross_attn > 1`.
- Added module logging instead of using `print(...)` inside the model path.
- Replaced the in-place scatter `x[present_idx] = updated` used by the
  subbatching path with `x = x.index_copy(0, present_idx, updated)` to keep the
  mixed-batch update compatible with PyTorch autograd.

### 2) `healnet/tests/test_healnet.py`
- Added a smoke test covering a mixed batch where different samples are missing
  different modalities. The model must still return logits for the full batch.

### 3) Related integration changes outside the vendored repo
- `m3trics/models/healnet_wrapper.py`
  - Replaced the previous local forked implementation with a thin wrapper that
    imports `HealNet` from this vendored repo.
  - The wrapper now forwards sample-level modality masks instead of collapsing a
    modality to `None` whenever one sample in the batch is missing it.
- `m3trics/dataset/dataset.py`
  - Kept the existing `HealNetMaskAwareBatchSampler` but updated it to a hybrid
    policy:
    - exact-pattern full batches are emitted first
    - leftovers are packed into mixed batches by mask similarity
    - those residual mixed batches rely on the model-side subbatching described
      above
  - This preserves constant batch size whenever possible while still using the
    Marta-style per-sample missingness path for underfilled residual batches.

## Notes
- These changes are intentionally minimal and limited to the core model/test
  path used by the `methods` benchmark wrapper.
- The external pipeline wrapper that imports this repo lives in
  `m3trics/models/healnet_wrapper.py` and is not part of this vendored repo.
