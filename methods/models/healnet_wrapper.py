from pathlib import Path
import importlib
import sys

import torch
import torch.nn as nn


_HEALNET_REPO_ROOT = Path(__file__).resolve().parent / "healnet-adoption"
if str(_HEALNET_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HEALNET_REPO_ROOT))

_HEALNET_MODULE = importlib.import_module("healnet.models.healnet")
Attention = _HEALNET_MODULE.Attention
HealNet = _HEALNET_MODULE.HealNet


class HealNetBinaryWrapper(nn.Module):
    """Binary-classification adapter around the vendored HealNet repo.

    This pipeline provides one vector per modality with shape ``[B, D]`` plus a
    ``present_mask`` of shape ``[B, M]``. The wrapper maps each modality to
    ``[B, 1, D]`` and forwards sample-level availability so HealNet can skip
    fusion updates only for the missing samples, not for the whole batch.
    """

    def __init__(
        self,
        input_dims,
        depth=3,
        num_freq_bands=2,
        max_freq=10.0,
        num_latents=128,
        latent_dim=128,
        cross_heads=8,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        attn_dropout=0.0,
        ff_dropout=0.0,
        self_per_cross_attn=1,
        weight_tie_layers=False,
        fourier_encode_data=True,
        snn=True,
    ):
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims must contain at least one modality.")

        self.n_modalities = len(input_dims)
        self.input_dims = [int(d) for d in input_dims]

        self.model = HealNet(
            n_modalities=self.n_modalities,
            channel_dims=self.input_dims,
            num_spatial_axes=[1] * self.n_modalities,
            out_dims=1,
            depth=int(depth),
            num_freq_bands=int(num_freq_bands),
            max_freq=float(max_freq),
            l_c=int(num_latents),
            l_d=int(latent_dim),
            x_heads=int(cross_heads),
            l_heads=int(latent_heads),
            cross_dim_head=int(cross_dim_head),
            latent_dim_head=int(latent_dim_head),
            attn_dropout=float(attn_dropout),
            ff_dropout=float(ff_dropout),
            weight_tie_layers=bool(weight_tie_layers),
            fourier_encode_data=bool(fourier_encode_data),
            self_per_cross_attn=int(self_per_cross_attn),
            final_classifier_head=True,
            snn=bool(snn),
        )

    def forward(self, Xs, present_mask=None):
        if len(Xs) != self.n_modalities:
            raise ValueError(f"Expected {self.n_modalities} modalities, got {len(Xs)}")

        if present_mask is not None:
            if present_mask.ndim != 2 or present_mask.shape[1] != self.n_modalities:
                raise ValueError(
                    f"present_mask must have shape [B, {self.n_modalities}], got {tuple(present_mask.shape)}."
                )
            present_mask = present_mask.to(dtype=torch.bool)

        tensors = []
        for i, Xi in enumerate(Xs):
            if Xi is None:
                tensors.append(None)
                continue

            if Xi.ndim != 2:
                raise ValueError(
                    f"HealNet wrapper expects [B, D], got {tuple(Xi.shape)} for modality {i}."
                )
            if present_mask is not None and not bool(torch.any(present_mask[:, i])):
                tensors.append(None)
                continue

            tensors.append(Xi.unsqueeze(1))

        if present_mask is None:
            return self.model(tensors)
        if not bool(torch.any(present_mask)):
            return None

        masks = [present_mask[:, i].unsqueeze(1) for i in range(self.n_modalities)]
        return self.model(tensors, masks=masks)


__all__ = ["Attention", "HealNet", "HealNetBinaryWrapper"]
