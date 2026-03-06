import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillDyAM(nn.Module):
    """DyAM variant for distillation.

    Differences vs plain DyAM:
    - Can concatenate present-mask to each modality branch input.
    - Exposes fused representation for representation distillation.
    """

    def __init__(
        self,
        input_dims,
        bottleneck_dim=16,
        dropout_p=0.4,
        temperature=2.0,
        concat_masks_input=True,
    ):
        super().__init__()

        if not input_dims:
            raise ValueError("input_dims must contain at least one modality")

        self.input_dims = list(input_dims)
        self.n_modalities = len(self.input_dims)
        self.bottleneck_dim = int(bottleneck_dim)
        self.T = float(temperature)
        self.concat_masks_input = bool(concat_masks_input)
        if self.T <= 0:
            raise ValueError("temperature must be > 0")

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d + (self.n_modalities if self.concat_masks_input else 0), bottleneck_dim),
                    nn.BatchNorm1d(bottleneck_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p),
                )
                for d in self.input_dims
            ]
        )

        self.risk_layers = nn.ModuleList(
            [nn.Linear(bottleneck_dim, 1, bias=False) for _ in range(self.n_modalities)]
        )
        self.attn_layers = nn.ModuleList(
            [nn.Linear(bottleneck_dim, 1, bias=False) for _ in range(self.n_modalities)]
        )

    def _infer_mask_from_input(self, Xs):
        batch_size = Xs[0].shape[0]
        device = Xs[0].device
        masks = torch.zeros((batch_size, self.n_modalities), device=device, dtype=torch.float32)

        for i, Xi in enumerate(Xs):
            if Xi is None:
                continue
            present = (Xi != 0).any(dim=1).float()
            masks[:, i] = present
        return masks

    def forward(self, Xs, present_mask=None, return_aux=False):
        if len(Xs) != self.n_modalities:
            raise ValueError(f"Expected {self.n_modalities} modalities, got {len(Xs)}")

        first_valid = next((x for x in Xs if x is not None), None)
        if first_valid is None:
            raise ValueError("All modalities in this batch are None.")

        batch_size = first_valid.size(0)
        device = first_valid.device
        risk_scores = []
        attn_logits = []
        modality_feats = []

        if present_mask is None:
            masks = self._infer_mask_from_input(Xs)
        else:
            masks = present_mask.to(device=device, dtype=torch.float32)

        for i in range(self.n_modalities):
            Xi = Xs[i]
            if Xi is None:
                risk_scores.append(torch.zeros(batch_size, 1, device=device))
                attn_logits.append(torch.full((batch_size, 1), -1e9, device=device))
                modality_feats.append(torch.zeros(batch_size, self.bottleneck_dim, device=device))
                continue

            if self.concat_masks_input:
                Xi = torch.cat([Xi, masks], dim=1)

            feat = self.projections[i](Xi)
            modality_feats.append(feat)
            risk_scores.append(self.risk_layers[i](feat))
            attn_logits.append(self.attn_layers[i](feat))

        R = torch.cat(risk_scores, dim=1)
        A_logits = torch.cat(attn_logits, dim=1)

        A_softplus = F.softplus(A_logits) / self.T
        masked_scores = A_softplus * masks
        R_masked = R * masks
        attn_weights = masked_scores / (masked_scores.sum(dim=1, keepdim=True) + 1e-9)
        output = (attn_weights * R_masked).sum(dim=1, keepdim=True)

        feat_stack = torch.stack(modality_feats, dim=1)  # [B, M, D]
        fused_repr = (attn_weights.unsqueeze(-1) * feat_stack).sum(dim=1)  # [B, D]

        if return_aux:
            num_active = masks.sum(dim=1, keepdim=True)
            alpha = attn_weights * num_active
            return output, attn_weights, alpha, R_masked, fused_repr
        return output

