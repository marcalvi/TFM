import torch
import torch.nn as nn
import torch.nn.functional as F


class DyAM(nn.Module):
    """Dynamic Attention over Modalities for binary classification."""

    def __init__(self, input_dims, bottleneck_dim=16, dropout_p=0.4, temperature=2.0):
        super().__init__()

        if not input_dims:
            raise ValueError("input_dims must contain at least one modality")

        self.input_dims = list(input_dims)
        self.n_modalities = len(self.input_dims)
        self.T = float(temperature)
        if self.T <= 0:
            raise ValueError("temperature must be > 0")

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, bottleneck_dim),
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
        """Infer present mask from non-zero modality vectors when mask is not provided."""
        batch_size = Xs[0].shape[0]
        device = Xs[0].device
        masks = torch.zeros((batch_size, self.n_modalities), device=device, dtype=torch.float32)

        for i, Xi in enumerate(Xs):
            if Xi is None:
                continue
            # A modality is considered present if any feature differs from zero.
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

        if present_mask is None:
            masks = self._infer_mask_from_input(Xs)
        else:
            masks = present_mask.to(device=device, dtype=torch.float32)

        for i in range(self.n_modalities):
            Xi = Xs[i]
            if Xi is None:
                risk_scores.append(torch.zeros(batch_size, 1, device=device))
                attn_logits.append(torch.full((batch_size, 1), -1e9, device=device))
                continue

            feat = self.projections[i](Xi)
            risk_scores.append(self.risk_layers[i](feat))
            attn_logits.append(self.attn_layers[i](feat))

        R = torch.cat(risk_scores, dim=1)
        A_logits = torch.cat(attn_logits, dim=1)

        A_softplus = F.softplus(A_logits) / self.T
        masked_scores = A_softplus * masks
        R_masked = R * masks

        denom = masked_scores.sum(dim=1, keepdim=True) + 1e-9
        attn_weights = masked_scores / denom

        output = (attn_weights * R_masked).sum(dim=1, keepdim=True)

        if return_aux:
            num_active = masks.sum(dim=1, keepdim=True)
            alpha = attn_weights * num_active
            return output, attn_weights, alpha, R_masked
        return output
