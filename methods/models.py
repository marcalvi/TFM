import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- MLP ------------------------------

def _build_hidden_stack(input_dim, hidden_dim, n_hidden_layers, dropout_p):
    if n_hidden_layers < 1:
        raise ValueError("n_hidden_layers must be >= 1")

    layers = []
    in_dim = input_dim
    for _ in range(n_hidden_layers):
        layers.extend(
            [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
            ]
        )
        in_dim = hidden_dim
    return layers

class MultimodalMLP(nn.Module):
    def __init__(
        self,
        input_dims,
        modality_hidden_dim=16,
        modality_hidden_layers=1,
        fusion_hidden_dim=32,
        fusion_hidden_layers=1,
        dropout_p=0.2,
        use_mask=True,
    ):
        super().__init__()

        if not input_dims:
            raise ValueError("input_dims must contain at least one modality")

        self.input_dims = list(input_dims)
        self.n_modalities = len(self.input_dims)
        self.use_mask = use_mask

        self.modality_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *_build_hidden_stack(
                        input_dim=dim,
                        hidden_dim=modality_hidden_dim,
                        n_hidden_layers=modality_hidden_layers,
                        dropout_p=dropout_p,
                    )
                )
                for dim in self.input_dims
            ]
        )

        fusion_input_dim = self.n_modalities * modality_hidden_dim
        if self.use_mask:
            fusion_input_dim += self.n_modalities

        fusion_layers = _build_hidden_stack(
            input_dim=fusion_input_dim,
            hidden_dim=fusion_hidden_dim,
            n_hidden_layers=fusion_hidden_layers,
            dropout_p=dropout_p,
        )
        fusion_layers.append(nn.Linear(fusion_hidden_dim, 1))
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, Xs, present_mask=None):
        if len(Xs) != self.n_modalities:
            raise ValueError(f"Expected {self.n_modalities} modalities, got {len(Xs)}")

        if present_mask is None:
            batch_size = Xs[0].shape[0] # Files de la primera modalitat = nº pacients al batch 
            present_mask = torch.ones(batch_size, self.n_modalities, device=Xs[0].device, dtype=torch.bool)

        encoded = []
        for i, (block, Xi) in enumerate(zip(self.modality_blocks, Xs)):
            feat = block(Xi)
            feat = feat * present_mask[:, i].unsqueeze(1).float()
            encoded.append(feat)

        fused = torch.cat(encoded, dim=1)
        if self.use_mask:
            fused = torch.cat([fused, present_mask.float()], dim=1)

        return self.fusion(fused)




# ---------------------------- DyAM ------------------------------

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

        # Harmonize each modality into the same bottleneck space.
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

        # Positive attention scores with temperature scaling.
        A_softplus = F.softplus(A_logits) / self.T
        masked_scores = A_softplus * masks
        R_masked = R * masks

        # Normalize across modalities, guarding against zero denominator.
        denom = masked_scores.sum(dim=1, keepdim=True) + 1e-9
        attn_weights = masked_scores / denom

        # Final risk score (logit).
        output = (attn_weights * R_masked).sum(dim=1, keepdim=True)

        if return_aux:
            num_active = masks.sum(dim=1, keepdim=True)
            alpha = attn_weights * num_active
            return output, attn_weights, alpha, R_masked
        return output
