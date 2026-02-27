import torch
import torch.nn as nn

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
