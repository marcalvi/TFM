import torch
import torch.nn as nn


def _build_hidden_stack(input_dim, hidden_dim, n_hidden_layers, dropout_p, use_batchnorm=True):
    if n_hidden_layers < 1:
        raise ValueError("n_hidden_layers must be >= 1")

    layers = []
    in_dim = input_dim
    for _ in range(n_hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend(
            [
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
        fusion_batchnorm=False,
        output_dim=1,
    ):
        super().__init__()

        if not input_dims:
            raise ValueError("input_dims must contain at least one modality")

        self.input_dims = list(input_dims)
        self.n_modalities = len(self.input_dims)
        self.use_mask = use_mask
        self.fusion_hidden_layers = int(fusion_hidden_layers)
        self.output_dim = int(output_dim)
        if self.fusion_hidden_layers < 1:
            raise ValueError("fusion_hidden_layers must be >= 1")
        if self.output_dim < 1:
            raise ValueError("output_dim must be >= 1")

        self.modality_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *_build_hidden_stack(
                        input_dim=dim,
                        hidden_dim=modality_hidden_dim,
                        n_hidden_layers=modality_hidden_layers,
                        dropout_p=dropout_p,
                        use_batchnorm=True,
                    )
                )
                for dim in self.input_dims
            ]
        )

        fusion_input_dim = self.n_modalities * modality_hidden_dim
        if self.use_mask:
            fusion_input_dim += self.n_modalities

        self.fusion_linears = nn.ModuleList()
        self.fusion_batchnorms = nn.ModuleList()
        self.fusion_dropouts = nn.ModuleList()
        in_dim = fusion_input_dim
        for _ in range(self.fusion_hidden_layers):
            self.fusion_linears.append(nn.Linear(in_dim, fusion_hidden_dim))
            self.fusion_batchnorms.append(
                nn.BatchNorm1d(fusion_hidden_dim) if bool(fusion_batchnorm) else nn.Identity()
            )
            self.fusion_dropouts.append(nn.Dropout(p=dropout_p))
            in_dim = fusion_hidden_dim
        self.fusion_output = nn.Linear(fusion_hidden_dim, self.output_dim)

    def forward(self, Xs, present_mask=None, return_aux=False):
        if len(Xs) != self.n_modalities:
            raise ValueError(f"Expected {self.n_modalities} modalities, got {len(Xs)}")

        if present_mask is None:
            batch_size = Xs[0].shape[0]
            present_mask = torch.ones(
                batch_size,
                self.n_modalities,
                device=Xs[0].device,
                dtype=torch.bool,
            )

        encoded = []
        for i, (block, Xi) in enumerate(zip(self.modality_blocks, Xs)):
            feat = block(Xi)
            feat = feat * present_mask[:, i].unsqueeze(1).float()
            encoded.append(feat)

        fused = torch.cat(encoded, dim=1)
        if self.use_mask:
            fused = torch.cat([fused, present_mask.float()], dim=1)

        hidden = fused
        final_pre_activation = None
        for linear, batchnorm, dropout in zip(
            self.fusion_linears,
            self.fusion_batchnorms,
            self.fusion_dropouts,
        ):
            hidden = linear(hidden)
            hidden = batchnorm(hidden)
            final_pre_activation = hidden
            hidden = torch.relu(hidden)
            hidden = dropout(hidden)

        logits = self.fusion_output(hidden)
        if return_aux:
            return logits, final_pre_activation
        return logits
