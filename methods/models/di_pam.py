import torch
import torch.nn as nn
import torch.nn.functional as F


def _student_mlp_block(input_dim, output_dim, dropout_p):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout_p),
    )


class DiPAM(nn.Module):
    """Student network for pAM distillation.

    The student still computes one unimodal risk logit per modality, but instead
    of aggregating them with attention it learns:
    1. a compensated 5-D representation from masked unimodal logits + modality mask
    2. a final patient logit from that compensated representation

    Distillation is handled in the training loop against a plain pAM teacher.
    """

    def __init__(
        self,
        input_dims,
        bottleneck_dim=16,
        dropout_p=0.4,
        temperature=2.0,
        student_repr_hidden_dim=16,
        student_head_hidden_dim=8,
    ):
        super().__init__()

        if not input_dims:
            raise ValueError("input_dims must contain at least one modality")

        self.input_dims = list(input_dims)
        self.n_modalities = len(self.input_dims)
        self.bottleneck_dim = int(bottleneck_dim)
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

        student_input_dim = self.n_modalities * 2
        self.student_repr_projector = nn.Sequential(
            _student_mlp_block(student_input_dim, int(student_repr_hidden_dim), dropout_p),
            nn.Linear(int(student_repr_hidden_dim), self.n_modalities),
        )
        self.student_head = nn.Sequential(
            nn.Linear(self.n_modalities, int(student_head_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(student_head_hidden_dim), 1),
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

        if present_mask is None:
            masks = self._infer_mask_from_input(Xs)
        else:
            masks = present_mask.to(device=device, dtype=torch.float32)

        for i in range(self.n_modalities):
            Xi = Xs[i]
            if Xi is None:
                risk_scores.append(torch.zeros(batch_size, 1, device=device))
                continue

            feat = self.projections[i](Xi)
            risk_scores.append(self.risk_layers[i](feat))

        raw_risks = torch.cat(risk_scores, dim=1)
        masked_risks = raw_risks * masks

        student_input = torch.cat([masked_risks, masks], dim=1)
        compensated_repr = self.student_repr_projector(student_input)
        output = self.student_head(compensated_repr)

        if return_aux:
            # Provide a pAM-like proxy for downstream logging tables.
            proxy_scores = (F.softplus(raw_risks) / self.T) * masks
            attn_proxy = proxy_scores / (proxy_scores.sum(dim=1, keepdim=True) + 1e-9)
            num_active = masks.sum(dim=1, keepdim=True)
            alpha_proxy = attn_proxy * num_active
            return output, attn_proxy, alpha_proxy, masked_risks, compensated_repr
        return output
