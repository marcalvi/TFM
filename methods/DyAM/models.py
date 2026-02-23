import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DyAM(nn.Module):
    def __init__(self, input_dims, bottleneck_dim=16, dropout_p=0.4, temperature=2):
        super().__init__()
        self.n_modalities = len(input_dims)
        self.T = temperature 
        self.dropout = nn.Dropout(p=dropout_p)

        # Bottleneck projections: Harmonizes all modalities to the same size
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_p)
            ) for d in input_dims
        ])

        # Layers now take bottleneck_dim instead of raw input_dims
        self.risk_layers = nn.ModuleList([
            nn.Linear(bottleneck_dim, 1, bias=False) for _ in range(self.n_modalities)
        ])

        self.attn_layers = nn.ModuleList([
            nn.Linear(bottleneck_dim, 1, bias=False) for _ in range(self.n_modalities)
        ])

    def forward(self, Xs, masks=None):
        first_valid = next((x for x in Xs if x is not None), None)
        if first_valid is None:
            raise ValueError("All modalities in this batch are None.")
        
        B = first_valid.size(0)
        device = first_valid.device
        risk_scores = []
        attn_logits = []

        masks = torch.zeros((B, self.n_modalities), device = device, dtype=torch.float32)

        for i in range(self.n_modalities):
            Xi = Xs[i]

            if Xi is None:
                
                risk_scores.append(torch.zeros(B, 1, device=device))
                attn_logits.append(torch.full((B, 1), -1e9, device=device))

            else:
                for j in range(Xi.size(0)):
                    tensor = Xi[j,:]
                    if not torch.all(tensor == 0):
                        masks[j,i] = 1

                # Project raw features to bottleneck space
                feat = self.projections[i](Xi)
                
                # Calculate scores from the bottlenecked representation
                risk_scores.append(self.risk_layers[i](feat))
                attn_logits.append(self.attn_layers[i](feat))

        R = torch.cat(risk_scores, dim=1)
        R_masked = R * masks
        A_logits = torch.cat(attn_logits, dim=1)
        A_softplus = F.softplus(A_logits)
        A_softplus_T = A_softplus / self.T
        masked_logits = A_softplus_T * masks
        a = masked_logits / masked_logits.sum(dim=1, keepdim=True)

        # The count of available modalities per patient
        num_active = masks.sum(dim=1, keepdim=True)
        # The Attention Share (scaled by count)
        alpha = a * num_active

        output = (a * R_masked).sum(dim=1, keepdim=True)

        return output, a, alpha, R_masked
