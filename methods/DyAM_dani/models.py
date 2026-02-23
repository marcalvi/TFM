import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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
    


class HierarchicalDyAM(nn.Module):
    def __init__(self, input_dims, bottleneck_dim=16, dropout_p=0.4, temperature=2):
        super().__init__()
        self.n_modalities = len(input_dims)
        self.T = temperature 
        
        # 1. Projections (Harmonize all 9 modalities to bottleneck_dim)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_p)
            ) for d in input_dims
        ])

        # 2. Risk Layers (One per modality)
        self.risk_local = nn.ModuleList([
            nn.Linear(bottleneck_dim, 1, bias=False) for _ in range(self.n_modalities)
        ])

        # 3. Local Attention (Intra-group)
        self.attn_local = nn.ModuleList([
            nn.Linear(bottleneck_dim, 1, bias=False) for _ in range(self.n_modalities)
        ])

        # 4. New Group-Level Risk Layers
        # These map the weighted modality risks into a single group-level risk
        self.risk_global_rad  = nn.Linear(5, 1, bias=True) # 5 Radiology modalities
        self.risk_global_path = nn.Linear(2, 1, bias=True) # 2 Pathology modalities
        self.risk_global_clin = nn.Linear(2, 1, bias=True) # 2 Clinical modalities
        
        # 4. Global Attention (Inter-group: Rad vs Path vs Clin)
        self.attn_global_rad = nn.Linear(5, 1, bias=False)
        self.attn_global_path = nn.Linear(2, 1, bias=False)
        self.attn_global_clin = nn.Linear(2, 1, bias=False)
    

    def forward(self, Xs):
        first_valid = next((x for x in Xs if x is not None), None)
        if first_valid is None: raise ValueError("All modalities in this batch are None.")
        
        B, device = first_valid.size(0), first_valid.device
        risk_scores, attn_logits = [], []
        masks = torch.zeros((B, self.n_modalities), device=device, dtype=torch.float32)

        # 1. Feature Extraction & Local Scoring
        for i in range(self.n_modalities):
            Xi = Xs[i]
            if Xi is None:
                risk_scores.append(torch.zeros(B, 1, device=device))
                attn_logits.append(torch.full((B, 1), -1e9, device=device))
            else:
                for j in range(Xi.size(0)):
                    if not torch.all(Xi[j,:] == 0): masks[j,i] = 1
                feat = self.projections[i](Xi)
                risk_scores.append(self.risk_local[i](feat))
                attn_logits.append(self.attn_local[i](feat))

        R_local = torch.cat(risk_scores, dim=1) # [B, 9]
        A_logits_local = torch.cat(attn_logits, dim=1)
        A_val_local = (F.softplus(A_logits_local) / self.T) * masks 

        # 2. Local Group Aggregation
        rad_idx, path_idx, clin_idx = [0,1,2,3,4], [5,6], [7,8]
        
        def get_weighted_risks(indices):
            group_A = A_val_local[:, indices]
            group_R = R_local[:, indices]
            group_masks = masks[:, indices]
            
            group_sum = group_A.sum(dim=1, keepdim=True) + 1e-9
            a_local = group_A / group_sum
            
            # Attn Share Local: a * num_available_in_group
            num_active_group = group_masks.sum(dim=1, keepdim=True)
            attn_share_local = a_local * num_active_group
            
            return a_local * group_R, a_local, attn_share_local

        out_rad_local, a_rad, share_rad_local = get_weighted_risks(rad_idx)
        out_path_local, a_path, share_path_local = get_weighted_risks(path_idx)
        out_clin_local, a_clin, share_clin_local = get_weighted_risks(clin_idx)

        # 3. Global Level Computations
        # Calculate Group-level Risks
        R_rad_global = self.risk_global_rad(out_rad_local)
        R_path_global = self.risk_global_path(out_path_local)
        R_clin_global = self.risk_global_clin(out_clin_local)
        R_global = torch.cat([R_rad_global, R_path_global, R_clin_global], dim=1) # [B, 3]

        # Calculate Group-level Attention Logits
        A_rad_global = self.attn_global_rad(out_rad_local)
        A_path_global = self.attn_global_path(out_path_local)
        A_clin_global = self.attn_global_clin(out_clin_local)
        A_logits_global = torch.cat([A_rad_global, A_path_global, A_clin_global], dim=1)

        # 4. Global Masking & Normalization
        # Group mask is 1 if at least one modality in family is present
        masks_global = torch.cat([
            (masks[:, rad_idx].sum(dim=1, keepdim=True) > 0).float(),
            (masks[:, path_idx].sum(dim=1, keepdim=True) > 0).float(),
            (masks[:, clin_idx].sum(dim=1, keepdim=True) > 0).float()
        ], dim=1)

        A_val_global = (F.softplus(A_logits_global) / self.T) * masks_global
        a_global = A_val_global / (A_val_global.sum(dim=1, keepdim=True) + 1e-9)
        
        # Global Attention Share
        num_active_global = masks_global.sum(dim=1, keepdim=True)
        share_global = a_global * num_active_global

        # 5. Final Output
        output = (a_global * R_global).sum(dim=1, keepdim=True)

        # 6. Reconstruct Final a for metrics (Full 9-modality attention)
        # This represents the total contribution of each modality to the final output
        a_final = torch.zeros((B, 9), device=device)
        a_final[:, rad_idx] = a_global[:, 0:1] * a_rad
        a_final[:, path_idx] = a_global[:, 1:2] * a_path
        a_final[:, clin_idx] = a_global[:, 2:3] * a_clin

                # total_active is the count of ALL modalities present for this patient (out of 9)
        total_active = masks.sum(dim=1, keepdim=True) 

        # share_final represents importance relative to the whole model
        share_final = a_final * total_active

        # Return bundle for evaluation
        # Note: Return local a and share for the 9 modalities, and global versions for the 3 groups
        return output, a_final, a_global, R_local, share_final, (share_rad_local, share_path_local, share_clin_local), share_global
