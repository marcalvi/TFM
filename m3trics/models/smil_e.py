import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.utils import compute_survival_loss_from_logits, normalize_task_type


class UniversalReconstructionNet(nn.Module):
    """Global reconstruction module for arbitrary subsets of present modalities."""

    def __init__(self, latent_dim, num_priors, num_modalities, num_heads=4, dropout=0.1):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads for MultiheadAttention.")

        self.latent_dim = int(latent_dim)
        self.num_priors = int(num_priors)
        self.num_modalities = int(num_modalities)

        self.input_projection = nn.Linear(self.latent_dim, self.latent_dim)
        self.modality_embedding = nn.Embedding(self.num_modalities, self.latent_dim)

        self.self_attention = nn.MultiheadAttention(
            self.latent_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(self.latent_dim)

        self.cross_attention = nn.MultiheadAttention(
            self.latent_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(self.latent_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )
        self.ffn_norm = nn.LayerNorm(self.latent_dim)

        self.sigma_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_priors),
            nn.Softplus(),
        )

    def forward(self, h_present, present_mask):
        """Predict reconstruction scales for every modality.

        Args:
            h_present: [B, M, D] latent tensor where missing modalities are zeroed.
            present_mask: [B, M] bool tensor with modality availability.
        """
        if h_present.ndim != 3:
            raise ValueError(f"Expected h_present with shape [B, M, D], got {tuple(h_present.shape)}.")
        if present_mask.ndim != 2:
            raise ValueError(
                f"Expected present_mask with shape [B, M], got {tuple(present_mask.shape)}."
            )
        if h_present.shape[:2] != present_mask.shape:
            raise ValueError(
                f"h_present and present_mask shape mismatch: {tuple(h_present.shape)} vs "
                f"{tuple(present_mask.shape)}."
            )
        if not torch.all(present_mask.any(dim=1)):
            raise ValueError("UniversalReconstructionNet received a sample with all modalities missing.")

        batch_size, num_modalities, _ = h_present.shape
        device = h_present.device

        modality_ids = torch.arange(num_modalities, device=device)
        modality_emb = self.modality_embedding(modality_ids).unsqueeze(0).expand(batch_size, -1, -1)

        h = self.input_projection(h_present) + modality_emb
        key_padding_mask = ~present_mask

        h_self, _ = self.self_attention(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = self.self_attn_norm(h + h_self)

        h_ffn = self.ffn(h)
        h = self.ffn_norm(h + h_ffn)

        sigma_all = []
        for missing_idx in range(num_modalities):
            query = self.modality_embedding(
                torch.tensor([missing_idx], device=device)
            ).view(1, 1, -1).expand(batch_size, 1, -1)
            h_cross, _ = self.cross_attention(
                query,
                h,
                h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            h_cross = self.cross_attn_norm(query + h_cross)
            sigma_all.append(self.sigma_predictor(h_cross.squeeze(1)))

        return torch.stack(sigma_all, dim=1), h


class SMILNoiseNet(nn.Module):
    """Feature perturbation network matching the original SMIL training style."""

    def __init__(self, fused_dim, hidden_dim):
        super().__init__()
        self.fc0 = nn.Linear(int(fused_dim), int(fused_dim))
        self.fc1 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.softplus = nn.Softplus()

    def forward(self, x, layer_name, meta_train=True):
        if layer_name == "fc0":
            mu = self.fc0(x)
        elif layer_name == "fc1":
            mu = self.fc1(x)
        else:
            raise ValueError(f"Unknown layer_name '{layer_name}'. Expected 'fc0' or 'fc1'.")

        if meta_train:
            return self.softplus(torch.randn_like(mu) + mu)
        return self.softplus(mu)


class SMILE(nn.Module):
    """SMIL extended to M modalities by replacing pairwise reconstruction with global attention."""

    def __init__(
        self,
        input_dims,
        latent_dim=64,
        num_priors=64,
        num_heads=4,
        dropout=0.1,
        classifier_hidden_dim=256,
        alpha=1e-2,
        beta=1e-2,
        output_dim=1,
    ):
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims must contain at least one modality.")

        self.input_dims = list(input_dims)
        self.num_modalities = len(self.input_dims)
        self.latent_dim = int(latent_dim)
        self.num_priors = int(num_priors)
        self.classifier_hidden_dim = int(classifier_hidden_dim)
        self.output_dim = int(output_dim)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.encoders = nn.ModuleList(
            [nn.Linear(dim, self.latent_dim) for dim in self.input_dims]
        )
        self.register_buffer(
            "priors",
            torch.randn(self.num_modalities, self.num_priors, self.latent_dim),
        )

        self.reconstruction_net = UniversalReconstructionNet(
            latent_dim=self.latent_dim,
            num_priors=self.num_priors,
            num_modalities=self.num_modalities,
            num_heads=num_heads,
            dropout=dropout,
        )

        fused_dim = self.latent_dim * self.num_modalities
        self.noise_net = SMILNoiseNet(fused_dim=fused_dim, hidden_dim=self.classifier_hidden_dim)
        self.classifier_fc1 = nn.Linear(fused_dim, self.classifier_hidden_dim)
        self.classifier_fc2 = nn.Linear(self.classifier_hidden_dim, self.output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def set_priors(self, priors):
        priors = torch.as_tensor(priors, dtype=self.priors.dtype, device=self.priors.device)
        if priors.shape != self.priors.shape:
            raise ValueError(
                f"Invalid priors shape {tuple(priors.shape)}; expected {tuple(self.priors.shape)}."
            )
        self.priors.copy_(priors)

    def _normalize_mask(self, present_mask, Xs):
        if present_mask is None:
            first_valid = next((x for x in Xs if x is not None), None)
            if first_valid is None:
                raise ValueError("All modalities are None.")
            batch_size = first_valid.size(0)
            masks = torch.zeros(
                (batch_size, self.num_modalities),
                device=first_valid.device,
                dtype=torch.bool,
            )
            for idx, Xi in enumerate(Xs):
                if Xi is None:
                    continue
                masks[:, idx] = (Xi != 0).any(dim=1)
            return masks

        if present_mask.ndim == 1:
            present_mask = present_mask.unsqueeze(0)
        if present_mask.size(1) != self.num_modalities:
            raise ValueError(
                f"present_mask has width {present_mask.size(1)}, expected {self.num_modalities}."
            )
        return present_mask.to(dtype=torch.bool)

    def _encode_modalities(self, Xs, masks, device):
        encoded_modalities = []
        first_valid = next((x for x in Xs if x is not None), None)
        for modality_idx, Xi in enumerate(Xs):
            if Xi is None:
                Xi = torch.zeros(
                    (first_valid.size(0), self.input_dims[modality_idx]),
                    dtype=first_valid.dtype,
                    device=device,
                )
            h = self.encoders[modality_idx](Xi)
            h = h * masks[:, modality_idx].unsqueeze(1).to(h.dtype)
            encoded_modalities.append(h)
        return torch.stack(encoded_modalities, dim=1)

    def reconstruct_missing_modalities(self, h_encoded, present_mask, meta_train=False):
        sigma_all, attended_present = self.reconstruction_net(h_encoded, present_mask)
        reconstructed = h_encoded.clone()
        missing_mask = ~present_mask

        for modality_idx in range(self.num_modalities):
            modality_missing = missing_mask[:, modality_idx]
            if not torch.any(modality_missing):
                continue
            sigma = sigma_all[modality_missing, modality_idx, :].clamp_min(1e-6)
            # Match the original SMIL behavior: sample only during meta-train,
            # use the mean at validation/test time.
            if meta_train:
                omega = torch.randn_like(sigma) * sigma + 1.0
            else:
                omega = torch.ones_like(sigma)
            priors_m = self.priors[modality_idx]
            reconstructed_vals = omega @ priors_m
            reconstructed[modality_missing, modality_idx, :] = reconstructed_vals

        return reconstructed, sigma_all, attended_present

    def _classify(self, h_modalities, add_noise, meta_train):
        fused = h_modalities.reshape(h_modalities.size(0), -1)
        if add_noise:
            fused = fused + self.noise_net(fused, layer_name="fc0", meta_train=meta_train)
        fusion_feature = fused

        hidden = self.classifier_fc1(fused)
        if add_noise:
            hidden = hidden + self.noise_net(hidden, layer_name="fc1", meta_train=meta_train)
        hidden_feature = hidden

        logits = self.classifier_fc2(self.dropout(self.relu(hidden)))
        return logits, fusion_feature, hidden_feature

    def forward(self, Xs, present_mask=None, mode="incomplete", meta_train=False, return_aux=False):
        if len(Xs) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {len(Xs)}.")

        first_valid = next((x for x in Xs if x is not None), None)
        if first_valid is None:
            raise ValueError("All modalities in this batch are None.")

        if mode not in {"incomplete", "complete"}:
            raise ValueError("mode must be 'incomplete' or 'complete'.")

        device = first_valid.device
        masks = self._normalize_mask(present_mask, Xs).to(device=device)
        if not torch.all(masks.any(dim=1)):
            raise ValueError("SMIL-E received a sample with all modalities missing.")

        h_encoded = self._encode_modalities(Xs, masks, device=device)

        if mode == "incomplete":
            h_modalities, sigma_all, attended_present = self.reconstruct_missing_modalities(
                h_encoded,
                masks,
                meta_train=meta_train,
            )
            add_noise = True
        else:
            h_modalities = h_encoded
            sigma_all = None
            attended_present = None
            add_noise = False

        logits, fusion_feature, hidden_feature = self._classify(
            h_modalities,
            add_noise=add_noise,
            meta_train=meta_train,
        )

        if not return_aux:
            return logits

        aux = {
            "present_mask": masks,
            "missing_mask": ~masks,
            "recon_sigma": sigma_all,
            "recon_context": attended_present,
            "encoded_modalities": h_encoded,
            "fused_modalities": h_modalities,
            "fusion_feature": fusion_feature,
            "hidden_feature": hidden_feature,
        }
        return logits, aux


def learn_priors(
    base_dataset,
    encoders,
    num_modalities,
    num_priors=100,
    batch_size=512,
    device=None,
):
    """Learn per-modality priors from modality slots visible to SMILe during training."""
    from sklearn.cluster import MiniBatchKMeans

    # `base_dataset` can be either the raw MultimodalBaseDataset or the
    # MultimodalDatasetWithMissing wrapper used by the training loader. When the
    # wrapper is provided, use its fixed masks so synthetic missingness does not
    # leak hidden modalities into the reconstruction priors.
    visible_masks = None
    if hasattr(base_dataset, "base_dataset"):
        missing_dataset = base_dataset
        source_dataset = missing_dataset.base_dataset
        if getattr(missing_dataset, "fixed_present_masks", None) is not None:
            visible_masks = [mask.detach().cpu().numpy().astype(bool) for mask in missing_dataset.fixed_present_masks]
    else:
        source_dataset = base_dataset

    if not hasattr(source_dataset, "indexed") or not hasattr(source_dataset, "patient_ids"):
        raise ValueError("base_dataset must expose 'indexed' and 'patient_ids'.")

    device = device or next(encoders.parameters()).device
    patient_ids = list(source_dataset.patient_ids)
    modality_frames = list(source_dataset.indexed.values())

    priors = []
    rng = np.random.default_rng(0)

    for modality_idx in range(num_modalities):
        modality_frame = modality_frames[modality_idx]
        available_patient_ids = []
        for sample_idx, pid in enumerate(patient_ids):
            if pid not in modality_frame.index:
                continue
            if visible_masks is not None and not bool(visible_masks[sample_idx][modality_idx]):
                continue
            available_patient_ids.append(pid)
        if not available_patient_ids:
            latent_dim = int(getattr(encoders[modality_idx], "out_features", 0))
            if latent_dim <= 0:
                raise ValueError(
                    f"Cannot infer latent dimension for SMILe modality index {modality_idx}."
                )
            priors.append(torch.zeros(int(num_priors), latent_dim, dtype=torch.float32))
            continue
        arr = modality_frame.loc[available_patient_ids].to_numpy(dtype=np.float32, copy=True)

        features = []
        encoders[modality_idx].eval()
        with torch.no_grad():
            for start in range(0, arr.shape[0], batch_size):
                batch = torch.from_numpy(arr[start : start + batch_size]).to(device)
                features.append(encoders[modality_idx](batch).cpu().numpy())
        features = np.concatenate(features, axis=0)

        n_clusters = min(int(num_priors), features.shape[0])
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=0,
            batch_size=min(max(n_clusters, 1), features.shape[0]),
            n_init=3,
        )
        kmeans.fit(features)
        centers = kmeans.cluster_centers_.astype(np.float32, copy=False)

        if n_clusters < int(num_priors):
            extra_idx = rng.integers(0, n_clusters, size=int(num_priors) - n_clusters)
            centers = np.concatenate([centers, centers[extra_idx]], axis=0)

        priors.append(torch.from_numpy(centers[: int(num_priors)]))

    return torch.stack(priors, dim=0)


def smile_alignment_loss(
    noisy_logits,
    clean_logits,
    noisy_aux,
    clean_aux,
    targets,
    alpha=1e-2,
    beta=1e-2,
    bce_criterion=None,
    mse_criterion=None,
    task_config=None,
):
    """SMIL-style meta-validation loss: classification on noisy view + feature alignment."""
    task_type = normalize_task_type((task_config or {}).get("task_type", "binary_classification"))
    if noisy_logits.ndim == 2 and noisy_logits.size(1) == 1:
        noisy_logits = noisy_logits.squeeze(1)
    if clean_logits.ndim == 2 and clean_logits.size(1) == 1:
        clean_logits = clean_logits.squeeze(1)

    if bce_criterion is None:
        bce_criterion = nn.BCEWithLogitsLoss()
    if mse_criterion is None:
        mse_criterion = nn.MSELoss()

    if task_type == "survival":
        loss_ce_noise = compute_survival_loss_from_logits(
            logits=noisy_logits,
            y_disc=targets["y_disc"],
            censorship=targets["censorship"],
            loss_name=str((task_config or {}).get("survival_loss", "nll")).strip().lower(),
        )
    else:
        loss_ce_noise = bce_criterion(noisy_logits, targets)
    loss_map_1 = mse_criterion(clean_aux["fusion_feature"], noisy_aux["fusion_feature"])
    loss_map_2 = mse_criterion(clean_aux["hidden_feature"], noisy_aux["hidden_feature"])
    total = loss_ce_noise + (float(alpha) * loss_map_1) + (float(beta) * loss_map_2)
    return total, {
        "ce_noise": loss_ce_noise.detach(),
        "align_fusion": loss_map_1.detach(),
        "align_hidden": loss_map_2.detach(),
    }


def meta_train_step(
    model,
    optimizer,
    incomplete_train_batch,
    incomplete_val_batch,
    complete_val_batch,
    inner_steps=1,
    inner_lr=1e-2,
    alpha=1e-2,
    beta=1e-2,
    task_config=None,
):
    """First-order SMIL-style meta update inside a benchmark inner-train split."""
    if inner_steps < 1:
        raise ValueError("inner_steps must be >= 1")

    bce_criterion = nn.BCEWithLogitsLoss()
    adapted_model = copy.deepcopy(model)
    adapted_model.train()
    inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=float(inner_lr))

    X_train_inc, mask_train_inc, y_train_inc = incomplete_train_batch
    last_meta_train_loss = None
    for _ in range(int(inner_steps)):
        noisy_logits, _ = adapted_model(
            X_train_inc,
            mask_train_inc,
            mode="incomplete",
            meta_train=True,
            return_aux=True,
        )
        if normalize_task_type((task_config or {}).get("task_type", "binary_classification")) == "survival":
            last_meta_train_loss = compute_survival_loss_from_logits(
                logits=noisy_logits if noisy_logits.ndim != 2 or noisy_logits.size(1) != 1 else noisy_logits.squeeze(1),
                y_disc=y_train_inc["y_disc"],
                censorship=y_train_inc["censorship"],
                loss_name=str((task_config or {}).get("survival_loss", "nll")).strip().lower(),
            )
        else:
            last_meta_train_loss = bce_criterion(noisy_logits.squeeze(1), y_train_inc)
        inner_optimizer.zero_grad()
        last_meta_train_loss.backward()
        inner_optimizer.step()

    X_val_inc, mask_val_inc, y_val_inc = incomplete_val_batch
    X_val_full, mask_val_full, y_val_full = complete_val_batch

    adapted_model.eval()
    noisy_logits, noisy_aux = adapted_model(
        X_val_inc,
        mask_val_inc,
        mode="incomplete",
        meta_train=False,
        return_aux=True,
    )
    clean_logits, clean_aux = adapted_model(
        X_val_full,
        mask_val_full,
        mode="complete",
        meta_train=False,
        return_aux=True,
    )
    meta_val_loss, meta_val_dict = smile_alignment_loss(
        noisy_logits=noisy_logits,
        clean_logits=clean_logits,
        noisy_aux=noisy_aux,
        clean_aux=clean_aux,
        targets=y_val_inc,
        alpha=alpha,
        beta=beta,
        bce_criterion=bce_criterion,
        task_config=task_config,
    )

    noisy_train_eval, _ = adapted_model(
        X_train_inc,
        mask_train_inc,
        mode="incomplete",
        meta_train=False,
        return_aux=True,
    )
    if normalize_task_type((task_config or {}).get("task_type", "binary_classification")) == "survival":
        meta_train_eval_loss = compute_survival_loss_from_logits(
            logits=noisy_train_eval if noisy_train_eval.ndim != 2 or noisy_train_eval.size(1) != 1 else noisy_train_eval.squeeze(1),
            y_disc=y_train_inc["y_disc"],
            censorship=y_train_inc["censorship"],
            loss_name=str((task_config or {}).get("survival_loss", "nll")).strip().lower(),
        )
    else:
        meta_train_eval_loss = bce_criterion(noisy_train_eval.squeeze(1), y_train_inc)
    total_meta_loss = meta_train_eval_loss + meta_val_loss

    optimizer.zero_grad()
    adapted_model.zero_grad()
    total_meta_loss.backward()

    for base_param, adapted_param in zip(model.parameters(), adapted_model.parameters()):
        if adapted_param.grad is None:
            base_param.grad = None
        else:
            base_param.grad = adapted_param.grad.detach().clone()
    optimizer.step()

    stats = {
        "meta_train_loss": float(meta_train_eval_loss.item()),
        "meta_val_loss": float(meta_val_loss.item()),
        "align_fusion": float(meta_val_dict["align_fusion"].item()),
        "align_hidden": float(meta_val_dict["align_hidden"].item()),
        "ce_noise": float(meta_val_dict["ce_noise"].item()),
        "total_meta_loss": float(total_meta_loss.item()),
    }
    return stats
