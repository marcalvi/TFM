import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils import select_device

def _require_patient_ids(base_dataset):
    if not hasattr(base_dataset, "patient_ids"):
        raise ValueError("base_dataset must expose 'patient_ids'.")
    return list(base_dataset.patient_ids)

def _extract_modality_arrays(base_dataset, patient_ids):
    modality_arrays = []
    modality_dims = []
    for df in base_dataset.indexed.values():
        arr = df.loc[patient_ids].to_numpy(dtype=np.float32, copy=True)
        modality_arrays.append(arr)
        modality_dims.append(int(arr.shape[1]))
    return modality_arrays, modality_dims

def _normalize_reference_masks(reference_present_masks, n_samples, n_modalities):
    if reference_present_masks is None:
        return np.ones((n_samples, n_modalities), dtype=bool)

    if isinstance(reference_present_masks, torch.Tensor):
        masks_np = reference_present_masks.detach().cpu().numpy().astype(bool, copy=False)
    else:
        mask_rows = []
        for mask in reference_present_masks:
            if isinstance(mask, torch.Tensor):
                mask_rows.append(mask.detach().cpu().numpy().astype(bool, copy=False))
            else:
                mask_rows.append(np.asarray(mask, dtype=bool))
        masks_np = np.stack(mask_rows, axis=0)

    if masks_np.shape != (n_samples, n_modalities):
        raise ValueError(
            "reference_present_masks has invalid shape. "
            f"Expected {(n_samples, n_modalities)}, got {masks_np.shape}."
        )
    return masks_np

def _expand_feature_mask(present_mask, modality_dims):
    present_bool = np.asarray(present_mask, dtype=bool).reshape(-1)
    if len(present_bool) != len(modality_dims):
        raise ValueError(
            f"present_mask length {len(present_bool)} does not match "
            f"{len(modality_dims)} modalities."
        )
    parts = [
        np.full(dim, bool(is_present), dtype=np.float32)
        for is_present, dim in zip(present_bool, modality_dims)
    ]
    return np.concatenate(parts, axis=0)

def _get_self_index(sample_index, sample_id, patient_id_to_index, n_samples):
    if sample_id is not None and sample_id in patient_id_to_index:
        return int(patient_id_to_index[sample_id])
    if sample_index is not None and 0 <= int(sample_index) < n_samples:
        return int(sample_index)
    return None

class KNNModalityImputer:
    """Patient-level KNN imputer fit on one train split only.

    Distances are computed on modalities observed in both the query and each
    reference sample. Only reference samples where the target modality is
    available can contribute to that modality's imputation.
    """

    def __init__(self, base_dataset, reference_present_masks=None, k=5):
        if not hasattr(base_dataset, "indexed"):
            raise ValueError("base_dataset must expose 'indexed'.")

        self.k = int(k)
        if self.k < 1:
            raise ValueError("k must be >= 1")

        self.patient_ids = _require_patient_ids(base_dataset)
        self.n_samples = len(self.patient_ids)
        self.patient_id_to_index = {pid: i for i, pid in enumerate(self.patient_ids)}

        self.modality_arrays, self.modality_dims = _extract_modality_arrays(
            base_dataset,
            self.patient_ids,
        )
        self.n_modalities = len(self.modality_arrays)
        self.reference_present_masks = _normalize_reference_masks(
            reference_present_masks,
            self.n_samples,
            self.n_modalities,
        )

        self.modality_means = []
        for modality_idx, arr in enumerate(self.modality_arrays):
            present_rows = self.reference_present_masks[:, modality_idx]
            if present_rows.any():
                self.modality_means.append(arr[present_rows].mean(axis=0, dtype=np.float32))
            else:
                self.modality_means.append(arr.mean(axis=0, dtype=np.float32))

    def impute_modalities(self, modalities, present_mask, sample_index=None, sample_id=None):
        present = np.asarray(present_mask.detach().cpu().numpy(), dtype=bool)
        missing_idx = np.where(~present)[0]
        if missing_idx.size == 0:
            return modalities

        observed_idx = np.where(present)[0]
        dist2 = np.full(self.n_samples, np.inf, dtype=np.float32)

        if observed_idx.size > 0 and self.n_samples > 0:
            common_counts = np.zeros(self.n_samples, dtype=np.float32)
            accum = np.zeros(self.n_samples, dtype=np.float32)

            for modality_idx in observed_idx:
                available = self.reference_present_masks[:, modality_idx]
                if not np.any(available):
                    continue
                query = modalities[modality_idx].detach().cpu().numpy().astype(np.float32).reshape(1, -1)
                diff = self.modality_arrays[modality_idx] - query
                sq_dist = np.sum(diff * diff, axis=1)
                accum[available] += sq_dist[available]
                common_counts[available] += 1.0

            valid = common_counts > 0
            dist2[valid] = accum[valid] / common_counts[valid]

        self_idx = _get_self_index(
            sample_index=sample_index,
            sample_id=sample_id,
            patient_id_to_index=self.patient_id_to_index,
            n_samples=self.n_samples,
        )
        if self_idx is not None:
            dist2[self_idx] = np.inf

        for modality_idx in missing_idx:
            valid_candidates = np.where(
                self.reference_present_masks[:, modality_idx] & np.isfinite(dist2)
            )[0]

            if valid_candidates.size > 0:
                k_eff = min(self.k, valid_candidates.size)
                ordered = valid_candidates[np.argsort(dist2[valid_candidates])[:k_eff]]
                imputed_np = self.modality_arrays[modality_idx][ordered].mean(
                    axis=0,
                    dtype=np.float32,
                )
            else:
                imputed_np = self.modality_means[modality_idx]

            modalities[modality_idx] = torch.as_tensor(
                imputed_np,
                dtype=modalities[modality_idx].dtype,
                device=modalities[modality_idx].device,
            )

        return modalities


class _MaskedTabularVAE(nn.Module):
    """Small masked VAE for joint tabular multimodal reconstruction."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, masked_x, feature_mask):
        hidden = self.encoder(torch.cat([masked_x, feature_mask], dim=1))
        return self.mu_head(hidden), self.logvar_head(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, feature_mask):
        return self.decoder(torch.cat([z, feature_mask], dim=1))

    def forward(self, masked_x, feature_mask):
        mu, logvar = self.encode(masked_x, feature_mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, feature_mask)
        return recon, mu, logvar


class VAEModalityImputer:
    """Joint masked VAE imputer fit on one train split only.

    Inputs are the split's train samples after the train-missing simulation.
    Targets remain the complete split data, so the model learns to reconstruct
    missing modalities from observed modalities and the explicit mask.
    """

    def __init__(
        self,
        base_dataset,
        reference_present_masks=None,
        latent_dim=16,
        hidden_dim=128,
        epochs=30,
        batch_size=64,
        lr=1e-3,
        beta=1e-3,
        device=None,
        seed=0,
    ):
        if not hasattr(base_dataset, "indexed"):
            raise ValueError("base_dataset must expose 'indexed'.")

        self.patient_ids = _require_patient_ids(base_dataset)
        self.n_samples = len(self.patient_ids)
        self.patient_id_to_index = {pid: i for i, pid in enumerate(self.patient_ids)}

        self.modality_arrays, self.modality_dims = _extract_modality_arrays(
            base_dataset,
            self.patient_ids,
        )
        self.n_modalities = len(self.modality_arrays)
        self.reference_present_masks = _normalize_reference_masks(
            reference_present_masks,
            self.n_samples,
            self.n_modalities,
        )

        self.joint_slices = []
        cursor = 0
        for dim in self.modality_dims:
            self.joint_slices.append(slice(cursor, cursor + dim))
            cursor += dim
        self.input_dim = int(cursor)

        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.beta = float(beta)
        self.seed = int(seed)
        self.device = select_device() if device is None else torch.device(device)

        full_matrix = np.concatenate(self.modality_arrays, axis=1).astype(np.float32, copy=False)
        feature_mask = np.stack(
            [_expand_feature_mask(mask, self.modality_dims) for mask in self.reference_present_masks],
            axis=0,
        ).astype(np.float32, copy=False)
        masked_matrix = full_matrix * feature_mask

        self.model = _MaskedTabularVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        ).to(self.device)
        self._fit(masked_matrix, feature_mask, full_matrix)

    def _loss_function(self, recon_x, target_x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, target_x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + (self.beta * kld)

    def _fit(self, masked_matrix, feature_mask, full_matrix):
        dataset = TensorDataset(
            torch.from_numpy(masked_matrix),
            torch.from_numpy(feature_mask),
            torch.from_numpy(full_matrix),
        )
        batch_size = min(max(self.batch_size, 1), max(len(dataset), 1))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(max(self.epochs, 1)):
            for masked_x, mask_x, target_x in loader:
                masked_x = masked_x.to(self.device)
                mask_x = mask_x.to(self.device)
                target_x = target_x.to(self.device)

                optimizer.zero_grad()
                recon_x, mu, logvar = self.model(masked_x, mask_x)
                loss = self._loss_function(recon_x, target_x, mu, logvar)
                loss.backward()
                optimizer.step()

        self.model.eval()

    def impute_modalities(self, modalities, present_mask, sample_index=None, sample_id=None):
        present = np.asarray(present_mask.detach().cpu().numpy(), dtype=bool)
        missing_idx = np.where(~present)[0]
        if missing_idx.size == 0:
            return modalities

        joint_vector = np.concatenate(
            [mod.detach().cpu().numpy().astype(np.float32, copy=False) for mod in modalities],
            axis=0,
        )
        feature_mask = _expand_feature_mask(present, self.modality_dims)
        masked_vector = joint_vector * feature_mask

        with torch.no_grad():
            recon_x, _, _ = self.model(
                torch.from_numpy(masked_vector).unsqueeze(0).to(self.device),
                torch.from_numpy(feature_mask).unsqueeze(0).to(self.device),
            )
        recon_np = recon_x.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        for modality_idx in missing_idx:
            recon_slice = self.joint_slices[modality_idx]
            modalities[modality_idx] = torch.as_tensor(
                recon_np[recon_slice],
                dtype=modalities[modality_idx].dtype,
                device=modalities[modality_idx].device,
            )

        return modalities

def build_imputer(
    imputation_method,
    reference_base_dataset,
    reference_present_masks=None,
    knn_k=5,
    vae_kwargs=None,
    imputer_device=None,
    imputer_seed=0,
):
    method = str(imputation_method).strip().lower()
    if method == "zero":
        return None
    if method == "knn":
        return KNNModalityImputer(
            base_dataset=reference_base_dataset,
            reference_present_masks=reference_present_masks,
            k=knn_k,
        )
    if method == "vae":
        vae_kwargs = dict(vae_kwargs or {})
        return VAEModalityImputer(
            base_dataset=reference_base_dataset,
            reference_present_masks=reference_present_masks,
            device=imputer_device,
            seed=imputer_seed,
            **vae_kwargs,
        )
    raise ValueError("imputation_method must be one of: zero, knn, vae")
