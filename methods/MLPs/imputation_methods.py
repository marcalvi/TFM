import numpy as np
import torch


class KNNModalityImputer:
    """KNN imputer for missing modalities at patient level.

    Distances are computed with currently observed modalities only.
    Missing modality vectors are replaced by the mean vector of k nearest
    neighbors (excluding the sample itself).
    """

    def __init__(self, base_dataset, k=5):
        if not hasattr(base_dataset, "indexed") or not hasattr(base_dataset, "patients"):
            raise ValueError("base_dataset must expose 'indexed' and 'patients'.")

        self.k = int(k)
        if self.k < 1:
            raise ValueError("k must be >= 1")

        self.patient_ids = list(base_dataset.patients)
        self.n_samples = len(self.patient_ids)

        self.modality_arrays = []
        self.modality_means = []

        for df in base_dataset.indexed.values():
            # Keep row order aligned with dataset indexing.
            arr = df.loc[self.patient_ids].to_numpy(dtype=np.float32, copy=True)
            self.modality_arrays.append(arr)
            self.modality_means.append(arr.mean(axis=0, dtype=np.float32))

    def impute_modalities(self, modalities, present_mask, sample_index):
        present = np.asarray(present_mask.detach().cpu().numpy(), dtype=bool)
        missing_idx = np.where(~present)[0]
        if missing_idx.size == 0:
            return modalities

        observed_idx = np.where(present)[0]
        neighbors = np.array([], dtype=int)

        if observed_idx.size > 0 and self.n_samples > 1:
            dist2 = np.zeros(self.n_samples, dtype=np.float32)
            for m in observed_idx:
                query = modalities[m].detach().cpu().numpy().astype(np.float32).reshape(1, -1)
                arr = self.modality_arrays[m]
                diff = arr - query
                dist2 += np.sum(diff * diff, axis=1)

            dist2[int(sample_index)] = np.inf
            k_eff = min(self.k, self.n_samples - 1)
            if k_eff > 0:
                neighbors = np.argpartition(dist2, kth=k_eff - 1)[:k_eff]
                neighbors = neighbors[np.argsort(dist2[neighbors])]

        for m in missing_idx:
            if neighbors.size > 0:
                imputed_np = self.modality_arrays[m][neighbors].mean(axis=0, dtype=np.float32)
            else:
                imputed_np = self.modality_means[m]

            modalities[m] = torch.as_tensor(
                imputed_np,
                dtype=modalities[m].dtype,
                device=modalities[m].device,
            )

        return modalities
