import numpy as np
import torch
from torch.utils.data import Dataset


class MLPDataset(Dataset):
    """Multimodal patient-level dataset with complete modalities."""

    def __init__(self, dfs, label_df, label_col, id_col="patient"):
        self.label_col = label_col

        # `label_df` is  indexed by patient in main.py
        self.label_df = label_df
        self.patients = sorted(self.label_df.index.tolist())
        self.indexed = {}

        # Keep one feature-only dataframe per modality (drop patient id column)
        for name, df in dfs.items():
            feat_df = df.drop(columns=[id_col], errors="ignore")
            missing_ids = set(self.patients) - set(feat_df.index.tolist())
            if missing_ids:
                raise ValueError(
                    f"Base dataset has missing modality '{name}' for {len(missing_ids)} patients."
                )
            self.indexed[name] = feat_df

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p_id = self.patients[idx]
        Xs = []

        for df in self.indexed.values():
            x = torch.tensor(df.loc[p_id].values, dtype=torch.float32)
            Xs.append(x)

        y = torch.tensor(float(self.label_df.loc[p_id, self.label_col]), dtype=torch.float32)
        return Xs, y, p_id

class MissingModalitySimulator:
    def __init__(
        self,
        num_modalities,
        modality_names,
        missing_prob=0,
        missing_location="global",
        mild_weight=4.0 / 9.0,
        moderate_weight=3.0 / 9.0,
        severe_weight=2.0 / 9.0,
    ):
        if num_modalities < 1:
            raise ValueError("num_modalities must be >= 1")
        if missing_prob < 0 or missing_prob > 1:
            raise ValueError("missing_prob must be in [0, 1]")

        names = list(modality_names)
        if len(names) != num_modalities:
            raise ValueError("Length of modality_names must match num_modalities")

        self.modality_names = names
        self.modality_name_to_idx = {str(n).lower(): i for i, n in enumerate(self.modality_names)}
        self.num_modalities = int(num_modalities)
        self.missing_prob = float(missing_prob)
        self.missing_location = str(missing_location).lower()

        if self.missing_location == "global":
            missing_weights = np.array([mild_weight, moderate_weight, severe_weight], dtype=float)
            if np.any(missing_weights < 0):
                raise ValueError("Global missing severity weights must be non-negative")
            if not np.isclose(missing_weights.sum(), 1.0):
                raise ValueError("Global missing severity weights must sum to 1")
            self.complete_prob = 1.0 - self.missing_prob
            self.mild_prob = self.missing_prob * missing_weights[0]
            self.moderate_prob = self.missing_prob * missing_weights[1]
            self.severe_prob = self.missing_prob * missing_weights[2]
            self.specific_missing_idx = None
        else:
            if self.missing_location not in self.modality_name_to_idx:
                valid = ", ".join(["global"] + sorted(self.modality_name_to_idx.keys()))
                raise ValueError(
                    f"Invalid missing_location='{missing_location}'. Expected one of: {valid}"
                )
            self.specific_missing_idx = self.modality_name_to_idx[self.missing_location]

    def generate_missing_pattern(self):
        """Generate a modality-presence mask for one sample."""
        pattern = np.ones(self.num_modalities, dtype=bool)

        # Modality-specific missingness: only one selected modality can be dropped.
        if self.specific_missing_idx is not None:
            if np.random.random() < self.missing_prob:
                pattern[self.specific_missing_idx] = False
            return pattern

        # Global missingness: missing_prob is split into mild/moderate/severe with baseline ratios.
        r = np.random.random()

        if r < self.complete_prob:
            return pattern

        if r < self.complete_prob + self.mild_prob:
            missing_idx = np.random.randint(self.num_modalities)
            pattern[missing_idx] = False
            return pattern

        if r < self.complete_prob + self.mild_prob + self.moderate_prob:
            if self.num_modalities == 1:
                return pattern
            missing_idxs = np.random.choice(self.num_modalities, min(2, self.num_modalities), replace=False)
            pattern[missing_idxs] = False
            return pattern

        min_missing = min(3, self.num_modalities)
        num_missing = np.random.randint(min_missing, self.num_modalities + 1)
        missing_idxs = np.random.choice(self.num_modalities, num_missing, replace=False)
        pattern[missing_idxs] = False
        return pattern


class MultimodalDatasetWithMissing(Dataset):
    """Wrap a complete base dataset and inject simulated missing modalities."""

    def __init__(self, base_dataset, simulator, apply_missing=True):
        self.base_dataset = base_dataset
        self.simulator = simulator
        self.apply_missing = apply_missing

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        Xs, label, pid = self.base_dataset[idx]
        if len(Xs) != self.simulator.num_modalities:
            raise ValueError("Number of modalities in sample does not match simulator.")

        if self.apply_missing:
            # Clone tensors so masking does not mutate base dataset outputs.
            Xs_missing = [m.clone() for m in Xs]
            present_mask = torch.as_tensor(self.simulator.generate_missing_pattern(), dtype=torch.bool)
            # Zero-out modalities marked as missing by the simulator.
            for i, present in enumerate(present_mask):
                if not bool(present):
                    Xs_missing[i] = torch.zeros_like(Xs_missing[i])
        else:
            # Validation/test path: keep all modalities as present.
            present_mask = torch.ones(self.simulator.num_modalities, dtype=torch.bool)
            Xs_missing = Xs

        return Xs_missing, present_mask, label, pid


def multimodal_collate(batch):
    """Collate multimodal samples into batched tensors per modality."""
    Xs_batch, present_masks, ys, pids = zip(*batch)
    n_modalities = len(Xs_batch[0])

    Xs_out = []
    for m in range(n_modalities):
        modality_samples = []
        for i in range(len(batch)):
            x = Xs_batch[i][m]
            if x.ndim != 1:
                raise ValueError(
                    f"Expected 1D modality vectors, got shape {tuple(x.shape)} "
                    f"for modality index {m} in batch sample {i}."
                )
            modality_samples.append(x.to(dtype=torch.float32))
        Xs_out.append(torch.stack(modality_samples, dim=0))

    y = torch.stack(ys).to(dtype=torch.float32)
    present_mask = torch.stack(present_masks).to(dtype=torch.bool)

    return Xs_out, present_mask, y, list(pids)
