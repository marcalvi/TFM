import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imputation_methods import KNNModalityImputer

# ------------------------ MM SIMULATOR --------------------------

class MissingModalitySimulator:
    def __init__(
        self,
        num_modalities,
        modality_names,
        missing_prob=0,
        missing_location="global",
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
        else:
            # Global missingness: each modality is independently removed
            # with probability `missing_prob`.
            missing_draw = np.random.random(self.num_modalities) < self.missing_prob
            pattern[missing_draw] = False

        # Always keep at least one modality present.
        if not pattern.any():
            keep_idx = np.random.randint(self.num_modalities)
            pattern[keep_idx] = True

        return pattern

# ------------------------ DATASET CLASSES --------------------------

class MultimodalDatasetWithMissing(Dataset):
    """Wrap a complete base dataset and inject simulated missing modalities."""

    def __init__(
        self,
        base_dataset,
        simulator,
        apply_missing=True,
        imputation_method="zero",
        knn_k=5,
    ):
        self.base_dataset = base_dataset
        self.simulator = simulator
        self.apply_missing = apply_missing
        self.imputation_method = str(imputation_method).strip().lower()
        if self.imputation_method not in {"zero", "knn"}:
            raise ValueError("imputation_method must be one of: zero, knn")

        self.imputer = None
        if self.apply_missing and self.imputation_method == "knn":
            self.imputer = KNNModalityImputer(base_dataset=self.base_dataset, k=knn_k)

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
            if self.imputation_method == "zero":
                # Replace missing modalities by zeros.
                for i, present in enumerate(present_mask):
                    if not bool(present):
                        Xs_missing[i] = torch.zeros_like(Xs_missing[i])
            else:
                # Replace missing modalities with KNN mean imputation.
                Xs_missing = self.imputer.impute_modalities(
                    modalities=Xs_missing,
                    present_mask=present_mask,
                    sample_index=idx,
                )
        else:
            # Validation/test path: keep all modalities as present.
            present_mask = torch.ones(self.simulator.num_modalities, dtype=torch.bool)
            Xs_missing = Xs

        return Xs_missing, present_mask, label, pid
    

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

# ------------------------ BUILD DATALOADER FUNCTION --------------------------

# Function to build data loaders with missing data for training and evaluation
def build_loaders(
    dfs_train_scaled,
    inst_df_train,
    dfs_eval_scaled,
    inst_df_eval,
    label_col,
    missing_simulator,
    batch_size,
    train_missing=False,
    val_missing=False,
    imputation_method="zero",
):
    train_base = MLPDataset(dfs=dfs_train_scaled, label_df=inst_df_train, label_col=label_col)
    val_base = MLPDataset(dfs=dfs_eval_scaled, label_df=inst_df_eval, label_col=label_col)

    train_ds = MultimodalDatasetWithMissing(
        base_dataset=train_base,
        simulator=missing_simulator,
        apply_missing=train_missing,
        imputation_method=imputation_method,
    )
    val_ds = MultimodalDatasetWithMissing(
        base_dataset=val_base,
        simulator=missing_simulator,
        apply_missing=val_missing,
        imputation_method=imputation_method,
    )

    n_train = len(train_ds)
    if n_train < 2:
        raise ValueError(
            f"Inner-train split has only {n_train} sample(s). "
            "At least 2 are required with BatchNorm."
        )
    # Drop only when the last batch would have size 1 (BatchNorm issue).
    drop_last_train = (n_train % batch_size) == 1

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multimodal_collate,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_collate,
        drop_last=False,
    )

    return train_loader, val_loader