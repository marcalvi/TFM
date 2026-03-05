import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from imputation_methods import KNNModalityImputer
import zlib

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

    def generate_missing_pattern(self, rng=None):
        """Generate a modality-presence mask for one sample."""
        if rng is None:
            rng = np.random.default_rng()
        pattern = np.ones(self.num_modalities, dtype=bool)

        # Modality-specific missingness: only one selected modality can be dropped.
        if self.specific_missing_idx is not None:
            if float(rng.random()) < self.missing_prob:
                pattern[self.specific_missing_idx] = False
        else:
            # Global missingness: each modality is independently removed
            # with probability `missing_prob`.
            missing_draw = rng.random(self.num_modalities) < self.missing_prob
            pattern[missing_draw] = False

        # Always keep at least one modality present.
        if not pattern.any():
            keep_idx = int(rng.integers(0, self.num_modalities))
            pattern[keep_idx] = True

        return pattern

# ------------------------ DATASET CLASSES --------------------------

class MultimodalBaseDataset(Dataset):
    """Multimodal patient-level dataset with complete modalities."""

    def __init__(self, dfs, label_df, label_col, id_col="patient"):
        self.label_col = label_col

        # `label_df` is indexed by patient ids in main.py
        self.label_df = label_df
        self.patient_ids = sorted(self.label_df.index.tolist())
        self.indexed = {}

        # Keep one feature-only dataframe per modality (drop patient id column)
        for name, df in dfs.items():
            feat_df = df.drop(columns=[id_col], errors="ignore")
            missing_ids = set(self.patient_ids) - set(feat_df.index.tolist())
            if missing_ids:
                raise ValueError(
                    f"Base dataset has missing modality '{name}' for {len(missing_ids)} patients."
                )
            self.indexed[name] = feat_df

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        p_id = self.patient_ids[idx]
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


class MultimodalDatasetWithMissing(Dataset):
    """Wrap a complete base dataset and inject simulated missing modalities."""

    def __init__(
        self,
        base_dataset,
        simulator,
        apply_missing=True,
        imputation_method="zero",
        knn_k=5,
        missing_pattern_seed=0,
    ):
        self.base_dataset = base_dataset
        self.simulator = simulator
        self.apply_missing = apply_missing
        self.imputation_method = str(imputation_method).strip().lower()
        self.missing_pattern_seed = int(missing_pattern_seed)
        if self.imputation_method not in {"zero", "knn"}:
            raise ValueError("imputation_method must be one of: zero, knn")

        self.imputer = None
        if self.apply_missing and self.imputation_method == "knn":
            self.imputer = KNNModalityImputer(base_dataset=self.base_dataset, k=knn_k)

        self.fixed_present_masks = None
        if self.apply_missing:
            self.fixed_present_masks = self._precompute_present_masks()

    def __len__(self):
        return len(self.base_dataset)

    def _stable_seed_for_patient(self, patient_id):
        key = (
            f"{self.missing_pattern_seed}|"
            f"{self.simulator.missing_location}|"
            f"{self.simulator.missing_prob:.12f}|"
            f"{patient_id}"
        )
        return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF

    def _precompute_present_masks(self):
        if hasattr(self.base_dataset, "patient_ids"):
            patient_ids = list(self.base_dataset.patient_ids)
        else:
            raise ValueError(
                "base_dataset must expose patient_ids "
                "for deterministic missing-pattern simulation."
            )

        masks = []
        for patient_id in patient_ids:
            rng = np.random.default_rng(self._stable_seed_for_patient(patient_id))
            mask = torch.as_tensor(self.simulator.generate_missing_pattern(rng=rng), dtype=torch.bool)
            masks.append(mask)
        return masks

    def __getitem__(self, idx):
        Xs, label, pid = self.base_dataset[idx]
        if len(Xs) != self.simulator.num_modalities:
            raise ValueError("Number of modalities in sample does not match simulator.")

        if self.apply_missing:
            # Clone tensors so masking does not mutate base dataset outputs.
            Xs_missing = [m.clone() for m in Xs]
            present_mask = self.fixed_present_masks[idx].clone()
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


class HealNetMaskAwareBatchSampler(Sampler):
    """Batch sampler that groups samples by missing-pattern compatibility.

    Goal: maximize batches where at least one modality is present for all
    samples in the batch, so HealNet can perform non-skipped modality updates.
    """

    def __init__(self, dataset, batch_size, shuffle=True, seed=0, drop_last=False):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._iter_idx = 0

        n_samples = len(dataset)
        if n_samples == 0:
            raise ValueError("HealNetMaskAwareBatchSampler received an empty dataset.")

        if getattr(dataset, "fixed_present_masks", None) is not None:
            mask_rows = [m.detach().cpu().numpy().astype(bool) for m in dataset.fixed_present_masks]
            self.sample_masks = np.stack(mask_rows, axis=0)
        else:
            n_modalities = int(dataset.simulator.num_modalities)
            self.sample_masks = np.ones((n_samples, n_modalities), dtype=bool)

        self.batches, self._batch_trainable = self._build_batches(self.seed)
        self.total_batches = len(self.batches)
        self.trainable_batches = int(sum(self._batch_trainable))
        self.trainable_batch_pct = (
            float(self.trainable_batches) / float(self.total_batches)
            if self.total_batches > 0
            else 0.0
        )

    def _build_batches(self, seed):
        rng = np.random.default_rng(int(seed))
        mask_tuples = [tuple(row.tolist()) for row in self.sample_masks]

        groups = {}
        for idx, mtuple in enumerate(mask_tuples):
            groups.setdefault(mtuple, []).append(idx)

        batches = []
        batch_trainable = []
        leftovers = []

        group_items = list(groups.items())
        if self.shuffle:
            rng.shuffle(group_items)

        for mtuple, indices in group_items:
            idxs = list(indices)
            if self.shuffle:
                rng.shuffle(idxs)

            n_full = len(idxs) // self.batch_size
            for b_i in range(n_full):
                batch = idxs[b_i * self.batch_size : (b_i + 1) * self.batch_size]
                batches.append(batch)
                batch_trainable.append(any(mtuple))

            rem_start = n_full * self.batch_size
            leftovers.extend(idxs[rem_start:])

        if leftovers:
            leftovers.sort(key=lambda ix: int(self.sample_masks[ix].sum()), reverse=True)
            while leftovers:
                first = leftovers.pop(0)
                batch = [first]
                common = self.sample_masks[first].copy()

                cand_i = 0
                while len(batch) < self.batch_size and cand_i < len(leftovers):
                    cand_idx = leftovers[cand_i]
                    cand_mask = self.sample_masks[cand_idx]
                    candidate_common = np.logical_and(common, cand_mask)
                    if candidate_common.any():
                        batch.append(cand_idx)
                        common = candidate_common
                        leftovers.pop(cand_i)
                    else:
                        cand_i += 1

                if self.drop_last and len(batch) < self.batch_size:
                    continue

                batches.append(batch)
                batch_trainable.append(bool(common.any()))

        if self.shuffle and len(batches) > 1:
            order = np.arange(len(batches))
            rng.shuffle(order)
            batches = [batches[i] for i in order]
            batch_trainable = [batch_trainable[i] for i in order]

        return batches, batch_trainable

    def __iter__(self):
        if self._iter_idx == 0:
            self._iter_idx += 1
            for b in self.batches:
                yield b
            return

        iter_seed = self.seed + self._iter_idx
        self._iter_idx += 1
        batches, _ = self._build_batches(iter_seed)
        for b in batches:
            yield b

    def __len__(self):
        return self.total_batches

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
    missing_pattern_seed=0,
    model_name="mlp",
    loader_seed=0,
    id_col="patient",
):
    train_base = MultimodalBaseDataset(
        dfs=dfs_train_scaled,
        label_df=inst_df_train,
        label_col=label_col,
        id_col=id_col,
    )
    val_base = MultimodalBaseDataset(
        dfs=dfs_eval_scaled,
        label_df=inst_df_eval,
        label_col=label_col,
        id_col=id_col,
    )

    train_ds = MultimodalDatasetWithMissing(
        base_dataset=train_base,
        simulator=missing_simulator,
        apply_missing=train_missing,
        imputation_method=imputation_method,
        missing_pattern_seed=missing_pattern_seed,
    )
    val_ds = MultimodalDatasetWithMissing(
        base_dataset=val_base,
        simulator=missing_simulator,
        apply_missing=val_missing,
        imputation_method=imputation_method,
        missing_pattern_seed=missing_pattern_seed,
    )

    model_name_l = str(model_name).strip().lower()
    train_batch_size = int(batch_size)
    eval_batch_size = 1 if model_name_l == "healnet" else train_batch_size
    n_train = len(train_ds)
    if model_name_l in {"mlp", "dyam"} and n_train < 2:
        raise ValueError(
            f"Inner-train split has only {n_train} sample(s). "
            "At least 2 are required with BatchNorm."
        )

    if model_name_l == "healnet" and bool(train_missing) and train_batch_size > 1:
        train_sampler = HealNetMaskAwareBatchSampler(
            dataset=train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            seed=int(loader_seed),
            drop_last=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=multimodal_collate,
        )
    else:
        drop_last_train = model_name_l in {"mlp", "dyam"} and (n_train % train_batch_size) == 1
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=multimodal_collate,
            drop_last=drop_last_train,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=multimodal_collate,
        drop_last=False,
    )

    return train_loader, val_loader
