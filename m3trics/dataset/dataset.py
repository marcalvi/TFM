import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from dataset.imputation_methods import build_imputer
from scripts.utils import normalize_model_name, normalize_task_type
import zlib

# ------------------------ MM SIMULATOR --------------------------

class MissingModalitySimulator:
    def __init__(
        self,
        num_modalities,
        modality_names,
        missing_prop=0,
        missing_location="global",
    ):
        if num_modalities < 1:
            raise ValueError("num_modalities must be >= 1")
        if missing_prop < 0 or missing_prop > 1:
            raise ValueError("missing_prop must be in [0, 1]")

        names = list(modality_names)
        if len(names) != num_modalities:
            raise ValueError("Length of modality_names must match num_modalities")

        self.modality_names = names
        self.modality_name_to_idx = {str(n).lower(): i for i, n in enumerate(self.modality_names)}
        self.num_modalities = int(num_modalities)
        # Semantics: exact fraction of modality values to remove from the dataset,
        # not an independent Bernoulli probability per value.
        self.missing_prop = float(missing_prop)
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

    def _stable_slot_score(self, patient_id, modality_idx, missing_pattern_seed):
        key = (
            f"{int(missing_pattern_seed)}|"
            f"{self.missing_location}|"
            f"{patient_id}|"
            f"{int(modality_idx)}"
        )
        return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF

    def generate_dataset_missing_masks(self, patient_ids, missing_pattern_seed=0):
        """Generate deterministic dataset-level masks with an exact missing fraction."""
        patient_ids = list(patient_ids)
        n_samples = len(patient_ids)
        if n_samples == 0:
            return np.zeros((0, self.num_modalities), dtype=bool)

        masks = np.ones((n_samples, self.num_modalities), dtype=bool)
        missing_fraction = float(self.missing_prop)
        if missing_fraction <= 0.0:
            return masks

        if self.specific_missing_idx is not None:
            target_missing = int(round(missing_fraction * n_samples))
            target_missing = min(max(target_missing, 0), n_samples)
            ranked = sorted(
                (
                    self._stable_slot_score(pid, self.specific_missing_idx, missing_pattern_seed),
                    sample_idx,
                )
                for sample_idx, pid in enumerate(patient_ids)
            )
            for _, sample_idx in ranked[:target_missing]:
                masks[sample_idx, self.specific_missing_idx] = False
            return masks

        total_slots = n_samples * self.num_modalities
        target_missing = int(round(missing_fraction * total_slots))
        max_missing = n_samples * (self.num_modalities - 1)
        if target_missing > max_missing:
            raise ValueError(
                f"Global missing fraction {missing_fraction:g} is too large for "
                f"{self.num_modalities} modalities while keeping at least one "
                "modality present per sample."
            )

        row_missing_counts = np.zeros(n_samples, dtype=np.int64)
        ranked_slots = sorted(
            (
                self._stable_slot_score(pid, modality_idx, missing_pattern_seed),
                sample_idx,
                modality_idx,
            )
            for sample_idx, pid in enumerate(patient_ids)
            for modality_idx in range(self.num_modalities)
        )

        removed = 0
        max_missing_per_row = self.num_modalities - 1
        for _, sample_idx, modality_idx in ranked_slots:
            if removed >= target_missing:
                break
            if row_missing_counts[sample_idx] >= max_missing_per_row:
                continue
            masks[sample_idx, modality_idx] = False
            row_missing_counts[sample_idx] += 1
            removed += 1

        if removed != target_missing:
            raise RuntimeError(
                f"Unable to assign exact global missing fraction. "
                f"Requested {target_missing} missing slots, assigned {removed}."
            )

        return masks

# ------------------------ DATASET CLASSES --------------------------

class MultimodalBaseDataset(Dataset):
    """Multimodal patient-level dataset with complete modalities."""

    def __init__(
        self,
        dfs,
        label_df,
        label_col,
        id_col="patient",
        task_type="binary_classification",
        survival_time_col=None,
        survival_event_col=None,
        survival_censorship_col=None,
        survival_y_disc_col=None,
    ):
        self.label_col = label_col
        self.task_type = normalize_task_type(task_type)
        self.survival_time_col = survival_time_col
        self.survival_event_col = survival_event_col
        self.survival_censorship_col = survival_censorship_col
        self.survival_y_disc_col = survival_y_disc_col

        # `label_df` is indexed by patient ids in the training launcher
        self.label_df = label_df
        self.patient_ids = sorted(self.label_df.index.tolist())
        self.patient_id_to_index = {pid: i for i, pid in enumerate(self.patient_ids)}
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

        if self.task_type == "survival":
            y = {
                "event_time": torch.tensor(
                    float(self.label_df.loc[p_id, self.survival_time_col]),
                    dtype=torch.float32,
                ),
                "event": torch.tensor(
                    float(self.label_df.loc[p_id, self.survival_event_col]),
                    dtype=torch.float32,
                ),
                "censorship": torch.tensor(
                    float(self.label_df.loc[p_id, self.survival_censorship_col]),
                    dtype=torch.float32,
                ),
                "y_disc": torch.tensor(
                    int(self.label_df.loc[p_id, self.survival_y_disc_col]),
                    dtype=torch.long,
                ),
            }
        else:
            y = torch.tensor(float(self.label_df.loc[p_id, self.label_col]), dtype=torch.float32)
        return Xs, y, p_id

    def get_by_patient_id(self, patient_id):
        if patient_id not in self.patient_id_to_index:
            raise KeyError(f"Patient id '{patient_id}' not found in base dataset.")
        return self.__getitem__(self.patient_id_to_index[patient_id])
    

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

    if isinstance(ys[0], dict):
        y = {
            "event_time": torch.stack([item["event_time"] for item in ys]).to(dtype=torch.float32),
            "event": torch.stack([item["event"] for item in ys]).to(dtype=torch.float32),
            "censorship": torch.stack([item["censorship"] for item in ys]).to(dtype=torch.float32),
            "y_disc": torch.stack([item["y_disc"] for item in ys]).to(dtype=torch.long),
        }
    else:
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
        prefit_imputer=None,
        imputer_kwargs=None,
        imputer_device=None,
        imputer_seed=0,
    ):
        self.base_dataset = base_dataset
        self.simulator = simulator
        self.apply_missing = apply_missing
        self.imputation_method = str(imputation_method).strip().lower()
        self.missing_pattern_seed = int(missing_pattern_seed)
        if self.imputation_method not in {"zero", "knn", "vae"}:
            raise ValueError("imputation_method must be one of: zero, knn, vae")
        self.knn_k = int(knn_k)
        self.imputer_kwargs = dict(imputer_kwargs or {})
        self.imputer_device = imputer_device
        self.imputer_seed = int(imputer_seed)

        self.imputer = prefit_imputer

        self.fixed_present_masks = None
        if self.apply_missing:
            self.fixed_present_masks = self._precompute_present_masks()

    def __len__(self):
        return len(self.base_dataset)

    def _precompute_present_masks(self):
        if hasattr(self.base_dataset, "patient_ids"):
            patient_ids = list(self.base_dataset.patient_ids)
        else:
            raise ValueError(
                "base_dataset must expose patient_ids "
                "for deterministic missing-pattern simulation."
            )

        mask_array = self.simulator.generate_dataset_missing_masks(
            patient_ids=patient_ids,
            missing_pattern_seed=self.missing_pattern_seed,
        )
        return [torch.as_tensor(mask_row, dtype=torch.bool) for mask_row in mask_array]

    def set_imputer(self, imputer, imputation_method=None):
        if imputation_method is not None:
            method = str(imputation_method).strip().lower()
            if method not in {"zero", "knn", "vae"}:
                raise ValueError("imputation_method must be one of: zero, knn, vae")
            self.imputation_method = method
        self.imputer = imputer

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
                if self.imputer is None:
                    raise RuntimeError(
                        f"Imputer for method '{self.imputation_method}' has not been initialized."
                    )
                # Replace missing modalities with the selected fitted imputer.
                Xs_missing = self.imputer.impute_modalities(
                    modalities=Xs_missing,
                    present_mask=present_mask,
                    sample_index=idx,
                    sample_id=pid,
                )
        else:
            # Validation/test path: keep all modalities as present.
            present_mask = torch.ones(self.simulator.num_modalities, dtype=torch.bool)
            Xs_missing = Xs

        return Xs_missing, present_mask, label, pid


class HealNetMaskAwareBatchSampler(Sampler):
    """Hybrid batch sampler for HealNet under sample-level missingness.

    Strategy:
    1. Build as many full homogeneous batches as possible from exact missing
       patterns. These batches preserve constant batch size and maximize
       full-batch modality updates.
    2. Pool the leftover samples from underfilled groups into residual mixed
       batches. Those mixed batches are ordered by mask similarity, but they do
       not require one modality to be shared by the whole batch.
    3. The residual batches rely on model-side subbatching so each modality is
       updated only for the subset of samples where it is available.
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

        self.batches, self._batch_trainable, self._batch_homogeneous = self._build_batches(self.seed)
        self.total_batches = len(self.batches)
        self.trainable_batches = int(sum(self._batch_trainable))
        self.homogeneous_batches = int(sum(self._batch_homogeneous))
        self.mixed_residual_batches = int(self.total_batches - self.homogeneous_batches)
        self.trainable_batch_pct = (
            float(self.trainable_batches) / float(self.total_batches)
            if self.total_batches > 0
            else 0.0
        )
        self.homogeneous_batch_pct = (
            float(self.homogeneous_batches) / float(self.total_batches)
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
        batch_homogeneous = []
        leftovers = []

        group_items = list(groups.items())
        if self.shuffle:
            rng.shuffle(group_items)

        for mtuple, indices in group_items:
            idxs = list(indices)
            if self.shuffle:
                rng.shuffle(idxs)

            # First fill exact-mask batches so HealNet sees as many fully
            # homogeneous batches as possible before resorting to subbatching.
            n_full = len(idxs) // self.batch_size
            for b_i in range(n_full):
                batch = idxs[b_i * self.batch_size : (b_i + 1) * self.batch_size]
                batches.append(batch)
                batch_trainable.append(any(mtuple))
                batch_homogeneous.append(True)

            rem_start = n_full * self.batch_size
            leftovers.extend(idxs[rem_start:])

        if leftovers:
            leftovers.sort(key=lambda ix: int(self.sample_masks[ix].sum()), reverse=True)
            while leftovers:
                first = leftovers.pop(0)
                batch = [first]
                common = self.sample_masks[first].copy()
                union = self.sample_masks[first].copy()

                # Residual mixed batches are assembled by mask similarity; the
                # model-side subbatching will then use only the rows that are
                # actually present for each modality inside these leftovers.
                while len(batch) < self.batch_size and leftovers:
                    best_pos = None
                    best_score = None
                    for cand_i, cand_idx in enumerate(leftovers):
                        cand_mask = self.sample_masks[cand_idx]
                        shared_common = int(np.logical_and(common, cand_mask).sum())
                        shared_union = int(np.logical_and(union, cand_mask).sum())
                        score = (
                            shared_common,
                            shared_union,
                            int(cand_mask.sum()),
                        )
                        if best_score is None or score > best_score:
                            best_score = score
                            best_pos = cand_i

                    cand_idx = leftovers.pop(best_pos)
                    cand_mask = self.sample_masks[cand_idx]
                    batch.append(cand_idx)
                    common = np.logical_and(common, cand_mask)
                    union = np.logical_or(union, cand_mask)

                if self.drop_last and len(batch) < self.batch_size:
                    continue

                batches.append(batch)
                batch_trainable.append(bool(union.any()))
                batch_homogeneous.append(False)

        if self.shuffle and len(batches) > 1:
            order = np.arange(len(batches))
            rng.shuffle(order)
            batches = [batches[i] for i in order]
            batch_trainable = [batch_trainable[i] for i in order]
            batch_homogeneous = [batch_homogeneous[i] for i in order]

        return batches, batch_trainable, batch_homogeneous

    def __iter__(self):
        if self._iter_idx == 0:
            self._iter_idx += 1
            for b in self.batches:
                yield b
            return

        iter_seed = self.seed + self._iter_idx
        self._iter_idx += 1
        batches, _, _ = self._build_batches(iter_seed)
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
    prefit_imputer=None,
    imputer_kwargs=None,
    task_type="binary_classification",
    survival_time_col=None,
    survival_event_col=None,
    survival_censorship_col=None,
    survival_y_disc_col=None,
):
    train_base = MultimodalBaseDataset(
        dfs=dfs_train_scaled,
        label_df=inst_df_train,
        label_col=label_col,
        id_col=id_col,
        task_type=task_type,
        survival_time_col=survival_time_col,
        survival_event_col=survival_event_col,
        survival_censorship_col=survival_censorship_col,
        survival_y_disc_col=survival_y_disc_col,
    )
    val_base = MultimodalBaseDataset(
        dfs=dfs_eval_scaled,
        label_df=inst_df_eval,
        label_col=label_col,
        id_col=id_col,
        task_type=task_type,
        survival_time_col=survival_time_col,
        survival_event_col=survival_event_col,
        survival_censorship_col=survival_censorship_col,
        survival_y_disc_col=survival_y_disc_col,
    )

    method_l = str(imputation_method).strip().lower()
    defer_imputer_fit = prefit_imputer is None and method_l != "zero" and (train_missing or val_missing)
    dataset_init_method = "zero" if defer_imputer_fit else method_l

    train_ds = MultimodalDatasetWithMissing(
        base_dataset=train_base,
        simulator=missing_simulator,
        apply_missing=train_missing,
        imputation_method=dataset_init_method,
        missing_pattern_seed=missing_pattern_seed,
        prefit_imputer=prefit_imputer,
        imputer_kwargs=imputer_kwargs,
        imputer_seed=loader_seed,
    )
    val_ds = MultimodalDatasetWithMissing(
        base_dataset=val_base,
        simulator=missing_simulator,
        apply_missing=val_missing,
        imputation_method=dataset_init_method,
        missing_pattern_seed=missing_pattern_seed,
        prefit_imputer=prefit_imputer,
        imputer_kwargs=imputer_kwargs,
        imputer_seed=loader_seed,
    )

    shared_imputer = prefit_imputer
    if defer_imputer_fit:
        shared_imputer = build_imputer(
            imputation_method=method_l,
            reference_dataset=train_ds,
            knn_k=5,
            vae_kwargs=imputer_kwargs,
            imputer_seed=loader_seed,
        )
        train_ds.set_imputer(shared_imputer, method_l)
        val_ds.set_imputer(shared_imputer, method_l)

    model_name_l = normalize_model_name(model_name)
    train_batch_size = int(batch_size)
    eval_batch_size = 1 if model_name_l == "healnet" else train_batch_size
    n_train = len(train_ds)
    if model_name_l in {"mlp", "di_mmlp", "pam", "dipam"} and n_train < 2:
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
        drop_last_train = model_name_l in {"mlp", "di_mmlp", "pam", "dipam"} and (n_train % train_batch_size) == 1
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

    return train_loader, val_loader, shared_imputer
