import pandas as pd
import torch
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    def __init__(self, modality_data_list, labels):
        """
        modality_data_list: List of DataFrames (one per selected modality)
        labels: Labels DataFrame
        """
        self.modalities = [torch.tensor(modality.values, dtype=torch.float32) for modality in modality_data_list]
        self.labels = torch.tensor(labels.values, dtype=torch.float32).squeeze()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        modality_tensors = [modality[idx] for modality in self.modalities]
        label = self.labels[idx]
        return modality_tensors, label

class DyAMDataset(torch.utils.data.Dataset):
    def __init__(self, dfs, label_df, label_col, id_col="patient"):
        self.dfs = dfs
        self.id_col = id_col
        self.label_col = label_col
        self.label_df = label_df

        # Get unique ids across all modalities
        all_ids = set()
        for df in dfs.values():
            all_ids |= set(df[id_col])

        self.patients = sorted(list(all_ids))
        self.indexed = {name: df.set_index(id_col) for name, df in dfs.items()}

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p_id = self.patients[idx]
        Xs = []
        for name, df in self.indexed.items():
            if p_id in df.index:
                x = torch.tensor(df.loc[p_id].values, dtype=torch.float32)
            else:
                x = None   
            Xs.append(x)

        y = torch.tensor(self.label_df.loc[pid, self.label_col], dtype=torch.float32)

        # Return the patient ID as the third element
        return Xs, y, p_id
        

def dyam_collate(batch):
    # Now unpacking three items: Xs (list of tensors), y (tensor), pid (string)
    Xs_batch, ys, pids = zip(*batch)

    n_modalities = len(Xs_batch[0])
    B = len(batch)

    Xs_out = []
    for m in range(n_modalities):
        modality_samples = [Xs_batch[i][m] for i in range(B)]

        # Find the first valid sample to get dimensions
        first_valid = next((x for x in modality_samples if x is not None), None)

        if first_valid is None:
            Xs_out.append(None)
        else:
            # Get the feature dimension
            dims = first_valid.shape[-1]
            tensor = torch.zeros(B, dims)
            for i, x in enumerate(modality_samples):
                if x is not None:
                    # Keep your existing logic for multi-dimensional inputs
                    if x.ndim > 1:
                        x = x.mean(dim=0) if x.shape[0] > 0 else torch.zeros(dims)
                    tensor[i] = x.view(-1)
            Xs_out.append(tensor)

    y = torch.stack(ys)
    
    # Return the patient IDs as a list
    return Xs_out, y, list(pids)


def healnet_collate(batch):
    Xs_batch, ys, pids = zip(*batch)
    n_modalities = len(Xs_batch[0])
    B = len(batch)

    Xs_out = []
    masks_out = [] # New: Explicitly track missingness for HEALNet

    for m in range(n_modalities):
        modality_samples = [Xs_batch[i][m] for i in range(B)]
        first_valid = next((x for x in modality_samples if x is not None), None)

        if first_valid is None:
            # If a whole modality is missing in a batch (rare but possible)
            Xs_out.append(None)
            masks_out.append(torch.zeros(B, dtype=torch.bool))
        else:
            dims = first_valid.shape[-1]
            tensor = torch.zeros(B, dims)
            mask = torch.zeros(B, dtype=torch.bool) # 0 = missing, 1 = present
            
            for i, x in enumerate(modality_samples):
                if x is not None:
                    if x.ndim > 1:
                        x = x.mean(dim=0) if x.shape[0] > 0 else torch.zeros(dims)
                    tensor[i] = x.view(-1)
                    mask[i] = True 
            
            Xs_out.append(tensor)
            masks_out.append(mask)

    return Xs_out, masks_out, torch.stack(ys), list(pids)