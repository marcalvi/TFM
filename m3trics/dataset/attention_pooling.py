from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _AttentionMILPooler(nn.Module):
    """Attention pooling over a bag of lesion-level embeddings."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(int(input_dim), 1)

    def attention_weights(self, bag_norm):
        logits = self.attn(bag_norm).squeeze(-1)
        return torch.softmax(logits, dim=0)

    def classify_pooled(self, pooled_norm):
        return self.classifier(pooled_norm).squeeze(-1)


class AttentionPooler:
    """Fit a supervised attention pooling module on inner-train bags only."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        dropout=0.1,
        epochs=25,
        lr=1e-3,
        weight_decay=1e-4,
        seed=0,
        device="cpu",
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.device = torch.device(device)

        self.model = _AttentionMILPooler(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        self.feature_mean = None
        self.feature_std = None
        self.feature_cols = None
        self.id_col = None

    def _prepare_bags(self, df, labels_df, id_col, label_col, feature_cols):
        df = self._ensure_unambiguous_id_column(df, id_col)
        labels_indexed = labels_df[[id_col, label_col]].drop_duplicates(id_col).set_index(id_col)
        grouped = OrderedDict()
        for patient_id, g in df.groupby(id_col, sort=False):
            if patient_id not in labels_indexed.index:
                continue
            bag_np = g[feature_cols].to_numpy(dtype=np.float32, copy=True)
            grouped[patient_id] = {
                "bag": torch.from_numpy(bag_np).to(self.device),
                "label": torch.tensor(
                    float(labels_indexed.loc[patient_id, label_col]),
                    dtype=torch.float32,
                    device=self.device,
                ),
            }

        if not grouped:
            raise ValueError("AttentionPooler received no train bags after patient alignment.")
        return grouped

    def _normalize_bag(self, bag):
        return (bag - self.feature_mean) / self.feature_std

    @staticmethod
    def _ensure_unambiguous_id_column(df, id_col):
        work_df = df.copy()
        index_names = [
            name for name in getattr(work_df.index, "names", [work_df.index.name])
            if name is not None
        ]

        if id_col in work_df.columns and id_col in index_names:
            return work_df.reset_index(drop=True)

        if id_col not in work_df.columns and id_col in index_names:
            return work_df.reset_index()

        return work_df

    def fit(self, df_train, labels_df, id_col="patient", label_col="label"):
        df_train = self._ensure_unambiguous_id_column(df_train, id_col)
        if id_col not in df_train.columns:
            raise ValueError(f"Input dataframe must contain '{id_col}'.")
        if label_col not in labels_df.columns:
            raise ValueError(f"Labels dataframe must contain '{label_col}'.")

        self.id_col = str(id_col)
        self.feature_cols = [c for c in df_train.columns if c != id_col]
        if not self.feature_cols:
            raise ValueError("AttentionPooler found no feature columns to pool.")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        flat_train = df_train[self.feature_cols].to_numpy(dtype=np.float32, copy=True)
        mean_np = flat_train.mean(axis=0, keepdims=False)
        std_np = flat_train.std(axis=0, keepdims=False)
        std_np = np.where(std_np < 1e-6, 1.0, std_np)
        self.feature_mean = torch.from_numpy(mean_np).to(self.device)
        self.feature_std = torch.from_numpy(std_np).to(self.device)

        bags = self._prepare_bags(
            df=df_train,
            labels_df=labels_df,
            id_col=id_col,
            label_col=label_col,
            feature_cols=self.feature_cols,
        )
        labels = torch.stack([item["label"] for item in bags.values()])

        positives = float(labels.sum().item())
        negatives = float(labels.numel() - positives)
        if positives > 0.0 and negatives > 0.0:
            pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=self.device)
        else:
            pos_weight = None

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.model.train()
        for _ in range(self.epochs):
            logits = []
            targets = []
            for item in bags.values():
                bag_raw = item["bag"]
                bag_norm = self._normalize_bag(bag_raw)
                weights = self.model.attention_weights(bag_norm)
                pooled_norm = torch.sum(weights.unsqueeze(-1) * bag_norm, dim=0)
                logits.append(self.model.classify_pooled(pooled_norm))
                targets.append(item["label"])

            logits = torch.stack(logits)
            targets = torch.stack(targets)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        return self

    def transform(self, df):
        df = self._ensure_unambiguous_id_column(df, self.id_col)
        if self.feature_cols is None or self.id_col is None:
            raise RuntimeError("AttentionPooler.transform called before fit().")
        if self.id_col not in df.columns:
            raise ValueError(f"Input dataframe must contain '{self.id_col}'.")

        rows = []
        with torch.no_grad():
            for patient_id, g in df.groupby(self.id_col, sort=False):
                bag_raw = torch.from_numpy(
                    g[self.feature_cols].to_numpy(dtype=np.float32, copy=True)
                ).to(self.device)
                bag_norm = self._normalize_bag(bag_raw)
                weights = self.model.attention_weights(bag_norm)
                pooled_raw = torch.sum(weights.unsqueeze(-1) * bag_raw, dim=0)

                row = {self.id_col: patient_id}
                row.update(
                    {
                        feat_name: float(value)
                        for feat_name, value in zip(self.feature_cols, pooled_raw.cpu().numpy().tolist())
                    }
                )
                rows.append(row)

        out = pd.DataFrame(rows, columns=[self.id_col] + list(self.feature_cols))
        if out.empty:
            raise ValueError("AttentionPooler.transform produced an empty dataframe.")
        return out
