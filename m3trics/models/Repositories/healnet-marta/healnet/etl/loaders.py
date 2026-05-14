"""
    Adapted from https://github.com/konst-int-i/healnet by Marta Buetas-Arcas for multimodal survival datasets.
"""

import logging
import einops
from torch.utils.data import Dataset
from torchvision import transforms
from healnet.utils import Config
from openslide import OpenSlide
import os
from multiprocessing import Lock
from multiprocessing import Manager
import h5py
import torch
import pprint
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
from box import Box
import logging

logger = logging.getLogger(__name__)

class SyntheticMultiModalSurvival(Dataset):
    """
    Small synthetic multimodal survival dataset compatible with Pipeline / collate_fn.

    Returns tuples: ([mod1, mod2, mod3], censorship, event_time, y_disc)
    - each modality tensor shape: (1, D_mod)  (single-channel embedding)
    - censorship convention: 1 == censored, 0 == event observed (matches Pipeline expectations)
    - y_disc: discretised bin index 0..n_bins-1
    """
    def __init__(
        self,
        n_samples: int = 2000,
        dims: Tuple[int, int, int] = (30, 700, 512),
        n_bins: int = 6,
        seed: int = 0,
        censoring_prob: float = 0.3,
    ):
        super().__init__()
        rnd = np.random.RandomState(seed)
        self.n_samples = int(n_samples)
        self.dims = tuple(map(int, dims))
        self.n_bins = int(n_bins)

        # generate modality features
        X1 = rnd.normal(loc=0.0, scale=1.0, size=(self.n_samples, self.dims[0])).astype(np.float32)
        X2 = rnd.normal(loc=0.0, scale=1.0, size=(self.n_samples, self.dims[1])).astype(np.float32)
        X3 = rnd.normal(loc=0.0, scale=1.0, size=(self.n_samples, self.dims[2])).astype(np.float32)

        # build a known linear risk so model has signal to learn
        w1 = rnd.normal(scale=1.0, size=(self.dims[0],)).astype(np.float32)
        w2 = rnd.normal(scale=1.0, size=(self.dims[1],)).astype(np.float32)
        w3 = rnd.normal(scale=1.0, size=(self.dims[2],)).astype(np.float32)
        linear_risk = (X1 @ w1) * 0.6 + (X2 @ w2) * 0.5 + (X3 @ w3) * 0.4
        rates = np.exp(linear_risk - linear_risk.mean())

        # sample event times from exponential(rate)
        event_times = rnd.exponential(scale=1.0 / (rates + 1e-8)).astype(np.float32)

        # independent censoring indicator (1 = censored)
        censored = (rnd.rand(self.n_samples) < censoring_prob).astype(np.int64)


        # discretize event_times using quantiles into n_bins (0..n_bins-1)
        quantiles = np.quantile(event_times, q=np.linspace(0, 1, self.n_bins + 1))
        if len(np.unique(quantiles)) < len(quantiles):
            quantiles = quantiles + np.linspace(0, 1e-6, len(quantiles))
        y_disc = np.digitize(event_times, bins=quantiles[1:-1], right=False).astype(np.int64)
        y_disc = np.clip(y_disc, 0, self.n_bins - 1)

        # store tensors and add token axis (1) to match MMDataset modality shape (1, D)
        self.modalities = [
            torch.from_numpy(X1).unsqueeze(1),  # (N, 1, D1)
            torch.from_numpy(X2).unsqueeze(1),  # (N, 1, D2)
            torch.from_numpy(X3).unsqueeze(1),  # (N, 1, D3)
        ]
        self.event_times = torch.from_numpy(event_times)
        self.censored = torch.from_numpy(censored)
        self.y_disc = torch.from_numpy(y_disc)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        mods = [m[idx] for m in self.modalities]  # each is shape (1, D)
        return mods, self.censored[idx], self.event_times[idx], self.y_disc[idx]


class MMDataset(Dataset):
    """
    Generic torch dataset object for supervised multi-modal data.
    This class expects CSV file paths to be provided via the `config` Box.

    Notes:
    - __getitem__ returns: ([modalities], censorship, event_time, y_disc)
    """
    def __init__(self,
                 config: Box,
                 filter_overlap: bool = True,
                 survival_analysis: bool = True,
                 sources: List = ["clinical", "pathology", "radiology"],
                 n_bins: int = 4,
                 id_col: str = "id",
                 time_col: str = "mOS_months",
                 event_col: str = "mOS_event",
                 log_dir=None
                 ):
        """
        Args:
            filter_overlap: filter omic data and/or slides that do not have a corresponding sample in the other modality
            n_bins: number of discretised bins for survival analysis
        """
        self.config = config
        self.sources = sources
        self.filter_overlap = filter_overlap
        self.survival_analysis = survival_analysis
        self.n_bins = n_bins
        self.id_col = id_col # "id"
        self.time_col = time_col # "OS_months"
        self.event_col = event_col # "OS_event"
        self.log_dir = log_dir

        # initialize standardization params and clip value (config override supported)
        self.standardization_params = {}
        self.clip_value = float(self.config.get("data.clip_value", 10.0))

        # Loader paths (must exist in config)
        self.clinical_path = Path(self.config.clinical)
        self.pathology_path = Path(self.config.pathology)
        self.radiology_path = Path(self.config.radiology)
        self.target_path = Path(self.config.target_path)

        # pre-load and transform data
        self.target = self.load_target()
        self.clinical_df = self.load_clinical()  
        self.pathology_df = self.load_pathology()
        self.radiology_df = self.load_radiology()

        # Ensure id column exists and is string in each df
        for name, df in (("target", self.target), ("clinical", self.clinical_df), ("pathology", self.pathology_df), ("radiology", self.radiology_df)):
            if df is None or df.empty:
                continue
            if self.id_col not in df.columns:
                raise KeyError(f"Expected id column '{self.id_col}' in {name} dataframe; columns: {df.columns.tolist()}")
            df[self.id_col] = df[self.id_col].astype(str)
        
        # Create y_disc in target if not present (use target's time_col)
        endpoint = event_col.split('_')[0]  # e.g., 'mOS' from 'mOS_event'
        y_disc_col = f'y_disc_{endpoint}'
        if y_disc_col not in self.target.columns:
            self._create_y_disc(self.target, time_col=self.time_col, n_bins=self.n_bins)
        else:
            self.n_bins = int(self.target[y_disc_col].nunique())
            
            logger.info("Using existing y_disc column '%s' with n_bins=%d", y_disc_col, self.n_bins)
            self.target["y_disc"] = self.target[y_disc_col].astype(int)
      

        # --------- cleaning & imputation of clinical df, drop columns with >70% NaNs ----------

        # cleaning & imputation of clinical df
        self.clinical_df, dropped_clinical = self._clean_impute_df(self.clinical_df, modality_name="clinical")
        if dropped_clinical:
            logger.info("Clinical columns dropped (>70%% NaN): %s", dropped_clinical)
        still_nan = [c for c in self.clinical_df.columns if c != self.id_col and self.clinical_df[c].isna().any()]
        if still_nan:
            logger.warning("Clinical columns still with NaNs after imputation (unexpected): %s", still_nan)

        # --------- end cleaning / imputation ----------

        # build id sets
        clinical_ids = set(self.clinical_df[self.id_col].astype(str).values) if len(self.clinical_df) > 0 else set()
        pathology_ids = set(self.pathology_df[self.id_col].astype(str).values) if len(self.pathology_df) > 0 else set()
        radiology_ids = set(self.radiology_df[self.id_col].astype(str).values) if len(self.radiology_df) > 0 else set()

        if self.filter_overlap:
            # require presence in ALL modalities
            sample_ids = list(sorted(clinical_ids & pathology_ids & radiology_ids))
        else:
            # union of all sample ids
            sample_ids = list(sorted(clinical_ids | pathology_ids | radiology_ids))

        self.sample_ids = [str(s) for s in sample_ids]
        if len(self.sample_ids) == 0:
            raise ValueError("No samples found after applying overlap filter. Check ids and filter_overlap setting.")
        
        # Store feature column lists for later standardization
        self.clinical_feature_cols = [c for c in self.clinical_df.columns if c != self.id_col]
        self.pathology_feature_cols = [c for c in self.pathology_df.columns if c != self.id_col]
        self.radiology_feature_cols = [c for c in self.radiology_df.columns if c != self.id_col]

        # Build modality maps (id -> tensor) for each modality
        self.modality_maps = {'clinical': {}, 'pathology': {}, 'radiology': {}}
        self._build_modality_maps()

        # Targets per sample 
        self.censorship = []
        self.survival_months = []
        self.y_disc = []

        target_ids = set(self.target[self.id_col].astype(str).values)

        for sid in self.sample_ids:
            if sid in target_ids:
                row = self.target[self.target[self.id_col].astype(str) == sid].iloc[0]
                 # Pipeline convention: censorship column should be 1 if censored, 0 if event occurred
                self.censorship.append(1-int(row[self.event_col]))
                self.survival_months.append(float(row[self.time_col]))
                self.y_disc.append(int(row['y_disc']))
            else:
                # fallback defaults
                self.censorship.append(0)
                self.survival_months.append(0.0)
                self.y_disc.append(0)

        # convert to tensors
        self.censorship = torch.tensor(self.censorship, dtype=torch.long)
        self.survival_months = torch.tensor(self.survival_months, dtype=torch.float32)
        self.y_disc = torch.tensor(self.y_disc, dtype=torch.long)

        # store metadata for model creation
        self.modalities_list = ['clinical', 'pathology', 'radiology']
        self.feature_dims = {
            'clinical': len(self.clinical_feature_cols),
            'pathology': len(self.pathology_feature_cols),
            'radiology': len(self.radiology_feature_cols)
        }

        # multiprocessing cache placeholder
        manager = Manager()
        self.patch_cache = manager.dict()

        # Store raw copies of dataframes to allow resetting in cross-validation (prevent cumulative standardization)
        self.raw_clinical_df = self.clinical_df.copy() if hasattr(self, "clinical_df") and self.clinical_df is not None else None
        self.raw_pathology_df = self.pathology_df.copy() if hasattr(self, "pathology_df") and self.pathology_df is not None else None
        self.raw_radiology_df = self.radiology_df.copy() if hasattr(self, "radiology_df") and self.radiology_df is not None else None

        logger.info("MMDataset initialised. samples: %d. filter_overlap=%s", len(self.sample_ids), self.filter_overlap)

    # -----------------------
    # Helper methods
    # -----------------------
    def _clean_impute_df(self, df: pd.DataFrame, modality_name: str):
        """
        - Drops columns where > 70% values are NaN
        - For remaining columns:
            - If column has NaNs and appears binary (values in {0,1}), fill NaN with 0.5
            - Else if numeric, fill NaN with column mean
            - Else attempt numeric coercion then mean; fallback fill with 0
        - Prints which columns had NaNs (for modality == 'clinical'/0 print header names)
        Returns cleaned df and list of dropped columns.
        """
        if df is None or df.empty:
            return df, []

        # Do not consider id column for feature operations
        cols = [c for c in df.columns if c != self.id_col]
        if len(cols) == 0:
            return df, []

        n_rows = len(df)
        nan_counts = df[cols].isna().sum()
        nan_frac = nan_counts / float(max(1, n_rows))

        # drop columns with >70% NaNs
        drop_cols = nan_frac[nan_frac > 0.7].index.tolist()
        if len(drop_cols) > 0:
            print(f"[MMDataset] Dropping columns for modality '{modality_name}' due to >70% NaNs: {drop_cols}")
            df = df.drop(columns=drop_cols)
            # update cols
            cols = [c for c in cols if c not in drop_cols]

        imputed_cols = []
        # For remaining columns, fill NaNs according to type
        for c in cols:
            n_nan = int(df[c].isna().sum())
            if n_nan == 0:
                continue

            imputed_cols.append(c)
            # Try to detect binary (0/1 or True/False)
            non_null = df[c].dropna()
            is_binary = False
            try:
                # coerce non-null values to numeric then check unique values subset
                non_null_num = pd.to_numeric(non_null, errors='coerce').dropna()
                unique_vals = set(non_null_num.unique())
                if unique_vals.issubset({0, 1}):
                    is_binary = True
            except Exception:
                is_binary = False

            if is_binary:
                # fill NaN with 0.5
                df[c] = df[c].fillna(0.5)
            else:
                # if column is numeric dtype, fill with mean
                if pd.api.types.is_numeric_dtype(df[c]):
                    mean_val = float(df[c].mean())
                    df[c] = df[c].fillna(mean_val)
                else:
                    # try coercion to numeric then fill with mean
                    coerced = pd.to_numeric(df[c], errors='coerce')
                    if coerced.notna().any():
                        mean_val = float(coerced.mean())
                        df[c] = coerced.fillna(mean_val)
                    else:
                        # fallback: replace NaN with 0
                        df[c] = df[c].fillna(0)

        if len(imputed_cols) > 0:
            print(f"[MMDataset] Modality '{modality_name}' - imputed columns (NaN replaced): {imputed_cols}")

        return df, drop_cols

    def _build_modality_maps(self):
        # CLINICAL
        clinical_cols = self.clinical_feature_cols
        if hasattr(self, "clinical_df") and len(self.clinical_df) > 0:
            for _, row in self.clinical_df.iterrows():
                sid = str(row[self.id_col])
                vals = row[clinical_cols].values.astype(np.float32) if clinical_cols else np.array([], dtype=np.float32)
                t = torch.from_numpy(vals).unsqueeze(0)  # (1, D_clinical)
                self.modality_maps['clinical'][sid] = t

        # PATHOLOGY
        pathology_cols = self.pathology_feature_cols
        if hasattr(self, "pathology_df") and len(self.pathology_df) > 0:
            for _, row in self.pathology_df.iterrows():
                sid = str(row[self.id_col])
                vals = row[pathology_cols].astype(float).values.astype(np.float32) if pathology_cols else np.array([], dtype=np.float32)
                t = torch.from_numpy(vals).unsqueeze(0)
                self.modality_maps['pathology'][sid] = t

        # RADIOLOGY
        radiology_cols = self.radiology_feature_cols
        if hasattr(self, "radiology_df") and len(self.radiology_df) > 0:
            for _, row in self.radiology_df.iterrows():
                sid = str(row[self.id_col])
                vals = row[radiology_cols].astype(float).values.astype(np.float32) if radiology_cols else np.array([], dtype=np.float32)
                t = torch.from_numpy(vals).unsqueeze(0)  # (1, D_radiology)
                self.modality_maps['radiology'][sid] = t

    def _standardize_and_clip_df(self, df: pd.DataFrame, id_col: str, clip_value: float = 10.0):
        """
        Standardize numeric feature columns (zero mean, unit std) and clip extremes.
        - Leaves id_col unchanged.
        - Avoids dividing by zero by replacing zero std with 1.0.
        """
        if df is None or df.empty:
            return df

        if clip_value is None:
            clip_value = self.clip_value

        # select feature columns (exclude id, time/event/y_disc if present)
        exclude = {id_col, self.time_col, self.event_col, 'y_disc'}
        feature_cols = [c for c in df.columns if c not in exclude]
        if not feature_cols:
            return df

        # numeric columns only
        numeric_cols = df[feature_cols].select_dtypes(include=[float, int]).columns.tolist()
        if len(numeric_cols) == 0:
            return df

        # compute mean/std, protect zero std
        means = df[numeric_cols].mean()
        stds = df[numeric_cols].std().replace(0, 1.0)

        # standardize and clip to [-clip_value, clip_value]
        df[numeric_cols] = (df[numeric_cols] - means) / stds
        df[numeric_cols] = df[numeric_cols].clip(-clip_value, clip_value)

        return df
    
    # Compute standardization from train indices and apply to all splits
    def compute_standardization_from_indices(self, train_indices: List[int], clip_value: float = None):
        """
        Compute per-modality means/stds using only the given train_indices,
        apply the standardization to the whole dataframes and rebuild modality maps.
        Stores the params in self.standardization_params.

        Args:
            train_indices: list-like of integer indices referencing positions in self.sample_ids
            clip_value: optional clip value (defaults to self.clip_value)
        """
        if clip_value is None:
            clip_value = self.clip_value

        if not isinstance(train_indices, (list, tuple, np.ndarray)):
            raise ValueError("train_indices must be a list/tuple/ndarray of dataset indices")

        # Map indices -> sample_ids used in dataframes
        train_sample_ids = [self.sample_ids[i] for i in train_indices]

        # Reset dataframes to raw state before applying new standardization (fix for fold data leakage)
        if hasattr(self, "raw_clinical_df") and self.raw_clinical_df is not None:
            self.clinical_df = self.raw_clinical_df.copy()
        if hasattr(self, "raw_pathology_df") and self.raw_pathology_df is not None:
            self.pathology_df = self.raw_pathology_df.copy()
        if hasattr(self, "raw_radiology_df") and self.raw_radiology_df is not None:
            self.radiology_df = self.raw_radiology_df.copy()

        # Helper to compute means/stds for a dataframe and feature list
        def compute_and_apply(df, feature_cols):
            if df is None or df.empty or len(feature_cols) == 0:
                return None
            mask = df[self.id_col].astype(str).isin(train_sample_ids)
            train_df = df[mask]
            if train_df.empty:
                logger.warning("No rows found for provided train indices for df.")
                means = pd.Series(0, index=feature_cols, dtype=float)
                stds = pd.Series(1.0, index=feature_cols, dtype=float)
            else:
                means = train_df[feature_cols].mean()
                stds = train_df[feature_cols].std().replace(0, 1.0)
            df[feature_cols] = (df[feature_cols] - means) / stds
            df[feature_cols] = df[feature_cols].clip(-clip_value, clip_value)
            return {"means": means.to_numpy(dtype=float), "stds": stds.to_numpy(dtype=float), "cols": feature_cols}

        # Clinical
        self.standardization_params['clinical'] = compute_and_apply(self.clinical_df, self.clinical_feature_cols)
        # Pathology
        self.standardization_params['pathology'] = compute_and_apply(self.pathology_df, self.pathology_feature_cols)
        # Radiology
        self.standardization_params['radiology'] = compute_and_apply(self.radiology_df, self.radiology_feature_cols)

        # Rebuild modality maps since underlying dataframes changed values
        self.modality_maps = {'clinical': {}, 'pathology': {}, 'radiology': {}}
        self._build_modality_maps()

        logger.info("Computed and applied standardization from train indices. Params keys: %s", [k for k, v in self.standardization_params.items() if v is not None])
        return self.standardization_params

    
    # -----------------------
    # Dataset interface
    # -----------------------
    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        # For each modality either return tensor (1, D) or None (if not present)
        out_modalities = []
        for mod in self.modalities_list:
            t = self.modality_maps.get(mod, {}).get(sid, None)
            out_modalities.append(t)
        censorship = self.censorship[idx]
        event_time = self.survival_months[idx]
        y_disc = self.y_disc[idx]
        return out_modalities, censorship, event_time, y_disc

    # helper to discretize survival times into y_disc
    def _create_y_disc(self, df, time_col: str, n_bins: int = 4, eps: float = 1e-6):
        """
        Discretize continuous survival time into n_bins using quantiles (pd.qcut).
        Stores bin edges on self.bin_edges and optionally writes:
        - target_hist.png : histogram of the continuous target (already present)
        - y_disc_hist.png  : bar plot (counts) of the discrete y_disc labels
        - y_disc_distribution.csv : counts per discrete bin
        - y_disc_bin_edges.csv : numeric bin edges (if available)

        Returns:
            pd.DataFrame with 'y_disc' column added
        """
        
        label_col = time_col
        try:
            disc_labels, bins = pd.qcut(df[label_col], q=n_bins, retbins=True, labels=False, duplicates="drop")
            df["y_disc"] = disc_labels.astype(int).values
            self.bin_edges = np.asarray(bins)
        except Exception as e:
            logger.warning("pd.qcut failed with q=%s; falling back to pd.cut: %s", n_bins, e)
            df["y_disc"], bins = pd.cut(df[label_col], bins=n_bins, retbins=True, labels=False, include_lowest=True)
            df["y_disc"] = df["y_disc"].astype(int)
            self.bin_edges = np.asarray(bins)

        # Build discrete distribution info
        try:
            counts = df["y_disc"].value_counts().sort_index()
            logger.info("y_disc distribution (bin_index: count): %s", counts.to_dict())
        except Exception as e:
            logger.warning("Failed to compute y_disc distribution: %s", e)
            counts = pd.Series(dtype=int)
        
        # If a log_dir is provided, save artifacts for inspection (csv + plots + bin edges)
        if self.log_dir is not None:
            out_dir = Path(self.log_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save counts CSV and bin edges
            try:
                dist_df = pd.DataFrame({"bin": counts.index.astype(int), "count": counts.values})
                dist_df.to_csv(out_dir.joinpath("y_disc_distribution.csv"), index=False)
                if self.bin_edges is not None:
                    np.savetxt(out_dir.joinpath("y_disc_bin_edges.csv"), self.bin_edges, delimiter=",")
            except Exception as e:
                logger.warning("Failed to save y_disc artifacts: %s", e)


            # Plot and save histograms
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # continuous histogram
                try:
                    ax = axes[0]
                    df[label_col].hist(bins=50, ax=ax)
                    ax.set_title("Continuous target distribution")
                    ax.set_xlabel(label_col)
                    ax.set_ylabel("count")
                except Exception as e_cont:
                    logger.debug("Failed to draw continuous histogram: %s", e_cont)

                # discrete bar plot
                try:
                    ax = axes[1]
                    bin_indices = counts.index.astype(int).tolist()
                    bin_counts = counts.values.tolist()
                    ax.bar(bin_indices, bin_counts, align="center", color="C1")
                    ax.set_xticks(bin_indices)
                    ax.set_xlabel("y_disc (bin index)")
                    ax.set_ylabel("count")
                    ax.set_title(f"Discrete y_disc distribution (n_bins={n_bins})")
                except Exception as e_disc:
                    logger.debug("Failed to draw discrete histogram: %s", e_disc)

                fig.tight_layout()
                fig_path = out_dir.joinpath("y_disc_vs_target_hist.png")
                fig.savefig(fig_path)
                plt.close(fig)
            except Exception as e:
                logger.debug("Failed to create/save histograms: %s", e)

        return df
    
    # loader functions
    def load_clinical(self) -> pd.DataFrame:
        path = self.clinical_path
        df = pd.read_csv(path, header=0, low_memory=False)
        return df

    def load_pathology(self) -> pd.DataFrame:
        path = self.pathology_path
        df = pd.read_csv(path, header=0, low_memory=False)
        return df

    def load_radiology(self) -> pd.DataFrame:
        path = self.radiology_path
        df = pd.read_csv(path, header=0, low_memory=False)
        return df

    def load_target(self) -> pd.DataFrame:
        path = self.target_path
        df = pd.read_csv(path, header=0, low_memory=False)
        return df

    # pretty print info
    def get_info(self, full_detail: bool = False):
        logger.info("Dataset: %s", self.config.get("dataset", "custom"))
        logger.info("Samples: %d", len(self.sample_ids))
        logger.info("Feature dims: %s", self.feature_dims)
        logger.info("Filter overlap: %s", self.filter_overlap)
        try:
            unique, counts = np.unique(self.y_disc.numpy(), return_counts=True)
            logger.info("Survival bins distribution: %s", dict(zip(unique.tolist(), counts.tolist())))
        except Exception:
            pass

class TCGADataset(Dataset):
    """
    Main dataset class for TCGA data. Loads in omic data and WSI data and returns a tuple of tensors when
    __getitem__ is called along with survival information (censorship, event_time, discretised survival).
    """

    def __init__(self, dataset: str,
                 config: Box,
                 level: int=2,
                 filter_overlap: bool = True,
                 survival_analysis: bool = True,
                 num_classes: int = 2,
                 n_bins: int = 4,
                 sources: List = ["omic", "slides"],
                 log_dir = None,
                 ):
        """
        Dataset wrapper to load different TCGA data modalities (omic and WSI data).
        Args:
            dataset (str): TCGA dataset to load (e.g. "brca", "blca")
            config (Box): Config object
            filter_overlap: filter omic data and/or slides that do not have a corresponding sample in the other modality
            n_bins: number of discretised bins for survival analysis

        Examples:
            >>> from healnet.etl.loaders import TCGADataset
            >>> from healnet.utils import Config
            >>> config = Config("config/main.yml").read()
            >>> dataset = TCGADataset("blca", config)
            # get omic data
            >>> dataset.omic_df
            # get sample slide
            >>> slide, tensor = dataset.load_omic(blca.sample_slide_id, resolution="lowest")
            # get overall sample
            >>> (slide, tensor), censorship, event_time, y_disc = next(iter(dataset))
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset
        self.log_dir = log_dir
        self.sources = sources
        self.filter_overlap = filter_overlap
        self.survival_analysis = survival_analysis
        self.sample_missing = False
        self.num_classes = num_classes
        self.n_bins = n_bins
        self.subset = self.config["survival.subset"]
        self.raw_path = Path(config.tcga_path).joinpath(f"wsi/{dataset}")
        prep_path = Path(config.tcga_path).joinpath(f"wsi/{dataset}_preprocessed_level{level}")
        self.prep_path = prep_path
        # create patch feature directory for first-time run
        os.makedirs(self.prep_path.joinpath("patch_features"), exist_ok=True)
        self.slide_ids = [slide_id.rsplit(".", 1)[0] for slide_id in os.listdir(prep_path.joinpath("patches"))]



        # for early fusion baseline, we need to concatenate omic and slide features into a single tensor
        self.concat = True if self.config.model in ["fcnn", "healnet_early"] and len(self.sources) > 1 else False

        valid_sources = ["omic", "slides"]
        assert all([source in valid_sources for source in sources]), f"Invalid source specified. Valid sources are {valid_sources}"
        self.wsi_paths: dict = self._get_slide_dict() # {slide_id: path}
        self.sample_slide_id = self.slide_ids[0] + ".svs"
        self.sample_slide = OpenSlide(self.wsi_paths[self.sample_slide_id])
        # pre-load and transform omic data
        self.omic_df = self.load_omic()
        self.features = self.omic_df.drop(["site", "oncotree_code", "case_id", "slide_id", "train", "censorship", "survival_months", "y_disc"], axis=1)
        self.omic_tensor = torch.Tensor(self.features.values)
        if self.config.model in ["healnet", "healnet_early"]:
            # Healnet expects inputs of the shape (batch_size, input_dim, channels)
            if self.config.omic_attention:
                self.omic_tensor = einops.repeat(self.omic_tensor, "n feat -> n channels feat", channels=1)
            else:
                self.omic_tensor = einops.repeat(self.omic_tensor, "n feat -> n feat channels", channels=1)


        self.level = level
        self.slide_idx: dict = self._get_slide_idx() # {idx (molecular_df): slide_id}
        self.wsi_width, self.wsi_height = self.get_resize_dims(level=self.level, override=config["data.resize"])
        self.censorship = self.omic_df["censorship"].values
        self.survival_months = self.omic_df["survival_months"].values
        self.y_disc = self.omic_df["y_disc"].values

        manager = Manager()
        self.patch_cache = manager.dict()
        # self.patch_cache = SharedLRUCache(capacity=256) # capacity should be multiple of num_workers
        print(f"Dataloader initialised for {dataset} dataset")
        self.get_info(full_detail=False)

    def __getitem__(self, index):
        y_disc = self.y_disc[index]
        censorship = self.censorship[index]
        event_time = self.survival_months[index]


        if len(self.sources) == 1 and self.sources[0] == "omic":
            omic_tensor = self.omic_tensor[index]
            return [omic_tensor], censorship, event_time, y_disc

        elif len(self.sources) == 1 and self.sources[0] == "slides":
            slide_id = self.omic_df.iloc[index]["slide_id"].rsplit(".", 1)[0]

           
            if index not in self.patch_cache:
                slide_tensor = self.load_patch_features(slide_id)
                self.patch_cache[index] = slide_tensor

            else:
                slide_tensor = self.patch_cache[index]
            if self.config.model == "fcnn": # for fcnn baseline
                slide_tensor = torch.flatten(slide_tensor)

            return [slide_tensor], censorship, event_time, y_disc

        else: # both
            omic_tensor = self.omic_tensor[index]
            slide_id = self.omic_df.iloc[index]["slide_id"].rsplit(".", 1)[0]

            if index not in self.patch_cache:
                slide_tensor = self.load_patch_features(slide_id)
                self.patch_cache[index] = slide_tensor
            else:
                slide_tensor = self.patch_cache[index]

            if self.concat: # for early fusion baseline
                slide_flat = torch.flatten(slide_tensor)
                omic_flat = torch.flatten(omic_tensor)
                concat_tensor = torch.cat([omic_flat, slide_flat], dim=0)
                if self.config.model == "healnet_early":
                    concat_tensor = concat_tensor.unsqueeze(0)
                return [concat_tensor], censorship, event_time, y_disc
            else: # keep separate for HEALNet
                return [omic_tensor, slide_tensor], censorship, event_time, y_disc

    def get_resize_dims(self, level: int, patch_height: int = 128, patch_width: int = 128, override=False):
        # TODO - use TIA to handle resizing
        if override is False:
            width = self.sample_slide.level_dimensions[level][0]
            height = self.sample_slide.level_dimensions[level][1]
            # take nearest multiple of 128 of height and width (for patches)
            width = round(width/patch_width)*patch_width
            height = round(height/patch_height)*patch_height
        else:
            width = self.config["data.resize_width"]
            height = self.config["data.resize_height"]
        return width, height

    def _get_slide_idx(self):
        # filter slide index to only include samples with WSIs availables
        filter_keys = [slide_id + ".svs" for slide_id in self.slide_ids]
        tmp_df = self.omic_df[self.omic_df.slide_id.isin(filter_keys)]
        return dict(zip(tmp_df.index, tmp_df["slide_id"]))

    def __len__(self):
        if self.sources == ["omic"]:
            # use all omic samples when running single modality
            return self.omic_df.shape[0]
        else:
            # only use overlap otherwise
            return len(self.slide_ids)
    def _get_slide_dict(self):
        """
        Given the download structure of the gdc-client, each slide is stored in a folder
        with a non-meaningful name. This function returns a dictionary of slide_id to
        the path of the slide.
        Returns:
            svs_dict (dict): Dictionary of slide_id to path of slide
        """
        slide_path = Path(self.config.tcga_path).joinpath(f"wsi/{self.dataset}")
        svs_files = list(slide_path.glob("**/*.svs"))
        svs_dict = {path.name: path for path in svs_files}
        return svs_dict

    # def _load_patch_coords(self):
    #     """
    #     Loads all patch coordinates for the dataset and level specified in the config and writes it to a dictionary
    #     with key: slide_id and value: patch coordinates (where each coordinate is a x,y tupe)
    #     """
    #     coords = {}
    #     for slide_id in self.slide_ids:
    #         patch_path = self.prep_path.joinpath(f"patches/{slide_id}.h5")
    #         h5_file = h5py.File(patch_path, "r")
    #         patch_coords = h5_file["coords"][:]
    #         coords[slide_id] = patch_coords
    #     return coords

    def get_info(self, full_detail: bool = False):
        """
        Logging util to print some basic dataset information. Normally called at the start of a pipeline run
        Args:
            full_detail (bool): Print all slide properties

        Returns:
            None
        """
        slide_path = Path(self.config.tcga_path).joinpath(f"wsi/{self.dataset}/")
        print(f"Dataset: {self.dataset.upper()}")
        print(f"Molecular data shape: {self.omic_df.shape}")
        sample_overlap = (set(self.omic_df["slide_id"]) & set(self.wsi_paths.keys()))
        print(f"Molecular/Slide match: {len(sample_overlap)}/{len(self.omic_df)}")
        # print(f"Slide dimensions: {slide.dimensions}")
        print(f"Slide level count: {self.sample_slide.level_count}")
        print(f"Slide level dimensions: {self.sample_slide.level_dimensions}")
        print(f"Slide resize dimensions: w: {self.wsi_width}, h: {self.wsi_height}")
        print(f"Sources selected: {self.sources}")
        print(f"Censored share: {np.round(len(self.omic_df[self.omic_df['censorship'] == 1])/len(self.omic_df), 3)}")
        print(f"Survival_bin_sizes: {dict(self.omic_df['y_disc'].value_counts().sort_values())}")

        if full_detail:
            pprint(dict(self.sample_slide.properties))

    def show_samples(self, n=1):
        """
        Logging util to show some detailed sample stats and render the whole slide image (e.g., in a notebook)
        Args:
            n (int): Number of samples to show

        Returns:
            None
        """
        # sample_df = self.omic_df.sample(n=n)
        sample_df = self.omic_df[self.omic_df["slide_id"].isin(self.wsi_paths.keys())].sample(n=n)
        for idx, row in sample_df.iterrows():
            print(f"Case ID: {row['case_id']}")
            print(f"Patient age: {row['age']}")
            print(f"Gender: {'female' if row['is_female'] else 'male'}")
            print(f"Survival months: {row['survival_months']}")
            print(f"Survival years:  {np.round(row['survival_months']/12, 1)}")
            print(f"Censored (survived follow-up period): {'yes' if row['censorship'] else 'no'}")
            # print(f"Risk: {'high' if row['high_risk'] else 'low'}")
            # plot wsi
            slide, slide_tensor = self.load_wsi(row["slide_id"], level=self.level)
            print(f"Shape:", slide_tensor.shape)
            plt.figure(figsize=(10, 10))
            plt.imshow(slide_tensor)
            plt.show()




    def load_omic(self,
                  eps: float = 1e-6
                  ) -> pd.DataFrame:
        """
        Loads in omic data and returns a dataframe and filters depending on which whole slide images
        are available, such that only samples with both omic and WSI data are kept.
        Also calculates the discretised survival time for each sample.
        Args:
            eps (float): Epsilon value to add to min and max survival time to ensure all samples are included

        Returns:
            pd.DataFrame: Dataframe with omic data and discretised survival time (target)
        """
        data_path = Path(self.config.tcga_path).joinpath(f"omic/tcga_{self.dataset}_all_clean.csv.zip")
        df = pd.read_csv(data_path, compression="zip", header=0, index_col=0, low_memory=False)
        valid_subsets = ["all", "uncensored", "censored"]
        assert self.subset in valid_subsets, "Invalid cut specified. Must be one of 'all', 'uncensored', 'censored'"

        # handle missing values
        num_nans = df.isna().sum().sum()
        nan_counts = df.isna().sum()[df.isna().sum() > 0]
        df = df.fillna(df.mean(numeric_only=True))
        print(f"Filled {num_nans} missing values with mean")
        print(f"Missing values per feature: \n {nan_counts}")

        # filter samples for which there are no slides available
        if self.filter_overlap:
            slides_available = self.slide_ids
            omic_available = [id[:-4] for id in df["slide_id"]]
            overlap = set(slides_available) & set(omic_available)
            print(f"Slides available: {len(slides_available)}")
            print(f"Omic available: {len(omic_available)}")
            print(f"Overlap: {len(overlap)}")
            if len(slides_available) < len(omic_available):
                print(f"Filtering out {len(omic_available) - len(slides_available)} samples for which there are no omic data available")
                overlap_filter = [id + ".svs" for id in overlap]
                df = df[df["slide_id"].isin(overlap_filter)]
            elif len(slides_available) > len(omic_available):
                print(f"Filtering out {len(slides_available) - len(omic_available)} samples for which there are no slides available")
                self.slide_ids = overlap
            else:
                print("100% modality overlap, no samples filtered out")

        # assign target column (high vs. low risk in equal parts of survival)
        label_col = "survival_months"
        if self.subset == "all":
            df["y_disc"] = pd.qcut(df[label_col], q=self.n_bins, labels=False).values
        else:
            if self.subset == "censored":
                subset_df = df[df["censorship"] == 1]
            elif self.subset == "uncensored":
                subset_df = df[df["censorship"] == 0]
            # take q_bins from uncensored patients
            disc_labels, q_bins = pd.qcut(subset_df[label_col], q=self.n_bins, retbins=True, labels=False)
            q_bins[-1] = df[label_col].max() + eps
            q_bins[0] = df[label_col].min() - eps
            # use bin cuts to discretize all patients
            df["y_disc"] = pd.cut(df[label_col], bins=q_bins, retbins=False, labels=False, right=False, include_lowest=True).values

        df["y_disc"] = df["y_disc"].astype(int)

        if self.log_dir is not None:
            df.to_csv(self.log_dir.joinpath(f"{self.dataset}_omic_overlap.csv.zip"), compression="zip")

        return df

    def load_wsi(self, slide_id: str, level: int = None) -> Tuple:
        """
        Load in single slide and get region at specified resolution level
        Args:
            slide_id:
            level:
            resolution:

        Returns:
            Tuple (openslide object, tensor of region)
        """

        slide = OpenSlide(self.raw_path.joinpath(f"{slide_id}.svs"))

        # specify resolution level
        if level is None:
            level = slide.level_count # lowest resolution by default
        if level > slide.level_count - 1:
            level = slide.level_count - 1
        # load in region
        size = slide.level_dimensions[level]
        region = slide.read_region((0,0), level, size)
        # add transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]), # remove alpha channel
            transforms.Resize((self.wsi_height, self.wsi_width)),
            RearrangeTransform("c h w -> h w c") # rearrange for Healnet architecture
        ])
        region_tensor = transform(region)
        return slide, region_tensor

    def load_patch_features(self, slide_id: str) -> torch.Tensor:
        """
        Loads patch features for a single slide from torch.pt file
        Args:
            slide_id (str): Slide ID

        Returns:
            torch.Tensor: Patch features
        """
        load_path = self.prep_path.joinpath(f"patch_features/{slide_id}.pt")
        with open(load_path, "rb") as file:
            patch_features = torch.load(file, weights_only=True)
        patch_features = patch_features.permute(1, 0)
        return patch_features



class SharedLRUCache:
    """
    Shared LRU cache for multiprocessing
    """
    def __init__(self, capacity: int):
        """

        Args:
            capacity (int): Number of items to be stored in the cache
        """
        manager = Manager()
        self.capacity = capacity
        self.cache = manager.dict()
        self.order = manager.list()
        self.lock = Lock()
    def get(self, key: int):
        with self.lock:
            if key in self.cache:
                # Move key to end to show it was recently used.
                self.order.remove(key)
                self.order.append(key)
                return self.cache[key]
            else:
                return None

    def set(self, key: int, value):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
            else:
                if len(self.order) >= self.capacity:
                    removed_key = self.order.pop(0)  # Remove the first (least recently used) item.
                    del self.cache[removed_key]

            self.order.append(key)
            self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache


class RearrangeTransform(object):
    """
    Wrapper for einops.rearrange to pass into torchvision.transforms.Compose
    """
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, img):
        img = rearrange(img, self.pattern)
        return img

class RepeatTransform(object):
    """
    Wrapper for einops.repeat to pass into torchvision.transforms.Compose
    """
    def __init__(self, pattern, b):
        self.pattern = pattern
        self.b = b
    def __call__(self, img):
        img = repeat(img, self.pattern, b=self.b)
        return img


if __name__ == '__main__':
    # os.chdir("../../")
    # config = Config("config/main.yml").read()
    # brca = TCGADataset("brca", config)
    # blca = TCGADataset("blca", config)
    # print(config)
    # print(brca.omic_df.shape, blca.omic_df.shape)
    # blca.load_wsi("TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F", resolution="lowest")

    from torch.utils.data import DataLoader
    
    n=50
    tab_tensor = torch.rand(size=(n, 1, 10))
    img_tensor = torch.rand(size=(n, 224, 224, 1))
    vid_tensor = torch.rand(size=(n, 12, 224, 224, 1))
    
    target = torch.rand(size=(n,))
    
    data = MMDataset([tab_tensor, img_tensor, vid_tensor], target)
    
    loader = DataLoader(data, batch_size=4, shuffle=True)
    
    # fetch batch
    tensors, target = next(iter(loader))
    
    print([t.shape for t in tensors])


# if __name__ == "__main__": 
    
