"""
    Adapted from https://github.com/konst-int-i/healnet by Marta Buetas-Arcas for multimodal survival datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold, train_test_split
import multiprocessing
import argparse
from argparse import Namespace
import yaml
from tqdm import tqdm
import os
import sys
import random
import hashlib
import json

HEALNET_ADOPT_PATH = os.environ.get("HEALNET_ADOPT_PATH", "/nfs/rnas/projects/mmCRC/git/healnet-adoption")
if HEALNET_ADOPT_PATH not in sys.path:
    sys.path.insert(0, HEALNET_ADOPT_PATH)

from healnet.utils import EarlyStopping, calc_reg_loss, pickle_obj
from healnet.models.survival_loss import CrossEntropySurvLoss, CoxPHSurvLoss, nll_loss
from healnet.baselines import RegularizedFCNN, MMPrognosis, MCAT, SNN, MILAttentionNet, MultiModNModule
from healnet.baselines.multimodn import MLPEncoder, PatchEncoder, ClassDecoder
from healnet.models import HealNet
from healnet.utils import Config, flatten_config
from healnet.etl import TCGADataset, MMDataset

from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sksurv.metrics import concordance_index_censored
from scipy.stats import spearmanr
from torch import optim
import pandas as pd
from box import Box
from pathlib import Path
from datetime import datetime
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
import wandb
import logging

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # older Pythons or nonstandard streams might not support reconfigure
    pass

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def safe_set_mp_start_method(method="fork"):
    try:
        import torch.multiprocessing as mp

        current = mp.get_start_method(allow_none=True)
        if current is None:
            mp.set_start_method(method)
            logger.info("Set multiprocessing start method to %s", method)
        else:
            logger.debug("Multiprocessing start method already set to %s", current)
    except Exception as e:
        logger.warning("Could not set multiprocessing start method: %s", e)

class Pipeline:
    """
    Main experimental pipeline class for training and evaluating models, config handling, and logging
    """

    def __init__(self, config: Box, args: Namespace):
        self.config = flatten_config(config)
        self.dataset = self.config.dataset
        self.args = args
        self.log_dir = None
        self.hp_dir = None  # top-level folder for this hyperparameter configuration (sweep group)
        self._check_config()
        self.wandb_name = self.config.get("wandb_name", None)
        # WandB enabled if config enables wandb or running sweep mode
        self._wandb_enabled = bool(self.config.get("wandb", False)) or (getattr(self.args, "mode", None) == "sweep")
        self.output_dims = int(self.config[f"model_params.output_dims"])
        self.sources = self.config.sources

        # date
        self.local_run_id = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self._wandb_enabled:
            self.wandb_setup()

        # Setting the seed
        if self.args.seed is None:
            self.args.seed = 42
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.args.seed)

        if self.config.get("explainer", False):
            run_name = None
            if self._wandb_enabled:
                try:
                    run_name = getattr(wandb, "run", None) and wandb.run.name
                except Exception:
                    run_name = None
            if run_name is None:
                run_name = self.local_run_id
            self.log_dir = Path(self.config.log_path).joinpath(f"{run_name}")
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def wandb_setup(self) -> None:
        if not bool(self.config.get("wandb", False)):
            logger.debug("WandB not enabled in config; skipping setup.")
            return None

        api_key = getattr(self.args, "api_key", None) or os.environ.get("WANDNB_API_KEY") or os.environ.get("WANDB_API_KEY")
        if api_key:
            # ensure env var is set so any sub-process inherits it
            os.environ["WANDB_API_KEY"] = api_key
            os.environ["WANDNB_API_KEY"] = api_key

        if self.args.mode == "sweep":
            with open(self.args.sweep_config, "r") as f:
                sweep_config = yaml.safe_load(f)
            try:
                if api_key:
                    try:
                        wandb.login(key=api_key)
                    except Exception:
                        logger.debug("wandb.login with key failed or not necessary.")

                sweep_id = wandb.sweep(sweep=sweep_config, project=self.config.get("wandb_project", self.args.project_name or "rename"))
                logger.info("Created wandb sweep with id: %s", sweep_id)
                logger.info("Launch agents with: `wandb agent %s` (or use wandb.agent programmatically)", sweep_id)

                try:
                    wandb.agent(sweep_id, function=self.main)
                except Exception:
                    logger.exception("wandb.agent failed or blocked; if running in a different environment, run `wandb agent %s` manually", sweep_id)
            
            except Exception as e:
                logger.exception("Failed to initialize wandb sweep: %s", e)

        else:
            try:
                if api_key:
                    try:
                        wandb.login(key=api_key)
                    except Exception:
                        logger.debug("wandb.login with key failed or not necessary.")
                wandb_config = dict(self.config)
                
                wandb.init(project=self.config.get("wandb_project", self.args.project_name or "rename"),
                        name=self.wandb_name,
                        config=wandb_config,
                        resume=False)
                logger.info("Initialized wandb run (name=%s)", self.wandb_name)
            except Exception as e:
                logger.exception("Failed to initialize wandb: %s", e)
        return None

    def _check_config(self) -> None:
        """
        Assert that the config only contains valid arguments
        Returns:
            None
        """
        valid_sources = ["clinical", "pathology", "radiology"]
        assert all([source in valid_sources for source in self.config.sources]), f"Invalid source specified. Valid sources are {valid_sources}"

        valid_survival_losses = ["nll", "ce_survival", "cox"]
        assert self.config["survival.loss"] in valid_survival_losses, f"Invalid survival loss specified. " \
                                                                   f"Valid losses are {valid_survival_losses}"

        valid_datasets = ["mmCRC"]
        assert self.config.dataset in valid_datasets, f"Invalid dataset specified. Valid datasets are {valid_datasets}"

        valid_models = ["healnet", "fcnn", "healnet_early", "mcat", "mm_prognosis", "multimodn"]
        assert self.config.model in valid_models, f"Invalid model specified. Valid models are {valid_models}"

        valid_class_weights = ["inverse", "inverse_root", "None"]
        assert self.config[f"model_params.class_weights"] in valid_class_weights, f"Invalid class weight specified. Valid weights are {valid_class_weights}"

        return None


    def _compute_hp_group(self, extra_config: dict = None):
        """
        Compute a deterministic fingerprint for this hyperparameter configuration (used to group folder per HP combo).
        extra_config: optional dict (e.g., wandb.config) that overrides self.config for hashing when present.
        Returns (hp_hash, hp_serializable_dict)
        """
        try:
            hp_dict = dict(extra_config) if extra_config else dict(self.config)
        except Exception:
            hp_dict = {str(k): str(v) for k, v in (extra_config or self.config).items()}

        def _sanitize(o):
            try:
                json.dumps(o)
                return o
            except Exception:
                return str(o)

        hp_sanitized = {k: _sanitize(v) for k, v in sorted(hp_dict.items(), key=lambda x: x[0])}
        hp_serialized = json.dumps(hp_sanitized, sort_keys=True, default=str)
        hp_hash = hashlib.sha1(hp_serialized.encode("utf-8")).hexdigest()[:10]
        return hp_hash, hp_sanitized


    def main(self):
        # Sweep initialization handled in wandb_setup above.
        #ensure an explicit wandb.init() exists
        try:
            if getattr(self.args, "mode", None) == "sweep":
                run = getattr(wandb, "run", None)
                # If there's no active run (or it's offline), explicitly init one for the agent
                if run is None or getattr(run, "mode", None) == "offline":
                    try:
                        # Force an online run for sweep agents (set resume=False)
                        wandb.init(project=self.config.get("wandb_project", self.args.project_name or "rename"),
                                   config=dict(self.config),
                                   resume=False,
                                   mode="online")
                        logger.info("Initialized wandb run inside sweep agent: id=%s", wandb.run.id)
                    except Exception as e:
                        logger.exception("Failed to init wandb inside sweep agent: %s", e)
        except Exception:
            logger.debug("Sweep-mode wandb.init check skipped (ignored)")

        try:
            if getattr(self.args, "mode", None) == "sweep" and getattr(wandb, "run", None) is not None:
                try:
                    wc = dict(wandb.config)
                    # Merge into self.config (self.config uses dot-keys like "model_params.depth")
                    for k, v in wc.items():
                        # skip any W&B internal keys if present
                        if k.startswith("_"):
                            continue
                        # ensure key is string (W&B sometimes returns AttrDicts)
                        self.config[str(k)] = v
                    logger.info("Merged wandb.config into runtime config: %s", {k: self.config.get(str(k)) for k in wc.keys()})
                    # Helpful one-line debug for immediate inspection
                    print("Merged W&B config:", wc)

                    # IMPORTANT: If the sweep provides a 'seed' in wandb.config, propagate it to args.seed
                    if "seed" in wc:
                        try:
                            new_seed = int(wc["seed"])
                            self.args.seed = new_seed
                            random.seed(self.args.seed)
                            np.random.seed(self.args.seed)
                            torch.manual_seed(self.args.seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(self.args.seed)
                            logger.info("Updated args.seed from wandb.config to %s and reseeded RNGs", self.args.seed)
                        except Exception as e:
                            logger.warning("Failed to apply seed from wandb.config: %s", e)

                except Exception as e:
                    logger.exception("Failed to merge wandb.config into self.config: %s", e)
        except Exception:
            logger.debug("Sweep-mode wandb.config merge skipped (ignored)")

        # create HP (hyperparameter combo) folder
        try:
            extra_cfg = None
            if getattr(self.args, "mode", None) == "sweep" and self._wandb_enabled:
                try:
                    extra_cfg = dict(wandb.config)
                except Exception:
                    extra_cfg = None

            hp_hash, hp_serializable = self._compute_hp_group(extra_cfg)
            base_log = Path(self.config.get("log_path", "."))
            self.hp_dir = base_log.joinpath(f"hp_{hp_hash}")
            self.hp_dir.mkdir(parents=True, exist_ok=True)

            hyperparams_path = self.hp_dir.joinpath("hyperparams.yml")
            try:
                # Coerce any Box/BoxList/numpy/Path/etc into native JSON-serializable types
                safe_obj = json.loads(json.dumps(hp_serializable, default=str))
                with open(hyperparams_path, "w") as fh:
                    yaml.safe_dump(safe_obj, fh, sort_keys=True)
            except Exception as e:
                logger.warning("Failed to write hyperparams YAML to %s: %s", hyperparams_path, e)

            run_id = (getattr(getattr(wandb, "run", None), "id", None) or self.local_run_id)
            self.log_dir = self.hp_dir.joinpath(str(run_id))
            self.log_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Logging to HP folder: %s and run folder: %s", self.hp_dir, self.log_dir)
        except Exception as e:
            logger.exception("Failed to set up HP-specific logging directory: %s", e)
            if self.log_dir is None:
                run_name = getattr(wandb, "run", None) and getattr(wandb.run, "name", None)
                if run_name is None:
                    run_name = self.local_run_id
                self.log_dir = Path(self.config.get("log_path", ".")).joinpath(run_name)
                self.log_dir.mkdir(parents=True, exist_ok=True)

        train_c_indeces, val_c_indeces, test_c_indeces = [], [], []
        test_data_indices = []
        missing_perfs = []
        models = []

        # Get dataset once
        dataset = self.load_data()
        y_disc = dataset.y_disc
        # handle tensor vs list/array
        if isinstance(y_disc, torch.Tensor):
            y_disc_np = y_disc.numpy()
        else:
            y_disc_np = np.array(y_disc)

        # Stratified K-Fold
        cv = StratifiedKFold(n_splits=self.config["n_folds"], shuffle=True, random_state=self.args.seed)

        # Enumerate folds and split the indices
        for fold, (train_val_idx, test_idx) in enumerate(cv.split(np.zeros(len(dataset)), y_disc_np), 1):
            logger.info("***** FOLD %d *****", fold)

            n_total = len(dataset)
            n_train_val = len(train_val_idx)
            val_desired = int(0.15 * n_total)
            
            if val_desired <= 0 or val_desired >= n_train_val:
                # fallback if dataset is very small
                val_size = 0.2
            else:
                val_size = val_desired / n_train_val
            
            # Stratified inner split
            train_idx, val_idx = train_test_split(train_val_idx, 
                                                  test_size=val_size, 
                                                  stratify=y_disc_np[train_val_idx], 
                                                  random_state=self.args.seed)

            # Create dataloaders
            train_data, val_data, test_data = self.get_dataloaders(dataset, train_idx, val_idx, test_idx)
            
            # Store indices similar to original logic
            test_data_indices.append(test_data.dataset.indices)

            model = self.make_model(train_data)
            if self._wandb_enabled:
                try:
                    wandb.watch(model)
                except Exception:
                    logger.debug("wandb.watch() failed (ignored)")

            model, _, train_c_index, _, val_c_index, _, test_c_index, missing_performance = self.train_survival_fold(
                model, train_data, val_data, test_data, fold=fold
            )
            train_c_indeces.append(train_c_index)
            val_c_indeces.append(val_c_index)
            test_c_indeces.append(test_c_index)
            missing_perfs.append(missing_performance)
            models.append(model)

        # log summary statistics
        if self._wandb_enabled:
            try:
                wandb.log(
                    {
                        "mean_train_c_index": np.mean(train_c_indeces),
                        "mean_val_c_index": np.mean(val_c_indeces),
                        "std_train_c_index": np.std(train_c_indeces),
                        "std_val_c_index": np.std(val_c_indeces),
                        "mean_test_c_index": np.mean(test_c_indeces),
                        "std_test_c_index": np.std(test_c_indeces),
                    }
                )
                wandb.run.summary["mean_val_c_index"] = float(np.mean(val_c_indeces))
                wandb.run.summary["std_val_c_index"] = float(np.std(val_c_indeces))
            except Exception:
                logger.debug("Failed to log fold summary to wandb (ignored)")

        best_fold = int(np.argmax(test_c_indeces))
        best_model = models[best_fold]

        if self.config.get("missing_ablation", False):
            missing_50_c_index, missing_omic_c_index, missing_wsi_c_index = np.mean(missing_perfs, axis=0)
            if self._wandb_enabled:
                try:
                    wandb.log(
                        {"missing_50_c_index": missing_50_c_index, "missing_omic_c_index": missing_omic_c_index, "missing_wsi_c_index": missing_wsi_c_index}
                    )
                except Exception:
                    logger.debug("Failed to log missing ablation results")

        if self.config.get("explainer", False):
            torch.save(best_model.state_dict(), self.log_dir.joinpath("best_model.pt"))
            pickle_obj(self.config, self.log_dir.joinpath("config.pkl"))
            pickle_obj(test_data_indices[best_fold], self.log_dir.joinpath("test_data_indices.pkl"))

        if self._wandb_enabled:
            try:
                wandb.finish()
            except Exception:
                logger.debug("wandb.finish() failed (ignored)")
    
    def make_collate_fn(self): # NEW: for missing modalities at sample-level
        """
        Collate function for per-sample missing modalities.

        Expects dataset __getitem__ to return:
        ([mod1, mod2, ...], censorship, event_time, y_disc)
        where each modality element is either:
        - torch.Tensor with shape (tokens, channels) or (channels,) or (1, channels)
        - None if modality missing for that sample

        Returns batched:
        modalities_batched: list of length M with tensors of shape (B, tokens, channels) or None
        masks: list of length M with boolean tensors of shape (B, tokens) or None
        censorship, event_time, y_disc: stacked tensors
        """
        import torch

        def normalize_tensor(x):
            if x is None:
                return None
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return x

        def collate(batch):
            # batch: list of tuples ([mod1,...], censorship, event_time, y_disc)
            modalities_per_sample = [item[0] for item in batch]
            B = len(batch)

            # stack scalar targets
            censorship = torch.stack([torch.as_tensor(item[1]) for item in batch])
            event_time = torch.stack([torch.as_tensor(item[2]) for item in batch])
            y_disc = torch.stack([torch.as_tensor(item[3]) for item in batch])

            M = len(modalities_per_sample[0])
            modalities_batched = []
            masks = []

            for i in range(M):
                ref = None
                for sample_modalities in modalities_per_sample:
                    t = normalize_tensor(sample_modalities[i])
                    if t is not None:
                        ref = t
                        break
                if ref is None:
                    # modality absent for all samples in this batch (original approach of HealNet)
                    modalities_batched.append(None)
                    masks.append(None)
                    continue

                token_count = ref.shape[0] 
                channel_count = ref.shape[-1]
                stacked = []
                mask_i = torch.zeros((B, token_count), dtype=torch.bool)

                for b_idx, sample_modalities in enumerate(modalities_per_sample):
                    t = normalize_tensor(sample_modalities[i])
                    if t is None:
                        stacked.append(torch.zeros_like(ref))
                    else:
                        # pad or truncate channels if necessary
                        if t.shape[-1] != channel_count:
                            last_diff = channel_count - t.shape[-1]
                            if last_diff > 0:
                                pad = torch.zeros(*t.shape[:-1], last_diff, dtype=t.dtype, device=t.device)
                                t = torch.cat([t, pad], dim=-1)
                            elif last_diff < 0:
                                t = t[..., :channel_count]
                      
                            if t.shape[0] != token_count:
                                if t.shape[0] == 1:
                                    t = t.expand(token_count, channel_count)
                                else:
                                    raise ValueError(
                                        f"Modality {i} token count mismatch for sample {b_idx}: "
                                        f"found {t.shape[0]} tokens but expected {token_count}. Collate cannot safely reshape this sample."
                                    )
                        stacked.append(t)
                        mask_i[b_idx, :] = True
                stacked_tensor = torch.stack(stacked, dim=0)
                stacked_tensor = stacked_tensor.to(dtype=torch.float32)
                assert stacked_tensor.shape == (B, token_count, channel_count), \
                    f"After stacking, modality {i} shape mismatch: {stacked_tensor.shape} != {(B, token_count, channel_count)}"

                modalities_batched.append(stacked_tensor)  # (B, tokens, channels)
                masks.append(mask_i)

            return modalities_batched, masks, censorship, event_time, y_disc

        return collate
   

    def load_data(self):
        """
        Initializes and returns the MMDataset **without splitting**.
        """
        data = MMDataset(config=self.config,
                         filter_overlap=self.config.get("filter_overlap", False),
                         survival_analysis=True,
                         sources=self.sources,
                         n_bins=self.output_dims,
                         log_dir=self.log_dir)
        return data

    def get_dataloaders(self, dataset, train_indices, val_indices, test_indices):
        """
        Creates DataLoaders for train, val, test splits based on indices.
        Computes standardization on train split only.
        """
        collate_fn = self.make_collate_fn()

        # Create subsets (this only points to the indices, not copying the data, so when thenn is standardized, it will affect the variable 'dataset')
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_indices)

        # Compute standardization using ONLY train indices
        try:
            if hasattr(dataset, "compute_standardization_from_indices"):
                dataset.compute_standardization_from_indices(train_indices)
                logger.info("Applied standardization computed from training split.")
            else:
                logger.debug("Dataset does not implement compute_standardization_from_indices(); no standardization applied.")
        except Exception as e:
            logger.warning("Failed to compute/apply standardization from train split: %s", e)

        # Calculate class weights based on current training subset
        if self.config[f"model_params.class_weights"] == "None":
            self.class_weights = None
        else:
            # temporary subset wrapper to reuse _calc_class_weights logic if needed, 
            # or just access dataset.y_disc directly with indices
            self.class_weights = torch.Tensor(self._calc_class_weights_from_indices(dataset, train_indices)).to(self.device)

        num_workers_train = int(min(8, max(0, self.config.get("dataloader.num_workers", 4))))
        
        train_data = DataLoader(
            train_subset,
            batch_size=self.config["train_loop.batch_size"],
            shuffle=True, # Shuffle training data
            num_workers=num_workers_train,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_data = DataLoader(
            val_subset,
            batch_size=self.config["train_loop.batch_size"],
            shuffle=False,
            num_workers=int(multiprocessing.cpu_count()),
            pin_memory=True,
            multiprocessing_context=MP_CONTEXT,
            collate_fn=collate_fn
        )

        test_data = DataLoader(
            test_subset,
            batch_size=self.config["train_loop.batch_size"],
            shuffle=False,
            num_workers=int(multiprocessing.cpu_count()),
            pin_memory=True,
            multiprocessing_context=MP_CONTEXT,
            collate_fn=collate_fn
        )

        return train_data, val_data, test_data

    def _calc_class_weights(self, train):
        if self.config[f"model_params.class_weights"] in ["inverse", "inverse_root"]:
            train_targets = np.array(train.dataset.y_disc)[train.indices]
            _, counts = np.unique(train_targets, return_counts=True)
            if self.config[f"model_params.class_weights"] == "inverse":
                class_weights = 1.0 / counts
            elif self.config[f"model_params.class_weights"] == "inverse_root":
                class_weights = 1.0 / np.sqrt(counts)
        else:
            class_weights = None
        return class_weights

    def _calc_class_weights_from_indices(self, dataset, indices):
        if self.config[f"model_params.class_weights"] in ["inverse", "inverse_root"]:
            y_disc = dataset.y_disc
            if isinstance(y_disc, torch.Tensor):
                y_disc = y_disc.numpy()
            train_targets = np.array(y_disc)[indices]
            _, counts = np.unique(train_targets, return_counts=True)
            if self.config[f"model_params.class_weights"] == "inverse":
                class_weights = 1.0 / counts
            elif self.config[f"model_params.class_weights"] == "inverse_root":
                class_weights = 1.0 / np.sqrt(counts)
        else:
            class_weights = None
        return class_weights

    def make_model(self, train_data: DataLoader):
        """
        Instantiates model and moves to CUDA device if available
        Args:
            train_data:

        Returns:
            nn.Module: model used for training
        """
        # get features from first batch
        batch_example = next(iter(train_data))
        feat = batch_example[0]  # feat is list of modality tensors (or None)
        if self.config.model in  ["healnet", "healnet_early"]:
            num_modalities = len(feat)
            underlying_ds = train_data.dataset.dataset if hasattr(train_data.dataset, "dataset") else train_data.dataset

            input_channels = []
            for i in range(num_modalities):
                f = feat[i]
                if f is not None:
                    # f shape: (batch, tokens, channels)
                    input_channels.append(int(f.shape[-1]))
                else:
                    # fallback: use dataset metadata
                    mod_name = underlying_ds.modalities_list[i]
                    input_channels.append(int(underlying_ds.feature_dims[mod_name]))

            input_axes = [1 for _ in range(num_modalities)]  # tabular modalities => 1 spatial axis
            modalities = num_modalities

            model = HealNet(
                n_modalities=modalities,
                channel_dims=input_channels,
                num_spatial_axes=input_axes,
                out_dims=self.output_dims,
                num_freq_bands=self.config[f"model_params.num_freq_bands"],
                depth=self.config[f"model_params.depth"],
                max_freq=self.config[f"model_params.max_freq"],
                l_c = self.config[f"model_params.num_latents"],
                l_d = self.config[f"model_params.latent_dim"],
                cross_dim_head = self.config[f"model_params.cross_dim_head"],
                latent_dim_head = self.config[f"model_params.latent_dim_head"],
                x_heads = self.config[f"model_params.cross_heads"],
                l_heads = self.config[f"model_params.latent_heads"],
                attn_dropout = self.config[f"model_params.attn_dropout"],
                ff_dropout = self.config[f"model_params.ff_dropout"],
                weight_tie_layers = self.config[f"model_params.weight_tie_layers"],
                fourier_encode_data = self.config[f"model_params.fourier_encode_data"],
                self_per_cross_attn = self.config[f"model_params.self_per_cross_attn"],
                final_classifier_head = True,
                snn = self.config[f"model_params.snn"],
            )
            model.float()
            model.to(self.device)

        elif self.config.model == "fcnn":
            model = RegularizedFCNN(output_dim=self.output_dims)
            model.to(self.device)

        elif self.config.model == "multimodn":
            l_d = 2000
            tab_features = feat[0].shape[1]
            patch_dims = feat[1].shape[2]
            encoders = [
                MLPEncoder(state_size=l_d, hidden_layers=[1024, 256, 128, 64], n_features=tab_features),
                PatchEncoder(state_size=l_d, hidden_layers=[512, 256, 128, 64], n_features=patch_dims)
            ]
            decoders = [ClassDecoder(state_size=l_d, n_classes=self.output_dims, activation=torch.sigmoid)]

            model = MultiModNModule(
                state_size=l_d,
                encoders=encoders,
                decoders=decoders
            )
            model.float()
            model.to(self.device)

        elif self.config.model == "mm_prognosis":
            if len(self.config["sources"]) == 1:
                input_dim = feat[0].shape[1]
                # input_dim = feat[0].shape[2] + feat[1].shape[2]
            model = MMPrognosis(sources=self.sources,
                                output_dims=self.output_dims,
                                config=self.config
                                )
            model.float()
            model.to(self.device)

        elif self.config.model == "mcat":
            if len(self.config["sources"]) == 2:
                model = MCAT(
                    n_classes=self.output_dims,
                    omic_shape=feat[0].shape[1:],
                    wsi_shape=feat[1].shape[1:]
                )
            elif self.config["sources"][0] == "omic":
                model = SNN(
                    n_classes=self.output_dims,
                    input_dim=feat[0].shape[1]
                )
            elif self.config["sources"][0] == "slides":
                model = MILAttentionNet(
                    input_dim=feat[0].shape[1:],
                    n_classes=self.output_dims
                )
            model.float()
            model.to(self.device)

        return model
    


    def train_survival_fold(self, model: nn.Module, train_data: torch.utils.data.DataLoader, val_data: torch.utils.data.DataLoader, test_data: torch.utils.data.DataLoader, fold: int, gc: int = 16, **kwargs):
        """
        Trains model for survival analysis
        Args:
            model (nn.Module): model to train
            train_data (DataLoader): training data
            test_data (DataLoader): test data
            val_data (DataLoader): validation data
            **kwargs:

        Returns:
            Tuple: tuple of the model and all performance metrics for given fold
        """
        logger.info("Training survival model using %s", self.config.model)
        optimizer = optim.Adam(model.parameters(), lr=self.config["optimizer.lr"])
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                  max_lr=self.config["optimizer.max_lr"],
                                                  epochs=self.config["train_loop.epochs"],
                                                  steps_per_epoch=len(train_data))


        monitor = self.config.get("train_loop.monitor", "val_loss")
        if monitor == "val_c_index":
            # maximize c-index
            early_stopping = EarlyStopping(patience=self.config["train_loop.patience"], mode="max", verbose=True)
        else:
            # minimize loss
            early_stopping = EarlyStopping(patience=self.config["train_loop.patience"], mode="min", verbose=True)

        model.train()

        def param_norm(m):
            return sum(p.detach().cpu().norm().item() for p in m.parameters())

        init_norm = param_norm(model)
        logger.debug("Initial param norm: %0.6f", init_norm)

        # track best validation c-index so we can always evaluate the best model at the end
        best_val_c_index = -float("inf")
        # store best state_dict in memory and write to disk only once per fold at the end
        best_state_dict = None
        best_state_run_id = None

    
        for epoch in range(1, self.config["train_loop.epochs"]+1):
            logger.info("Epoch %d", epoch)
            risk_scores = []
            censorships = []
            event_times = []
            train_loss_surv, train_loss = 0.0, 0.0 
            grad_norms = []

            if self.log_dir is None:
                run_name = getattr(wandb, "run", None) and getattr(wandb.run, "name", None)
                if run_name is None:
                    run_name = self.local_run_id
                self.log_dir = Path(self.config.get("log_path", ".")).joinpath(run_name)
                self.log_dir.mkdir(parents=True, exist_ok=True)

            for batch, (features, masks, censorship, event_time, y_disc) in enumerate(tqdm(train_data)):
                features = [f.to(self.device) if f is not None else None for f in features]
                masks = [m.to(self.device) if m is not None else None for m in masks]
                censorship = censorship.to(self.device)
                event_time = event_time.to(self.device)
                y_disc = y_disc.to(self.device)

                """ Biggest change with respect to the original repository, in order to tackle with missing modalities per sample in batch """
                # determine batch size b robustly (prefer features, then masks, then targets)
                b = None
                for f in features:
                    if f is not None:
                        b = f.shape[0]
                        break
                if b is None and masks is not None:
                    for m in masks:
                        if m is not None:
                            b = m.shape[0]
                            break
                if b is None:
                    if isinstance(y_disc, torch.Tensor):
                        b = y_disc.shape[0]
                    elif isinstance(censorship, torch.Tensor):
                        b = censorship.shape[0]
                    else:
                        raise ValueError("Cannot infer batch size: all feature tensors and masks are None and no scalar targets available.")

                # Build 'present' list from collated `features` and `masks`
                present = []
                for i in range(len(features)):
                    mi = masks[i] if masks is not None and len(masks) > i else None
                    fi = features[i]

                    if fi is None:
                        # If the feature tensor itself is None (collate set modality to None),
                        # mark all samples absent for that modality.
                        present.append(torch.zeros(b, dtype=torch.bool, device=self.device))
                        continue

                    # If a per-modality mask is provided by collate, use it (preferred).
                    if mi is not None:
                        # mi shape: (B, tokens) -> True if any token valid for that sample
                        present_i = mi.any(dim=1).to(dtype=torch.bool, device=self.device)
                        present.append(present_i)
                        continue

                    # Fallback: derive presence by checking nonzero rows in fi (handles padding-by-zeros)
                    flat = fi.flatten(start_dim=1)
                    present_i = (flat.abs().sum(dim=1) > 0).to(dtype=torch.bool, device=self.device)
                    present.append(present_i)
                
                optimizer.zero_grad()
                
                for i, feat in enumerate(features):
                    if feat is None and present[i].any():
                        logger.debug("modality %d: feat is None but present indicates some samples present", i)
              

                # Single forward pass -> pass per-modality masks and presence to HealNet
                if self.config["model"] == "multimodn":
                    model_loss, logits = model.forward(features, F.one_hot(y_disc, num_classes=self.output_dims))
                else:
                    if self.config["model"] in ["healnet", "healnet_early"]:
                        logits = model(features, masks=masks, present=present)
                    else:
                        logits = model.forward(features)
                    model_loss = 0.0
            
                if batch == 0 and epoch == 1:
                    logger.debug("Modality shapes: %s", [f.shape if f is not None else None for f in features])
                    logger.debug("Modality dtypes: %s", [f.dtype if f is not None else None for f in features])

                y_hat = torch.topk(logits, k=1, dim=1)[1].squeeze(1)
                hazards = torch.sigmoid(logits)  # sigmoid to get hazards from predictions for surv analysis
                survival = torch.cumprod(1-hazards, dim=1)  # as per paper, survival = cumprod(1-hazards)

                # Risk scalar: collapse the survival curve into a single number for ranking
                # Larger numeric risk (less negative) implies higher hazard / earlier event
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()  # risk = -sum(survival)
                

                if self.config["survival.loss"] == "nll":
                    surv_loss = nll_loss(hazards=hazards, S=survival, Y=y_disc, c=censorship, weights=self.class_weights)
                elif self.config["survival.loss"] == "ce_survival":
                    loss_fn = CrossEntropySurvLoss()
                    surv_loss = loss_fn(hazards=hazards, survival=survival, y_disc=y_disc, censorship=censorship)
                elif self.config["survival.loss"] == "cox":
                    loss_fn = CoxPHSurvLoss()
                    surv_loss = loss_fn(hazards=hazards, survival=survival, censorship=censorship)

                reg_loss = calc_reg_loss(model, self.config[f"model_params.l1"], self.config.model, self.config.sources)

                # log risk, censorship and event time for concordance index
                risk_scores.append(risk)
                censorships.append(censorship.detach().cpu().numpy())
                event_times.append(event_time.detach().cpu().numpy())

                loss_value = surv_loss.item()
                train_loss_surv += loss_value
                train_loss += loss_value + reg_loss

                # backward pass
                surv_loss = surv_loss / gc + reg_loss + model_loss  # gradient accumulation step
                surv_loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            final_norm = param_norm(model)
            logger.debug("Final param norm: %0.6f (delta %0.6f)", final_norm, final_norm - init_norm)
  
            train_loss /= len(train_data)
            train_loss_surv /= len(train_data)

            # compute lr for logging (OneCycleLR)
            try:
                current_lr = optimizer.param_groups[0]['lr']
            except Exception:
                current_lr = None

            risk_scores_full = np.concatenate(risk_scores)
            censorships_full = np.concatenate(censorships)
            event_times_full = np.concatenate(event_times)

            # calculate epoch-level concordance index
            train_c_index = concordance_index_censored((1-censorships_full).astype(bool), event_times_full, risk_scores_full, tied_tol=1e-08)[0]
            
            if self._wandb_enabled:
                try:
                    log_prefix = f"fold{fold}/"
                    wandb.log({
                        f"{log_prefix}train/loss": train_loss_surv, 
                        f"{log_prefix}train/total_loss": train_loss, 
                        f"{log_prefix}train/c_index": train_c_index, 
                        f"{log_prefix}train/lr": current_lr, 
                        f"{log_prefix}epoch": epoch
                    }, step=epoch)
                except Exception:
                    logger.debug("Failed to log training metrics to wandb (ignored)")

            logger.info("Epoch: %d, train_loss: %.4f, train_c_index: %.4f", epoch, train_loss_surv, train_c_index)

            logger.info("Running validation")
            logger.info("Running validation")
            val_loss, val_c_index = self.evaluate_survival_epoch(epoch, model, val_data, log_prefix=f"fold{fold}/")
            logger.info("Epoch: %d, val_loss: %.4f, val_c_index: %.4f", epoch, val_loss, val_c_index)
            
            monitor_value = val_c_index if monitor == "val_c_index" else val_loss
            logger.debug("monitor_value: %s (%s)", monitor_value, type(monitor_value))

            if self.config["train_loop.early_stopping"] and early_stopping.step(monitor_value, model):
                logger.info("Early stopping at epoch %d (monitor reached best value)", epoch)
                model = early_stopping.load_best_weights(model)
                break
            
             # ---------- checkpoint on validation c-index (maximize) ----------
            try:
                if val_c_index is not None and val_c_index > best_val_c_index:
                    best_val_c_index = float(val_c_index)
                    sd = getattr(model, "module", model).state_dict()
                    try:
                        sd_cpu = {k: v.cpu().clone() for k, v in sd.items()}
                    except Exception:
                        sd_cpu = sd
                    best_state_dict = sd_cpu
                    best_state_run_id = (getattr(getattr(wandb, "run", None), "id", None) or self.local_run_id)
                    if self._wandb_enabled:
                        try:
                            wandb.run.summary[f"best_val_c_index_fold{fold}"] = best_val_c_index
                        except Exception:
                            logger.debug("Failed to update wandb run summary (ignored)")
                    logger.info("Found new best val for fold %d (val_c_index=%.4f) -> stored best state in memory", fold, best_val_c_index)
            except Exception as e:
                logger.warning("Failed to register best model in memory: %s", e)
            # ----------------------------------------------------------------


        # After training completes, write the single best state dict (if any) to disk ONCE for this fold.
        try:
            if best_state_dict is not None:
                run_id = best_state_run_id or (getattr(getattr(wandb, "run", None), "id", None) or self.local_run_id)
                if self.log_dir is None:
                    if self.hp_dir is None:
                        base_log = Path(self.config.get("log_path", "."))
                        self.hp_dir = base_log.joinpath(f"hp_manual_{run_id}")
                        self.hp_dir.mkdir(parents=True, exist_ok=True)
                    self.log_dir = self.hp_dir.joinpath(str(run_id))
                    self.log_dir.mkdir(parents=True, exist_ok=True)

                best_model_path = str(self.log_dir.joinpath(f"best_model_{run_id}_fold{fold}.pt"))
                torch.save(best_state_dict, best_model_path)
                if self._wandb_enabled:
                    try:
                        wandb.run.summary[f"best_model_path_fold{fold}"] = best_model_path
                    except Exception:
                        logger.debug("Failed to set wandb run summary path (ignored)")
                logger.info("Saved final best model for fold %d to %s (val_c_index=%.4f)", fold, best_model_path, best_val_c_index)
        except Exception as e:
            logger.warning("Failed to save final best model for fold %d: %s", fold, e)

        
        # Ensure best model loaded for evaluation
        try:
            if best_state_dict is not None:
                try:
                    model.load_state_dict(best_state_dict)
                except RuntimeError:
                    try:
                        model.module.load_state_dict(best_state_dict)
                    except Exception:
                        model.load_state_dict(best_state_dict, strict=False)
                logger.info("Loaded best model from memory for final test evaluation (val_c_index=%.4f)", best_val_c_index)
        except Exception as e:
            logger.warning("Failed to load best model checkpoint: %s", e)
        
        # Final test evaluation: once stopped and best model is loaded, evaluate on test set
        logger.info("Running test set evaluation")
        # Final test evaluation: once stopped and best model is loaded, evaluate on test set
        logger.info("Running test set evaluation")
        test_loss, test_c_index = self.evaluate_survival_epoch(epoch, model, test_data, save_preds=True, log_prefix=f"fold{fold}/")
        logger.info("Epoch: %d, test_loss: %.4f, test_c_index: %.4f", epoch, test_loss, test_c_index)

        # run ablation
        missing_performance = None
        if self.config.missing_ablation:
            _, missing_50_c_index = self.evaluate_survival_epoch(epoch=self.config["train_loop.epochs"],
                                                                               model=model,
                                                                               test_data=test_data,
                                                                               missing_mode="50")
            _, missing_path_c_index = self.evaluate_survival_epoch(epoch=self.config["train_loop.epochs"],
                                                                               model=model,
                                                                               test_data=test_data,
                                                                               missing_mode="pathology")
            _, missing_clin_c_index = self.evaluate_survival_epoch(epoch=self.config["train_loop.epochs"],
                                                                               model=model,
                                                                               test_data=test_data,
                                                                               missing_mode="clinical")
            _, missing_rad_c_index = self.evaluate_survival_epoch(epoch=self.config["train_loop.epochs"],
                                                                               model=model,
                                                                               test_data=test_data,
                                                                               missing_mode="radiology")

            missing_performance = (missing_50_c_index, missing_path_c_index, missing_clin_c_index, missing_rad_c_index)

        # return values of final epoch
        return model, train_loss, train_c_index, val_loss, val_c_index, test_loss, test_c_index, missing_performance

    def _sample_missing(self, features, use_omic, mode):
        assert mode in ["50", "omic", "wsi"], "Invalid missing ablation mode"

        if mode == "50":
            if use_omic:
                use_omic = False
                return [features[0]], use_omic
            else:
                use_omic = True
                return [features[1]], use_omic
        elif mode == "omic":
            # return only WSIs
            return [features[1]], None
        elif mode == "wsi":
            # return only omic
            return [features[0]], None
    
    def _apply_missing_mode_to_batch(self, features, masks, missing_mode, rng_seed= None, p: float = 0.5, exact_one: bool = False):
        """
        Adaptation to missing sample_missing function
        Apply a missing-modality simulation to a single batch (features, masks)
        Args:
            features: list of length M of tensors or None; each tensor shape (B, n_tokens, channels)
            masks: list of length M of boolean masks (B, n_tokens) or None (may be None)
            missing_mode:
                - None: no change
                - "50": per-sample random dropping with probability p (default p=0.5)
                - "exact_one": drop exactly one randomly chosen modality per sample (if exact_one=True)
                - any modality name in self.sources (e.g., "clinical", "pathology", "radiology"):
                    drop that modality for the whole batch (set features[i]=None, masks[i]=None)
            rng_seed: optional int seed for deterministic behavior
            p: per-modality drop probability for "50" mode
            exact_one: if True and missing_mode == "50", drop exactly one modality per sample instead of independent drops
        Returns:
            (features_mod, masks_mod): modified lists
        """
        if missing_mode is None:
            return features, masks

        M = len(features)
        # map modality name -> index (use self.sources if present)
        try:
            modal_names = list(self.sources)
        except Exception:
            # fallback default order if self.sources missing
            modal_names = ["clinical", "pathology", "radiology"]

        name_to_idx = {n: i for i, n in enumerate(modal_names)}

        rng = np.random.RandomState(int(rng_seed) if rng_seed is not None else int(self.args.seed or 0))

        # shallow-copy lists so we don't mutate inputs unexpectedly
        feats = list(features)
        masks_out = list(masks) if masks is not None else [None] * M

        # helper to ensure we have a mask for a modality: derive from features if None
        def _ensure_mask(i, f):
            if masks_out is None:
                return None
            if masks_out[i] is not None:
                return masks_out[i]
            # derive token-validity mask from feature tensor if present
            if f is None:
                return None
            # f shape expected (B, n_tokens, channels)
            token_valid = (f.abs().sum(dim=-1) > 0).to(dtype=torch.bool, device=f.device)
            masks_out[i] = token_valid
            return masks_out[i]

        # Whole-modality drop by name
        if missing_mode in name_to_idx:
            drop_idx = name_to_idx[missing_mode]
            feats[drop_idx] = None
            masks_out[drop_idx] = None
            return feats, masks_out

        B = None
        # find batch size
        for f in feats:
            if f is not None:
                B = int(f.shape[0])
                break
        if B is None:
            # try masks
            for m in masks_out:
                if m is not None:
                    B = int(m.shape[0])
                    break
        if B is None:
            # nothing we can do
            return feats, masks_out

        # Randomized drop modes
        if missing_mode == "50":
            if exact_one:
                # for each sample choose one modality to drop (only among modalities present)
                for b_idx in range(B):
                    # build list of present modality indices for this sample
                    present_idxs = []
                    for i in range(M):
                        f_i = feats[i]
                        m_i = masks_out[i] if masks_out is not None and len(masks_out) > i else None
                        if f_i is None:
                            continue
                        # derive token validity for this sample row
                        if m_i is not None:
                            present_flag = bool(m_i[b_idx].any().item())
                        else:
                            # sum over channels for this row
                            present_flag = (f_i[b_idx].abs().sum().item() > 0)
                        if present_flag:
                            present_idxs.append(i)
                    if len(present_idxs) == 0:
                        continue
                    drop_i = rng.choice(present_idxs)
                    # drop sample-row: zero-out features row and set mask False if exists
                    if feats[drop_i] is not None:
                        feats[drop_i][b_idx] = torch.zeros_like(feats[drop_i][b_idx])
                    if masks_out[drop_i] is not None:
                        masks_out[drop_i][b_idx] = False
            else:
                # independent per-modality, per-sample drop with prob p
                for i in range(M):
                    f_i = feats[i]
                    if f_i is None:
                        continue
                    # get or build mask
                    m_i = _ensure_mask(i, f_i)
                    n_tokens = f_i.shape[1]
                    for b_idx in range(B):
                        if rng.rand() < p:
                            # drop this modality for this sample (zero row + mask False)
                            f_i[b_idx] = torch.zeros_like(f_i[b_idx])
                            if m_i is not None:
                                m_i[b_idx, :] = False
                            else:
                                # create mask on CPU for consistency
                                if masks_out is not None:
                                    masks_out[i] = torch.zeros((B, n_tokens), dtype=torch.bool, device=self.device)
                                    masks_out[i][b_idx, :] = False

            return feats, masks_out

        # if a different mode is detected: return unchanged
        return feats, masks_out

    def evaluate_survival_epoch(self,
                                epoch: int,
                                model: nn.Module,
                                test_data: DataLoader,
                                missing_mode: str=None,
                                save_preds: bool = False,
                                log_prefix: str = "",
                                # loss_reg: float=0.0,
                                **kwargs):

        model.eval()
        risk_scores = []
        censorships = []
        event_times = []
        predictions = []
        labels = []
        val_loss_surv, val_loss = 0.0, 0.0
        # use_omic = True

        for batch, (features, masks, censorship, event_time, y_disc) in enumerate(tqdm(test_data)):
            if missing_mode is not None: # handle for missing modality ablation
                # features, use_omic = self._sample_missing(features, use_omic, missing_mode)
                features, masks = self._apply_missing_mode_to_batch(features, masks, missing_mode, rng_seed=int(self.args.seed or 0) + batch)

            features = [f.to(self.device) if f is not None else None for f in features]
            masks = [m.to(self.device) if m is not None else None for m in masks]
            censorship = censorship.to(self.device)
            event_time = event_time.to(self.device)
            y_disc = y_disc.to(self.device)

            # determine batch size b robustly
            b = None
            for f in features:
                if f is not None:
                    b = f.shape[0]
                    break
            if b is None and masks is not None:
                for m in masks:
                    if m is not None:
                        b = m.shape[0]
                        break
            if b is None:
                if isinstance(y_disc, torch.Tensor):
                    b = y_disc.shape[0]
                elif isinstance(censorship, torch.Tensor):
                    b = censorship.shape[0]
                else:
                    raise ValueError("Cannot infer batch size in evaluation: all features and masks are None and no scalar targets available.")

            present = []
            for i in range(len(features)):
                fi = features[i]
                mi = masks[i] if masks is not None and len(masks) > i else None

                if fi is None:
                    present.append(torch.zeros(b, dtype=torch.bool, device=self.device))
                    continue

                if mi is not None:
                    present_i = mi.any(dim=1).to(dtype=torch.bool, device=self.device)
                    present.append(present_i)
                    continue

                flat = fi.flatten(start_dim=1)
                present_i = (flat.abs().sum(dim=1) > 0).to(dtype=torch.bool, device=self.device)
                present.append(present_i)
            
            if self.config["model"] == "multimodn":
                model_loss, logits = model.forward(features, F.one_hot(y_disc, num_classes=self.output_dims))
            else:
                if self.config["model"] in ["healnet", "healnet_early"]:
                    logits = model(features, masks=masks, present=present)
                else:
                    logits = model.forward(features)
                model_loss = 0.0

            y_hat = torch.topk(logits, k=1, dim=1)[1].squeeze(1) # y_hat is the bin with the largest hazard (bc sigmoid is monotonic)
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1-hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

            if self.config["survival.loss"] == "nll":
                loss = nll_loss(hazards=hazards, S=survival, Y=y_disc, c=censorship, weights=self.class_weights)
            elif self.config["survival.loss"] == "ce_survival":
                loss_fn = CrossEntropySurvLoss()
                loss = loss_fn(hazards=hazards, survival=survival, y_disc=y_disc, censorship=censorship)
            elif self.config["survival.loss"] == "cox":
                loss_fn = CoxPHSurvLoss()
                loss = loss_fn(hazards=hazards, survival=survival, censorship=censorship)

            reg_loss = calc_reg_loss(model, self.config[f"model_params.l1"], self.config.model, self.config.sources)

            # log risk, censorship and event time for concordance index
            risk_scores.append(risk)
            censorships.append(censorship.detach().cpu().numpy())
            event_times.append(event_time.detach().cpu().numpy())

            loss_value = float(loss.item())
            val_loss_surv += loss_value
            val_loss += loss_value + reg_loss + model_loss

            predictions.append(y_hat.detach().cpu().tolist())
            labels.append(y_disc.detach().cpu().tolist())

        # calculate epoch-level stats
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        val_loss_surv /= len(test_data)
        val_loss /= len(test_data)

        risk_scores_full = np.concatenate(risk_scores)
        censorships_full = np.concatenate(censorships)
        event_times_full = np.concatenate(event_times)

        # ----- SAVE PREDICTIONS ONLY WHEN REQUESTED (e.g., for the test set) -----
        if save_preds:
            try:
                # get dataset-level indices and sample ids where possible
                subset = test_data.dataset 
                if hasattr(subset, "indices"):
                    orig_indices = list(subset.indices)
                    underlying = subset.dataset if hasattr(subset, "dataset") else subset
                else:
                    orig_indices = list(range(len(predictions)))
                    underlying = subset

                # try to get sample ids mapping from the underlying dataset if available
                sample_ids = None
                if hasattr(underlying, "sample_ids"):
                    # map each original index to sample id if lengths align
                    try:
                        sample_ids = [underlying.sample_ids[i] for i in orig_indices]
                    except Exception:
                        sample_ids = None

                N = len(predictions)
                if len(orig_indices) != N:
                    logger.warning("orig_indices len %d != N preds %d. Using sequential indices.", len(orig_indices), N)
                    orig_indices = list(range(N))
                    sample_ids = None

                df_out = pd.DataFrame({
                    "dataset_index": orig_indices,
                    "sample_id": sample_ids if sample_ids is not None else [None] * N,
                    "prediction": predictions.tolist() if predictions.size else [None] * N,
                    "true_label": labels.tolist() if labels.size else [None] * N,
                    "risk": risk_scores_full.tolist() if risk_scores_full.size else [None] * N,
                    "censorship": censorships_full.tolist() if censorships_full.size else [None] * N,
                    "event_time": event_times_full.tolist() if event_times_full.size else [None] * N
                })


                # Ensure a logging directory is available
                if self.log_dir is None:
                    try:
                        run_name = getattr(wandb, "run", None) and getattr(wandb.run, "name", None)
                    except Exception:
                        run_name = None
                    if run_name is None:
                        run_name = self.local_run_id
                    self.log_dir = Path(self.config.get("log_path", ".")).joinpath(run_name)
                    self.log_dir.mkdir(parents=True, exist_ok=True)

                csv_name = self.log_dir.joinpath(f"test_predictions_epoch{epoch}.csv")
                df_out.to_csv(csv_name, index=False)
                if self._wandb_enabled:
                    try:
                        wandb.log({"test/predictions_table": wandb.Table(dataframe=df_out)}, step=epoch)
                        wandb.save(str(csv_name))
                    except Exception:
                        logger.debug("Failed to upload predictions to wandb")
                logger.info("Saved test predictions CSV to %s", csv_name)
            except Exception as e:
                logger.warning("Failed to save test predictions/labels to CSV: %s", e)
        
        if self._wandb_enabled:
            try:
                wandb.log({
                    f"{log_prefix}val/loss_epoch": val_loss_surv, 
                    f"{log_prefix}val/c_index_epoch": concordance_index_censored((1 - censorships_full).astype(bool), event_times_full, risk_scores_full)[0], 
                    f"{log_prefix}epoch": epoch
                }, step=epoch)
            except Exception:
                logger.debug("Failed to log validation epoch metrics to wandb (ignored)")

        model.train()
        return val_loss_surv, concordance_index_censored((1 - censorships_full).astype(bool), event_times_full, risk_scores_full)[0]


    def calc_gradient_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def make_test_model(self, test_data: torch.utils.data.DataLoader):
        batch_example = next(iter(test_data))
        feat = batch_example[0]
        # reuse make_model logic by creating a temporary small config slice
        return self.make_model(test_data)    

    def test_trained_model(self, artifact_path: str = None, fold: int = 1, save_preds: bool = True):
        """
        Instantiate a model compatible with the test set, load weights from artifact_path (file or directory),
        evaluate on the test split and optionally save predictions.

        Args:
            artifact_path: Path to a checkpoint file (best_model_foldX.pt) or a directory containing such a file.
            fold: fold id used only to decide random seeds / logging (not used to change data splits here).
            save_preds: whether to save test predictions CSV via evaluate_survival_epoch.

        Returns:
            model (nn.Module): the model loaded (in eval mode)
            test_loss (float): survival loss on test set
            test_c_index (float): concordance index on test set
        """

        dataset = self.load_data()
        y_disc = dataset.y_disc
        if isinstance(y_disc, torch.Tensor):
            y_disc_np = y_disc.numpy()
        else:
            y_disc_np = np.array(y_disc)

        skf = StratifiedKFold(n_splits=self.config["n_folds"], shuffle=True, random_state=self.args.seed)
        splits = list(skf.split(np.zeros(len(dataset)), y_disc_np))
        
        if fold < 1 or fold > len(splits):
             raise ValueError(f"Fold {fold} out of range (1-{len(splits)})")
        
        train_val_idx, test_idx = splits[fold-1]
        
        # Inner split for standardization
        fold_seed = int(self.args.seed) + int(fold)
        n_total = len(dataset)
        n_train_val = len(train_val_idx)
        val_desired = int(0.15 * n_total)
        
        if val_desired <= 0 or val_desired >= n_train_val:
            val_size = 0.2
        else:
            val_size = val_desired / n_train_val
            
        train_idx, _ = train_test_split(train_val_idx, 
                                        test_size=val_size, 
                                        stratify=y_disc_np[train_val_idx], 
                                        random_state=fold_seed)

        # Apply standardisation from train_idx
        if hasattr(dataset, "compute_standardization_from_indices"):
             dataset.compute_standardization_from_indices(train_idx)
        
        # Create validation/test loader (we only need test_data here)
        collate_fn = self.make_collate_fn()
        test_subset = Subset(dataset, test_idx)
        test_data = DataLoader(
            test_subset,
            batch_size=self.config["train_loop.batch_size"],
            shuffle=False,
            num_workers=int(multiprocessing.cpu_count()),
            pin_memory=True,
            multiprocessing_context=MP_CONTEXT,
            collate_fn=collate_fn
        )
        model = self.make_test_model(test_data)

        if artifact_path is None:
            raise ValueError("artifact_path must be provided")

        p = Path(artifact_path)
        if p.is_dir():
            candidates = list(p.glob("**/*.pt")) + list(p.glob("**/*.pth"))
            if len(candidates) == 0:
                raise FileNotFoundError(f"No .pt/.pth files found under artifact dir {artifact_path}")
            candidates_sorted = sorted(candidates, key=lambda x: ("best" not in x.name, x.name))
            ckpt_path = str(candidates_sorted[0])
        elif p.is_file():
            ckpt_path = str(p)
        else:
            raise FileNotFoundError(f"artifact_path {artifact_path} does not exist")
        
        try:
            state = torch.load(ckpt_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed loading checkpoint {ckpt_path}: {e}")

        state_dict = None
        if isinstance(state, dict):
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state_dict = state["state_dict"]
            elif all(isinstance(v, torch.Tensor) for v in state.values()):
                state_dict = state
            else:
                for key in ("model_state_dict", "model", "net", "state"):
                    if key in state and isinstance(state[key], dict):
                        state_dict = state[key]
                        break
                if state_dict is None:
                    for k, v in state.items():
                        if isinstance(v, dict) and len(v) > 0 and all(hasattr(it, "shape") for it in v.values() if hasattr(it, "__class__")):
                            state_dict = v
                            break
        else:
            raise RuntimeError("Loaded checkpoint is not a dict/state_dict")
        
        if state_dict is None:
            raise RuntimeError("Could not find a state_dict inside the checkpoint. Inspect the checkpoint keys.")


        # fix 'module.' prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith("module."):
                new_key = k.replace("module.", "", 1)
            new_sd[new_key] = v

        # load into model
        try:
            model.load_state_dict(new_sd)
        except RuntimeError as e:
            try:
                model.load_state_dict(new_sd, strict=False)
                logger.warning("Loaded checkpoint %s with strict=False due to: %s", ckpt_path, e)
            except Exception as e2:
                raise RuntimeError(f"Failed to load checkpoint into model: {e2}")

        model.to(self.device)
        model.eval()

        # Evaluate on test set, optionally saving predictions
        logger.info("Running test evaluation with checkpoint %s", ckpt_path)
        test_loss, test_c_index = self.evaluate_survival_epoch(epoch=0, model=model, test_data=test_data, save_preds=save_preds)

        # update wandb summary if enabled
        if self._wandb_enabled:
            try:
                wandb.run.summary["test_loss"] = float(test_loss)
                wandb.run.summary["test_c_index"] = float(test_c_index)
                wandb.run.summary["loaded_checkpoint"] = ckpt_path
            except Exception:
                logger.debug("Failed to update wandb run summary (ignored)")

        return model, test_loss, test_c_index

if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run main training pipeline of healnet")

    # assumes execution
    parser.add_argument("--absolut_path", type=str, default='/nfs/rnas/projects/mmCRC/git/healnet-adoption', help="Path to healnet repo")
    parser.add_argument("--config_path", type=str, default="config/main_gpu.yml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="sweep", choices=["single_run", "sweep", "run_plan", "test_trained_model", "sanity_check"])# single_run for a fixed congig (like config_gpu.yml), sweep for hyperparameter sweep
    parser.add_argument("--sweep_config", type=str, default = 'config/sweep_grid.yaml', help="Hyperparameter sweep configuration") # default="config/sweep_grid_2.yaml"
    parser.add_argument("--project_name", type=str, default="rename", help="wandb project name")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset for run plan")
    parser.add_argument("--datasets", type=list, default=["blca", "brca", "ucec", "kirp", "mmCRC"], help="Datasets for run plan")
    parser.add_argument("--api_key", type=str, default=None, help="wandb api key name")
    parser.add_argument("--artifact_dir", type=str, default=None, help="Path to trained model artifact for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # call config
    args = parser.parse_args()
    args.sweep_config = os.path.join(args.absolut_path, args.sweep_config)
    logger.info("Using sweep config path: %s", args.sweep_config)
    MP_CONTEXT = "fork"
    # set up multiprocessing context for PyTorch
    torch.multiprocessing.set_start_method(MP_CONTEXT)

    config_path = os.path.join(args.absolut_path, args.config_path)
    config = Config(config_path).read()

    if args.dataset is not None: # for command line sweeps
        config["dataset"] = args.dataset

    # get best hyperparameters for dataset
    hp_box = Config(config["hyperparams"]).read()
    dataset_key = str(config.dataset)
    # if the dataset key isn't present, try removing surrounding quotes (e.g. "'mmCRC'" or '"mmCRC"')
    if dataset_key not in hp_box:
        if (dataset_key.startswith("'") and dataset_key.endswith("'")) or (dataset_key.startswith('"') and dataset_key.endswith('"')):
            dataset_key = dataset_key[1:-1]

    if dataset_key not in hp_box:
        raise KeyError(f"Dataset '{config.dataset}' not found in hyperparams file {config['hyperparams']}. Available keys: {list(hp_box.keys())}")

    hyperparams = hp_box[dataset_key]
    config["model_params"] = hyperparams

    if args.mode in ("single_run", "sweep"):
        pipeline = Pipeline(config=config, args=args)
        pipeline.main()
    elif args.mode == "test_trained_model":
        pipeline = Pipeline(config=config, args=args)
        if args.artifact_dir is None:
            raise ValueError("Please provide --artifact_dir pointing to checkpoint file or artifact download dir")
        model, test_loss, test_c_index = pipeline.test_trained_model(artifact_path=args.artifact_dir, fold=1, save_preds=True)
        logger.info("Test c-index: %.4f, loss: %.4f", test_c_index, test_loss)

    elif args.mode == "sanity_check":
        pipeline = Pipeline(config=config, args=args)
        from healnet.etl.loaders import SyntheticMultiModalSurvival

        synth_dims = (30, 700, 512)
        n_bins = int(config["model_params"]["output_dims"]) if "output_dims" in config["model_params"] else int(pipeline.output_dims)

        data = SyntheticMultiModalSurvival(n_samples=2000, dims=synth_dims, n_bins=n_bins, seed=args.seed, censoring_prob=0.3)
        n = len(data)

        train_frac, val_frac, test_frac = 0.7, 0.15, 0.15
        train_n = int(train_frac * n)
        val_n = int(val_frac * n)
        test_n = n - train_n - val_n

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        train_subset, val_subset, test_subset = torch.utils.data.random_split(data, [train_n, val_n, test_n], generator=generator)

        collate_fn = pipeline.make_collate_fn()
        batch_size = int(pipeline.config.get("train_loop.batch_size", 4))

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

        logger.info("Sanity check: train/val/test sizes: %d/%d/%d", len(train_subset), len(val_subset), len(test_subset))

        batch = next(iter(train_loader))
        mods, masks, cens, times, y_disc = batch
        logger.debug("Batch modality shapes: %s", [m.shape if m is not None else None for m in mods])
        logger.debug("Censorship unique: %s", torch.unique(cens, return_counts=True))
        logger.debug("y_disc unique counts: %s", np.unique(y_disc.numpy(), return_counts=True))

        model = pipeline.make_model(train_loader)
        init_norm = sum(p.detach().cpu().norm().item() for p in model.parameters())
        logger.info("Initial param norm: %0.6f", init_norm)

        model, train_loss, train_c_index, val_loss, val_c_index, test_loss, test_c_index, missing_performance = pipeline.train_survival_fold(
            model=model, train_data=train_loader, val_data=val_loader, test_data=test_loader, fold=1
        )

        logger.info("Sanity check finished. Train c-index: %.4f, Val c-index: %.4f, Test c-index: %.4f", train_c_index, val_c_index, test_c_index)

    