"""
Adapted Explainer for MMDataset + tabular modalities (clinical, pathology, radiology).

This script was rewritten to:
- use the MMDataset loader (multimodal tabular inputs)
- work with arbitrary tabular modalities defined in `config.sources`
- provide per-modality feature attributions using simple gradient saliency (gradient * input magnitude)
  as a robust fallback for modalities that are represented as a single token (most tabular modalities).
- still attempt to surface attention-based importance when the model's attention has >1 context tokens
  for a modality (e.g., patch-based WSI inputs).
- save per-sample PNGs in the explanations/{run_name} folder.

Usage:
- Ensure Pipeline wrote config.pkl and best_model.pt into the run log folder (explainer expects them).
- Instantiate Explainer with the run log folder: e = Explainer(log_dir="logs/your-run", show=False)
- Run explanations: e.run(n_samples=5)

This file replaces the previous TCGA-specific explainer and focuses on tabular multimodal datasets
served by MMDataset (clinical/pathology/radiology).

"""
from pathlib import Path
from box import Box
import random
# from torch.utils.data import DataLoader
import sys
import os

HEALNET_ADOPT_PATH = os.environ.get("HEALNET_ADOPT_PATH")
if not HEALNET_ADOPT_PATH:
    # default to repository root relative to this file if env var is missing/empty
    HEALNET_ADOPT_PATH = str(Path(__file__).resolve().parents[2])
print("Initial HEALNET_ADOPT_PATH =", HEALNET_ADOPT_PATH)

# ensure repo root is first on sys.path
if HEALNET_ADOPT_PATH not in sys.path:
    sys.path.insert(0, HEALNET_ADOPT_PATH)

import importlib
import importlib.util
importlib.invalidate_caches()

# If a conflicting "healnet" module exists and is not a package, remove it so
# we can force-load the repository package (prevents "'healnet' is not a package").
if 'healnet' in sys.modules:
    mod = sys.modules['healnet']
    mod_file = getattr(mod, '__file__', None)
    if mod_file and not Path(mod_file).is_dir():
        del sys.modules['healnet']

# Prefer loading the local `healnet` package from the repository to avoid
# shadowing by other installed modules named `healnet`.
pkg_init = Path(HEALNET_ADOPT_PATH) / 'healnet' / '__init__.py'
if pkg_init.exists():
    spec = importlib.util.spec_from_file_location('healnet', str(pkg_init))
    if spec and spec.loader:
        healnet_pkg = importlib.util.module_from_spec(spec)
        sys.modules['healnet'] = healnet_pkg
        spec.loader.exec_module(healnet_pkg)

from healnet.etl import MMDataset
import os

print("HEALNET_ADOPT_PATH =", HEALNET_ADOPT_PATH)
print("MMDataset loaded from:", getattr(MMDataset, "__file__", getattr(MMDataset, "__module__", "unknown")))

from copy import deepcopy
from healnet.models import HealNet
from healnet.utils import unpickle
import seaborn as sns
import numpy as np
import torch
from typing import *
import h5py
import matplotlib.pyplot as plt
import pandas as pd

class Explainer(object):
    def __init__(self, log_dir: str, show=False):
        self.log_dir = Path(log_dir)
        # run name
        self.show = show
        self.expl_dir = Path(f"{self.log_dir}/explanations/{self.log_dir.name}")
        self.expl_dir.mkdir(parents=True, exist_ok=True)

        # load saved pipeline config and test indices
        self.config = unpickle(self.log_dir.joinpath("config.pkl"))
        print("CONFIG",self.config)
        self.dataset = self.config.dataset
        self.test_data_indices = unpickle(self.log_dir.joinpath("test_data_indices.pkl"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Initializing dataset...")
        # load dataset (use same sources as in config)
        self.sources = list(self.config.get("sources", []))
        self.data = MMDataset(
            config=self.config,
            filter_overlap=self.config.get("filter_overlap", False),
            survival_analysis=True,
            sources=self.sources,
            n_bins=int(self.config.get("model_params", {}).get("output_dims", 4)),
            log_dir=None,
        )

        print("Loading model...")
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

        # gather feature name lists for modalities if present
        self.feature_cols = {}
        if hasattr(self.data, "clinical_feature_cols"):
            self.feature_cols["clinical"] = self.data.clinical_feature_cols
        if hasattr(self.data, "pathology_feature_cols"):
            self.feature_cols["pathology"] = self.data.pathology_feature_cols
        if hasattr(self.data, "radiology_feature_cols"):
            self.feature_cols["radiology"] = self.data.radiology_feature_cols

    def _load_model(self) -> torch.nn.Module:
        """
        Reloads HealNet from best_model.pt saved in the run directory.
        Falls back to state_dict keys heuristics similar to Pipeline.test_trained_model.
        """
        ckpt_path = self.log_dir.joinpath("best_model.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Could not find best_model.pt in {self.log_dir}")

        state = torch.load(str(ckpt_path), map_location="cpu")

        # infer a state_dict from loaded checkpoint
        state_dict = None
        if isinstance(state, dict):
            candidates = ("state_dict", "model_state_dict", "model", "net", "state")
            for key in candidates:
                if key in state and isinstance(state[key], dict):
                    state_dict = state[key]
                    break
            if state_dict is None:
                # maybe checkpoint is already a state dict mapping to tensors
                if all(isinstance(v, torch.Tensor) for v in state.values()):
                    state_dict = state
        if state_dict is None:
            # last resort: assume file contains raw state_dict-like structure
            raise RuntimeError("Could not infer state_dict from checkpoint at %s" % str(ckpt_path))

        # clean module. prefixes
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            if k.startswith("module."):
                nk = k.replace("module.", "", 1)
            new_sd[nk] = v

        latent_size = new_sd['latents'].shape
        latent_dim_ld = latent_size[1] 

        print(f"Inferred Latent Dimension (l_d) from weights: {latent_dim_ld}")

        # create model with same architecture config saved in pipeline config
        # build input shapes from sample
        sample_feats, _, _, _ = self.data[0]
        num_sources = len(sample_feats)
        input_channels = []
        input_axes = []
        modalities = num_sources

        for i, f in enumerate(sample_feats):
            if f is None:
                # fallback: try dataset metadata
                mod_name = self.data.modalities_list[i] if hasattr(self.data, "modalities_list") else self.sources[i]
                input_channels.append(int(self.data.feature_dims.get(mod_name, 1)))
                input_axes.append(1)
            else:
                # f shape expected (1, tokens, channels) or (1, tokens) or (1, D)
                ts = list(f.shape)
                # if f is (1, D) treat D as channels and tokens=1
                if len(ts) == 2:
                    tokens = 1
                    channels = int(ts[1])
                else:
                    tokens = int(ts[1])
                    channels = int(ts[-1])
                input_channels.append(channels)
                input_axes.append(1)  # tabular modalities -> 1 spatial axis

        model = HealNet(
            n_modalities=modalities,
            channel_dims=input_channels,
            num_spatial_axes=input_axes,
            out_dims = int(self.config["model_params.output_dims"]),
            num_freq_bands = int(self.config["model_params.num_freq_bands"]),
            depth = int(self.config["model_params.depth"]),
            max_freq = float(self.config["model_params.max_freq"]),
            l_c=int(self.config["model_params.num_latents"]),
            l_d=int(self.config["model_params.latent_dim"]),
            cross_dim_head = int(self.config["model_params.cross_dim_head"]),
            latent_dim_head=int(self.config["model_params.latent_dim_head"]),
            x_heads = int(self.config["model_params.cross_heads"]),     # cross attention heads
            l_heads = int(self.config["model_params.latent_heads"]),    # latent/self heads
            attn_dropout = float(self.config["model_params.attn_dropout"]),
            ff_dropout = float(self.config["model_params.ff_dropout"]),
            weight_tie_layers = bool(self.config["model_params.weight_tie_layers"]),
            fourier_encode_data = bool(self.config["model_params.fourier_encode_data"]),
            self_per_cross_attn = int(self.config["model_params.self_per_cross_attn"]),
            final_classifier_head = True,
            snn = bool(self.config["model_params.snn"])
        )

        print("num_freq_bands", int(self.config["model_params.num_freq_bands"]))
        try:
            model.load_state_dict(new_sd, strict=True)
        except Exception:
            # try relaxed load
            # model.load_state_dict(new_sd, strict=False)
            print("Failed to load state_dict into model from", str(ckpt_path))

        return model

    def run(self, n_samples: int = 5, random_state: int = 0):
        """
        Run explanations for n_samples drawn from the test set (or entire dataset if no test indices).
        For each sample, compute:
         - attention-based importances if attention has >1 context tokens for a modality
         - gradient-saliency based importances per modality (gradient magnitude across channels)
        Outputs per-sample PNGs into explanations/<run>/
        """
        if self.test_data_indices:
            indices = list(self.test_data_indices)
        else:
            indices = list(range(len(self.data)))

        rng = np.random.RandomState(int(random_state))
        chosen = rng.choice(indices, size=min(n_samples, len(indices)), replace=False)

        for idx in chosen:
            try:
                self.explain_sample(idx)
            except Exception as e:
                print(f"[Explainer] Failed to explain sample {idx}: {e}")

    def explain_sample(self, idx: int):
        """
        Explain a single dataset index. Saves plots to disk.
        """
        
        mods, censorship, event_time, y_disc = self.data[idx]
        # mods: list of tensors or None, each tensor typically shape (1, tokens, channels) or (1, D)
        # Move to device and ensure batch dimension present
        inputs = []
        for m in mods:
            if m is None:
                inputs.append(None)
            else:
                t = m.to(self.device).float()
                # ensure 3D shape: (b, tokens, channels)
                # 1. ALWAYS ensure a batch dimension (index 0)
                # Most datasets return (D,) or (Tokens, Channels)
                if t.ndim == 1: # (D,) -> (1, D)
                    t = t.unsqueeze(0)
                elif t.ndim >= 2 and t.shape[0] != 1: # (Tokens, Channels) -> (1, Tokens, Channels)
                    t = t.unsqueeze(0)
                    
                # 2. Ensure spatial dimensions match the model's expectations
                # HealNet expects (b, *spatial, channels). For tabular (axes=1), it needs 3D.
                # If after adding batch it's still 2D (1, D), add the token dimension.
                if t.ndim == 2: 
                    t = t.unsqueeze(1) # (1, D) -> (1, 1, D)
                inputs.append(t)

        # Forward once (no grads) to capture any attention maps if available
        with torch.no_grad():
            logits = self.model(list(inputs))
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

        predicted_bin = int(torch.topk(logits, k=1, dim=1)[1].squeeze(1).detach().cpu().item())

        sample_name = f"idx{idx}_predbin{predicted_bin}_risk{risk[0]:.4f}"
        print(f"[Explainer] Explaining sample {idx} -> {sample_name}")

        # Attempt to get attention-based maps when context token count > 1 for modalities
        attn_weights = self.model.get_attention_weights()
        if attn_weights:
            # attn_weights is list of tensors shaped (b*h, n_q, n_k)
            # Collect for each modality those weights where n_k equals that modality's token count
            token_counts = []
            for t in inputs:
                if t is None:
                    token_counts.append(0)
                else:
                    try:
                        token_counts.append(int(t.shape[1]))
                    except Exception:
                        print("Failed to get token count for input:", t)

            modality_attn = {m: [] for m in range(len(inputs))}
            for w in attn_weights:
                if w is None: continue
                try:
                    # w shape: (b*h, n_q, n_k)
                    n_k = int(w.shape[2])
                except Exception:
                    print("Failed to get n_k from attention weight:", w)
                    continue
                # find matching modality indices
                for mi, tc in enumerate(token_counts):
                    if tc > 1 and n_k == tc:
                        modality_attn[mi].append(w.detach().cpu())

            # For modalities with collected attention layers, plot heatmap or aggregate
            for mi, wlist in modality_attn.items():
                if len(wlist) == 0:
                    continue
                # aggregate attention: mean over layers, heads, and queries
                # convert to shape (b, h, n_q, n_k) by reshaping (b*h, n_q, n_k)
                agg = torch.stack(wlist).mean(dim=0)  # mean layers
                # reshape to (b, h, n_q, n_k)
                b_h, n_q, n_k = agg.shape
                # attempt to infer heads from model config: we can't easily recover h reliably here,
                # so collapse batch-head dimension
                agg = agg.mean(dim=0)  # mean over (b*h) -> (n_q, n_k)
                attn_map = agg.mean(axis=0)  # mean over queries -> (n_k,)
                # save numpy
                arr = attn_map.numpy()
                # simple visualization: barplot over tokens
                plt.figure(figsize=(8, 3))
                plt.bar(np.arange(len(arr)), arr, color="C1")
                mod_name = self.sources[mi] if mi < len(self.sources) else f"mod{mi}"
                plt.title(f"{sample_name} - Attention tokens for modality {mod_name}")
                plt.xlabel("token idx")
                plt.ylabel("attention (avg)")
                fname = self.expl_dir.joinpath(f"{sample_name}_attn_mod{mi}_{mod_name}.png")
                plt.savefig(fname, bbox_inches="tight", dpi=200)
                if self.show:
                    plt.show()
                plt.close()

        # Gradient-saliency based per-modality feature importance
        # Build clones of inputs with requires_grad=True
        inputs_grad = []
        for t in inputs:
            if t is None:
                inputs_grad.append(None)
            else:
                tg = t.detach().clone().to(self.device).float()
                tg.requires_grad_(True)
                inputs_grad.append(tg)

        # Forward and compute scalar risk for backprop
        logits = self.model(inputs_grad)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        # risk scalar: negative sum of survival (standard used in pipeline)
        risk_tensor = -torch.sum(survival, dim=1)  # shape (b,)
        score = risk_tensor[0]  # single-sample

        # Backprop
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # For each modality, compute importance = sum_abs_grad over tokens -> per-channel importance
        for mi, tg in enumerate(inputs_grad):
            if tg is None:
                continue
            grad = tg.grad  # shape (1, tokens, channels)
            if grad is None:
                continue
            # aggregate across tokens -> (channels,)
            abs_imp = grad.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
            # optionally weight by input magnitude (gradient * input)
            inp = tg.detach().cpu().numpy().squeeze(0)  # (tokens, channels)
            inp_mag = np.abs(inp).sum(axis=0)
            # grad * input magnitude (elementwise)
            importance = abs_imp * (inp_mag + 1e-8)
            # normalize for plotting
            if importance.max() > 0:
                importance = importance / float(np.max(importance))
            else:
                importance = importance

            mod_name = self.sources[mi] if mi < len(self.sources) else f"mod{mi}"
            feature_names = self.feature_cols.get(mod_name, None)

            # If channels count matches a stored feature name list, use them, else fall back to numeric indices
            try:
                channels = importance.shape[0]

            except Exception:
                print("Failed to get channels from importance array:", importance)
                continue
            
            if feature_names and len(feature_names) == channels:
                names = feature_names
            else:
                names = [f"f_{i}" for i in range(channels)]

            # Build pandas series and plot top-k
            df = pd.DataFrame({"feature": names, "importance": importance})
            df = df.sort_values("importance", ascending=False).head(40)

            plt.figure(figsize=(6, max(3, 0.2 * len(df))))
            sns.barplot(data=df, x="importance", y="feature", palette="viridis")
            plt.title(f"{sample_name} - Gradient importance - {mod_name}")
            plt.xlabel("normalized importance")
            plt.tight_layout()
            fname = self.expl_dir.joinpath(f"{sample_name}_gradimp_mod{mi}_{mod_name}.png")
            plt.savefig(fname, dpi=200)
            if self.show:
                plt.show()
            plt.close()

        # Save a small CSV with per-modality importance arrays for potential downstream inspection
        save_dict = {}
        for mi, tg in enumerate(inputs_grad):
            if tg is None or tg.grad is None:
                continue

            # Calculate importances
            grad = tg.grad.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
            inp = tg.detach().cpu().numpy().squeeze(0)
            imp = (grad * (np.abs(inp).sum(axis=0) + 1e-8))
            # Normalising
            if imp.max() > 0:
                imp = imp / float(imp.max())

            # Getting the name of the features:
            mod_name = self.sources[mi] if mi < len(self.sources) else f"mod{mi}"
            feature_names = self.feature_cols.get(mod_name, None)

            channels = imp.shape[0]
            if feature_names and len(feature_names) == channels:
                names = feature_names
            else:
                names= [f"f_{i}" for i in range(channels)]

            # Saving in the csv both the feature names and the importances
            save_dict[f"{mod_name}_feature"] = names        
            save_dict[f"{mod_name}_importance"] = imp.tolist()

        if save_dict:
            csv_path = self.expl_dir.joinpath(f"{sample_name}_importances.csv")
            pd.DataFrame(dict([(k, pd.Series(v)) for k, v in save_dict.items()])).to_csv(csv_path, index=False)
            print(f"[Explainer] Saved importances CSV to {csv_path}")

        print(f"[Explainer] Saved explanations for sample {idx} to {self.expl_dir}")

if __name__ == "__main__":
    # quick CLI: pass path to log dir
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Path to run log directory containing config.pkl and best_model.pt")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to explain")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    args = parser.parse_args()

    expl = Explainer(args.log_dir, show=args.show)
    print([ (i, [None if m is None else tuple(m.shape) for m in expl.data[i][0]]) for i in range(5)])
    print('feature_dims', getattr(expl.data,'feature_dims',None))
    expl.run(n_samples=args.n)