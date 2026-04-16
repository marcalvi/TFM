from .preprocess_dataset import load_or_preprocess_dataset

try:
    from .dataset import (
        MissingModalitySimulator,
        MultimodalBaseDataset,
        MultimodalDatasetWithMissing,
        HealNetMaskAwareBatchSampler,
        multimodal_collate,
        build_loaders,
    )
except ModuleNotFoundError as exc:
    if getattr(exc, "name", None) != "torch":
        raise
    MissingModalitySimulator = None
    MultimodalBaseDataset = None
    MultimodalDatasetWithMissing = None
    HealNetMaskAwareBatchSampler = None
    multimodal_collate = None
    build_loaders = None

__all__ = [
    "MissingModalitySimulator",
    "MultimodalBaseDataset",
    "MultimodalDatasetWithMissing",
    "HealNetMaskAwareBatchSampler",
    "multimodal_collate",
    "build_loaders",
    "load_or_preprocess_dataset",
]
