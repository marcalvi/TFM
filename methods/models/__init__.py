from .mlp import MultimodalMLP
from .dyam import DyAM
from .distill_dyam import DistillDyAM
_HEALNET_IMPORT_ERROR = None
try:
    from .healnet import HealNet, HealNetBinaryWrapper
except Exception as _healnet_import_error:
    _HEALNET_IMPORT_ERROR = _healnet_import_error
    HealNet = None

    class HealNetBinaryWrapper:  # pragma: no cover - import fallback
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "HealNet dependencies are not available. "
                "Install missing packages (e.g. einops) to use model='HealNet'."
            ) from _HEALNET_IMPORT_ERROR

__all__ = [
    "MultimodalMLP",
    "DyAM",
    "DistillDyAM",
    "HealNet",
    "HealNetBinaryWrapper",
]
