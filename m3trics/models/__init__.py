from .mlp import MultimodalMLP
from .di_mmlp import DiMMLP
from .pam import PAM
from .dipam import DiPAM
from .smil_e import SMILE, learn_priors, smile_alignment_loss, meta_train_step
_HEALNET_IMPORT_ERROR = None
try:
    from .healnet_wrapper import HealNet, HealNetBinaryWrapper
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
    "DiMMLP",
    "PAM",
    "DiPAM",
    "SMILE",
    "learn_priors",
    "smile_alignment_loss",
    "meta_train_step",
    "HealNet",
    "HealNetBinaryWrapper",
]
