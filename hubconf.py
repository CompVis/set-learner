"""PyTorch Hub entry points for VICIS."""

from __future__ import annotations

from huggingface_hub import hf_hub_download
from torch import bfloat16
from torch.hub import get_dir

dependencies = [
    "diffusers",
    "einops",
    "huggingface_hub",
    "torch",
    "torchvision",
]

_HF_REPO_ID = "CompVis/set-learner"
_WEIGHTS_FILE = "model.pt"


def vicis(pretrained: bool = True, device="cuda", dtype=bfloat16, **kwargs):
    """Load the fixed VICIS model from the ECCV 2026 paper.

    Args:
        pretrained: Load the published weights from Hugging Face.
        device: Device on which the returned model is placed.
        dtype: Parameter dtype of the returned model.
        **kwargs: Reserved for forward-compatible Hub calls. Unknown arguments
            are rejected to avoid silently changing the fixed architecture.
    """
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected VICIS arguments: {unknown}")

    from vicis.model import build_model, load_model

    if not pretrained:
        model = build_model().to(device=device, dtype=dtype)
        model.direction_in_proj.float()
        return model.eval()

    checkpoint = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_WEIGHTS_FILE,
        cache_dir=get_dir(),
    )
    return load_model(checkpoint, device=device, dtype=dtype)
