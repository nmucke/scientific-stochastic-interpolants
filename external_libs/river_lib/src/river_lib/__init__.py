"""Frozen RIVER VQGAN autoencoder for 64x64 KTH frames (see README.md for origin and license)."""

import torch

from .autoencoder import VQModelInterface


class _LightningCallbackKey:
    """Stand-in for the one pytorch-lightning class the checkpoint pickles (as a dict key under "callbacks")."""


# Allow-listing the stand-in under the pickled name lets `torch.load(..., weights_only=True)` read the file
# without pytorch-lightning installed and without unpickling arbitrary code.
_SAFE_GLOBALS = [
    (
        _LightningCallbackKey,
        "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
    )
]


class KTHAutoencoder(torch.nn.Module):
    """Frozen RIVER VQGAN for 64x64 KTH frames."""

    def __init__(self, ckpt_path: str = "data/KTH/vqvae.ckpt") -> None:
        super().__init__()
        self.ae = VQModelInterface()

        with torch.serialization.safe_globals(_SAFE_GLOBALS):
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)[
                "state_dict"
            ]
        # The checkpoint also holds the training loss (LPIPS + discriminator), which is not needed here.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss.")}
        self.ae.load_state_dict(state_dict, strict=True)

        self.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [N, 1, 64, 64] float in [0, 1] -> latents [N, 4, 8, 8] (continuous, as used by RIVER)."""
        # RIVER feeds grayscale KTH as RGB with three identical channels, scaled to [-1, 1].
        rgb = (2.0 * frames - 1.0).expand(-1, 3, -1, -1)
        return self.ae.encode(rgb)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents [N, 4, 8, 8] -> frames [N, 1, 64, 64] float in [0, 1] (clamped)."""
        # As in RIVER's generation, the latents are snapped to the codebook before decoding.
        rgb = self.ae.decode(latents)
        return ((rgb.mean(dim=1, keepdim=True) + 1.0) / 2.0).clamp(0.0, 1.0)
