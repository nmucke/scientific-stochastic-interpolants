# Vendored from RIVER (https://github.com/Araachie/river, commit 9ecde339e4c19713485fc1ab31bcb524d6772f6e,
# GPL-3.0), files model/vqgan/taming/autoencoder.py and model/vqgan/taming/quantize.py, which themselves derive
# from taming-transformers (https://github.com/CompVis/taming-transformers, MIT).
#
# Trimmed to inference with the `f8_small` config that RIVER's configs/kth.yaml selects: no losses, no
# discriminator, no pytorch-lightning, no index remapping. Module and parameter names are unchanged so the
# RIVER checkpoint loads as is.
import torch
import torch.nn as nn

from .modules import Decoder, Encoder

# RIVER: vq_f8_small_ddconfig (64x64 RGB image -> 4x8x8 latent, 16384 codebook entries of dimension 4)
EMBED_DIM = 4
N_EMBED = 16384
Z_CHANNELS = 4
DDCONFIG = dict(
    ch=128, ch_mult=[1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], resolution=64
)


class VectorQuantizer(nn.Module):
    """Nearest-codebook-entry lookup (inference part of RIVER/taming `VectorQuantizer2`)."""

    def __init__(self, n_e, e_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_e, e_dim)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1)  # b h w c
        z_flat = z.reshape(-1, z.shape[-1])
        e = self.embedding.weight
        # squared distances (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flat**2).sum(dim=1, keepdim=True) + (e**2).sum(dim=1) - 2 * z_flat @ e.T
        z_q = self.embedding(d.argmin(dim=1)).view(z.shape)
        return z_q.permute(0, 3, 1, 2).contiguous()


class VQModelInterface(nn.Module):
    """RIVER's `VQModelInterface`: continuous latents out of `encode`, quantization inside `decode`."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_channels=3, z_channels=Z_CHANNELS, **DDCONFIG)
        self.decoder = Decoder(out_ch=3, z_channels=Z_CHANNELS, **DDCONFIG)
        self.quantize = VectorQuantizer(N_EMBED, EMBED_DIM)
        self.quant_conv = nn.Conv2d(Z_CHANNELS, EMBED_DIM, 1)
        self.post_quant_conv = nn.Conv2d(EMBED_DIM, Z_CHANNELS, 1)

    def encode(self, x):
        return self.quant_conv(self.encoder(x))

    def decode(self, h):
        # also go through quantization layer
        return self.decoder(self.post_quant_conv(self.quantize(h)))
