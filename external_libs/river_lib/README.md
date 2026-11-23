# river_lib

The pretrained VQGAN autoencoder that [RIVER](https://github.com/Araachie/river) (Davtyan et al., ICCV 2023) uses
for 64x64 KTH, reduced to a small inference-only package that depends on `torch` alone. It is the latent space
used for the KTH experiment of "Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes"
(arXiv 2403.13724).

```python
from river_lib import KTHAutoencoder

ae = KTHAutoencoder("data/KTH/vqvae.ckpt")   # frozen, eval mode
z = ae.encode(frames)                        # [N, 1, 64, 64] in [0, 1] -> [N, 4, 8, 8]
x = ae.decode(z)                             # [N, 4, 8, 8] -> [N, 1, 64, 64] in [0, 1]
```

## Weights

    curl -L -o data/KTH/vqvae.ckpt https://huggingface.co/cvg-unibe/river_kth_64/resolve/main/vqvae.ckpt

(about 860 MB; it is a pytorch-lightning checkpoint that also contains optimizer state and the training loss.
Only `state_dict` is read, with `weights_only=True`; the 55 `loss.*` entries — LPIPS and discriminator — are
dropped, every other entry is loaded with `strict=True`.)

## Conventions (taken from the RIVER code)

- Input: RIVER converts the KTH frames to RGB (`dataset/convert_to_h5.py`, three identical channels) and scales
  them as `x / 127.5 - 1` (`dataset/video_dataset.py`). `KTHAutoencoder.encode` therefore maps `[0, 1]` gray
  frames to `[-1, 1]` and replicates the channel; `decode` averages the three output channels and maps back.
- Latents: `encode` returns `quant_conv(encoder(x))`, the continuous pre-quantization latent, with no rescaling
  (`model/model.py` trains the flow directly on these).
- Decoding: `decode` snaps the latents to the nearest codebook entry before decoding, as RIVER's
  `Model.generate_frames` does.

## Origin and license

- `src/river_lib/modules.py`, `src/river_lib/autoencoder.py`: vendored and trimmed from RIVER, commit
  `9ecde339e4c19713485fc1ab31bcb524d6772f6e`, files `model/vqgan/taming/{modules,autoencoder,quantize}.py`
  (config `vq_f8_small_ddconfig`, class `VQModelInterface`). RIVER is licensed under GPL-3.0; its license text
  is in `LICENSE`, and this package is distributed under the same license.
- Those RIVER files derive from [taming-transformers](https://github.com/CompVis/taming-transformers)
  (Esser, Rombach, Ommer), which is MIT licensed.

Removed relative to RIVER: pytorch-lightning, losses and discriminator, training code, the other autoencoder
configs and variants, codebook index remapping, and code paths that the `f8_small` config never takes (timestep
embedding, non-convolutional resampling). Module and parameter names are unchanged.
