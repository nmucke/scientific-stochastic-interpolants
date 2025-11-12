import contextlib
import dataclasses
import pdb
from datetime import timedelta

import torch
import torch.nn as nn
from aurora import AuroraSmallPretrained, Batch
from aurora.model.fourier import lead_time_expansion
from aurora.model.swin3d import Swin3DTransformerBackbone


class Swin3DTransformerBackboneWrapper(nn.Module):
    def __init__(self, backbone: Swin3DTransformerBackbone):
        super().__init__()

        self.backbone = backbone

        # MLP to expand the pseudo-time to the same dimension as the lead time.
        self.pseudo_time_mlp = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        lead_time: timedelta,
        pseudo_time: torch.Tensor,
        rollout_step: int,
        patch_res: tuple[int, int, int],
    ) -> torch.Tensor:
        """Run the backbone.

        Args:
            x (torch.Tensor): Input tokens of shape `(B, L, D)`.
            lead_time (datetime.timedelta): Lead time.
            pseudo_time (torch.Tensor): Pseudo-time.
            rollout_step (int): Roll-out step.
            patch_res (tuple[int, int, int]): Patch resolution of the form `(C, H, W)`.

        Returns:
            torch.Tensor: Output tokens of shape `(B, L, D)`.
        """
        _msg = "Input shape does not match patch size."
        assert x.shape[1] == patch_res[0] * patch_res[1] * patch_res[2], _msg

        # It's costly to pad across the level dimension, so we should not even though our model
        # supports it.
        _msg = f"Patch height ({patch_res[0]}) must be divisible by ws[0] ({self.backbone.window_size[0]})"
        assert patch_res[0] % self.backbone.window_size[0] == 0, _msg

        all_enc_res, padded_outs = self.backbone.get_encoder_specs(patch_res)

        lead_hours = lead_time / timedelta(hours=1)
        lead_times = lead_hours * torch.ones(
            x.shape[0], dtype=torch.float32, device=x.device
        )

        c = self.backbone.time_mlp(
            lead_time_expansion(lead_times, self.backbone.embed_dim).to(dtype=x.dtype)
        )

        # Expand the pseudo-time to the same dimension as the lead time.
        pseudo_time = self.pseudo_time_mlp(
            lead_time_expansion(pseudo_time, self.backbone.embed_dim).to(dtype=x.dtype)
        )

        c = c + pseudo_time

        skips = []
        for i, layer in enumerate(self.backbone.encoder_layers):
            x, x_unscaled = layer(x, c, all_enc_res[i], rollout_step=rollout_step)
            skips.append(x_unscaled)
        for i, layer in enumerate(self.backbone.decoder_layers):
            index = self.backbone.num_decoder_layers - i - 1
            x, _ = layer(
                x,
                c,
                all_enc_res[index],
                padded_outs[index - 1],
                rollout_step=rollout_step,
            )

            if 0 < i < self.backbone.num_decoder_layers - 1:
                # For the intermediate stages, we use additive skip connections.
                x = x + skips[index - 1]
            elif i == self.backbone.num_decoder_layers - 1:
                # For the last stage, we perform concatentation like in Pangu.
                x = torch.cat([x, skips[0]], dim=-1)
        return x


class AuroraModelWrapper(AuroraSmallPretrained):
    def __init__(self) -> None:
        super().__init__()

        self.load_checkpoint()

        self.backbone: Swin3DTransformerBackboneWrapper = (
            Swin3DTransformerBackboneWrapper(self.backbone)
        )

    def forward(self, batch: Batch, pseudo_time: float) -> Batch:
        """Forward pass.

        Args:
            batch (:class:`Batch`): Batch to run the model on.

        Returns:
            :class:`Batch`: Prediction for the batch.
        """
        batch = self.batch_transform_hook(batch)

        # Get the first parameter. We'll derive the data type and device from this parameter.
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)
        batch = batch.crop(patch_size=self.patch_size)
        batch = batch.to(p.device)

        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        # Insert batch and history dimension for static variables.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={
                k: v[None, None].repeat(B, T, 1, 1)
                for k, v in batch.static_vars.items()
            },
        )

        # Apply some transformations before feeding `batch` to the encoder. We'll later want to
        # refer to the original batch too, so rename the variable.
        transformed_batch = batch

        # Clamp positive variables.
        if self.positive_surf_vars:
            transformed_batch = dataclasses.replace(
                transformed_batch,
                surf_vars={
                    k: v.clamp(min=0) if k in self.positive_surf_vars else v
                    for k, v in batch.surf_vars.items()
                },
            )
        if self.positive_atmos_vars:
            transformed_batch = dataclasses.replace(
                transformed_batch,
                atmos_vars={
                    k: v.clamp(min=0) if k in self.positive_atmos_vars else v
                    for k, v in batch.atmos_vars.items()
                },
            )

        transformed_batch = self._pre_encoder_hook(transformed_batch)

        # The encoder is always just run.
        x = self.encoder(
            transformed_batch,
            lead_time=self.timestep,
        )

        if self.autocast:
            if torch.cuda.is_available():
                device_type = "cuda"
            elif torch.xpu.is_available():
                device_type = "xpu"
            else:
                device_type = "cpu"
            context = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        else:
            context = contextlib.nullcontext()
        with context:
            x = self.backbone(
                x,
                lead_time=self.timestep,
                pseudo_time=pseudo_time,
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
            )

        pred = self.decoder(
            x,
            batch,
            lead_time=self.timestep,
            patch_res=patch_res,
        )

        # Remove batch and history dimension from static variables.
        pred = dataclasses.replace(
            pred,
            static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
        )

        # Insert history dimension in prediction. The time should already be right.
        pred = dataclasses.replace(
            pred,
            surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
            atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
        )

        pred = self._post_decoder_hook(batch, pred)

        # Clamp positive variables.
        clamp_at_rollout_step = (
            pred.metadata.rollout_step >= 1
            if self.clamp_at_first_step
            else pred.metadata.rollout_step > 1
        )
        if self.positive_surf_vars and clamp_at_rollout_step:
            pred = dataclasses.replace(
                pred,
                surf_vars={
                    k: v.clamp(min=0) if k in self.positive_surf_vars else v
                    for k, v in pred.surf_vars.items()
                },
            )
        if self.positive_atmos_vars and clamp_at_rollout_step:
            pred = dataclasses.replace(
                pred,
                atmos_vars={
                    k: v.clamp(min=0) if k in self.positive_atmos_vars else v
                    for k, v in pred.atmos_vars.items()
                },
            )

        pred = pred.unnormalise(surf_stats=self.surf_stats)

        return pred
