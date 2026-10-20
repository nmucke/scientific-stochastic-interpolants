"""Small adapters that turn an existing single-time backbone into a map network."""

import torch
from torch import nn


class TwoTimeEncoder(nn.Module):
    """Preserve a single-time embedding and add a learnable interval embedding."""

    def __init__(self, original: nn.Module, embedding_dim: int):
        super().__init__()
        self.original = original
        self.gap = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        nn.init.zeros_(self.gap[-1].weight)
        nn.init.zeros_(self.gap[-1].bias)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        if times.ndim != 2 or times.shape[-1] != 2:
            raise ValueError("Map conditioning must have shape [batch, 2].")
        gap = times[:, 1:2] - times[:, :1]
        return self.original(times[:, :1]) + gap * self.gap(gap)


class PathConditionedInit(nn.Module):
    """Inject path features after the original input/history processing block."""

    def __init__(self, original: nn.Module, path_channels: int, hidden_channels: int):
        super().__init__()
        self.original = original
        self.path_channels = path_channels
        self.projection = nn.Conv2d(path_channels, hidden_channels, 1, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, x, field_history=None, field_cond=None):
        if field_cond is None or field_cond.shape[1] < self.path_channels:
            raise ValueError("Missing Brownian path feature channels.")
        physical = field_cond[:, : -self.path_channels]
        path = field_cond[:, -self.path_channels :]
        result = self.original(
            x,
            field_history=field_history,
            field_cond=physical if physical.shape[1] else None,
        )
        return result + self.projection(path)


class MapConditionedNetwork(nn.Module):
    """Reuse a UNet with two times and optional spatial Brownian conditioning.

    Construct ``backbone`` with its original single-time configuration. The
    added branches start at zero, so loading a teacher reproduces its drift
    exactly before map training. The teacher itself is never stored.
    """

    def __init__(self, backbone: nn.Module, path_channels: int = 0):
        super().__init__()
        if path_channels < 0:
            raise ValueError("path_channels must be nonnegative.")
        self.backbone = backbone
        self.path_channels = path_channels
        linear_layers = [
            m for m in backbone.cond_encoder.modules() if isinstance(m, nn.Linear)
        ]
        if not linear_layers:
            raise ValueError("The backbone must expose a learned cond_encoder.")
        embedding_dim = linear_layers[-1].out_features
        backbone.cond_encoder = TwoTimeEncoder(backbone.cond_encoder, embedding_dim)
        if path_channels:
            # UNet's original initial block always ends in the first hidden width.
            original = backbone.init_conv
            convolutions = [m for m in original.modules() if isinstance(m, nn.Conv2d)]
            hidden_channels = convolutions[-1].out_channels
            backbone.init_conv = PathConditionedInit(
                original, path_channels, hidden_channels
            )

    def forward(
        self,
        x,
        cond,
        field_history=None,
        field_cond=None,
        pars_cond=None,
        path_features=None,
    ):
        if self.path_channels:
            if path_features is None:
                path_features = x.new_zeros(
                    x.shape[0], self.path_channels, *x.shape[2:]
                )
            if path_features.shape != (x.shape[0], self.path_channels, *x.shape[2:]):
                raise ValueError(
                    "Brownian features do not match configured path_channels/state shape."
                )
            field_cond = (
                path_features
                if field_cond is None
                else torch.cat([field_cond, path_features], 1)
            )
        elif path_features is not None and path_features.numel():
            raise ValueError("This network was configured without path channels.")
        return self.backbone(x, cond, field_history, field_cond, pars_cond)

    def load_from_teacher(self, teacher: nn.Module) -> None:
        """Copy every original backbone weight, rejecting incompatible shapes."""
        source = getattr(teacher, "drift_model", teacher)
        if isinstance(source, MapConditionedNetwork):
            self.load_state_dict(source.state_dict())
            return
        destination = self.backbone.state_dict()
        mapped = {}
        for name, value in source.state_dict().items():
            target_name = name
            if name.startswith("cond_encoder."):
                target_name = "cond_encoder.original." + name[len("cond_encoder.") :]
            elif self.path_channels and name.startswith("init_conv."):
                target_name = "init_conv.original." + name[len("init_conv.") :]
            if (
                target_name not in destination
                or destination[target_name].shape != value.shape
            ):
                raise ValueError(
                    f"Teacher/backbone mismatch for {name} -> {target_name}."
                )
            mapped[target_name] = value
        expected = {
            k
            for k in destination
            if not k.startswith("cond_encoder.gap.")
            and not k.startswith("init_conv.projection.")
        }
        if set(mapped) != expected:
            raise ValueError(
                f"Teacher is missing backbone weights: {sorted(expected - set(mapped))}"
            )
        destination.update(mapped)
        self.backbone.load_state_dict(destination)
