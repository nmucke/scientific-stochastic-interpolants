import torch
import torch.nn as nn

POLAR_CONTRIBUTION = 1e-3


class LatitudeWeightedMSELoss(nn.Module):
    """
    Latitude weighted MSE loss.

    IMplementation is based on the following paper:
    https://arxiv.org/pdf/2507.20478v1
    """

    def __init__(
        self,
        latitudes: torch.Tensor,
        polar_contribution: float = POLAR_CONTRIBUTION,
    ):
        super(LatitudeWeightedMSELoss, self).__init__()
        self.latitudes = latitudes

        # Convert latitude from degrees to radians
        self.latitudes = torch.deg2rad(self.latitudes)

        self.polar_contribution = polar_contribution

        self.latitude_weights = self._compute_latitude_weights()

    def _compute_latitude_weights(self) -> torch.Tensor:
        """Compute the latitude weights."""

        expected_cos_latitudes = torch.mean(torch.cos(self.latitudes))

        weights = (
            self.polar_contribution
            + (1 - self.polar_contribution)
            * torch.cos(self.latitudes)
            / expected_cos_latitudes
        )

        return weights

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        latitude_weights = self.latitude_weights.repeat(
            pred.shape[0], pred.shape[1], 1
        ).unsqueeze(-1)
        latitude_weights = latitude_weights.to(pred.device)

        return torch.mean(latitude_weights * (pred - true) ** 2)
