"""Probabilistic-calibration metrics (spec Section 3b).

Conventions: the ensemble dimension is the leading axis ``E``. Spatial metrics
accept an optional boolean ``mask`` over the per-point dimensions where
``True`` keeps a cell and ``False`` excludes it (e.g. solid cells).
"""

from typing import Optional

import torch

from scisi.metrics.accuracy import _apply_mask, ensemble_mean_rmse

__all__ = ["crps", "spread_skill", "rank_histogram", "plot_rank_histogram"]


def crps(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Ensemble CRPS with the unbiased pairwise estimator, averaged spatially.

    Per grid point ``i`` (spec Section 3b):

        CRPS_i = mean_e |x_{e,i} - x*_i|
               - 0.5 * (1 / (E (E-1))) * sum_{e != e'} |x_{e,i} - x_{e',i}|

    The ``1 / (E (E - 1))`` normalisation (rather than the biased ``1 / E^2``)
    makes the spread term an unbiased estimator of ``E|X - X'|``. The per-point
    CRPS is then averaged over the kept grid points.

    Args:
        ensemble: Ensemble tensor ``[E, *spatial]``.
        target: Ground truth ``x*`` of shape ``[*spatial]`` (or broadcastable).
        mask: Optional boolean keep-mask over ``*spatial`` (``True`` = keep).

    Returns:
        Scalar tensor: the spatially-averaged CRPS.
    """
    E = ensemble.shape[0]
    if E < 1:
        raise ValueError("`ensemble` must have at least one member.")

    skill = (ensemble - target.unsqueeze(0)).abs().mean(dim=0)

    if E == 1:
        # No spread term defined; CRPS reduces to the MAE.
        spread = torch.zeros_like(skill)
    else:
        # Sum of all pairwise absolute differences per point. Computed from the
        # sorted members in O(E log E * d) memory rather than forming the full
        # [E, E, d] difference tensor:
        #   sum_{e, e'} |x_e - x_e'| = 2 * sum_e (2 e - E + 1) * x_(e)
        # with x_(e) the e-th order statistic (0-indexed).
        sorted_ens, _ = torch.sort(ensemble, dim=0)
        idx_shape = [E] + [1] * (ensemble.dim() - 1)
        coeff = (
            2.0 * torch.arange(E, device=ensemble.device) - E + 1
        ).reshape(idx_shape).to(sorted_ens.dtype)
        pairwise_sum = 2.0 * (coeff * sorted_ens).sum(dim=0)
        spread = 0.5 * pairwise_sum / (E * (E - 1))

    crps_point = skill - spread
    return _apply_mask(crps_point, mask).mean()


def spread_skill(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Spread-skill ratio with the finite-ensemble correction (spec 3b).

    ``ensStd_i = sqrt(var_e x_{e,i})`` (population variance). The reported
    spread is the spatial mean of ``ensStd_i`` multiplied by the
    ``sqrt((E + 1) / E)`` finite-ensemble correction; the skill is the
    ensemble-mean RMSE. A perfectly calibrated ensemble has a ratio of 1.

    Args:
        ensemble: Ensemble tensor ``[E, *spatial]``.
        target: Ground truth ``x*`` of shape ``[*spatial]``.
        mask: Optional boolean keep-mask over ``*spatial`` (``True`` = keep).

    Returns:
        Dict with keys:
            ``"spread"``    corrected mean ensemble standard deviation,
            ``"skill"``     ensemble-mean RMSE,
            ``"ratio"``     spread / skill,
            ``"deviation"`` ``|1 - ratio|`` (0 = perfectly calibrated).
    """
    E = ensemble.shape[0]
    if E < 2:
        raise ValueError("Spread-skill needs an ensemble of size >= 2.")

    correction = ((E + 1) / E) ** 0.5
    ens_std = ensemble.var(dim=0, unbiased=False).sqrt()
    spread = correction * _apply_mask(ens_std, mask).mean()
    skill = ensemble_mean_rmse(ensemble, target, mask=mask)

    ratio = spread / skill
    return {
        "spread": spread,
        "skill": skill,
        "ratio": ratio,
        "deviation": (1.0 - ratio).abs(),
    }


def rank_histogram(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rank (Talagrand) histogram counts (spec Section 3b).

    For each kept grid point the rank of ``x*_i`` among the ``E`` ensemble
    members is the number of members strictly below it, giving an integer in
    ``[0, E]`` (``E + 1`` possible bins). Ties are broken randomly so that a
    calibrated ensemble yields a flat histogram even with discrete-valued data.

    Args:
        ensemble: Ensemble tensor ``[E, *spatial]``.
        target: Ground truth ``x*`` of shape ``[*spatial]``.
        mask: Optional boolean keep-mask over ``*spatial`` (``True`` = keep).

    Returns:
        Long tensor of shape ``[E + 1]`` with the count in each rank bin.
    """
    E = ensemble.shape[0]
    target_b = target.unsqueeze(0).expand_as(ensemble)

    below = (ensemble < target_b).sum(dim=0)
    equal = (ensemble == target_b).sum(dim=0)
    # Randomly distribute ties: add a uniform integer in [0, equal].
    if bool((equal > 0).any()):
        rand = torch.rand_like(equal, dtype=torch.float64)
        tie_break = torch.floor(rand * (equal.to(torch.float64) + 1.0)).long()
    else:
        tie_break = torch.zeros_like(below)
    ranks = below + tie_break

    ranks_kept = _apply_mask(ranks, mask).long()
    return torch.bincount(ranks_kept.reshape(-1), minlength=E + 1)[: E + 1]


def plot_rank_histogram(
    counts: torch.Tensor,
    title: str = "Rank histogram",
    figure_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Bar plot of a rank-histogram, consistent with ``scisi.plotting`` style.

    Flat = calibrated, U = under-dispersed, inverted-U = over-dispersed.

    Args:
        counts: Rank-bin counts ``[E + 1]`` from :func:`rank_histogram`.
        title: Plot title.
        figure_path: If given, saves ``<figure_path>/rank_histogram.png``.
        show: If ``True`` displays the figure, otherwise closes it.
    """
    import matplotlib.pyplot as plt

    counts_np = counts.detach().cpu().numpy()
    n_bins = len(counts_np)
    flat = counts_np.sum() / n_bins

    plt.figure()
    plt.bar(range(n_bins), counts_np, color="tab:blue", edgecolor="black")
    plt.axhline(flat, color="tab:red", linestyle="--", label="uniform")
    plt.xlabel("Rank")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    if figure_path is not None:
        plt.savefig(f"{figure_path}/rank_histogram.png")
    if show:
        plt.show()
    else:
        plt.close()
