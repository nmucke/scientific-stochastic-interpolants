"""Conditional Föllmer maps with shared additive-noise Brownian paths."""

import torch

from scisi.models.base_map_model import BaseMapModel, batch_time
from scisi.sampling.brownian_path import has_affine_gamma, make_brownian_path

BROWNIAN_FEATURES = ("local", "karhunen_loeve")


class FollmerItoMapModel(BaseMapModel):
    """Learn average SDE drift while retaining exact martingale increments.

    Assumption: additive, state-independent diffusion gamma(t) C^{1/2} with a
    deterministic gamma and a fixed covariance C, both read from the
    interpolation (``noise_covariance`` and ``variance_clock``). Local path
    features approximate the drift's Brownian dependence; the explicit
    martingale increment is exact.

    ``brownian_features`` selects the path representation fed to the network:
    ``"local"`` (increments over equal subdivisions of [s,t]) or
    ``"karhunen_loeve"`` (the first ``num_path_features`` global KL
    coefficients of W on [0,1], the same for every interval of a path; needs
    gamma(t) = gamma_multiplier * (1 - t)).

    ``invariant`` (a ``scisi.models.invariants.LinearInvariant``, off by
    default) makes the map invariant-closed: the network output N becomes
    G = (Id - L^+ L) N + L^+ (l_t - l_s) / (t - s) with l_t = alpha_t L c +
    beta_t l(h) and c the latest history frame, so L Phi_{s,t}(x) =
    L x + l_t - l_s exactly when L C^{1/2} = 0. Not supported with masks.
    """

    def __init__(
        self,
        interpolation,
        map_model,
        num_path_features=4,
        mask_path=None,
        brownian_features="local",
        invariant=None,
    ):
        if invariant is not None and mask_path is not None:
            raise ValueError("Invariant-closed maps do not support masks.")
        super().__init__(interpolation, map_model, mask_path)
        if not isinstance(num_path_features, int) or num_path_features < 1:
            raise ValueError("num_path_features must be a positive integer.")
        if brownian_features not in BROWNIAN_FEATURES:
            raise ValueError(f"brownian_features must be one of {BROWNIAN_FEATURES}.")
        self.brownian_features = brownian_features
        if not torch.allclose(
            interpolation.transform_noise(torch.ones(1), torch.tensor([0.25])),
            torch.tensor([0.5]),
        ):
            raise ValueError("Föllmer Itô maps require wiener_process=True.")
        self.num_path_features = num_path_features
        self.gamma_multiplier = float(interpolation.gamma_multiplier)
        if brownian_features == "karhunen_loeve" and not has_affine_gamma(
            interpolation
        ):
            raise ValueError(
                "KarhunenLoeveBrownianPath requires gamma(t)=gamma_multiplier*(1-t)."
            )
        self.invariant = invariant
        self._invariant_checked = False

    def make_brownian_path(self, x, generator=None):
        if self.invariant is not None and not self._invariant_checked:
            self._check_driver_annihilates_invariant(x)
        return make_brownian_path(
            self.interpolation,
            x,
            generator=generator,
            features=self.brownian_features,
            num_modes=self.num_path_features,
        )

    def _check_driver_annihilates_invariant(self, x):
        """Require L C^{1/2} = 0 on one probe draw (own RNG, global RNG untouched)."""
        probe_generator = torch.Generator(device=x.device).manual_seed(0)
        white = torch.randn(
            x[:1].shape, device=x.device, dtype=x.dtype, generator=probe_generator
        )
        coloured = self.interpolation.noise_covariance.sqrt(white)
        leak = self.invariant.value(coloured).abs().max()
        if leak > 1e-5 * coloured.abs().max().clamp_min(1e-30):
            raise ValueError(
                "The noise covariance must vanish on the invariant (L C^{1/2} = 0); "
                "wrap it in ProjectedCovariance or zero those modes."
            )
        self._invariant_checked = True

    def _close_invariant(self, velocity, s, t, field_history):
        """G = (Id - L^+ L) N + L^+ (l_t - l_s)/(t - s); l'_t on the diagonal."""
        if field_history is None:
            raise ValueError("Invariant-closed maps need the field history.")
        interpolation = self.interpolation
        base_value = self.invariant.value(field_history[..., -1])  # L c
        target_value = self.invariant.target(field_history)  # l(h)

        def shaped(time):
            return time.reshape(time.shape[0], *([1] * (base_value.ndim - 1)))

        def ell(time):  # l_t = alpha_t L c + beta_t l(h)
            time = shaped(time)
            return (
                interpolation.alpha(time) * base_value
                + interpolation.beta(time) * target_value
            )

        gap = shaped(t - s)
        diagonal_rate = (
            interpolation.alpha_diff(shaped(t)) * base_value
            + interpolation.beta_diff(shaped(t)) * target_value
        )
        interval_rate = (ell(t) - ell(s)) / torch.where(
            gap > 0, gap, torch.ones_like(gap)
        )
        rate = torch.where(gap > 0, interval_rate, diagonal_rate)
        return self.invariant.project_out(velocity) + self.invariant.lift(
            rate, like=velocity
        )

    def average_velocity(
        self,
        x,
        s,
        t,
        field_history=None,
        field_cond=None,
        pars_cond=None,
        brownian_path=None,
    ):
        s, t = batch_time(s, x), batch_time(t, x)
        if torch.any(t < s):
            raise ValueError("An Itô map requires s <= t.")
        if brownian_path is None:
            if torch.any(t != s):
                raise ValueError(
                    "Off-diagonal Itô maps require a shared brownian_path."
                )
            features = x.new_zeros(
                x.shape[0], self.num_path_features * x.shape[1], *x.shape[2:]
            )
        else:
            features = brownian_path.features(s, t, self.num_path_features)
        velocity = self.map_model(
            x=x,
            cond=torch.cat([s, t], -1),
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
            path_features=features,
        )
        if self.invariant is not None:
            velocity = self._close_invariant(velocity, s, t, field_history)
        return velocity

    def noise_increment(self, x, s, t, brownian_path=None):
        if brownian_path is None:
            if torch.any(t != s):
                raise ValueError(
                    "Off-diagonal Itô maps require a shared brownian_path."
                )
            return torch.zeros_like(x)
        return brownian_path.increment(s, t)

    def training_pair(self, base, target, t, noise):
        return (
            self.interpolation(base=base, target=target, t=t, noise=noise),
            self.interpolation.forward_diff(base=base, target=target, t=t, noise=noise),
        )
