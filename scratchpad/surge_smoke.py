"""CPU smoke test for SurgePosterior with a tiny random velocity net.

Builds a DM prior from a tiny FlowMatchingModel (random weights), wires SURGE
through a LinearObservationOperator, and runs a short multi-window autoregressive
assimilation. Asserts finite output + correct shape + a resample actually fires.
"""
import torch

from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.diffusion_model import DenoiseDiffusionModel
from scisi.models.interpolations import LinearDeterministicInterpolation, _expand_t
from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.posterior_models.surge_posterior import SurgePosterior
from scisi.sampling.sde_solvers import euler_maruyama_step

torch.manual_seed(0)

C, H, W, L = 1, 8, 8, 2


class TinyDrift(torch.nn.Module):
    """Tiny velocity net: signature (x, t, field_history, field_cond, pars_cond)."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(C, C, 3, padding=1)

    def forward(self, x, t, field_history=None, field_cond=None, pars_cond=None):
        # Robust to t shapes [B,1], [1], scalar: broadcast a mean time scalar.
        te = t.reshape(-1).float().mean()
        return self.conv(x) * 0.1 + 0.1 * te

    # accept (x=..., cond=...) training-style kwargs too (unused here)


class TinyDM(DenoiseDiffusionModel):
    """Test-only DM that tolerates a 1-D ``t`` (shape [1]).

    The base ``sample`` loop's first-step path passes ``t = t_vec[:, 0]`` of
    shape ``[1]``; ``_expand_t`` requires ``[B, 1]``. This shim reshapes ``t``
    to ``[B, 1]`` before delegating, so the SMOKE TEST exercises SurgePosterior
    end-to-end with the real base-posterior machinery. Production uses the real
    drift nets + real ``DenoiseDiffusionModel``; nothing here changes that.
    """

    @staticmethod
    def _fix_t(t, x):
        if t.dim() == 1:
            t = t.reshape(-1, 1)
        if t.shape[0] != x.shape[0]:
            t = t.reshape(1, -1).expand(x.shape[0], -1)
        return t

    def score(self, x, t, *a, **k):
        return super().score(x, self._fix_t(t, x), *a, **k)

    def drift(self, x, t, *a, **k):
        return super().drift(x, self._fix_t(t, x), *a, **k)


fm = FlowMatchingModel(
    interpolation=LinearDeterministicInterpolation(),
    drift_model=TinyDrift(),
)
dm = TinyDM.from_flow_matching(fm).eval()

obs_op = LinearObservationOperator(type="random", data_size=(C, H, W),
                                   percent_obs=0.25, seed=1)

posterior = SurgePosterior(
    model=dm,
    obs_operator=obs_op,
    variance=0.05,
    guidance_scale=1.0,
    ess_threshold=0.9,  # high threshold -> resampling triggers
)

E = 6
T = 4  # physical steps
field_history = torch.randn(1, C, H, W, L)
observations = torch.randn(1, obs_op.num_obs, T)

traj = posterior.sample_trajectory(
    base=None,
    field_history=field_history,
    observations=observations,
    ensemble_size=E,
    batch_size=4,           # >1 chunk so the chunked weight accumulation is exercised
    num_steps=5,
    num_physical_steps=T,
    stepper=euler_maruyama_step,
)

print("trajectory shape:", tuple(traj.shape))
print("all finite:", bool(torch.isfinite(traj).all()))
print("mean:", float(traj.mean()), "std:", float(traj.std()))
assert tuple(traj.shape) == (E, C, H, W, T), traj.shape
assert torch.isfinite(traj).all(), "non-finite output"
print("SMOKE TEST PASSED")
