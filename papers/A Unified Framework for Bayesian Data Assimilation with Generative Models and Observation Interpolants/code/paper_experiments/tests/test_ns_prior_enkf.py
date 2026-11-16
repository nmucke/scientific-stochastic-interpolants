"""Generative prior + EnKF on Navier--Stokes: only the forecast differs from the EnKF.

Needs jax + jax-cfd (real EnKF classes) but no checkpoints, data or GPU: the
256^2 jax-cfd solver is replaced by a cheap stochastic stub and the learned prior
by deterministic stand-ins, on a 3-step, E=4 case with the paper's sparse
(Gaspari-Cohn r=20) and super-resolution (no localisation) operators.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

os.environ.setdefault("ENKF_JAX_PLATFORM", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
pytest.importorskip("jax")
pytest.importorskip("jax_cfd")
pytest.importorskip("jax_cfd_lib")

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cases.navier_stokes import _ns_pipeline, enkf_baseline as eb  # noqa: E402
from cases.navier_stokes._ns_pipeline import TruthAndObs  # noqa: E402

SCEN = EXPERIMENT / "configs" / "scenario"
L, T, E, VAR = 5, 8, 4, 0.01  # 3 assimilation steps
# (scenario config, localisation radius): the paper's settings (run_ns_grid.sh).
CASES = [("sparse_1p5", 20.0), ("superres_16", None)]


def _operator(name):
    return _ns_pipeline.build_obs_operator(
        OmegaConf.load(SCEN / f"{name}.yaml"), (1, 128, 128), mask_seed=7
    )


def _truth(op, seed=0):
    g = torch.Generator().manual_seed(seed)
    history = torch.randn(1, 1, 128, 128, L, generator=g)
    truth = torch.randn(1, 1, 128, 128, T, generator=g)
    obs = torch.stack([op(truth[..., t]) for t in range(T)], dim=-1)
    obs = obs + 0.1 * torch.randn(obs.shape, generator=g)
    return TruthAndObs(
        init_base=history[..., -1].clone(), field_history=history, field_cond=None,
        pars_cond=None, true_trajectory=truth, observations=obs, obs_operator=op,
    )


def _stub_pde(u, key):
    """Cheap stochastic 'solver' on the 256^2 spectrum (physical noise added)."""
    noise = jnp.fft.rfft2(0.3 * jax.random.normal(key, (eb.HIGH_RES, eb.HIGH_RES)))
    return 0.9 * u + noise


def _identity_pde(u, key):
    return u


def _kwargs(op, truth, radius):
    return dict(
        truth_obs=truth, obs_operator=op, ensemble_size=E, variance=VAR,
        num_physical_steps=T, len_field_history=L, seed=3, localization_radius=radius,
    )


@pytest.mark.parametrize("scenario,radius", CASES)
def test_true_solver_forecast_reproduces_the_enkf_baseline_exactly(monkeypatch, scenario, radius):
    """The new loop with the baseline's forecast IS the baseline (bit for bit)."""
    monkeypatch.setattr(eb, "_build_dynamics", lambda: _stub_pde)
    op = _operator(scenario)
    truth = _truth(op)
    a = eb.run_enkf_baseline(**_kwargs(op, truth, radius))
    b = eb.run_prior_enkf_baseline(**_kwargs(op, truth, radius), forecast="true_solver")
    assert torch.isfinite(a.posterior_trajectory).all()
    assert not torch.equal(a.posterior_trajectory[..., L], a.posterior_trajectory[..., L + 1])
    torch.testing.assert_close(b.posterior_trajectory, a.posterior_trajectory, rtol=0, atol=0)
    assert not b.diverged and len(b.diagnostics["analysis_rmse"]) == T - L


@pytest.mark.parametrize("scenario,radius", CASES)
def test_prior_path_analysis_equals_true_solver_path_on_the_same_forecast(monkeypatch, scenario, radius):
    """Persistence via the prior path (128^2 history, lifted to 256^2) and via the
    true-solver path (identity solver on the 256^2 state) give the same analysis:
    the lift's odd-point fill does not reach the recorded 128^2 points, and the
    filter, localisation, R and perturbed observations are shared."""
    monkeypatch.setattr(eb, "_build_dynamics", lambda: _identity_pde)
    op = _operator(scenario)
    truth = _truth(op)
    a = eb.run_prior_enkf_baseline(**_kwargs(op, truth, radius), forecast="true_solver")
    b = eb.run_prior_enkf_baseline(
        **_kwargs(op, truth, radius), forecast=lambda h: h[..., -1].clone()
    )
    torch.testing.assert_close(b.posterior_trajectory, a.posterior_trajectory, rtol=1e-4, atol=1e-4)
    # And the update did something (the analysis moved off the forecast).
    assert (a.posterior_trajectory[..., L] - a.posterior_trajectory[..., L - 1]).abs().max() > 1e-2


def test_histories_are_per_member_and_slide_with_the_analysed_states(monkeypatch):
    op = _operator("sparse_1p5")
    truth = _truth(op)
    seen = []

    def sample_fn(history):
        seen.append(history.clone())
        # Uses every frame, so a wrong order or a stale frame changes the result.
        w = torch.linspace(0.1, 0.5, history.shape[-1])
        return (history * w).sum(-1) + 0.05 * torch.randn(history.shape[:-1])

    torch.manual_seed(0)
    out = eb.run_prior_enkf_baseline(**_kwargs(op, truth, 20.0), forecast=sample_fn)
    post = out.posterior_trajectory
    assert len(seen) == T - L
    first = seen[0]
    assert first.shape == (E, 1, 128, 128, L)
    # Frames 0..L-2: the IC trajectory, shared by every member.
    torch.testing.assert_close(first[..., :-1], truth.field_history[..., :-1].expand(E, -1, -1, -1, -1))
    # Frame L-1: each member's own (perturbed) initial state = the recorded IC column.
    torch.testing.assert_close(first[..., -1], post[..., L - 1], rtol=0, atol=0)
    assert (first[0, ..., -1] - first[1, ..., -1]).abs().max() > 0  # perturbed apart
    for k in range(1, T - L):
        h = seen[k]
        # Last frame = this member's analysis of the previous step (as recorded).
        torch.testing.assert_close(h[..., -1], post[..., L + k - 1], rtol=0, atol=0)
        # The window slides by one, member by member.
        torch.testing.assert_close(h[..., :-1], seen[k - 1][..., 1:], rtol=0, atol=0)
    assert torch.isfinite(post).all()


def test_localisation_is_the_enkf_baselines(monkeypatch):
    from jax_cfd_lib.ENKF import LocalizedSpectralEnKF, SpectralEnKF

    sparse = eb.build_enkf_filter(_operator("sparse_1p5"), E, VAR, localization_radius=20)
    assert type(sparse) is LocalizedSpectralEnKF
    assert sparse.localization_radius == 20.0 and sparse.localization_type == "distance"
    assert not sparse.adaptive_localization and sparse.grid_shape == (256, 256)
    assert sparse.obs_noise_std == pytest.approx(VAR ** 0.5)
    chw = _operator("sparse_1p5").obs_indices_c_h_w.numpy()
    np.testing.assert_array_equal(np.asarray(sparse.observation_operator.obs_coords), chw[:, 1:] * 2)
    superres = eb.build_enkf_filter(_operator("superres_16"), E, VAR, localization_radius=None)
    assert type(superres) is SpectralEnKF


def test_divergence_guard_nan_pads(monkeypatch):
    op = _operator("superres_16")
    truth = _truth(op)
    out = eb.run_prior_enkf_baseline(
        **_kwargs(op, truth, None), forecast=lambda h: 1e3 * torch.ones_like(h[..., -1]),
        divergence_rmse_threshold=10.0,
    )
    assert out.diverged
    assert torch.isfinite(out.posterior_trajectory[..., L]).all()
    assert torch.isnan(out.posterior_trajectory[..., L + 1:]).all()


# --------------------------------------------------------------------------- #
# Driver wiring: the prior is the Ours sampler, unguided; the EnKF keys are the
# EnKF branch's.
# --------------------------------------------------------------------------- #


def _runner(**cfg):
    from cases.navier_stokes.driver import NavierStokesRunner

    return NavierStokesRunner(OmegaConf.create({"case": {"device": "cpu"}, **cfg}), seeds=[0])


@pytest.mark.parametrize(
    "method,base,stepper_name",
    [("SI-SDE prior + EnKF", "Ours (SI-SDE)", "euler_maruyama_step"),
     ("DM-SDE prior + EnKF", "Ours (DM-SDE)", "euler_maruyama_step"),
     ("FM-ODE prior + EnKF", "Ours (FM-ODE)", "euler_step")],
)
def test_driver_builds_the_ours_prior_and_passes_the_enkf_settings(monkeypatch, tmp_path, method, base, stepper_name):
    from cases.navier_stokes import driver
    import scisi.posterior_models.forecast_correction as fc
    from results_schema import Method, Scenario

    built, captured, drawn = {}, {}, {}
    model = torch.nn.Linear(1, 1)

    def fake_build_posterior(**kw):
        built.update(kw)
        return model, SimpleNamespace(tag="posterior"), _ns_pipeline._STEPPERS[
            "ode" if "FM-ODE" in kw["method_name"] else "sde"]

    def fake_run(**kw):
        captured.update(kw)
        h = torch.zeros(E, 1, 128, 128, L)
        kw["forecast"](h)
        res = SimpleNamespace(diverged=False, nfe_per_step=1.0, seconds_per_step=2.0,
                              diagnostics={"forecast_spread": [0.1]})
        return res

    def fake_draw(posterior, history, num_steps, stepper, batch_size):
        drawn.update(posterior=posterior, num_steps=num_steps,
                     stepper=stepper.__name__, batch_size=batch_size)
        return history[..., -1]

    monkeypatch.setattr(driver._ns_pipeline, "build_posterior", fake_build_posterior)
    monkeypatch.setattr(fc, "draw_unguided_forecast", fake_draw)
    monkeypatch.setattr(driver._ns_pipeline, "attach_nfe_counter", lambda m: ("counter", m))
    fake_eb = SimpleNamespace(run_prior_enkf_baseline=fake_run)
    runner = _runner(enkf_localization_radius=20, enkf_inflation=1.0,
                     divergence_rmse_threshold=10.0, prior_enkf_diagnostics_dir=str(tmp_path))
    m = Method(method)
    assert m in driver.PRIOR_ENKF_METHODS and m not in driver.NS_METHODS
    ctx = SimpleNamespace(
        method=m, scenario=Scenario.SPARSE_1p5, seed=0, ensemble_size=E, num_steps=100,
        extra={"variance": VAR, "likelihood_ensemble_size": 8, "num_physical_steps": T,
               "test_index": 11},
    )
    prior = SimpleNamespace(len_field_history=L)
    scen_cfg = OmegaConf.load(SCEN / "sparse_1p5.yaml")
    truth = SimpleNamespace(field_cond=None, pars_cond=None)
    runner._run_prior_enkf(ctx, prior, scen_cfg, "OP", truth, fake_eb)

    assert built["method_name"] == base and built["obs_operator"] == "OP"
    assert built["likelihood_mode"] == "dps_jacobian_free"
    # Same EnKF keys and defaults as the Method.ENKF branch.
    assert captured["localization_radius"] == 20 and captured["inflation"] == 1.0
    assert captured["localization_type"] == "distance" and captured["corr_threshold"] is None
    assert captured["corr_inflation_max"] == 4.0 and captured["corr_inflation_beta"] == 0.5
    assert captured["ensemble_size"] == E and captured["seed"] == 0
    assert captured["divergence_rmse_threshold"] == 10.0
    assert captured["nfe_counter"] == ("counter", model)
    # Unguided forecast of the Ours posterior, M steps, its own stepper, all members at once.
    assert drawn["posterior"].tag == "posterior"
    assert (drawn["num_steps"], drawn["stepper"], drawn["batch_size"]) == (100, stepper_name, E)
    (diag,) = tmp_path.glob("*.json")
    assert "traj11__seed0__E4_M100" in diag.name and runner._variant(m) is None


def test_driver_forecast_step_override_is_tagged():
    from cases.navier_stokes import driver
    from results_schema import Method

    runner = _runner(prior_enkf_forecast_num_steps=25)
    assert runner._variant(Method.PRIOR_ENKF_DM_SDE) == "forecast_steps25"
    assert driver.METHOD_CONFIG_NAME[Method.PRIOR_ENKF_DM_SDE] == "prior_enkf_dm_sde"
    cfg = runner._method_cfgs()[Method.PRIOR_ENKF_DM_SDE]
    ours = runner._method_cfgs()[Method.OURS_DM_SDE]
    assert cfg.name == "DM-SDE prior + EnKF"
    assert cfg.stepper == ours.stepper and cfg.diffusion_term == ours.diffusion_term
