"""Validation of the exact nonlinear-observation posterior + its metrics.

* quadrature == closed-form Kalman posterior for g = identity (1e-8);
* quadrature == 1e6-particle self-normalised importance sampling (bootstrap PF)
  for arctan / cubic / square, within Monte-Carlo error;
* inverse-CDF samples reproduce the quadrature moments / mode weights;
* the kept metrics reproduce known answers on Gaussians (mean_err, var_err,
  standardised marginal_w2, 1-D kl_marginal_knn incl. its documented low bias
  when q under-covers p), mode_weight_err on a known mode split, unique_frac
  sees duplicated particles; exact-sample floors are small;
* the harness emits n_failed / n_seeds, kl_points only for identity cells,
  and passes the R-dependent tuned hyperparameters through;
* classical filters keep the linear path and accept a nonlinear h.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from cases.analytical.nonlinear_reference import (  # noqa: E402
    PointwiseNonlinearPosterior,
    g_numpy,
    importance_sampling_reference,
    kalman_posterior_moments,
)
from cases.analytical.nonlinear_metrics import nonlinear_metrics  # noqa: E402

CASES = [  # (nonlinearity, strength, x0 offset, R)
    ("arctan", 1.0, 2.0, 0.01),
    ("arctan", 4.0, 2.0, 0.25),
    ("cubic", 1.0, 0.0, 0.01),
    ("cubic", 1.0, 0.0, 0.25),
    ("square", 1.0, 0.5, 0.01),
    ("square", 1.0, 0.0, 0.25),
]


def _problem(nl, c, off, R, d=3, seed=1):
    rng = np.random.default_rng(seed)
    x0 = off + 0.5 * rng.standard_normal(d)
    x1 = x0 + rng.standard_normal(d)
    y = g_numpy(nl, x1, c) + np.sqrt(R) * rng.standard_normal(d)
    return PointwiseNonlinearPosterior(x0, y, obs_var=R, nonlinearity=nl, strength=c)


@pytest.mark.parametrize("R", [1.0, 0.25, 0.01])
@pytest.mark.parametrize("prior_var", [1.0, 0.3])
def test_identity_matches_closed_form(R, prior_var):
    x0 = np.array([0.3, -1.2, 2.0, 5.0])
    y = np.array([1.1, 0.4, -0.5, -3.0])
    post = PointwiseNonlinearPosterior(x0, y, prior_var=prior_var, obs_var=R)
    mean, var = kalman_posterior_moments(x0, y, prior_var, R)
    np.testing.assert_allclose(post.mean, mean, atol=1e-8, rtol=0)
    np.testing.assert_allclose(post.var, var, atol=1e-8, rtol=0)
    # exact Gaussian entropy and log-density
    np.testing.assert_allclose(post.entropy, 2.0 * np.log(2 * np.pi * np.e * var), atol=1e-8)
    xs = mean[None, :] + np.sqrt(var) * np.random.default_rng(0).standard_normal((5, 4))
    lp = (-0.5 * (xs - mean) ** 2 / var - 0.5 * np.log(2 * np.pi * var)).sum(1)
    np.testing.assert_allclose(post.log_prob(xs), lp, atol=1e-8)


@pytest.mark.parametrize("nl,c,off,R", CASES)
def test_importance_sampling_agrees(nl, c, off, R):
    post = _problem(nl, c, off, R)
    ref = importance_sampling_reference(post, n=1_000_000, seed=11)
    assert np.all(ref["ess"] > 5_000)
    tol = 5.0  # standard errors
    assert np.all(np.abs(post.mean - ref["mean"]) < tol * ref["se_mean"] + 1e-6)
    assert np.all(np.abs(post.var - ref["var"]) < tol * ref["se_var"] + 1e-6)
    above = np.array([post.mass_above(0.0, i) for i in range(post.d)])
    assert np.all(np.abs(above - ref["mass_above"]) < tol * ref["se_above"] + 1e-6)


@pytest.mark.parametrize("nl,c,off,R", CASES)
def test_samples_reproduce_moments(nl, c, off, R):
    post = _problem(nl, c, off, R)
    n = 400_000
    s = post.sample(n, seed=5)
    sd = np.sqrt(post.var)
    assert np.all(np.abs(s.mean(0) - post.mean) < 5 * sd / np.sqrt(n))
    # var of the sample variance <= (m4 - v^2)/n; use a generous kurtosis bound
    m4 = ((s - post.mean) ** 4).mean(0)
    assert np.all(np.abs(s.var(0) - post.var) < 5 * np.sqrt(m4 / n))
    above = np.array([post.mass_above(0.0, i) for i in range(post.d)])
    assert np.all(np.abs((s > 0).mean(0) - above) < 5 * np.sqrt(above * (1 - above) / n) + 1e-4)


def test_square_is_bimodal_and_covers_both_modes():
    post = PointwiseNonlinearPosterior(np.array([0.0]), np.array([4.0]), obs_var=0.01,
                                       nonlinearity="square", strength=1.0)
    m = post.marginals[0]
    assert m.grid[0] < -2.0 < 2.0 < m.grid[-1]
    assert abs(post.mass_above(0.0, 0) - 0.5) < 1e-8  # symmetric prior -> equal modes
    assert abs(post.mean[0]) < 1e-8 and abs(post.var[0] - 4.0) < 0.01


def test_marginal_wasserstein_exact_quantile():
    post = PointwiseNonlinearPosterior(np.array([0.0]), np.array([0.7]), obs_var=0.25,
                                       nonlinearity="arctan", strength=2.0)
    s = post.sample(2000, seed=3)[:, 0]
    # brute force: fine sub-grid of each empirical-quantile block
    n, K = s.size, 400
    u = (np.arange(n * K) + 0.5) / (n * K)
    q = post.quantile(u, 0)
    emp = np.repeat(np.sort(s), K)
    w2_bf = np.sqrt(np.mean((emp - q) ** 2))
    assert post.marginal_w2(s, 0) == pytest.approx(w2_bf, rel=2e-3)
    # a shifted sample is ~shift away
    assert post.marginal_w2(s + 0.5, 0) == pytest.approx(0.5, abs=0.03)


def test_grid_growth_gives_up_loudly(monkeypatch):
    import cases.analytical.nonlinear_reference as ref_mod

    monkeypatch.setattr(ref_mod, "_MAX_GROW", 0)
    with pytest.raises(RuntimeError, match="doublings"):
        PointwiseNonlinearPosterior(np.zeros(1), np.zeros(1))


def _gauss_kl_1d(mu, s):
    """KL(N(0, 1) || N(mu, s^2))."""
    return float(np.log(s) + (1.0 + mu**2) / (2.0 * s**2) - 0.5)


@pytest.mark.parametrize("d", [2, 10])
@pytest.mark.parametrize("mu,s", [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (0.0, 2.0), (0.0, 0.5), (3.0, 1.0)])
def test_metrics_gaussian_known_answers(d, mu, s):
    """q = N(m + mu sd, (s sd)^2) vs the exact Gaussian posterior (identity g)."""
    n = 4096
    rng = np.random.default_rng(7)
    x0 = rng.standard_normal(d)
    y = x0 + rng.standard_normal(d)
    post = PointwiseNonlinearPosterior(x0, y, obs_var=0.25)
    sd = np.sqrt(post.var)
    ref = post.sample(n, seed=1)
    q = post.mean + sd * (mu + s * rng.standard_normal((n, d)))
    out = nonlinear_metrics(post, ref, q)
    assert "mode_weight_err" not in out and "kl_points" not in out
    mc = 4.0 / np.sqrt(n)  # a few Monte-Carlo standard errors
    assert out["mean_err"] == pytest.approx(abs(mu), abs=mc * s + 0.01)
    assert out["var_err"] == pytest.approx(abs(s**2 - 1), abs=4 * np.sqrt(2.0 / n) * s**2 + 0.01)
    assert out["marginal_w2"] == pytest.approx(np.hypot(mu, s - 1.0), abs=0.05 * max(1.0, s))
    assert out["unique_frac"] == 1.0
    true_kl = _gauss_kl_1d(mu, s)
    if s >= 1.0 and mu <= 1.0:  # q covers p: the 1-D k-NN KL is accurate
        assert out["kl_marginal_knn"] == pytest.approx(true_kl, abs=0.03)
    else:  # q under-covers p: documented LOW bias, but still clearly non-zero
        assert 0.5 * true_kl < out["kl_marginal_knn"] < true_kl


def test_metric_floor_exact_samples():
    """Exact samples as q: every metric at its Monte-Carlo floor (documented values)."""
    for nl, c, off, R, d in [("identity", 1.0, 0.0, 0.01, 10), ("arctan", 4.0, 2.0, 0.01, 10),
                             ("cubic", 1.0, 0.0, 0.25, 2), ("square", 1.0, 0.5, 0.01, 2)]:
        post = _problem(nl, c, off, R, d=d, seed=4)
        ref, q = post.sample(4096, seed=10), post.sample(4096, seed=20)
        split = 0.0 if nl == "square" else None
        out = nonlinear_metrics(post, ref, q, split=split)
        assert out["mean_err"] < 0.05 and out["var_err"] < 0.08, (nl, out)
        assert abs(out["kl_marginal_knn"]) < 0.03, (nl, out)
        assert out["marginal_w2"] < (0.25 if nl == "square" else 0.07), (nl, out)
        if split is not None:
            assert out["mode_weight_err"] < 0.02


def test_mode_weight_err_and_unique_frac_known():
    post = _problem("square", 1.0, 0.5, 0.01, d=2, seed=4)
    exact = np.array([post.mass_above(0.0, i) for i in range(2)])
    n, w = 20_000, 0.5
    rng = np.random.default_rng(3)
    s = np.abs(post.sample(n, seed=8)) * np.where(rng.random((n, 2)) < w, 1.0, -1.0)
    out = nonlinear_metrics(post, post.sample(4096, seed=1), s, split=0.0)
    assert out["mode_weight_err"] == pytest.approx(np.mean(np.abs(w - exact)), abs=0.01)
    # Gaussian with the exact moments: right moments, wrong (unimodal) shape.
    g = post.mean + np.sqrt(post.var) * rng.standard_normal((4096, 2))
    gm = nonlinear_metrics(post, post.sample(4096, seed=1), g, split=0.0)
    assert gm["mean_err"] < 0.1 and gm["marginal_w2"] > 0.2 and gm["kl_marginal_knn"] > 0.2
    # A resampled cloud (PF-like): 64 distinct particles repeated -> unique_frac = 1/64.
    dup = np.repeat(post.sample(64, seed=2), 64, axis=0)
    assert nonlinear_metrics(post, post.sample(4096, seed=1), dup)["unique_frac"] == pytest.approx(1 / 64)
    # Non-finite draw: NaN accuracy, unique_frac still reported.
    bad = post.sample(100, seed=2)
    bad[0, 0] = np.nan
    nb = nonlinear_metrics(post, post.sample(100, seed=1), bad, split=0.0)
    assert np.isnan(nb["mean_err"]) and np.isnan(nb["mode_weight_err"]) and nb["unique_frac"] == 1.0


def test_classical_filters_linear_path_and_nonlinear_h():
    from cases.analytical.classical_baselines import (
        GaussianSystem, enkf_posterior, particle_filter_posterior,
    )

    sys_ = GaussianSystem(d=2, obs_var=0.25)
    x0, y = torch.tensor([0.5, -0.2]), torch.tensor([1.0, 0.3])
    ident = lambda x: x  # noqa: E731
    a = particle_filter_posterior(sys_, x0, y, ensemble_size=256, generator=torch.Generator().manual_seed(0))
    b = particle_filter_posterior(sys_, x0, y, ensemble_size=256, generator=torch.Generator().manual_seed(0),
                                  obs_fn=ident)
    assert torch.equal(a, b)
    a = enkf_posterior(sys_, x0, y, ensemble_size=256, generator=torch.Generator().manual_seed(0))
    b = enkf_posterior(sys_, x0, y, ensemble_size=256, generator=torch.Generator().manual_seed(0),
                       obs_fn=ident)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    # nonlinear PF converges to the quadrature posterior
    yy = torch.tensor([0.9, -0.4])
    pf = particle_filter_posterior(sys_, x0, yy, ensemble_size=200_000,
                                   generator=torch.Generator().manual_seed(1), obs_fn=torch.atan)
    post = PointwiseNonlinearPosterior(x0.double(), yy.double(), obs_var=0.25, nonlinearity="arctan")
    np.testing.assert_allclose(pf.mean(0).double().numpy(), post.mean, atol=0.02)


def test_harness_runs_classical_methods(tmp_path):
    from cases.analytical.nonlinear_driver import NonlinearAnalyticalHarness, NonlinearCell
    from results_schema import Method

    cells = [NonlinearCell("arctan", 1.0, 0.01, 2.0, 1), NonlinearCell("square", 1.0, 0.25, 0.5, 2),
             NonlinearCell("identity", 1.0, 0.25, 0.0, 2)]
    h = NonlinearAnalyticalHarness(cells, seeds=[0, 1], methods=[Method.ENKF, Method.PARTICLE_FILTER],
                                   n_eval=512, output_dir=tmp_path, verbose=False)
    recs = h.run(aggregate=True)
    assert not h.failures
    by = {(r.scenario, r.method, r.metric): r.value for r in recs}
    assert ("nl:square_c1_R0.25_x00.5_d2", "Particle filter", "mode_weight_err") in by
    assert by[("nl:square_c1_R0.25_x00.5_d2", "Particle filter", "n_failed")] == 0
    assert by[("nl:square_c1_R0.25_x00.5_d2", "Particle filter", "n_seeds")] == 2
    assert by[("nl:square_c1_R0.25_x00.5_d2", "Particle filter", "unique_frac")] < 1.0  # resampled
    # legacy Gaussian KL / sliced rows only on the identity control
    assert ("nl:identity_R0.25_x00_d2", "Exact reference", "kl_points") in by
    assert ("nl:identity_R0.25_x00_d2", "EnKF", "sliced_w2") in by
    assert not any(m in ("kl_points", "sliced_w2", "kl_knn", "kl_reverse_knn", "marginal_w1")
                   for (sc, _, m) in by if "identity" not in sc)
    assert list(tmp_path.glob("failures_*.json")) and list(tmp_path.glob("summary_*.md"))


def test_harness_counts_failures_and_passes_tuned_hparams(tmp_path, monkeypatch):
    from cases.analytical.nonlinear_driver import NonlinearAnalyticalHarness, NonlinearCell
    from results_schema import Method

    cell = NonlinearCell("cubic", 1.0, 0.01, 0.0, 2)
    tuned = {"R0.01": {"FlowDAS": {"guidance_scale": 0.123}}}
    h = NonlinearAnalyticalHarness([cell], seeds=[0, 1, 2], methods=[Method.FLOWDAS], n_eval=64,
                                   output_dir=tmp_path, tuned_hparams=tuned, verbose=False)
    seen = []

    def boom(method, sys_, x0, y, n, M, gen, obs_op=None, hparams=None):
        seen.append(hparams)
        if len(seen) == 2:
            return torch.full((n, 2), float("nan")), 1  # non-finite -> counted
        raise RuntimeError("boom")

    monkeypatch.setattr(h.base, "_sample_and_count", boom)
    recs = h.run(aggregate=True, write=False)
    by = {(r.method, r.metric): r.value for r in recs}
    assert seen == [{"guidance_scale": 0.123}] * 3
    assert by[("FlowDAS", "n_failed")] == 3 and by[("FlowDAS", "n_seeds")] == 3
    assert len(h.failures) == 3
    assert by[("Exact reference", "n_failed")] == 0
    # overrides (tuning) win over tuned_hparams
    h2 = NonlinearAnalyticalHarness([cell], seeds=[0], methods=[Method.FLOWDAS], tuned_hparams=tuned,
                                    overrides={Method.FLOWDAS: {"guidance_scale": 9.0}}, verbose=False)
    assert h2.method_hparams(Method.FLOWDAS, cell) == {"guidance_scale": 9.0}


def test_config_expands_with_identity_control_per_cell():
    from cases.analytical.nonlinear_driver import NonlinearAnalyticalHarness, load_config

    h = NonlinearAnalyticalHarness.from_config(load_config(), seeds=[0], verbose=False)
    tags = {c.tag for c in h.cells}
    for c in h.cells:
        ctrl = f"nl:identity_R{c.obs_variance:g}_x0{c.x0_offset:g}_d{c.d}"
        assert ctrl in tags, c.tag
    assert {c.d for c in h.cells} == {2, 10}
