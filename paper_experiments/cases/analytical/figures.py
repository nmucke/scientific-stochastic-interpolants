"""Figure ``fig:analytical_panels`` for the analytical linear--Gaussian case.

Seven panels (spec Section 4 / results.tex subfigures a-g):
  (a) prior conditional ``p(x1 | x0)``      -- 2-D density
  (b) likelihood ``p(y | x1)``               -- 2-D density
  (c) exact posterior                        -- 2-D density
  (d) sampled posterior (one sampler)        -- 2-D density
  (e) KL to exact vs diffusion strength g_tau (SDE samplers)
  (f) KL to exact vs number of steps M       (all samplers)
  (g) 1-D density slices: sampled vs exact

Run::

    python -m cases.analytical.figures            # writes the panels to disk

Saves PNG + PDF under ``paper_experiments/figures/results/analytical/`` to match
the ``\\includegraphics{figures/results/analytical/...}`` paths in results.tex.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .classical_baselines import GaussianSystem  # noqa: E402
from .driver import draw_interpolant_posterior, gaussian_kl_to_exact  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "figures" / "results" / "analytical"

_OBS_VAR = 1.0
_PRIOR_VAR = 1.0
_G0 = 1.0
_X0 = torch.tensor([5.0, 5.0])
_Y = torch.tensor([1.0, 1.0])
_N = 8000
_SAMPLERS = ("si_sde", "dm_sde", "fm_ode")
_SAMPLER_LABEL = {"si_sde": "SI-SDE", "dm_sde": "DM-SDE", "fm_ode": "FM-ODE"}


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def _hist2d(ax, samples: np.ndarray, title: str, rng) -> None:
    ax.hist2d(samples[:, 0], samples[:, 1], bins=80, range=rng, cmap="viridis")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def make_panels(num_steps: int = 50, ensemble_size: int = _N) -> list[Path]:
    """Render all seven panels (2-D for densities) and the convergence curves."""
    sys2 = GaussianSystem(d=2, obs_var=_OBS_VAR, prior_var=_PRIOR_VAR)
    g = torch.Generator().manual_seed(0)
    rng = [[-1.0, 7.0], [-1.0, 7.0]]
    written: list[Path] = []

    # (a) prior conditional p(x1 | x0) = N(x0, I).
    prior = _X0.unsqueeze(0) + torch.randn(_N, 2, generator=g)
    # (b) likelihood p(y | x1): as a function over x1, N(y, R) (proportional).
    like = _Y.unsqueeze(0) + (_OBS_VAR**0.5) * torch.randn(_N, 2, generator=g)
    # (c) exact posterior.
    exact = sys2.exact_posterior_samples(_X0, _Y, _N, g)
    # (d) sampled posterior (SI-SDE, the canonical sampler).
    sampled = draw_interpolant_posterior(
        sys2, _X0, _Y, sampler="si_sde", likelihood_mode="inflated",
        ensemble_size=ensemble_size, num_steps=num_steps, g0=_G0, seed=0,
    )

    for key, data, title in (
        ("an_prior", prior, r"(a) prior $p(x_1\,|\,x_0)$"),
        ("an_like", like, r"(b) likelihood $p(y\,|\,x_1)$"),
        ("an_true", exact, r"(c) exact posterior"),
        ("an_sampled", sampled, r"(d) sampled posterior (SI-SDE)"),
    ):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        _hist2d(ax, np.asarray(data), title, rng)
        written.append(_save(fig, key))

    # (e) KL vs diffusion strength g_tau (SDE samplers SI-SDE, DM-SDE).
    g_list = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for sampler in ("si_sde", "dm_sde"):
        kls = []
        for gv in g_list:
            s = draw_interpolant_posterior(
                sys2, _X0, _Y, sampler=sampler, likelihood_mode="inflated",
                ensemble_size=_N, num_steps=num_steps, g0=gv, seed=0,
            )
            kls.append(gaussian_kl_to_exact(sys2, _X0, _Y, s))
        ax.plot(g_list, kls, "-o", label=_SAMPLER_LABEL[sampler])
    ax.set_xlabel(r"diffusion strength $g_\tau$ base")
    ax.set_ylabel(r"KL to exact")
    ax.set_yscale("log")
    ax.set_title("(e) KL vs.\\ diffusion strength")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    written.append(_save(fig, "an_kl_diff"))

    # (f) KL vs number of steps M (all three samplers).
    M_list = [10, 25, 50, 100, 200, 400]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for sampler in _SAMPLERS:
        kls = []
        for M in M_list:
            s = draw_interpolant_posterior(
                sys2, _X0, _Y, sampler=sampler, likelihood_mode="inflated",
                ensemble_size=_N, num_steps=M, g0=_G0, seed=0,
            )
            kls.append(gaussian_kl_to_exact(sys2, _X0, _Y, s))
        ax.plot(M_list, kls, "-o", label=_SAMPLER_LABEL[sampler])
    ax.set_xlabel(r"integration steps $M$")
    ax.set_ylabel(r"KL to exact")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(f) KL vs.\\ steps $M$")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    written.append(_save(fig, "an_kl_steps"))

    # (g) 1-D density slices (first coordinate): sampled vs exact.
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    grid = np.linspace(-1, 7, 200)
    mean, cov = sys2.exact_posterior_moments(_X0, _Y)
    exact_pdf = np.exp(-0.5 * (grid - float(mean[0])) ** 2 / float(cov)) / (
        (2 * np.pi * float(cov)) ** 0.5
    )
    ax.plot(grid, exact_pdf, "k-", lw=2, label="exact")
    for sampler in _SAMPLERS:
        s = draw_interpolant_posterior(
            sys2, _X0, _Y, sampler=sampler, likelihood_mode="inflated",
            ensemble_size=_N, num_steps=num_steps, g0=_G0, seed=0,
        ).numpy()[:, 0]
        ax.hist(s, bins=60, range=(-1, 7), density=True, histtype="step",
                label=_SAMPLER_LABEL[sampler])
    ax.set_xlabel(r"$x_{1,1}$")
    ax.set_ylabel("density")
    ax.set_title("(g) 1-D density slices")
    ax.legend(fontsize=7)
    written.append(_save(fig, "an_slices"))

    return written


def main() -> None:
    paths = make_panels()
    for p in paths:
        print(f"[figures] wrote {p}")


if __name__ == "__main__":
    main()
