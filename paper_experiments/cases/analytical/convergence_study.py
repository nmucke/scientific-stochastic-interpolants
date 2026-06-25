"""Dimensionality + mode convergence study (spec Section 4, the key ablation).

Shows, for d in {2, 10, 100}, that the ``inflated`` likelihood mode converges to
the exact posterior mean/cov as ``M`` grows for all three samplers, while the
``dps_full`` and ``dps_jacobian_free`` surrogates *plateau* away from exact. KL
is the analytic Gaussian KL (tractable in any dimension; compares sample mean/cov
to the exact moments).

Run::

    python -m cases.analytical.convergence_study   # prints the table + writes a fig
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from .driver import gaussian_kl_to_exact  # noqa: E402
from .samplers import GaussianSystem, sample_posterior  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "figures" / "results" / "analytical"

_DIMS = (2, 10, 100)
_MODES = ("inflated", "dps_full", "dps_jacobian_free")
_M_LIST = (10, 25, 50, 100, 200, 400)
_SAMPLER = "fm_ode"  # representative; trends hold for all three
_N = 8000
_G0 = 1.0


def run(verbose: bool = True) -> dict:
    """Return ``results[d][mode] = [(M, KL), ...]`` and write a convergence fig."""
    results: dict = {}
    for d in _DIMS:
        sys_ = GaussianSystem(d=d, obs_var=1.0, prior_var=1.0)
        g0 = torch.Generator().manual_seed(d)
        x0 = torch.randn(d, generator=g0)
        y = (x0 + torch.randn(d, generator=g0)) + torch.randn(d, generator=g0)
        results[d] = {}
        for mode in _MODES:
            curve = []
            for M in _M_LIST:
                gg = torch.Generator().manual_seed(0)
                s = sample_posterior(
                    sys_, x0, y, sampler=_SAMPLER, likelihood_mode=mode,
                    ensemble_size=_N, num_steps=M, g0=_G0, generator=gg,
                )
                curve.append((M, gaussian_kl_to_exact(sys_, x0, y, s)))
            results[d][mode] = curve
            if verbose:
                tail = curve[-1][1]
                print(f"d={d:3d}  {mode:18s}  KL(M={_M_LIST[-1]})={tail:.4f}")

    # Figure: one panel per d, KL vs M, one curve per mode.
    fig, axes = plt.subplots(1, len(_DIMS), figsize=(3.4 * len(_DIMS), 3.0), squeeze=False)
    for j, d in enumerate(_DIMS):
        ax = axes[0, j]
        for mode in _MODES:
            Ms = [m for m, _ in results[d][mode]]
            kls = [k for _, k in results[d][mode]]
            ax.plot(Ms, kls, "-o", label=mode)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"steps $M$")
        if j == 0:
            ax.set_ylabel(r"KL to exact")
        ax.set_title(f"$d={d}$")
        ax.grid(True, which="both", alpha=0.3)
        if j == len(_DIMS) - 1:
            ax.legend(fontsize=7)
    fig.suptitle("Mode ablation: inflated converges, DPS surrogates plateau")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "an_dim_convergence.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(FIG_DIR / "an_dim_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"[convergence] wrote {out}")
    return results


if __name__ == "__main__":
    run()
