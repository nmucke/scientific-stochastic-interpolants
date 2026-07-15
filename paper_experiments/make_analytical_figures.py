"""Produce ALL figures for the analytical (linear--Gaussian) case.

* ``analytical_kl_vs_M``      -- KL to the exact posterior vs. sampler steps $M$,
                                 all methods (legend below).
* ``analytical_w2_vs_M``      -- sliced-$W_2$ to the exact posterior vs. $M$.
* the density / convergence PANELS (``an_prior``, ``an_like``, ``an_true``,
  ``an_sampled``, ``an_kl_diff``, ``an_kl_steps``, ``an_slices``) via
  ``cases.analytical.figures.make_panels``.

The metric-vs-M figures read the reduced-grid results
(``results/analytical/metrics/analytical__M<M>.csv`` or the aggregated file); the
panels draw fresh closed-form ensembles through the src/scisi posteriors.

    python paper_experiments/make_analytical_figures.py
    python paper_experiments/make_analytical_figures.py --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from figure_common import (  # noqa: E402
    FIGURES_DIR,
    load_metric_vs_M,
    make_vs_M_figure,
    mirror_figures,
    save_series_legend,
)

DEFAULT_OUT = _here.parent / "manuscript" / "figures" / "analytical"
CASE = "analytical"
SCEN = "analytical"  # Case 1 has a single joint scenario.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    written: list[Path] = []

    # (1) KL vs M and sliced-W2 vs M, all methods. The KL axis is capped so the
    # baselines that diverge at large M (SGLD / SURGE with fixed per-step
    # hyperparameters) exit the top instead of stretching the axis.
    legend_keys: set = set()  # every (method, variant) with data, for the legend file
    for metric, ylabel, stem, ycap in (
        ("kl_points", r"KL divergence to exact posterior", "analytical_kl_vs_M", 10.0),
        ("sliced_w2", r"Sliced-$W_2$ to exact posterior", "analytical_w2_vs_M", None),
    ):
        series = load_metric_vs_M(CASE, metric, SCEN)
        if series:
            legend_keys.update(k for k, s in series.items() if s)
            written += make_vs_M_figure([("", series)], ylabel, out / stem, ycap=ycap)
        else:
            print(f"[analytical] no data for {metric}; run the analytical grid first")

    # One shared legend file for the legend-free singles of this case.
    written += save_series_legend(legend_keys, out / "singles" / "analytical_legend")

    # (2) the density / convergence panels (fresh draws through src/scisi).
    try:
        from cases.analytical import figures as _panels

        _panels.FIG_DIR = out  # write the panels into the same figures dir
        written += _panels.make_panels()
    except Exception as exc:  # pragma: no cover - panels are optional
        print(f"[analytical] panels skipped ({exc})")

    # Mirror every figure into the in-repo paper_experiments/figures/ tree too
    # (singles/ keeps its subfolder).
    written += mirror_figures(written, FIGURES_DIR / CASE)

    for p in written:
        print(f"[fig] wrote {p}")


if __name__ == "__main__":
    main()
