"""Presentation copy of ``make_analytical_figures.py`` WITHOUT the Jac-free runs.

Identical figures to ``make_analytical_figures.py`` -- the metric-vs-$M$ curves
(``analytical_kl_vs_M``, ``analytical_w2_vs_M`` + their singles/legend) and the
density / convergence panels (``an_prior`` ... ``an_slices``) -- except that every
``(method, variant)`` series with the ``jacfree`` variant is dropped, so only the
shared-Jacobian "Ours" curves and the baselines are shown. The panels already draw
with ``likelihood_mode="inflated"``, so they are unchanged.

Figures land in ``<repo root>/root/`` (no mirroring into the in-repo
``paper_experiments/figures/`` tree -- this is a throwaway presentation build).

    python paper_experiments/temp_make_analytical_figures.py
    python paper_experiments/temp_make_analytical_figures.py --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import figure_common as fc  # noqa: E402
from figure_common import (  # noqa: E402
    _canon_variant,
    load_metric_vs_M,
    make_vs_M_figure,
    save_series_legend,
)

# With the Jac-free curves gone there is only one "Ours" variant left, so the
# "(shared)" qualifier carries no information -- drop it from the legend labels.
# Patched on the shared SERIES table (which both the plotter and the standalone
# legend read) so every figure this script writes stays consistent.
fc.SERIES = tuple(
    (m, v, label.replace(" (shared)", ""), *rest) for m, v, label, *rest in fc.SERIES
)

DEFAULT_OUT = _here.parent / "root"
CASE = "analytical"
SCEN = "analytical"  # Case 1 has a single joint scenario.
DROP_VARIANT = "jacfree"


def _without_jacfree(series: dict) -> dict:
    """Drop every ``(method, variant)`` series whose variant is Jacobian-free."""
    return {
        k: v for k, v in series.items() if _canon_variant(k[1]) != DROP_VARIANT
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    written: list[Path] = []

    # (1) KL vs M and sliced-W2 vs M, all methods except the Jac-free variants.
    legend_keys: set = set()
    for metric, ylabel, stem, ycap in (
        ("kl_points", r"KL divergence to exact posterior", "analytical_kl_vs_M", 10.0),
        ("sliced_w2", r"Sliced-$W_2$ to exact posterior", "analytical_w2_vs_M", None),
    ):
        series = _without_jacfree(load_metric_vs_M(CASE, metric, SCEN))
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

    for p in written:
        print(f"[fig] wrote {p}")


if __name__ == "__main__":
    main()
