"""Tune the baselines' main step / guidance knob for the nonlinear analytical sweep.

Protocol (no information about the nonlinearity is used):

* cell: the IDENTITY control (g = id, x0_offset 0, x0_spread 1, d = 2) at each
  R of the sweep -- its exact posterior is Gaussian;
* seeds: the tuning seeds 100, 101 (disjoint from the evaluation seeds 0-4);
* M = 50, n_eval = 4096, the same builders as the sweep (``NonlinearAnalyticalHarness``
  with a per-method hyperparameter override);
* criterion: the seed-mean standardised ``marginal_w2`` (a setting with any failed
  seed is ineligible); ties -> the smaller value;
* grid: ~5 log-spaced values per method. ``--stage probe`` runs a wide
  decade grid on seed 100 only to locate the basin; ``--stage final`` (default)
  runs the half-decade grids below on both tuning seeds and selects. If the final
  argmin sits on a grid edge the script says so (extend the grid and rerun).

Not tuned: DPS (``DPSGaussianLikelihood`` has no step knob), EnKF / PF (no
knob), the Ours samplers (no step / guidance knob; ``jacobian_damping`` is kept
at its default 1.0 by design).

Writes ``results/analytical_nonlinear/tuning/tuning_<stage>.csv`` (one row per
(R, method, value) with the seed-mean metrics and ``selected``) and prints the
``tuned_hparams`` YAML block for ``configs/case/analytical_nonlinear.yaml``.

    uv run python -m cases.analytical.nonlinear_tuning [--stage probe|final]   # from <PE>
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import torch

from results_schema import Method

from .nonlinear_driver import NonlinearAnalyticalHarness, NonlinearCell, load_config, r_key

TUNING_SEEDS = (100, 101)
R_VALUES = (0.01, 0.25)
# method -> (hyperparameter key, old analytical value at M = 50 for the record)
KNOBS: dict[Method, tuple[str, float]] = {
    Method.FLOWDAS: ("guidance_scale", 0.004),
    Method.SURGE_FLOWDAS: ("guidance_scale", 0.004),
    Method.SDA: ("gamma_sda", 0.01),
    Method.SURGE_SDA: ("gamma_sda", 0.01),
    Method.GUIDED_FM_FIG: ("guidance_scale", 1.0),
    Method.GUIDED_FM_OTODE: ("guidance_scale", 1.0),
    Method.SURGE: ("guidance_scale", 1.0),
    Method.D_FLOW_SGLD: ("step_size", 0.005),
}
_DEC = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
PROBE_GRIDS: dict[Method, tuple[float, ...]] = {
    Method.FLOWDAS: _DEC,
    Method.SURGE_FLOWDAS: _DEC,
    Method.SDA: _DEC,
    Method.SURGE_SDA: _DEC,
    Method.GUIDED_FM_FIG: (0.01, 0.1, 1.0, 10.0, 100.0),
    Method.GUIDED_FM_OTODE: (0.01, 0.1, 1.0, 10.0, 100.0),
    Method.SURGE: (0.01, 0.1, 1.0, 10.0, 100.0),
    Method.D_FLOW_SGLD: (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
}


def _grid(center: float, step: float = 0.5) -> tuple[float, ...]:
    """5 log-spaced values ``center * 10**(k * step)``, k = -2..2."""
    return tuple(float(f"{center * 10 ** (k * step):.3g}") for k in (-2, -1, 0, 1, 2))


# Final grids: half-decade (quarter-decade for OT-ODE's narrow basin) around the
# probe optimum (tuning_probe.csv, seed 100); where the probe argmin sat on a
# grid edge the grid is shifted past that edge.
FINAL_GRIDS: dict[str, dict[Method, tuple[float, ...]]] = {
    "R0.01": {
        Method.FLOWDAS: _grid(0.1),
        Method.SURGE_FLOWDAS: _grid(1e-3),
        Method.SDA: _grid(0.1),
        Method.SURGE_SDA: _grid(1e-3, 1.0),        # probe non-monotone (SMC noise)
        Method.GUIDED_FM_FIG: _grid(100.0),        # probe: plateau (collapsed) above 10
        Method.GUIDED_FM_OTODE: _grid(1.0, 0.25),
        Method.SURGE: _grid(0.1),
        Method.D_FLOW_SGLD: _grid(1.0),            # probe argmin on the upper edge
    },
    "R0.25": {
        Method.FLOWDAS: _grid(1.0),                # probe argmin on the upper edge
        Method.SURGE_FLOWDAS: _grid(1e-5),         # probe argmin on the lower edge
        Method.SDA: _grid(1e-4, 1.0),              # probe argmin on the lower edge (flat)
        Method.SURGE_SDA: _grid(0.1),
        Method.GUIDED_FM_FIG: _grid(100.0),        # probe: plateau (collapsed) above 1
        Method.GUIDED_FM_OTODE: _grid(1.0, 0.25),
        Method.SURGE: _grid(0.1),
        Method.D_FLOW_SGLD: _grid(1.0),            # probe argmin on the upper edge
    },
}


def tune(
    stage: str = "final",
    methods: Optional[Sequence[Method]] = None,
    r_values: Sequence[float] = R_VALUES,
    grids: Optional[dict[str, dict[Method, Sequence[float]]]] = None,
    out_dir: Optional[Path] = None,
) -> tuple[list[dict], dict[str, dict[str, dict[str, float]]]]:
    torch.set_default_dtype(torch.float32)
    cfg = load_config(num_steps=50, ensemble_size=64)
    out_dir = Path(out_dir or Path(__file__).resolve().parents[2] / "results" / "analytical_nonlinear" / "tuning")
    seeds = TUNING_SEEDS[:1] if stage == "probe" else TUNING_SEEDS
    methods = list(methods or KNOBS)
    rows: list[dict] = []
    selected: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for R in r_values:
        cell = NonlinearCell("identity", 1.0, R, 0.0, 2)
        for method in methods:
            key, old = KNOBS[method]
            if stage == "probe":
                grid = PROBE_GRIDS[method]
            else:
                grid = (grids or FINAL_GRIDS)[r_key(R)][method]
            mrows = []
            for v in grid:
                h = NonlinearAnalyticalHarness(
                    [cell], seeds=seeds, methods=[method], base_config=cfg,
                    overrides={method: {key: float(v)}}, include_reference=False,
                    output_dir=out_dir, verbose=False,
                )
                agg = h.run(aggregate=True, write=False)
                by = {r.metric: (r.value, r.std) for r in agg if r.method == method.value}
                err = "; ".join(sorted({f"{f['error']}: {f['message'][:80]}" for f in h.failures}))
                row = dict(stage=stage, R=R, method=method.value, param=key, value=float(v),
                           old_value=old, seeds=" ".join(map(str, seeds)),
                           n_failed=int(by.get("n_failed", (len(seeds), 0))[0]),
                           marginal_w2=by.get("marginal_w2", (math.nan, math.nan))[0],
                           marginal_w2_std=by.get("marginal_w2", (math.nan, math.nan))[1],
                           mean_err=by.get("mean_err", (math.nan,))[0],
                           var_err=by.get("var_err", (math.nan,))[0],
                           kl_marginal_knn=by.get("kl_marginal_knn", (math.nan,))[0],
                           kl_points=by.get("kl_points", (math.nan,))[0],
                           unique_frac=by.get("unique_frac", (math.nan,))[0],
                           seconds=sum(r.seconds or 0 for r in agg if r.metric == "marginal_w2"),
                           selected=False, note=err)
                mrows.append(row)
                print(f"R={R:g} {method.value:20s} {key}={v:<10g} w2={row['marginal_w2']:.4g} "
                      f"mean_err={row['mean_err']:.3g} var_err={row['var_err']:.3g} "
                      f"failed={row['n_failed']} {err}", flush=True)
            ok = [r for r in mrows if r["n_failed"] == 0 and math.isfinite(r["marginal_w2"])]
            if ok:
                best = min(ok, key=lambda r: (r["marginal_w2"], r["value"]))
                best["selected"] = True
                vals = [r["value"] for r in mrows]
                if best["value"] in (min(vals), max(vals)):
                    best["note"] = (best["note"] + "; " if best["note"] else "") + "argmin on grid edge"
                    print(f"  ! {method.value} R={R:g}: argmin {best['value']:g} on grid edge", flush=True)
                selected[r_key(R)][method.value] = {key: best["value"]}
            rows.extend(mrows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tuning_{stage}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
    print("tuned_hparams:")
    for rk, tab in selected.items():
        print(f"  {rk}:")
        for m, kv in tab.items():
            print(f"    \"{m}\": {{{', '.join(f'{k}: {v:g}' for k, v in kv.items())}}}")
    return rows, dict(selected)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=("probe", "final"), default="final")
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--R", nargs="*", type=float, default=list(R_VALUES))
    args = p.parse_args(argv)
    tune(args.stage, [Method(m) for m in args.methods] if args.methods else None, args.R)


if __name__ == "__main__":
    main()
