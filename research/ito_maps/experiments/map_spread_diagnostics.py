#!/usr/bin/env python
"""D2, D3, D4: where a learned map's one-step spread comes from, per scale.

All three use the validation cases of D1 (trajectories x origins, same order)
and the per-mode conventions of ``uncertainty_spectrum.py`` (rfft2, ortho,
normalised units). Uncertainty: whole-trajectory bootstrap, paired where two
quantities share cases.

* ``d2`` spread spectrum: per-mode unbiased member variance of the map at
  each budget over that of the multi-step SI reference saved by D1
  (``anomaly_si<steps>_<noise>_per_case``). Prediction: ratio below 1 at the
  forced scales, a flat excess tail at high wavenumber.
* ``d3`` path-feature ablation (Ito maps): the same with the path features
  replaced by zeros at sampling time (the explicit martingale increments are
  kept). Ratio ablated / full measures how much spread the feature branch
  creates. Full and ablated runs use the identical path realisation.
* ``d4`` noise leak (Ito maps): per shell, the regression coefficient of the
  member anomaly ``y_e - ybar`` on the driver anomaly ``M1_e - mean M1``,
  ``beta_k = sum Re(a_k conj m_k) / sum |m_k|^2``. ``beta = 1`` at high
  wavenumber means the white endpoint noise passes through uncancelled.

Same-path rule: a seed fixes a query sequence, not a path. Within a case the
budgets are run from finest to coarsest on ONE path object, so every coarse
query time is already cached (local features: interval subdivisions of a
dyadic grid) and all budgets see the same realisation. This is checked.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from phase0_common import (
    NS_VALIDATION,
    ZeroFeaturePath,
    atomic_json,
    atomic_npy,
    describe_model,
    load_model,
    load_trajectories,
    make_generator,
    normalise,
    preprocessing_affine,
    sample_next,
    sha256,
)
from uncertainty_spectrum import REPORT_SHELLS, FourierBasis, anomaly_power


def cached_times(path):
    times = getattr(path, "_times", None)
    return None if times is None else sum(len(t) for t in times)


def sample_case(model, history, budgets, generator, ablate=False):
    """Samples ``{budget: [E, C, H, W]}`` (and ``M1``) on one shared path.

    Budgets are run in decreasing order so coarse queries reuse cached times.
    Returns ``(samples, m1, same_path_verified)``.
    """
    order = sorted(budgets, reverse=True)
    x0 = history[..., -1].to(model.device)
    path = model.make_brownian_path(x0, generator)
    base = None
    if getattr(model, "gaussian_base", False):  # flow maps: one base for all budgets
        base = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    out, base_count, same = {}, None, True
    for b in order:
        out[b] = sample_next(
            model,
            history,
            b,
            generator=generator,
            brownian_path=path,
            path_wrapper=ZeroFeaturePath if ablate else None,
            base=base,
        ).double().numpy()
        count = cached_times(path) if path is not None else None
        if base_count is None:
            base_count = count
        elif count != base_count:
            same = False
    m1 = None if path is None else path.increment(0.0, 1.0).double().cpu().numpy()
    return out, m1, same


def cases(val, ids, origins, L, mean, std):
    for i, tid in enumerate(ids):
        x = normalise(val[i : i + 1], mean, std)
        for o in origins:
            yield tid, o, x[..., o - L : o], x[0, ..., o].double().numpy()


def shell_ratio_paired(num, den, basis, groups, n_boot=1000, seed=0):
    """Shell mean ratio ``num/den`` (per-case ``[n, C, modes]``), paired bootstrap."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    n_sh = np.stack([basis.shell_means(num[groups == g].mean(0))[0] for g in unique])
    d_sh = np.stack([basis.shell_means(den[groups == g].mean(0))[0] for g in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    boot = n_sh[draws].mean(1) / d_sh[draws].mean(1)
    ratio = n_sh.mean(0) / d_sh.mean(0)
    return ratio[0], np.quantile(boot, 0.05, 0)[0], np.quantile(boot, 0.95, 0)[0]


def table(ratio, lo, hi):
    return {str(k): {"ratio": float(ratio[k]), "q05": float(lo[k]), "q95": float(hi[k])} for k in REPORT_SHELLS if k < len(ratio)}


def leak_coefficients(anoms, m_anoms, basis):
    """Per-shell ``beta_k`` from stacked anomalies ``[n, E, C, H, W]``."""
    a = basis.coefficients(anoms)
    m = basis.coefficients(m_anoms)
    cross = (a.real * m.real + a.imag * m.imag).sum((0, 1))  # [C, modes]
    power = (m.real**2 + m.imag**2).sum((0, 1))
    c_sh = basis.shell_sums(cross[None])[0]
    p_sh = basis.shell_sums(power[None])[0]
    return c_sh / p_sh


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=["d2", "d3", "d4"])
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--budgets", nargs="+", type=int, default=[8, 4, 2, 1])
    p.add_argument("--reference", help="D1 npz (d2): d1_spectra_fft2.npz")
    p.add_argument("--reference-name", default="anomaly_si128_exact")
    p.add_argument("--val-ids", nargs=2, type=int, default=list(NS_VALIDATION))
    p.add_argument("--origins", nargs="+", type=int, default=[5, 25, 45, 65, 85])
    p.add_argument("--ensemble", type=int, default=16)
    p.add_argument(
        "--high-pass-shell",
        type=int,
        help="D4 high-pass threshold; default min(H, W) // 4 (32 at 128x128, 16 at 64x64)",
    )
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-cache")
    p.add_argument("--arrays-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    model, config = load_model(args.checkpoint_dir, args.device)
    info = describe_model(model, config)
    if args.mode in ("d3", "d4") and info["model_class"] != "FollmerItoMapModel":
        raise ValueError(f"{args.mode} needs an Ito map, got {info['model_class']}")
    mean, std = preprocessing_affine(config)
    L = int(config.len_field_history)
    ids = list(range(*args.val_ids))
    val = load_trajectories(config, tuple(args.val_ids), args.data_cache)
    basis = FourierBasis(*val.shape[2:4])
    if args.high_pass_shell is None:
        args.high_pass_shell = min(basis.height, basis.width) // 4
    arrays = Path(args.arrays_dir)
    arrays.mkdir(parents=True, exist_ok=True)
    budgets = sorted(set(args.budgets), reverse=True)
    var = {b: [] for b in budgets}
    var_ablated = {b: [] for b in budgets}
    anoms = {b: [] for b in budgets}
    m_anoms, groups, same_path = [], [], []
    gen = make_generator(args.device, args.seed)
    t0 = time.time()
    for tid, origin, hist, truth in cases(val, ids, args.origins, L, mean, std):
        history = hist.repeat(args.ensemble, 1, 1, 1, 1)
        state = gen.get_state()
        full, m1, same = sample_case(model, history, budgets, gen)
        same_path.append(same)
        if args.mode == "d3":
            gen.set_state(state)  # identical path realisation for the ablated run
            ablated, m1_ablated, _ = sample_case(model, history, budgets, gen, ablate=True)
            if not np.array_equal(m1, m1_ablated):
                raise RuntimeError("Ablated run did not reproduce the path")
        for b in budgets:
            v, _ = anomaly_power(full[b][None], truth[None], basis)
            var[b].append(v[0])
            if args.mode == "d3":
                var_ablated[b].append(anomaly_power(ablated[b][None], truth[None], basis)[0][0])
            if args.mode == "d4":
                anoms[b].append(full[b] - full[b].mean(0, keepdims=True))
        if args.mode == "d4":
            m_anoms.append(m1 - m1.mean(0, keepdims=True))
        groups.append(tid)
        atomic_npy(arrays / f"{args.mode}_traj{tid:03d}_o{origin:03d}.npy", np.stack([full[b] for b in budgets]).astype(np.float32))
    result = {
        "diagnostic": {"d2": "D2 spread spectrum vs SI reference", "d3": "D3 path-feature ablation", "d4": "D4 noise-leak regression"}[args.mode],
        "argv": sys.argv if argv is None else argv,
        "checkpoint": str(Path(args.checkpoint_dir) / "model.pth"),
        "checkpoint_sha256": sha256(Path(args.checkpoint_dir) / "model.pth"),
        "model": info,
        "cases": {"trajectories": ids, "origins": args.origins, "ensemble": args.ensemble},
        "budgets": budgets,
        "same_path_across_budgets_verified": bool(all(same_path)),
        "seconds": time.time() - t0,
        "units": "per-mode rfft2 ortho power of normalised fields; shells as in D1",
    }
    var = {b: np.stack(v) for b, v in var.items()}
    if args.mode == "d2":
        ref = np.load(args.reference)
        ref_case = ref[args.reference_name + "_per_case"].astype(np.float64)
        if len(ref_case) != len(groups):
            raise ValueError("Reference cases differ from these cases (origins/trajectories)")
        result["reference"] = {"file": args.reference, "name": args.reference_name}
        result["spread_ratio_by_budget"] = {}
        for b in budgets:
            ratio, lo, hi = shell_ratio_paired(var[b], ref_case, basis, groups)
            result["spread_ratio_by_budget"][str(b)] = {"table": table(ratio, lo, hi), "all_shells": ratio.tolist()}
    elif args.mode == "d3":
        result["ablated_over_full_by_budget"] = {}
        for b in budgets:
            ratio, lo, hi = shell_ratio_paired(np.stack(var_ablated[b]), var[b], basis, groups)
            result["ablated_over_full_by_budget"][str(b)] = {"table": table(ratio, lo, hi), "all_shells": ratio.tolist()}
    else:
        m = np.stack(m_anoms)
        result["leak_coefficient_by_budget"] = {}
        for b in budgets:
            beta = leak_coefficients(np.stack(anoms[b]), m, basis)[0]
            hp = basis.shell >= args.high_pass_shell
            a = basis.coefficients(np.stack(anoms[b]))
            mm = basis.coefficients(m)
            w = basis.multiplicity * hp
            cross = ((a.real * mm.real + a.imag * mm.imag).sum((0, 1))[0] * w).sum()
            power = ((mm.real**2 + mm.imag**2).sum((0, 1))[0] * w).sum()
            result["leak_coefficient_by_budget"][str(b)] = {
                "by_shell": {str(k): float(beta[k]) for k in REPORT_SHELLS if k < len(beta)},
                "all_shells": beta.tolist(),
                f"high_pass_shells_ge_{args.high_pass_shell}": float(cross / power),
            }
    atomic_json(args.output, result)
    print({k: v for k, v in result.items() if k.endswith("budget")}, flush=True)


if __name__ == "__main__":
    main()
