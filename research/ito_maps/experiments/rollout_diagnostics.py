#!/usr/bin/env python
"""D5, D6 and D7: rollout decomposition, budget residual, stationary statistics.

Works for any model with the solver ``sample`` or the map ``sample`` interface
(see ``phase0_common.sample_next``), so it can be rerun on retrained maps.

``short`` (D5 + D6, leads 1..N with truth, validation only). Per lead:

* ``variance_over_climatology``: pooled unbiased member variance (spatial mean)
  over the climatological per-pixel variance of the validation fields;
* ``pattern_corr_member_member`` / ``pattern_corr_member_truth``: mean
  centred pattern correlation over member pairs and member-truth pairs. For
  an exchangeable (calibrated) ensemble the two have the same expectation;
* injected vs propagated spread: one extra step from the lead-(n-1) member
  histories with (a) one shared Brownian path for all members (spread from
  the histories only: ``propagated``) and (b) E fresh draws from the first
  member's history (spread created in one step: ``injected``);
* D6: budget residual ``r = q_+(y_n) - q_-(y_{n-1}) - Delta I`` per member
  (enstrophy and energy, relative to the injection), the same for truth, and
  the unbiased U-statistic of ``E[r | h]^2`` on the same-history draws.

``long`` (D6 + D7 + D5 at long leads) reads a ``long_horizon_ns.py`` run saved
with ``--raw-every 1``: budget residual by lead and member; pooled enstrophy,
member shell spectra and lag-1 pattern autocorrelation after a burn-in,
against the validation data's whole-trajectory bootstrap band.

All statistics use denormalised fields in float64. Uncertainty bands
bootstrap whole trajectories (cases of one trajectory stay together).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ns_budget import build_ns_budgets
from phase0_common import (
    NS_VALIDATION,
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
from spatial_enstrophy_spectra import radial_enstrophy_shells


# ------------------------------------------------------------ statistics


def centred(x):
    """Remove each field's spatial mean: ``[..., H, W]``."""
    return x - x.mean(axis=(-2, -1), keepdims=True)


def pattern_corr(a, b):
    """Centred pattern correlation over the last two axes (broadcasting)."""
    a, b = centred(a), centred(b)
    num = (a * b).sum((-2, -1))
    return num / np.sqrt((a * a).sum((-2, -1)) * (b * b).sum((-2, -1)))


def member_member_corr(ens):
    """Mean pattern correlation over ordered pairs ``e != e'``: ``[E, H, W] -> float``."""
    a = centred(ens).reshape(len(ens), -1)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    gram = a @ a.T
    E = len(ens)
    return float((gram.sum() - np.trace(gram)) / (E * (E - 1)))


def member_variance(ens):
    """Spatial mean of the unbiased member variance of ``[E, ..., H, W]``."""
    return float(ens.var(0, ddof=1).mean())


def boot_band(values, groups, n_boot=2000, seed=0):
    """Mean and 90% whole-group bootstrap band of per-case ``values``."""
    values, groups = np.asarray(values, dtype=np.float64), np.asarray(groups)
    unique = np.unique(groups)
    per = np.array([values[groups == g].mean() for g in unique])
    rng = np.random.default_rng(seed)
    boot = per[rng.integers(0, len(per), size=(n_boot, len(per)))].mean(1)
    return {
        "mean": float(per.mean()),
        "q05": float(np.quantile(boot, 0.05)),
        "q95": float(np.quantile(boot, 0.95)),
    }


def residuals(budgets, y_next, y_prev):
    """Budget residual / injection per sample for physical ``[B, C, H, W]`` arrays."""
    yn = torch.as_tensor(y_next, dtype=torch.float64)
    yp = torch.as_tensor(y_prev, dtype=torch.float64)
    return {
        name: (b.residual(yn, yp) / b.injection).numpy()
        for name, b in budgets.items()
        if not name.startswith("_")
    }


def u_stat_relative(budgets, members, prev):
    """Unbiased ``(E[r|h]/inj)^2`` from same-history members ``[E, C, H, W]``."""
    from scisi.metrics import u_statistic

    r = residuals(budgets, members, np.ascontiguousarray(np.broadcast_to(prev, members.shape)))
    return {k: float(u_statistic(torch.as_tensor(v))) for k, v in r.items()}


def data_climatology(val_phys):
    """Per-pixel variance pooled over validation fields (spatial mean), mean field."""
    fields = np.moveaxis(val_phys, -1, 1).reshape(-1, *val_phys.shape[1:-1])
    mean = fields.mean(0)
    return float(((fields - mean) ** 2).mean()), mean


# ---------------------------------------------------------------- short


def run_short(args, model, config):
    mean, std = preprocessing_affine(config)
    L = int(config.len_field_history)
    ids = list(range(*args.val_ids))[: args.trajectories]
    val = load_trajectories(config, (NS_VALIDATION[0], NS_VALIDATION[1]), args.data_cache)
    val_phys = val.double().numpy()
    clim_var, _ = data_climatology(val_phys)
    budgets = build_ns_budgets(args.budget_config, *val.shape[2:4])
    local = [i - NS_VALIDATION[0] for i in ids]
    E, N = args.ensemble, args.leads
    rows, raw = [], {}
    phys = lambda x: x.double().numpy() * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
    out_dir = Path(args.arrays_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resumable per case: each finished case stores its rows next to its
    # ensemble array, with the settings that produced them, and is reused.
    # Each case has its own seed, so a resumed run equals an uninterrupted one.
    settings = {
        "checkpoint_sha256": sha256(Path(args.checkpoint_dir) / "model.pth"),
        "ensemble": E, "leads": N, "steps": args.steps, "si_noise": args.si_noise, "seed": args.seed,
    }
    for li, tid in zip(local, ids):
        x = normalise(val[li : li + 1], mean, std)
        for origin in args.origins:
            if origin + N > val.shape[-1]:
                raise ValueError("Not enough truth frames for the requested leads")
            case_rows = out_dir / f"short_traj{tid:03d}_o{origin:03d}_rows.json"
            if case_rows.exists() and (out_dir / f"short_traj{tid:03d}_o{origin:03d}.npy").exists():
                stored = json.loads(case_rows.read_text())
                if stored["settings"] != settings:
                    raise ValueError(f"{case_rows} was produced with other settings")
                rows.extend(stored["rows"])
                print(f"{time.strftime('%T')} traj {tid} origin {origin} reused", flush=True)
                continue
            gen = make_generator(args.device, args.seed + 1000 * tid + origin)
            first_row = len(rows)
            history = x[..., origin - L : origin].repeat(E, 1, 1, 1, 1)
            prev_members = phys(history[..., -1])
            truth_prev = val_phys[li, ..., origin - 1]
            ens_all = []
            for lead in range(1, N + 1):
                y = sample_next(model, history, args.steps, si_noise=args.si_noise, generator=gen).cpu()
                shared = sample_next(model, history, args.steps, si_noise=args.si_noise, shared=True, generator=gen).cpu()
                inj = sample_next(model, history[:1].repeat(E, 1, 1, 1, 1), args.steps, si_noise=args.si_noise, generator=gen).cpu()
                y_p, sh_p, inj_p = phys(y), phys(shared), phys(inj)
                truth = val_phys[li, ..., origin + lead - 1]
                r_mem = residuals(budgets, y_p, prev_members)
                r_true = residuals(budgets, truth[None], truth_prev[None])
                u_inj = u_stat_relative(budgets, inj_p, prev_members[:1])
                rows.append(
                    {
                        "trajectory": tid,
                        "origin": origin,
                        "lead": lead,
                        "variance_over_climatology": member_variance(y_p[:, 0]) / clim_var,
                        "propagated_variance_over_climatology": member_variance(sh_p[:, 0]) / clim_var,
                        "injected_variance_over_climatology": member_variance(inj_p[:, 0]) / clim_var,
                        "pattern_corr_member_member": member_member_corr(y_p[:, 0]),
                        "pattern_corr_member_truth": float(pattern_corr(y_p[:, 0], truth[0]).mean()),
                        "mse_ensemble_mean_over_climatology": float(((y_p.mean(0) - truth) ** 2).mean()) / clim_var,
                        **{f"residual_{k}_mean": float(v.mean()) for k, v in r_mem.items()},
                        **{f"residual_{k}_members": v.tolist() for k, v in r_mem.items()},
                        **{f"truth_residual_{k}": float(v[0]) for k, v in r_true.items()},
                        **{f"u_stat_same_history_{k}": v for k, v in u_inj.items()},
                    }
                )
                ens_all.append(y.numpy())
                prev_members, truth_prev = y_p, truth
                history = torch.cat([history[..., 1:], y.unsqueeze(-1).to(history.dtype)], -1)
            atomic_npy(out_dir / f"short_traj{tid:03d}_o{origin:03d}.npy", np.stack(ens_all, -1).astype(np.float32))
            # Plain JSON with NaN allowed, so a reused row equals the computed one.
            tmp = case_rows.with_name(case_rows.name + ".tmp")
            tmp.write_text(json.dumps({"settings": settings, "rows": rows[first_row:]}, allow_nan=True))
            os.replace(tmp, case_rows)
            print(f"{time.strftime('%T')} traj {tid} origin {origin} done", flush=True)
    keys = [k for k in rows[0] if k not in ("trajectory", "origin", "lead") and not k.endswith("_members")]
    by_lead = {}
    for lead in range(1, N + 1):
        sel = [r for r in rows if r["lead"] == lead]
        groups = [r["trajectory"] for r in sel]
        by_lead[str(lead)] = {k: boot_band([r[k] for r in sel], groups, seed=lead) for k in keys}
        by_lead[str(lead)]["finite_E_calibrated_mse_over_variance"] = 1 + 1 / E
    return {"climatology_variance": clim_var, "rows": rows, "by_lead": by_lead,
            "cases": {"trajectories": ids, "origins": args.origins, "ensemble": E}}


# ----------------------------------------------------------------- long


def data_reference(val_phys, burn=0, n_boot=2000):
    """Validation-data stationary statistics with whole-trajectory bootstrap."""
    fields = val_phys[:, 0]  # [n, H, W, T]
    fields = np.moveaxis(fields, -1, 1)[:, burn:]  # [n, T, H, W]
    z = 0.5 * (fields**2).mean((-2, -1))
    k, spec, _ = radial_enstrophy_shells(fields)
    lag1 = pattern_corr(fields[:, 1:], fields[:, :-1])
    rng = np.random.default_rng(1)
    draws = rng.integers(0, len(fields), size=(n_boot, len(fields)))

    def band(per_traj):
        boot = per_traj[draws].mean(1)
        return {"mean": per_traj.mean(0).tolist(), "q05": np.quantile(boot, 0.05, 0).tolist(), "q95": np.quantile(boot, 0.95, 0).tolist()}

    return {
        "enstrophy_mean": band(z.mean(1)),
        "enstrophy_pooled_q05_q95": [float(np.quantile(z, 0.05)), float(np.quantile(z, 0.95))],
        "enstrophy_std_pooled": float(z.std()),
        "shell_spectrum": band(spec.mean(1)),
        "lag1_pattern_autocorr": band(lag1.mean(1)),
        "k": k.tolist(),
    }


def run_long(args, config):
    run = Path(args.run_dir)
    protocol = json.loads((run / "protocol.json").read_text())
    diag = json.loads((run / "diagnostics.json").read_text())
    std = np.asarray(protocol["inverse_affine"]["std"])
    mean = np.asarray(protocol["inverse_affine"]["mean"])
    snaps = {s["lead"]: s for s in diag["raw_snapshots"]}
    completed = diag["completed_leads"]
    leads = [l for l in range(1, completed + 1) if l in snaps]
    if leads != list(range(1, completed + 1)):
        raise ValueError("Long-run diagnostics need raw states at every lead (--raw-every 1)")
    val = load_trajectories(config, NS_VALIDATION, args.data_cache)
    val_phys = val.double().numpy()
    clim_var, _ = data_climatology(val_phys)
    budgets = build_ns_budgets(args.budget_config, *val.shape[2:4])
    traj = protocol["trajectories"]
    origin = protocol["origin"]
    prev = np.stack([val_phys[t - NS_VALIDATION[0], ..., origin - 1] for t in traj])  # [B,C,H,W]
    B, E = len(traj), protocol["ensemble_size"]
    prev = np.repeat(prev[:, None], E, 1)
    per_lead, z_all, spec_all, lag_all = [], [], [], []
    last = None
    for lead in leads:
        raw = np.load(run / snaps[lead]["path"]).astype(np.float64)  # [B,E,C,H,W]
        y = raw * std.reshape(1, 1, -1, 1, 1) + mean.reshape(1, 1, -1, 1, 1)
        r = residuals(budgets, y.reshape(B * E, *y.shape[2:]), prev.reshape(B * E, *y.shape[2:]))
        row = {"lead": lead}
        for name, v in r.items():
            v = v.reshape(B, E)
            row[f"residual_{name}_mean"] = float(v.mean())
            row[f"residual_{name}_by_traj"] = v.mean(1).tolist()
        row["variance_over_climatology"] = float(np.mean([member_variance(y[b, :, 0]) for b in range(B)]) / clim_var)
        row["pattern_corr_member_member"] = float(np.mean([member_member_corr(y[b, :, 0]) for b in range(B)]))
        if lead <= protocol["available_truth_leads"]:
            truth = np.stack([val_phys[t - NS_VALIDATION[0], ..., origin + lead - 1] for t in traj])
            row["pattern_corr_member_truth"] = float(np.mean([pattern_corr(y[b, :, 0], truth[b, 0]).mean() for b in range(B)]))
        if lead > args.burn_in:
            z_all.append(0.5 * (y[:, :, 0] ** 2).mean((-2, -1)))  # [B,E]
            spec_all.append(radial_enstrophy_shells(y[:, :, 0])[1])  # [B,E,K]
            if last is not None:
                lag_all.append(pattern_corr(y[:, :, 0], last[:, :, 0]))
        per_lead.append(row)
        last, prev = y, y
    z_all, spec_all = np.stack(z_all, -1), np.stack(spec_all, -2)  # [B,E,T], [B,E,T,K]
    lag_all = np.stack(lag_all, -1)
    ref = data_reference(val_phys)
    chains = lambda a: a.reshape(B * E, *a.shape[2:])
    rng = np.random.default_rng(2)
    draws = rng.integers(0, B, size=(2000, B))

    def model_band(per_traj):  # bootstrap whole trajectories (all members of a case)
        boot = per_traj[draws].mean(1)
        return {"mean": per_traj.mean(0).tolist(), "q05": np.quantile(boot, 0.05, 0).tolist(), "q95": np.quantile(boot, 0.95, 0).tolist()}

    model_stats = {
        "enstrophy_mean": model_band(z_all.mean((1, 2))),
        "enstrophy_pooled_q05_q95": [float(np.quantile(z_all, 0.05)), float(np.quantile(z_all, 0.95))],
        "enstrophy_std_pooled": float(z_all.std()),
        "shell_spectrum": model_band(spec_all.mean((1, 2))),
        "lag1_pattern_autocorr": model_band(lag_all.mean((1, 2))),
    }
    inside = lambda m, r: bool(r["q05"] <= m <= r["q95"])
    spec_m, spec_r = np.asarray(model_stats["shell_spectrum"]["mean"]), ref["shell_spectrum"]
    in_band = (spec_m >= np.asarray(spec_r["q05"])) & (spec_m <= np.asarray(spec_r["q95"]))
    kmax = min(64, len(spec_m) - 1)
    verdict = {
        "enstrophy_mean_inside_data_band": inside(model_stats["enstrophy_mean"]["mean"], ref["enstrophy_mean"]),
        "lag1_inside_data_band": inside(model_stats["lag1_pattern_autocorr"]["mean"], ref["lag1_pattern_autocorr"]),
        "spectrum_fraction_of_shells_1_64_inside_band": float(in_band[1 : kmax + 1].mean()),
        "spectrum_ratio_model_over_data_shells_1_4_16_32_48_64": {
            str(k): float(spec_m[k] / spec_r["mean"][k]) for k in (1, 4, 16, 32, 48, 64) if k < len(spec_m)
        },
    }
    res_keys = [k for k in per_lead[0] if k.startswith("residual_") and k.endswith("_by_traj")]
    windows = {}
    for lo, hi in ((1, 10), (11, 95), (96, completed), (args.burn_in + 1, completed)):
        sel = [r for r in per_lead if lo <= r["lead"] <= hi]
        if not sel:
            continue
        windows[f"{lo}-{hi}"] = {
            k.replace("_by_traj", ""): model_band(np.mean([r[k] for r in sel], 0)[:, None])
            for k in res_keys
        }
        windows[f"{lo}-{hi}"]["variance_over_climatology"] = float(np.mean([r["variance_over_climatology"] for r in sel]))
        windows[f"{lo}-{hi}"]["pattern_corr_member_member"] = float(np.mean([r["pattern_corr_member_member"] for r in sel]))
    return {
        "run_dir": str(run),
        "protocol": {k: protocol[k] for k in ("model_class", "trajectories", "origin", "num_steps", "ensemble_size", "horizon") if k in protocol} | {"solver_stepper": protocol.get("solver_stepper", "euler_maruyama")},
        "completed_leads": completed,
        "burn_in": args.burn_in,
        "climatology_variance": clim_var,
        "per_lead": per_lead,
        "residual_windows": windows,
        "stationary_model": model_stats,
        "stationary_data_validation": ref,
        "d7_verdict": verdict,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=["short", "long"])
    p.add_argument("--checkpoint-dir", required=True, help="model (short) or dataset config source (long)")
    p.add_argument("--run-dir", help="long: long_horizon_ns.py output directory")
    p.add_argument("--val-ids", nargs=2, type=int, default=list(NS_VALIDATION))
    p.add_argument("--trajectories", type=int, default=4)
    p.add_argument("--origins", nargs="+", type=int, default=[5, 45, 85])
    p.add_argument("--ensemble", type=int, default=8)
    p.add_argument("--leads", type=int, default=10)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--si-noise", default="exact", choices=["em", "exact"])
    p.add_argument("--burn-in", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-cache")
    p.add_argument("--budget-config")
    p.add_argument("--arrays-dir", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    t0 = time.time()
    model, config = load_model(args.checkpoint_dir, args.device if args.mode == "short" else "cpu")
    result = {
        "diagnostic": "D5/D6 short rollout" if args.mode == "short" else "D5/D6/D7 long rollout",
        "argv": sys.argv if argv is None else argv,
        "checkpoint": str(Path(args.checkpoint_dir) / "model.pth"),
        "checkpoint_sha256": sha256(Path(args.checkpoint_dir) / "model.pth"),
        "model": describe_model(model, config),
        "num_steps": args.steps,
        "si_noise": args.si_noise,
        "units": "physical (denormalised) float64; residuals divided by the per-step injection Delta*I",
    }
    if args.mode == "short":
        if args.arrays_dir is None:
            raise ValueError("--arrays-dir is required for short mode")
        result.update(run_short(args, model, config))
    else:
        if args.run_dir is None:
            raise ValueError("--run-dir is required for long mode")
        result.update(run_long(args, config))
    result["seconds"] = time.time() - t0
    atomic_json(args.output, result)
    print(json.dumps(result.get("by_lead", result.get("d7_verdict")), indent=1)[:4000], flush=True)


if __name__ == "__main__":
    main()
