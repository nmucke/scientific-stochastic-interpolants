"""Per-method and cross-method diagnostic figures from saved NS ensemble states.

For every saved-state ``.npz`` (one per method x scenario, written by
``+save_states=true``) this builds, for a single seed:

  * a PER-METHOD figure ``method__<method>__<scenario>.png`` with
      - truth at the final assimilated step,
      - posterior mean at the final step,
      - per-grid-point posterior std at the final step,
      - the radial kinetic-energy spectrum at the final step (posterior vs truth),
      - RMSE / CRPS / spread-skill / KL-vs-EnKF as a function of assimilation step;
  * a per-scenario CROSS-METHOD figure ``compare__<scenario>.png`` overlaying all
      methods' RMSE, CRPS, KL, spread-skill vs step, plus every method's final-step
      energy spectrum against the truth.

KL is computed against the E=1000 non-localized EnKF ground-truth posterior
(``states_E1000_noloc``), at the SAME observed/unobserved point split the headline
``kl_points`` metric uses; methods/seeds with no matching EnKF reference get KL=NaN.

Usage:
    .venv/bin/python paper_experiments/make_method_figures.py \
        --states paper_experiments/results/states_gen \
        --ref    paper_experiments/results/states_E1000_noloc \
        --seed 0 --out paper_experiments/results/method_figures
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scisi.metrics.accuracy import ensemble_mean_rmse
from scisi.metrics.calibration import crps, spread_skill
from scisi.metrics.distributional import kl_at_points
from scisi.metrics.spectral import radial_kinetic_energy_spectrum

LEN_HISTORY = 5  # seeded history prefix; assimilated steps are LEN_HISTORY..T-1
KL_NUM_POINTS = 16

SCEN_DISPLAY = {
    "sparse_5": "sparse 5%",
    "sparse_1_5625": "sparse 1.5625%",
    "32_2_128_2": "32^2 -> 128^2",
    "16_2_128_2": "16^2 -> 128^2",
}
METHOD_DISPLAY = {
    "Ours_SI_SDE": "Ours (SI-SDE)",
    "Ours_FM_SDE": "Ours (FM-SDE)",
    "Ours_FM_ODE": "Ours (FM-ODE)",
    "FlowDAS": "FlowDAS",
    "Guided_FM": "Guided FM",
    "Guided_diffusion": "Guided diffusion",
    "SDA": "SDA",
    "EnKF": "EnKF (ref)",
    "Ensemble_score_filter": "EnSF",
    "Particle_filter": "Particle filter",
}
METHOD_ORDER = list(METHOD_DISPLAY.keys())


def _parse(fn: str) -> tuple[str, str, str]:
    """(method_slug, scenario_slug, seed_str) from a states filename."""
    parts = Path(fn).stem.split("__")
    return parts[1], parts[2], parts[3]


def _kl_point_split(obs_indices, c, h, w):
    """Replicate compute_metrics' fixed obs/unobs point selection for KL."""
    flat = c * h * w
    obs_mask = torch.zeros(flat, dtype=torch.bool)
    idx = np.asarray(obs_indices).reshape(-1).astype(np.int64)
    if idx.size:
        obs_mask[idx] = True
    obs_idx = torch.nonzero(obs_mask).reshape(-1)
    unobs_idx = torch.nonzero(~obs_mask).reshape(-1)
    half = max(KL_NUM_POINTS // 2, 1)

    def pick(ix, k):
        if ix.numel() == 0:
            return ix
        sel = torch.linspace(0, ix.numel() - 1, min(k, ix.numel())).long()
        return ix[sel]

    pidx = torch.cat([pick(obs_idx, half), pick(unobs_idx, half)])
    pis = torch.cat([
        torch.ones(pick(obs_idx, half).numel(), dtype=torch.bool),
        torch.zeros(pick(unobs_idx, half).numel(), dtype=torch.bool),
    ])
    return pidx, pis


def _series(post, true, ref, pidx, pis):
    """Per-step RMSE / CRPS / spread-skill ratio / KL-vs-ref over assimilated steps."""
    T, E = post.shape[-1], post.shape[0]
    steps = list(range(LEN_HISTORY, T))
    rmse, cr, ss, kl = [], [], [], []
    for t in steps:
        ens, tg = post[..., t], true[0, ..., t]
        rmse.append(float(ensemble_mean_rmse(ens, tg)))
        cr.append(float(crps(ens, tg)))
        try:
            ss.append(float(spread_skill(ens, tg)["ratio"]))
        except Exception:
            ss.append(float("nan"))
        if ref is not None and pidx is not None:
            ef = ens.reshape(E, -1)[:, pidx]
            rf = ref[..., t].reshape(ref.shape[0], -1)[:, pidx]
            kl.append(float(kl_at_points(sampled=ef, reference=rf,
                                         observed_mask=pis, method="gaussian")["mean"]))
        else:
            kl.append(float("nan"))
    return steps, rmse, cr, ss, kl


def _load(fn):
    d = np.load(fn, allow_pickle=True)
    return (torch.from_numpy(d["posterior_trajectory"]).float(),
            torch.from_numpy(d["true_trajectory"]).float(), d)


def per_method_figure(method, scen_disp, post, true, series, out):
    steps, rmse, cr, ss, kl = series
    t = post.shape[-1] - 1
    true_f = true[0, 0, :, :, t]
    mean_f = post[:, 0, :, :, t].mean(0)
    std_f = post[:, 0, :, :, t].std(0)
    vmin, vmax = float(true_f.min()), float(true_f.max())

    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])
    a0 = fig.add_subplot(gs[0, 0]); im0 = a0.imshow(true_f, vmin=vmin, vmax=vmax, cmap="RdBu_r"); a0.set_title("Truth (final step)"); plt.colorbar(im0, ax=a0, fraction=0.046)
    a1 = fig.add_subplot(gs[0, 1]); im1 = a1.imshow(mean_f, vmin=vmin, vmax=vmax, cmap="RdBu_r"); a1.set_title("Posterior mean"); plt.colorbar(im1, ax=a1, fraction=0.046)
    a2 = fig.add_subplot(gs[0, 2]); im2 = a2.imshow(std_f, cmap="viridis"); a2.set_title("Posterior std"); plt.colorbar(im2, ax=a2, fraction=0.046)
    for a in (a0, a1, a2):
        a.set_xticks([]); a.set_yticks([])

    # energy spectrum (final step): posterior mean vs truth
    a3 = fig.add_subplot(gs[1, 0])
    kk, ek = radial_kinetic_energy_spectrum(mean_f)
    kt, et = radial_kinetic_energy_spectrum(true_f)
    eps = 1e-12
    a3.loglog(np.asarray(kt) + eps, np.asarray(et) + eps, "k-", label="truth")
    a3.loglog(np.asarray(kk) + eps, np.asarray(ek) + eps, "r--", label="posterior")
    a3.set_xlabel("wavenumber k"); a3.set_ylabel("E(k)"); a3.set_title("Energy spectrum (final)"); a3.legend(fontsize=8)

    # metrics vs step: RMSE/CRPS/spread-skill on left, KL on right
    a4 = fig.add_subplot(gs[1, 1:])
    a4.plot(steps, rmse, "o-", label="RMSE", color="tab:blue")
    a4.plot(steps, cr, "s-", label="CRPS", color="tab:green")
    a4.plot(steps, ss, "^-", label="spread/skill", color="tab:orange")
    a4.axhline(1.0, color="tab:orange", ls=":", lw=1)
    a4.set_xlabel("assimilation step"); a4.set_ylabel("RMSE / CRPS / spread-skill")
    a4b = a4.twinx()
    a4b.plot(steps, kl, "d-", label="KL vs EnKF", color="tab:red")
    a4b.set_ylabel("KL vs EnKF", color="tab:red"); a4b.tick_params(axis="y", labelcolor="tab:red")
    l1, lab1 = a4.get_legend_handles_labels(); l2, lab2 = a4b.get_legend_handles_labels()
    a4.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper right")
    a4.set_title("Metrics vs assimilation step")

    fig.suptitle(f"{METHOD_DISPLAY.get(method, method)}  —  {scen_disp}", y=1.0, fontsize=14)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def compare_figure(scen_disp, per_method, truth_final, out):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    panels = [("rmse", "RMSE"), ("crps", "CRPS"), ("kl", "KL vs EnKF"), ("ss", "spread/skill")]
    for (key, title), ax in zip(panels, axes.flat[:4]):
        for m, d in per_method.items():
            ax.plot(d["steps"], d[key], "o-", ms=3, label=METHOD_DISPLAY.get(m, m))
        if key == "ss":
            ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xlabel("assimilation step"); ax.set_title(f"{title} vs step"); ax.legend(fontsize=7)

    # energy spectra at final step, all methods + truth
    ax = axes.flat[4]
    eps = 1e-12
    kt, et = radial_kinetic_energy_spectrum(truth_final)
    ax.loglog(np.asarray(kt) + eps, np.asarray(et) + eps, "k-", lw=2.0, label="truth")
    for m, d in per_method.items():
        kk, ek = d["spec"]
        ax.loglog(np.asarray(kk) + eps, np.asarray(ek) + eps, "--", lw=1.0, label=METHOD_DISPLAY.get(m, m))
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("E(k)"); ax.set_title("Energy spectrum (final)"); ax.legend(fontsize=7)
    axes.flat[5].axis("off")

    fig.suptitle(f"Cross-method comparison  —  {scen_disp}", y=1.0, fontsize=15)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="paper_experiments/results/states_gen")
    ap.add_argument("--ref", default="paper_experiments/results/states_E1000_noloc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper_experiments/results/method_figures")
    args = ap.parse_args()
    out_dir = Path(args.out)

    files = sorted(glob.glob(str(Path(args.states) / f"*__seed{args.seed}__*.npz")))
    # group by scenario
    by_scen: dict[str, list[str]] = {}
    for f in files:
        _, scen, _ = _parse(f)
        by_scen.setdefault(scen, []).append(f)

    for scen, fs in by_scen.items():
        scen_disp = SCEN_DISPLAY.get(scen, scen)
        # EnKF reference for this scenario (for KL)
        refg = glob.glob(str(Path(args.ref) / f"*__EnKF__{scen}__seed{args.seed}__*.npz"))
        ref = torch.from_numpy(np.load(refg[0], allow_pickle=True)["posterior_trajectory"]).float() if refg else None

        compare = {}
        truth_final = None
        for f in sorted(fs, key=lambda p: METHOD_ORDER.index(_parse(p)[0]) if _parse(p)[0] in METHOD_ORDER else 99):
            method, _, _ = _parse(f)
            post, true, data = _load(f)
            if not torch.isfinite(post[..., -1]).all():
                print(f"  skip {method}/{scen}: non-finite states")
                continue
            c, h, w = post.shape[1], post.shape[2], post.shape[3]
            pidx, pis = _kl_point_split(data["obs_indices"], c, h, w) if ref is not None else (None, None)
            series = _series(post, true, ref, pidx, pis)
            per_method_figure(method, scen_disp, post, true, series,
                              out_dir / f"method__{method}__{scen}.png")
            steps, rmse, cr, ss, kl = series
            kk, ek = radial_kinetic_energy_spectrum(post[:, 0, :, :, -1].mean(0))
            compare[method] = {"steps": steps, "rmse": rmse, "crps": cr, "ss": ss, "kl": kl, "spec": (kk, ek)}
            truth_final = true[0, 0, :, :, -1]
            print(f"  wrote method__{method}__{scen}.png")
        if compare:
            compare_figure(scen_disp, compare, truth_final, out_dir / f"compare__{scen}.png")
            print(f"  wrote compare__{scen}.png")


if __name__ == "__main__":
    main()
