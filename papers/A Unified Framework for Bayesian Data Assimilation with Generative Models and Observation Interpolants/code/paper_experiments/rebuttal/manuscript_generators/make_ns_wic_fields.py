#!/usr/bin/env python3
"""Field figure for Navier--Stokes from a poor initial condition, super-resolution 16^2:
truth, observations (dots), and posterior mean / ensemble std / |error| at the final
assimilation step of one held-out trajectory, for TMPD, FlowDAS + SURGE, D-Flow SGLD,
Ours (FM-ODE, Jac-free) and Ours (FM-ODE, Jac-free) + corr. (r=20).

Styling follows the manuscript's Figure 3 (figure_common.make_state_panel_singles /
save_field_panel): RdBu_r vorticity on the truth's symmetric scale; magma std and |error|
on a shared robust scale (each method's 99th percentile, capped at 4x their median);
observations as square markers over a faint grey truth; one vertical colour bar per row.

Writes singles to ./figures/ns_wic_fields/:
    ns_wic16__truth.pdf, ns_wic16__obs.pdf,
    ns_wic16__{mean,std,abserr}__<slug>.pdf, ns_wic16__cbar_{field,std,abserr}.pdf

Appendix field figures (18 methods, three blocks of six), one per scenario:
    poor IC (35 assimilation steps): ns_wic_<slug>__*  for slug in SCENARIOS (32^2, 16^2)
    true IC (15 assimilation steps): ns_tic_<slug>__*  for slug in TIC_SCENARIOS (32^2, 16^2,
        sparse 5%, sparse 1.5625%); file locations: the TIC_* table below. A state file that is
        not there yet is drawn as a grey "pending" panel and listed loudly on stdout.
The panel shown is posterior_trajectory[..., -1], i.e. the last assimilation step (it matches
per_step_rmse[-1]; the per-step figures number the steps 0..n-1).

Regenerate the six appendix figures (panels + LaTeX) with one command, from this directory:
    python3 make_ns_wic_fields.py && python3 make_ns_wic32_tex.py
(`make_ns_wic_fields.py legacy` re-renders the legacy ns_wic16/ns_wic16b/ns_wic32 panels;
not part of the default run. `make_ns_wic_fields.py tic` / `wic` / <slug> ... for subsets.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

NS = Path("/export/scratch2/ntm/delftblue_results/results/navier_stokes_wrong_ic_np40")
OUT = Path(__file__).resolve().parent / "figures" / "ns_wic_fields"
STEM = "ns_wic16"
SC = "16_2_128_2"
METHODS = (  # slug, file
    ("tmpd", NS / f"tmpd/navier_stokes/states/traj11/navier_stokes__TMPD__{SC}__seed0__E64_M100.npz"),
    ("flowdas_surge", NS / f"main/navier_stokes/states/traj11/navier_stokes__SURGE_FlowDAS__{SC}__seed0__E64_M100.npz"),
    ("d_flow_sgld", NS / f"main/navier_stokes/states/traj11/navier_stokes__D_Flow_SGLD__{SC}__seed0__E64_M100.npz"),
    ("fm_ode_jac_free", NS / f"main/navier_stokes/states/traj11/navier_stokes__Ours_FM_ODE__{SC}__jacfree__seed0__E64_M100.npz"),
    ("fm_ode_jac_free_corr", NS / f"ext/navier_stokes/states/traj11/navier_stokes__Ours_FM_ODE__{SC}__jacfree_fc25_8797d2ae__seed0__E64_M100.npz"),
)
# second figure: FM-ODE shared/Jac-free (+ corr.) and FM-ODE prior + EnKF (rows 1-2), five baselines (rows 3-4)
STEM_B = "ns_wic16b"
METHODS_B = (
    ("fm_ode_shared", NS / f"main/navier_stokes/states/traj11/navier_stokes__Ours_FM_ODE__{SC}__shared_jac10__seed0__E64_M100.npz"),
    ("fm_ode_shared_corr", NS / f"ext/navier_stokes/states/traj11/navier_stokes__Ours_FM_ODE__{SC}__shared_jac10_fc25_8797d2ae__seed0__E64_M100.npz"),
    ("fm_ode_jac_free", NS / f"main/navier_stokes/states/traj11/navier_stokes__Ours_FM_ODE__{SC}__jacfree__seed0__E64_M100.npz"),
    ("fm_ode_jac_free_corr", NS / f"ext/navier_stokes/states/traj11/navier_stokes__Ours_FM_ODE__{SC}__jacfree_fc25_8797d2ae__seed0__E64_M100.npz"),
    ("fm_ode_prior_enkf", NS / f"prior_enkf/navier_stokes/states/traj11/navier_stokes__FM_ODE_prior_EnKF__{SC}__seed0__E64_M100.npz"),
    ("tmpd", NS / f"tmpd/navier_stokes/states/traj11/navier_stokes__TMPD__{SC}__seed0__E64_M100.npz"),
    ("flowdas_surge", NS / f"main/navier_stokes/states/traj11/navier_stokes__SURGE_FlowDAS__{SC}__seed0__E64_M100.npz"),
    ("d_flow_sgld", NS / f"main/navier_stokes/states/traj11/navier_stokes__D_Flow_SGLD__{SC}__seed0__E64_M100.npz"),
    ("guided_fm_fig", NS / f"main/navier_stokes/states/traj11/navier_stokes__Guided_FM_FIG__{SC}__seed0__E64_M100.npz"),
    ("sda_surge", NS / f"main/navier_stokes/states/traj11/navier_stokes__SURGE_SDA__{SC}__seed0__E64_M100.npz"),
)
PANEL = 2.6          # design size (in), as figure_common
CBAR_H = 1.04        # printed colour-bar height (in), as figure_common


def style():
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 150})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200 if ext == "png" else None)
    plt.close(fig)


def panel(field, kw, name, scatter=None, background=None, pending=False):
    fig, ax = plt.subplots(figsize=(PANEL, PANEL))
    if pending:  # result file not there yet: same style as the "diverged (NaN)" panel
        H, W = background.shape
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_facecolor("0.9")
        ax.text(0.5, 0.5, pending if isinstance(pending, str) else "pending", transform=ax.transAxes, ha="center", va="center",
                fontsize=18, color="0.25")
        background = field = None
    else:
        H, W = (background if background is not None else field).shape
    if background is not None:
        ax.imshow(background, origin="lower", cmap="Greys", alpha=0.28, extent=(0, W, 0, H), aspect="equal")
    if field is not None:
        ax.imshow(field, origin="lower", **kw)
        if not np.isfinite(field).any():
            ax.set_facecolor("0.9")
            ax.text(0.5, 0.5, "diverged\n(NaN)", transform=ax.transAxes, ha="center", va="center",
                    fontsize=18, color="0.25")
    if scatter is not None:
        xs, ys, vals = scatter
        ax.scatter(xs, ys, c=vals, cmap=kw["cmap"], vmin=kw["vmin"], vmax=kw["vmax"], s=10.0,
                   linewidths=0.0, marker="s")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    save(fig, name)


def robust_vmax(fields):
    p99 = [float(np.percentile(f[np.isfinite(f)], 99)) for f in fields if np.isfinite(f).any()]
    if not p99:
        return 1.0
    med = float(np.median(p99))
    return (min(max(p99), 4.0 * med) if med > 0 else max(p99)) or 1.0


# ---- POOR-IC SEED OVERRIDES: (scenario, slug) -> (seed, state file, per-step file) -------------
# 32^2 SI-SDE (shared) + corr.: the seed-0 run diverged on traj 11 at the first step (DelftBlue V100,
# RMSE 16.2); rerun locally (RTX 3090) with seed 1, identical settings ($PE/results/_runs/si_corr_seeds).
_SICORR = Path("/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/A Unified Framework for "
               "Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/"
               "paper_experiments/results/_runs/si_corr_seeds/full")
WIC_SEED_OVERRIDE = {("32_2_128_2", "si_sde_shared_corr"): (
    1, _SICORR / "states/traj11/navier_stokes__Ours_SI_SDE__32_2_128_2__shared_jac10_fc25_8797d2ae__seed1__E64_M100.npz",
    _SICORR / "per_step/32_2_128_2__M100__traj11__ours_shared_jac10__fc25_8797d2ae__si_sde__seed1.csv")}


def methods_for(sc):
    """Appendix figures (poor IC): 18 methods in three blocks of six for scenario slug sc."""
    m = NS / "main/navier_stokes/states/traj11"
    e = NS / "ext/navier_stokes/states/traj11"
    out = []
    for cov, tag in (("shared_jac10", "shared"), ("jacfree", "jac_free")):
        for a, A in (("si_sde", "SI_SDE"), ("dm_sde", "DM_SDE"), ("fm_ode", "FM_ODE")):
            out.append((f"{a}_{tag}", m / f"navier_stokes__Ours_{A}__{sc}__{cov}__seed0__E64_M100.npz"))
            out.append((f"{a}_{tag}_corr", e / f"navier_stokes__Ours_{A}__{sc}__{cov}_fc25_8797d2ae__seed0__E64_M100.npz"))
    out += [("tmpd", NS / f"tmpd/navier_stokes/states/traj11/navier_stokes__TMPD__{sc}__seed0__E64_M100.npz"),
            ("flowdas_surge", m / f"navier_stokes__SURGE_FlowDAS__{sc}__seed0__E64_M100.npz"),
            ("sda_surge", m / f"navier_stokes__SURGE_SDA__{sc}__seed0__E64_M100.npz"),
            ("d_flow_sgld", m / f"navier_stokes__D_Flow_SGLD__{sc}__seed0__E64_M100.npz"),
            ("guided_fm_fig", m / f"navier_stokes__Guided_FM_FIG__{sc}__seed0__E64_M100.npz"),
            ("fm_ode_prior_enkf", NS / f"prior_enkf/navier_stokes/states/traj11/navier_stokes__FM_ODE_prior_EnKF__{sc}__seed0__E64_M100.npz")]
    return [(slug, WIC_SEED_OVERRIDE[(sc, slug)][1] if (sc, slug) in WIC_SEED_OVERRIDE else f) for slug, f in out]


def methods_32():
    """Appendix figure (poor IC, 32^2): 18 methods in three blocks of six."""
    return methods_for("32_2_128_2")


# the poor-IC scenarios of the appendix field figures (stem ns_wic_<slug>)
SCENARIOS = ("32_2_128_2", "16_2_128_2")
TRAJS = (10, 11, 12, 13, 14)
SHOWN_TRAJ = 11


def per_step_name(sc, traj, slug):
    """Per-step CSV basename for a method slug of methods_for()."""
    p = f"{sc}__M100__traj{traj}__"
    for a in ("si_sde", "dm_sde", "fm_ode"):
        for tag, cov in (("shared", "shared_jac10"), ("jac_free", "jacfree")):
            if slug == f"{a}_{tag}":
                return p + f"ours_{cov}__{a}.csv"
            if slug == f"{a}_{tag}_corr":
                return p + f"ours_{cov}__fc25_8797d2ae__{a}.csv"
    return p + {"tmpd": "baselines__tmpd.csv", "flowdas_surge": "baselines__surge_flowdas.csv",
                "sda_surge": "baselines__surge_sda.csv", "d_flow_sgld": "baselines__dflow_sgld.csv",
                "guided_fm_fig": "baselines__fig.csv", "fm_ode_prior_enkf": "prior_enkf__fm_ode.csv"}[slug]


def divergence(sc, thresh=10.0):
    """{slug: (diverged on SHOWN_TRAJ, #diverged of TRAJS)}: a trajectory is diverged if its per-step
    rmse has a NaN or exceeds `thresh`. Raises if a per-step file is missing or ambiguous."""
    import pandas as pd
    out = {}
    for slug, _ in methods_for(sc):
        div = {}
        for t in TRAJS:
            name = per_step_name(sc, t, slug)
            hits = [NS / d / "navier_stokes/per_step" / name for d in ("main", "ext", "tmpd", "prior_enkf")]
            hits = [h for h in hits if h.exists()]
            if len(hits) != 1:
                raise FileNotFoundError(f"{name}: {len(hits)} matches")
            df = pd.read_csv(hits[0])
            r = df.loc[df["metric"] == "rmse", "value"].to_numpy(float)
            assert r.size > 0 and (df["test_index"] == t).all(), hits[0]
            div[t] = bool(np.isnan(r).any() or (r > thresh).any())
        if (sc, slug) in WIC_SEED_OVERRIDE:   # shown panel = the rerun: judge it by its own per-step file
            df = pd.read_csv(WIC_SEED_OVERRIDE[(sc, slug)][2])
            r = df.loc[df["metric"] == "rmse", "value"].to_numpy(float)
            assert r.size > 0
            out[slug] = (bool(np.isnan(r).any() or (r > thresh).any()), sum(div.values()))
            continue
        out[slug] = (div[SHOWN_TRAJ], sum(div.values()))
    return out


# ---------------------------------------------------------------------------------------
# TRUE initial condition (stem ns_tic_<slug>), E=64, M=TIC_M (the M of Table 1's true-IC
# columns), held-out trajectory SHOWN_TRAJ.  EDITABLE TABLE: where the files are looked for.
# Each method's file is searched in its TIC_STATE_DIRS entry, in order; the first hit wins.
# If none hits, every results/navier_stokes* tree (except wrong_ic / backup / archive dirs) is
# searched for the same basename (a unique hit is used and reported). Otherwise: "pending".
# ---------------------------------------------------------------------------------------
RES = Path("/export/scratch2/ntm/delftblue_results/results")
TIC = RES / "navier_stokes"
TIC_M = 100
TIC_CORR_TAG = "fc25_8797d2ae"          # forecast-residual correction run tag (as poor IC)
TIC_SCENARIOS = ("32_2_128_2", "16_2_128_2", "sparse_5", "sparse_1_5625")
_GUESS = ("ext", "tmpd", "prior_enkf")  # sub-runs, laid out as for the poor IC
TIC_STATE_DIRS = {   # kind -> candidate states/traj<SHOWN_TRAJ> directories, in order
    "main": [TIC / "states", Path("/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/"
             "A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation "
             "Interpolants/code/paper_experiments/results/_runs/tic16_m100/states")],
    "corr": [TIC / "ext/states", TIC / "ext/navier_stokes/states", TIC / "states"]
            + [RES / f"navier_stokes_{r}/{k}/navier_stokes/states" for r in ("true_ic", "tic") for k in _GUESS],
    "tmpd": [TIC / "tmpd/states", TIC / "tmpd/navier_stokes/states", TIC / "states"]
            + [RES / f"navier_stokes_{r}/{k}/navier_stokes/states" for r in ("true_ic", "tic") for k in _GUESS],
    "prior_enkf": [TIC / "prior_enkf/states", TIC / "prior_enkf/navier_stokes/states", TIC / "states"]
                  + [RES / f"navier_stokes_{r}/{k}/navier_stokes/states" for r in ("true_ic", "tic") for k in _GUESS],
}
TIC_PER_STEP_DIRS = sorted({d.parent / "per_step" for ds in TIC_STATE_DIRS.values() for d in ds}, key=str)
TIC_PER_STEP_DIRS.remove(TIC / "per_step")
TIC_PER_STEP_DIRS.insert(0, TIC / "per_step")
TIC_TRAJS = (10, 11, 12, 13, 14)        # held-out trajectories with per-step files
TIC_EXPECTED = {}                       # (sc, slug) -> expected path (first candidate), for reports

# ---- TRUE-IC OVERRIDES: 16^2 traj11 fields never saved at M=100 ----------------------------
# (scenario, slug) -> M of the state file shown instead of TIC_M; the caption says so
# (tic_m_note) and the divergence note uses the per-step files at that M.
# Jac-free 16^2: only M=25 states exist. The original manuscript's Jac-free 16^2 panels were
# these same M=25 states (their colour-inverted means match the M=25 npz with corr 0.9996 and
# reproduce the M=25 last-step RMSE 0.407/0.380/0.348, not the M=250 0.361/0.340/0.311), so
# no field at a larger M can be recovered from them.
# 2026-11-17: Jac-free SI/DM/FM, FlowDAS + SURGE and Guided FM re-run locally at M=100 with states
# saved ($PE/results/_runs/tic16_m100; per-step RMSE identical to the grid except FlowDAS + SURGE,
# mean 0.470 vs 0.449: SURGE code changed since). Only D-Flow SGLD (not re-run) remains at M=250.
TIC_M_OVERRIDE = {("16_2_128_2", "d_flow_sgld"): 250}


def tic_files(sc):
    """[(slug, kind, state basename)] for the 18 methods, in methods_for() order."""
    sfx = f"seed0__E64_M{TIC_M}.npz"
    out = []
    for cov, tag in (("shared_jac10", "shared"), ("jacfree", "jac_free")):
        for a, A in (("si_sde", "SI_SDE"), ("dm_sde", "DM_SDE"), ("fm_ode", "FM_ODE")):
            out.append((f"{a}_{tag}", "main", f"navier_stokes__Ours_{A}__{sc}__{cov}__{sfx}"))
            out.append((f"{a}_{tag}_corr", "corr", f"navier_stokes__Ours_{A}__{sc}__{cov}_{TIC_CORR_TAG}__{sfx}"))
    out += [("tmpd", "tmpd", f"navier_stokes__TMPD__{sc}__{sfx}"),
            ("flowdas_surge", "main", f"navier_stokes__SURGE_FlowDAS__{sc}__{sfx}"),
            ("sda_surge", "main", f"navier_stokes__SURGE_SDA__{sc}__{sfx}"),
            ("d_flow_sgld", "main", f"navier_stokes__D_Flow_SGLD__{sc}__{sfx}"),
            ("guided_fm_fig", "main", f"navier_stokes__Guided_FM_FIG__{sc}__{sfx}"),
            ("fm_ode_prior_enkf", "prior_enkf", f"navier_stokes__FM_ODE_prior_EnKF__{sc}__{sfx}")]
    return out


def _fallback_search(rel):
    """Unique hit for relative path `rel` under results/navier_stokes* (true IC only)."""
    hits = []
    for root in sorted(RES.glob("navier_stokes*")):
        if "wrong_ic" in root.name:
            continue
        hits += [h for h in root.rglob(rel)
                 if not any(x in str(h) for x in ("backup", "archive", "_historical", "wrong_ic"))]
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous {rel}: " + ", ".join(map(str, hits)))
    return hits[0] if hits else None


def tic_methods(sc):
    """[(slug, path or None)] for the true IC; None = not there yet (pending)."""
    out = []
    for slug, kind, base in tic_files(sc):
        if (sc, slug) in TIC_M_OVERRIDE:
            base = base.replace(f"_M{TIC_M}.npz", f"_M{TIC_M_OVERRIDE[(sc, slug)]}.npz")
        cands = [d / f"traj{SHOWN_TRAJ}" / base for d in TIC_STATE_DIRS[kind]]
        TIC_EXPECTED[(sc, slug)] = cands[0]
        f = next((c for c in cands if c.exists()), None)
        if f is None:
            f = _fallback_search(f"states/traj{SHOWN_TRAJ}/{base}")
            if f is not None:
                print(f"NOTE: {slug} ({sc}) found outside TIC_STATE_DIRS: {f}")
        out.append((slug, f))
    return out


# per-step sources for the true IC: slug -> candidate (basename suffix, method filter or None);
# the main runs write one CSV per sampler family with a `method` column; the newer runs are
# expected to write one CSV per method as for the poor IC.
def tic_per_step_cands(slug):
    for a, lab in (("si_sde", "SI-SDE"), ("dm_sde", "DM-SDE"), ("fm_ode", "FM-ODE")):
        for tag, cov in (("shared", "shared_jac10"), ("jac_free", "jacfree")):
            if slug == f"{a}_{tag}":
                return [(f"ours_{cov}.csv", f"Ours ({lab})"), (f"ours_{cov}__{a}.csv", None)]
            if slug == f"{a}_{tag}_corr":
                return [(f"ours_{cov}__{TIC_CORR_TAG}__{a}.csv", None),
                        (f"ours_{cov}__{TIC_CORR_TAG}.csv", f"Ours ({lab})")]
    return {"tmpd": [("baselines__tmpd.csv", None), ("tmpd.csv", None), ("baselines.csv", "TMPD")],
            "flowdas_surge": [("baselines.csv", "SURGE (FlowDAS)"), ("baselines__surge_flowdas.csv", None)],
            "sda_surge": [("surge_sda.csv", "SURGE (SDA)"), ("baselines__surge_sda.csv", None)],
            "d_flow_sgld": [("baselines.csv", "D-Flow SGLD"), ("baselines__dflow_sgld.csv", None)],
            "guided_fm_fig": [("baselines.csv", "Guided FM (FIG)"), ("baselines__fig.csv", None)],
            "fm_ode_prior_enkf": [("prior_enkf_r20__fm_ode.csv", None), ("prior_enkf__fm_ode.csv", None)]}[slug]


def tic_shown_M(sc, slug):
    """Sampler steps M of the field shown for slug (TIC_M unless overridden / recovered)."""
    return TIC_M_OVERRIDE.get((sc, slug), TIC_M)


def tic_m_note(sc):
    """Caption clause naming the methods (with data) shown at an M other than TIC_M."""
    from make_ns_wic32_tex import NAMES
    ms = [(s, tic_shown_M(sc, s)) for s, f in tic_methods(sc) if f is not None and tic_shown_M(sc, s) != TIC_M]
    if not ms:
        return ""
    jl = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else xs[0]  # noqa: E731
    parts = []
    for M in sorted({M for _, M in ms}, reverse=True):
        ss = [s for s, MM in ms if MM == M]
        jf = [s for s in ss if s.endswith("_jac_free")]
        names = [NAMES[s].replace(" (FIG)", "") for s in ss if s not in jf] + (["Jac-free"] if len(jf) == 3 else [NAMES[s] for s in jf])
        parts.append(rf"{', '.join(names)}: $M={M}$")
    return "; ".join(parts) + rf" (no $M={TIC_M}$ fields saved). "


def tic_rmse(sc, traj, slug, M=None):
    """Per-step rmse of `slug` on trajectory `traj` (true IC), or None if no per-step file."""
    import pandas as pd
    M = TIC_M if M is None else M
    for suffix, meth in tic_per_step_cands(slug):
        name = f"{sc}__M{M}__traj{traj}__{suffix}"
        hits = [d / name for d in TIC_PER_STEP_DIRS if (d / name).exists()]
        if not hits:
            h = _fallback_search(f"per_step/{name}")
            hits = [h] if h is not None else []
        for h in hits:
            df = pd.read_csv(h)
            if meth is not None:
                df = df[df["method"] == meth]
            r = df.loc[df["metric"] == "rmse", "value"].to_numpy(float)
            if r.size:
                assert (df["test_index"] == traj).all() and (df["M"] == M).all(), h
                return r
    return None


def tic_divergence(sc, thresh=10.0):
    """As divergence(), true IC, only for methods whose state file exists (pending: skipped).
    Raises if such a method lacks a per-step file for one of TIC_TRAJS."""
    out = {}
    for slug, f in tic_methods(sc):
        if f is None:
            continue
        div = {}
        for t in TIC_TRAJS:
            r = tic_rmse(sc, t, slug, tic_shown_M(sc, slug))
            if r is None:
                raise FileNotFoundError(f"true IC {sc} {slug}: state file present but no per-step rmse "
                                        f"for traj{t} (candidates {tic_per_step_cands(slug)} in {TIC_PER_STEP_DIRS})")
            div[t] = bool(np.isnan(r).any() or (r > thresh).any())
        out[slug] = (div[SHOWN_TRAJ], sum(div.values()))
    return out


def n_assim_steps(ic, sc):
    """Number of assimilation steps (= per_step_rmse length) of the shown trajectory."""
    ms = methods_for(sc) if ic == "wic" else tic_methods(sc)
    ns = {int(np.load(f)["per_step_rmse"].size) for _, f in ms if f is not None and Path(f).exists()}
    assert len(ns) == 1, (ic, sc, ns)
    return ns.pop()


def render(STEM, METHODS, scale_groups=None, scale_exclude=()):
    """scale_groups: optional list of slug lists; std/|error| scales (and their colour bars,
    suffixed _g<i>) are then computed per group instead of over all methods. scale_exclude:
    slugs (diverged runs that are still finite) left out of the std/|error| scale computation."""
    rows, truth, obs, pending = [], None, None, []
    for slug, f in METHODS:
        if f is None or not Path(f).exists():
            pending.append(slug)
            continue
        z = np.load(f)
        post = z["posterior_trajectory"][:, 0, :, :, -1]          # [E, H, W], final step
        tr = z["true_trajectory"][0, 0, :, :, -1]
        if truth is None:
            truth, obs = tr, z["observations"][0, :, -1]
            obs_idx = z["obs_indices"] if "obs_indices" in z.files else None
            ic, tidx = int(z["ic_index"]) if "ic_index" in z.files else -1, int(z["test_index"])   # no ic_index: true IC
        else:  # same trajectory and truth for every method
            assert np.allclose(tr, truth) and int(z["test_index"]) == tidx and (int(z["ic_index"]) if "ic_index" in z.files else -1) == ic, slug
        mean = post.mean(axis=0)
        rows.append((slug, mean, post.std(axis=0), np.abs(mean - truth)))
    m = float(np.abs(truth).max())
    field_kw = dict(cmap="RdBu_r", vmin=-m, vmax=m)
    groups = scale_groups or [[r[0] for r in rows]]
    gkw = []   # per group: (suffix, std_kw, err_kw)
    for gi, g in enumerate(groups):
        rg = [r for r in rows if r[0] in g and r[0] not in scale_exclude]
        gkw.append(("" if scale_groups is None else f"_g{gi}",
                    dict(cmap="magma", vmin=0.0, vmax=robust_vmax([r[2] for r in rg])),
                    dict(cmap="magma", vmin=0.0, vmax=robust_vmax([r[3] for r in rg]))))
    group_of = {slug: gi for gi, g in enumerate(groups) for slug in g}

    panel(truth, field_kw, f"{STEM}__truth")
    H, W = truth.shape
    if obs_idx is not None and obs_idx.ndim == 1 and obs_idx.size == obs.size:
        # sparse: flat indices into the H x W grid; one dot at each observed grid point
        iy, ix = np.divmod(obs_idx, W)
        assert np.corrcoef(obs, truth[iy, ix])[0, 1] > 0.99, "obs / truth ordering mismatch"
        xy = (ix + 0.5, iy + 0.5)
    else:
        k = int(round(np.sqrt(obs.size)))             # super-resolution: row-major k x k block averages
        iy, ix = np.divmod(np.arange(obs.size), k)
        xy = ((ix + 0.5) * W / k, (iy + 0.5) * H / k)
    panel(None, field_kw, f"{STEM}__obs", background=truth, scatter=(*xy, obs))
    for slug, mean, std, err in rows:
        _, std_kw, err_kw = gkw[group_of[slug]]
        panel(mean, field_kw, f"{STEM}__mean__{slug}")
        panel(std, std_kw, f"{STEM}__std__{slug}")
        panel(err, err_kw, f"{STEM}__abserr__{slug}")
    for slug in pending:
        for q in ("mean", "std", "abserr"):
            # Jac-free + corr. was never run from the true IC: "not run" instead of "pending"
            lab = "not run" if STEM.startswith("ns_tic") and slug.endswith("_jac_free_corr") else True
            panel(None, None, f"{STEM}__{q}__{slug}", background=truth, pending=lab)
    bars = [("field", field_kw, "vorticity")]
    for suf, std_kw, err_kw in gkw:
        bars += [(f"std{suf}", std_kw, "vorticity std"), (f"abserr{suf}", err_kw, "$|$error$|$ (vorticity)")]
    for key, kw, lab in bars:
        fig = plt.figure(figsize=(0.62, CBAR_H * 1.1))
        cax = fig.add_axes([0.02, 0.05, 0.2, 0.9])
        sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(kw["vmin"], kw["vmax"]), cmap=kw["cmap"])
        cb = fig.colorbar(sm, cax=cax, orientation="vertical")
        cb.set_label(lab, fontsize=6.5, labelpad=2.0)
        cb.locator = mticker.MaxNLocator(nbins=4)
        cb.formatter = mticker.FuncFormatter(lambda v, _p: f"{v:g}")
        cb.update_ticks()
        cb.ax.tick_params(labelsize=6.0, length=2.0, width=0.5, pad=1.5)
        cb.outline.set_linewidth(0.5)
        save(fig, f"{STEM}__cbar_{key}")


def main(argv=()):
    """No args / 'all': poor-IC (SCENARIOS) and true-IC (TIC_SCENARIOS) appendix figures.
    'wic' / 'tic': only those. 'legacy': ns_wic16, ns_wic16b, ns_wic32 (not in the default run).
    Otherwise: the given scenario slugs, poor IC (stem ns_wic_<slug>); 'tic:<slug>' for true IC."""
    style()
    which = list(argv) or ["all"]
    if which[0] == "legacy":
        render(STEM, METHODS)
        render(STEM_B, METHODS_B)
        m32 = methods_32()
        render("ns_wic32", m32, scale_groups=[[s for s, _ in m32[:12]], [s for s, _ in m32[12:]]])
        print("wrote", OUT)
        return
    if which[0] in ("all", "wic", "tic"):
        jobs = ([("wic", sc) for sc in SCENARIOS] if which[0] != "tic" else []) + \
               ([("tic", sc) for sc in TIC_SCENARIOS] if which[0] != "wic" else [])
    else:
        jobs = [("tic", a[4:]) if a.startswith("tic:") else ("wic", a) for a in which]
    pending_all = []
    for ic, sc in jobs:
        if ic == "wic":
            ms = methods_for(sc)
            missing = [str(f) for _, f in ms if not f.exists()]
            if missing:
                raise FileNotFoundError("missing state files:\n" + "\n".join(missing))
            div = [s for s, (shown, _) in divergence(sc).items() if shown]
        else:
            ms = tic_methods(sc)
            pending_all += [(sc, s, TIC_EXPECTED[(sc, s)]) for s, f in ms if f is None]
            div = [s for s, (shown, _) in tic_divergence(sc).items() if shown]
        render(f"ns_{ic}_{sc}", ms, scale_groups=[[s for s, _ in ms[:12]], [s for s, _ in ms[12:]]],
               scale_exclude=div)
    print("wrote", OUT)
    if pending_all:
        bar = "!" * 100
        print(f"\n{bar}\n!!! {len(pending_all)} TRUE-IC STATE FILES MISSING -> drawn as 'pending' panels:")
        for sc, s, f in pending_all:
            print(f"!!!   {sc:15s} {s:22s} {f}")
        print("!!! when they arrive (or after editing the TIC_* table), rerun:\n"
              "!!!   python3 make_ns_wic_fields.py && python3 make_ns_wic32_tex.py\n" + bar)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
