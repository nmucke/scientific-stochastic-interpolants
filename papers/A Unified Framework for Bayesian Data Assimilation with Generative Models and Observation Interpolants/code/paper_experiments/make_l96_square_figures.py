#!/usr/bin/env python3
"""Lorenz 96 with a non-injective observation h(x) = c x^2: metrics table and
final-step posterior marginals (does a method capture the bimodal sign posterior?).

Methods: PF reference (N = 4e6, replicate 1 scored against replicate 0), EnKF (E = 64),
Ours SI-SDE (shared) and Ours SI-SDE (shared) + corr. (both at M = 50). The PF with 64
draws from replicate 1 (``pf_subsample``, the k-NN-KL sampling floor) is added when its
files exist.

Data (all under ``<root>/<obs>/``, the layout of ``cases/lorenz96/driver.py``):

    reference/test/sequence<i>_replicate<r>.pt   dict: samples [T, 4096, d] (equal-weight
                                                  PF draws), mean/variance [T, d], problem
    reference/test/sequence<i>_replicate<r>.pfterms.pt  cached PF k-NN radii / energy self-terms
    ensembles/<cell>[__M<M>]__seed<s>.pt          float32 [S, T, E, d]
    per_sequence/lorenz96__<cell>[__M<M>]__traj<s>__.json   {metric: [S] time averages}

Truth and observations are not stored: they are rebuilt with the driver's own
``make_sequence(problem, derive_seed("lorenz96-evaluation", "sequence", "test", i))``
from the problem stored in the reference (the truth and the noise do not depend on the
observation operator, so y = h(truth) + e is rebuilt with h from the stored problem).

Outputs (``--out``):
    metrics_<obs>.csv / .md / .tex     mean +- std over the test sequences
    pf_sign_mass_<obs>.csv             PF P(x_k > 0) per (sequence, site) at the final step
    final_step_bimodality_<obs>.csv    per (method, sequence, site) member fraction x > 0
    l96_<obs>_final_marginals.pdf/.png the figure

Table columns: time averages of sliced W2 / k-NN KL / energy distance to the PF
(the pipeline's per_sequence values); the same three at the final step, recomputed from
the saved ensembles with the driver's ``score_ensemble``; and a sign-mode diagnostic on
the observed sites: |frac(x>0) - P_PF(x>0)| at the final step (all observed sites, and
only the PF-bimodal ones, min(P, 1-P) >= --bimodal-mass) and the missed-mode rate
(PF-bimodal sites where the method puts < --missed-mass of its members on one side).
The time-averaged sign-mass error over all steps is reported too.

    .venv/bin/python make_l96_square_figures.py                  # square (falls back to arctan)
    .venv/bin/python make_l96_square_figures.py --obs arctan
    .venv/bin/python make_l96_square_figures.py --sequences 3 7 --sites 0 4 2 6

Needs torch (the .pt files): run with the repo venv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# paper_experiments: this file's directory (repo copy) or the original location
_HERE = Path(__file__).resolve().parent
PE = _HERE if (_HERE / "cases/lorenz96/driver.py").exists() else Path(
    "/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/A Unified Framework for "
    "Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments")
REPO_ROOT_L96 = PE / "results/lorenz96"
# optional second place to look for results (a local mirror of cluster results)
DELFTBLUE_ROOT_L96 = Path("/export/scratch2/ntm/delftblue_results/results/lorenz96")
OUT_DEFAULT = (PE / "results/figures/l96_square") if _HERE == PE else _HERE / "figures/l96_square"

# Score on the CPU unless asked otherwise (only the cached-PF-term-free path is costly).
os.environ.setdefault("SCISI_SCORING_DEVICE", "cpu")
sys.path.insert(0, str(PE))

import torch  # noqa: E402

# key, cell, uses M, label (table), short label (panel annotation), colour, line style
METHODS = (
    ("pf", "pf_reference", False, "PF reference", "PF", "#000000", "-"),
    ("pf_draws", "pf_subsample", False, "PF, $E$ draws (floor)", "PF$_E$", "#777777", ":"),
    ("enkf", "enkf", False, "EnKF ($E=64$)", "EnKF", "#E69F00", "-"),
    ("shared", "si_sde-shared", True, "Ours SI-SDE (shared)", "shared", "#0072B2", "-"),
    ("shared_corr", "si_sde_fc-shared", True, "Ours SI-SDE (shared) + corr.", "+corr.",
     "#009E73", "--"),
    ("jacfree", "si_sde", True, "Ours SI-SDE (Jac-free)", "Jac-free", "#CC79A7", "-."),
)
FIG_METHODS = ("enkf", "shared", "shared_corr", "jacfree")  # drawn over the PF in the figure
METRICS = (("sliced_w2", "sliced $W_2$"), ("kl_knn", "$k$NN-KL"), ("energy_distance", "energy dist."))
SEED_LABEL_FALLBACK = "lorenz96-evaluation"


# --------------------------------------------------------------------------- #
# Locating and loading
# --------------------------------------------------------------------------- #


def resolve_root(root: str | None, obs: str, allow_fallback: bool) -> tuple[Path, str]:
    """``(<sweep dir>, obs)``. ``root`` may be the sweep dir itself (holding reference/)
    or its parent (holding <obs>/). Without --root: the repo results, then DelftBlue."""
    candidates = []
    if root is not None:
        r = Path(root)
        candidates = [r] if (r / "reference").is_dir() else [r / obs]
    else:
        candidates = [REPO_ROOT_L96 / obs, DELFTBLUE_ROOT_L96 / obs]
    for c in candidates:
        if (c / "reference/test").is_dir() and (c / "ensembles").is_dir():
            return c, obs
    if allow_fallback and obs != "arctan":
        warnings.warn(f"no results for obs={obs!r} in {candidates}; falling back to arctan")
        parent = Path(root).parent if root and (Path(root) / "reference").is_dir() else None
        return resolve_root(str(parent) if parent else root, "arctan", False)
    raise FileNotFoundError(f"no reference/test + ensembles under {candidates}")


def reference_file(base: Path, index: int, replicate: int) -> Path:
    return base / "reference/test" / f"sequence{index}_replicate{replicate}.pt"


def count_sequences(base: Path) -> int:
    n = 0
    while reference_file(base, n, 0).exists():
        n += 1
    return n


def load_reference(base: Path, index: int, replicate: int) -> dict:
    """The stored PF reference plus its cached score terms (read-only; nothing is written)."""
    from cases.lorenz96._l96_divergence import pf_terms_match

    path = reference_file(base, index, replicate)
    ref = torch.load(path, weights_only=False)
    cache = path.with_suffix(".pfterms.pt")
    if cache.exists():
        terms = torch.load(cache, weights_only=False)
        if pf_terms_match(terms, ref["samples"]):
            ref["pf_terms"] = terms
    return ref


def cell_label(cell: str, uses_m: bool, m: int) -> str:
    return f"{cell}__M{m}" if uses_m else cell


def load_ensemble(base: Path, cell: str, uses_m: bool, m: int, seed: int):
    path = base / "ensembles" / f"{cell_label(cell, uses_m, m)}__seed{seed}.pt"
    if not path.exists():
        return None
    return torch.load(path, weights_only=False).double()  # [S, T, E, d]


def load_per_sequence(base: Path, cell: str, uses_m: bool, m: int, seed: int):
    path = base / "per_sequence" / f"lorenz96__{cell_label(cell, uses_m, m)}__traj{seed}__.json"
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- #
# Observation operator and sequences
# --------------------------------------------------------------------------- #


def obs_functions(observation: dict | None):
    """``(h, preimages)`` of the stored operator: h on the observed sites and the set of
    w with h(w) = y (two sign modes for ``square``, none when y is outside the range)."""
    op = (observation or {}).get("operator", "linear")
    c = (observation or {}).get("strength")
    c = 1.0 if c is None else float(c)
    if op == "linear":
        return (lambda w: w), (lambda y: [y])
    if op == "arctan":
        return (lambda w: np.arctan(c * w) / c), (
            lambda y: [math.tan(c * y) / c] if abs(c * y) < math.pi / 2 else [])
    if op == "square":
        return (lambda w: c * w ** 2), (
            lambda y: [-math.sqrt(y / c), math.sqrt(y / c)] if y > 0 else [])
    if op == "abs":
        return (lambda w: c * np.abs(w)), (
            lambda y: [-y / c, y / c] if y > 0 else [])
    raise ValueError(f"unknown observation operator {op!r}: add it to obs_functions()")


def rebuild_sequences(problem_dict: dict, count: int):
    """Truth [S, T+1, d] and observations [S, T, d_y] of the test sequences.

    The draws are those of the driver (``make_sequence`` with the linear dynamics: the
    truth and the noise do not depend on h). If the repo's ``Problem`` knows the stored
    operator, its ``observe`` is checked against the local h.
    """
    from cases.lorenz96._l96_system import Problem, make_sequence
    from common.seeding import derive_seed
    try:
        from cases.lorenz96.driver import SEED_LABEL
    except Exception:  # noqa: BLE001 - e.g. the driver refusing an unknown operator
        SEED_LABEL = SEED_LABEL_FALLBACK

    dyn = {k: v for k, v in problem_dict.items() if k != "observation"}
    linear = Problem(**dyn)
    h, _ = obs_functions(problem_dict.get("observation"))
    try:
        full = Problem(**problem_dict)
    except Exception:  # noqa: BLE001 - operator not (yet) in this checkout
        full = None
    idx = linear.obs_indices.numpy()
    truths, ys = [], []
    for i in range(count):
        seq = make_sequence(linear, derive_seed(SEED_LABEL, "sequence", "test", i))
        truth = seq.truth.numpy()
        noise = seq.observations.numpy() - truth[1:, idx]
        y = h(truth[1:, idx]) + noise
        if full is not None:
            y_repo = full.observe(seq.truth[1:]).numpy() + noise
            if not np.allclose(y, y_repo, rtol=1e-10, atol=1e-10):
                raise RuntimeError("local h disagrees with the repo's Problem.observe")
        truths.append(truth)
        ys.append(y)
    return np.stack(truths), np.stack(ys), idx


# --------------------------------------------------------------------------- #
# Scoring (the driver's score_ensemble on one step)
# --------------------------------------------------------------------------- #


def final_step_scores(ensemble_t: torch.Tensor, ref: dict, truth_t: np.ndarray, t: int) -> dict:
    """Pipeline metrics of members ``[E, d]`` at step ``t`` against PF replicate ``ref``.

    The reference is cut to step t (and its cached PF terms with it), so the driver's
    ``score_ensemble`` (member-divergence rule, sliced W2 with the same projections,
    k-NN KL on unique PF draws, energy distance) scores exactly that step.
    """
    from cases.lorenz96.driver import score_ensemble

    cut = {"mean": ref["mean"][t:t + 1], "variance": ref["variance"][t:t + 1],
           "samples": ref["samples"][t:t + 1]}
    if "pf_terms" in ref:
        cut["pf_terms"] = {"radii": [ref["pf_terms"]["radii"][t]],
                           "energy_self": [ref["pf_terms"]["energy_self"][t]]}
    out = score_ensemble(ensemble_t[None], cut, torch.as_tensor(truth_t[None]))
    return {k: v[0] for k, v in out.items()}


def alive(members: np.ndarray) -> np.ndarray:
    """The driver's rule: a member that is non-finite or leaves |x| < 100 is dropped."""
    from cases.lorenz96.driver import DIVERGED_ABS
    return np.isfinite(members).all(-1) & (np.abs(members) < DIVERGED_ABS).all(-1)


def frac_positive(members: np.ndarray) -> np.ndarray:
    """Fraction of (alive) members with x_k > 0 per site; members ``[..., E, d]``."""
    keep = alive(members)
    pos = (members > 0) & keep[..., None]
    return pos.sum(-2) / np.maximum(keep.sum(-1), 1)[..., None]


# --------------------------------------------------------------------------- #
# Bimodality of the PF marginal
# --------------------------------------------------------------------------- #


def kde(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    x = x[np.isfinite(x)]
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(1) / (len(x) * bw * math.sqrt(2 * math.pi))


def kde_bimodal(x: np.ndarray, bw_frac: float) -> bool:
    """Two KDE peaks (each >= 15% of the highest) with a dip <= 70% of the lower one."""
    s = x.std()
    if not s > 0:
        return False
    grid = np.linspace(x.min() - s, x.max() + s, 400)
    f = kde(x, grid, bw_frac * s)
    peaks = [i for i in range(1, len(f) - 1) if f[i] >= f[i - 1] and f[i] > f[i + 1]
             and f[i] >= 0.15 * f.max()]
    if len(peaks) < 2:
        return False
    top = sorted(sorted(peaks, key=lambda i: f[i])[-2:])
    return f[top[0]:top[1] + 1].min() <= 0.7 * min(f[top[0]], f[top[1]])


def pf_sign_table(ref_final: list[np.ndarray], obs_idx, bw_frac, mass) -> pd.DataFrame:
    rows = []
    for i, draws in enumerate(ref_final):
        p = (draws > 0).mean(0)
        for k in range(draws.shape[1]):
            m = min(p[k], 1 - p[k])
            rows.append(dict(sequence=i, site=k, observed=k in obs_idx, pf_p_pos=p[k],
                             minor_mass=m, kde_bimodal=kde_bimodal(draws[:, k], bw_frac),
                             bimodal=bool(m >= mass)))
    t = pd.DataFrame(rows)
    t["score"] = np.where(t.kde_bimodal, t.minor_mass, 0.0)
    return t


def choose_panels(table: pd.DataFrame, n_rows: int, sequences, sites, n_obs=2, n_unobs=2):
    """[(sequence, [sites])]: by default the sequences whose most bimodal observed site
    (KDE two-peaked, then largest minor sign mass) scores best; per sequence the n_obs
    most bimodal observed and the n_unobs most bimodal unobserved sites (or --sites)."""
    obs_t, un_t = table[table.observed], table[~table.observed]
    if sequences is None:
        best = obs_t.sort_values(["score", "minor_mass"], ascending=False).drop_duplicates("sequence")
        sequences = best.sequence.head(n_rows).tolist()
    panels = []
    for j, i in enumerate(sequences):
        if sites is not None:
            chosen = [int(k) for k in sites]
        else:
            pick = lambda t, n: [int(k) for k in t[t.sequence == i].sort_values(  # noqa: E731
                ["score", "minor_mass"], ascending=False).site.head(n)]
            chosen = sorted(pick(obs_t, n_obs)) + sorted(pick(un_t, n_unobs))
        panels.append((int(i), chosen))
    return panels


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #


def mean_std(values) -> tuple[float, float, int]:
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size == 0:
        return math.nan, math.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else math.nan, int(v.size)


def fmt(m: float, s: float, digits: int = 3) -> str:
    if not np.isfinite(m):
        return "n/a"
    f = lambda x: f"{x:.{digits}g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.1e}"  # noqa: E731
    return f(m) + (f" ± {f(s)}" if np.isfinite(s) else "")


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 9, "axes.linewidth": 0.7,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    })


def draw_panel(ax, i, k, observed, pf, members, truth, y_k, preimage, style, bw_frac,
               bw_members, p_pos, pf_p, pf_rep=None):
    s = pf.std()
    bw = bw_frac * s
    pts = [pf] + [m for m in members.values() if m is not None]
    lo = min(np.percentile(pf, 0.2), *(m.min() for m in pts[1:])) - 0.5 * s
    hi = max(np.percentile(pf, 99.8), *(m.max() for m in pts[1:])) + 0.5 * s
    lo, hi = min(lo, truth - 0.3 * s), max(hi, truth + 0.3 * s)
    grid = np.linspace(lo, hi, 500)
    f_pf = kde(pf, grid, bw)
    ax.fill_between(grid, f_pf, color="#BBBBBB", alpha=0.55, lw=0)
    ax.plot(grid, f_pf, color="#000000", lw=1.3)
    top = f_pf.max()
    if pf_rep is not None:   # second independent PF replicate: the reference's own error
        f_rep = kde(pf_rep, grid, bw)
        ax.plot(grid, f_rep, color="#444444", lw=1.1, ls=(0, (1, 1.2)))
        top = max(top, f_rep.max())
    for key, x in members.items():
        if x is None:
            continue
        _, _, _, _, _, colour, ls = style[key]
        f = kde(x, grid, bw_members * s)
        top = max(top, f.max())
        ax.plot(grid, f, color=colour, ls=ls, lw=1.2)
    # rugs below the axis line, one band per method
    for j, (key, x) in enumerate(members.items()):
        if x is None:
            continue
        y0 = -0.06 * top * (j + 1)
        ax.plot(x, np.full_like(x, y0), "|", color=style[key][5], ms=3.5, mew=0.6, alpha=0.8)
    ax.axvline(truth, color="#C0392B", lw=1.1)
    if observed:
        for w in preimage(y_k):
            ax.axvline(w, color="#555555", lw=0.9, ls=(0, (3, 2)))
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.06 * top * (len(members) + 0.8), 1.5 * top)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(f"sequence {i}, $x_{{{k}}}$ ({'observed' if observed else 'unobserved'})",
                 fontsize=8.5, pad=3)
    # P(x > 0) per method, in the upper corner on the side with less PF mass
    left = f_pf[: len(grid) // 3].sum() <= f_pf[-len(grid) // 3:].sum()
    x0, ha = (0.012, "left") if left else (0.988, "right")
    ax.text(x0, 0.985, "$P(x>0)$:", transform=ax.transAxes, fontsize=6.6, va="top", ha=ha)
    entries = [(f"PF {pf_p:.2f}", "#000000")] + (
        [(f"PF rep. 2 {float((pf_rep > 0).mean()):.2f}", "#444444")] if pf_rep is not None else []) + [
        (f"{style[key][4]} {p_pos[key]:.2f}", style[key][5]) for key in members if members[key] is not None]
    for j, (text, colour) in enumerate(entries):
        ax.text(x0, 0.985 - 0.095 * (j + 1), text, transform=ax.transAxes, fontsize=6.6,
                va="top", ha=ha, color=colour,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.75))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-sequences", type=int, default=None, help="use only the first n test sequences")
    ap.add_argument("--obs", default="square", help="observation sweep (default square; "
                    "falls back to arctan when square results are missing)")
    ap.add_argument("--no-fallback", action="store_true", help="fail instead of falling back")
    ap.add_argument("--root", default=None, help="results/lorenz96 dir (holding <obs>/) or the "
                    "sweep dir itself; default: repo results, then the DelftBlue copy")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=4, help="sequences in the figure")
    ap.add_argument("--sequences", type=int, nargs="+", default=None,
                    help="figure sequences (default: the most bimodal ones, automatically)")
    ap.add_argument("--n-observed", type=int, default=3, help="observed sites per sequence")
    ap.add_argument("--n-unobserved", type=int, default=3, help="unobserved sites per sequence")
    ap.add_argument("--sites", type=int, nargs="+", default=None,
                    help="figure sites, pairs per row (left, right), cycled; default: per row "
                         "the most bimodal observed and unobserved site")
    ap.add_argument("--step", type=int, default=-1, help="assimilation step index (default last)")
    ap.add_argument("--bw", type=float, default=0.12,
                    help="PF KDE bandwidth as a fraction of the PF marginal std (4096 draws)")
    ap.add_argument("--bw-members", type=float, default=0.25,
                    help="KDE bandwidth of the 64-member ensembles, fraction of the PF std "
                         "(small enough to keep two sign modes at about +-std apart)")
    ap.add_argument("--bimodal-mass", type=float, default=0.2,
                    help="PF site is bimodal when min(P(x>0), P(x<0)) >= this")
    ap.add_argument("--missed-mass", type=float, default=0.05,
                    help="a method misses a mode when < this fraction of members is on one side")
    ap.add_argument("--no-floor", action="store_true", help="leave out the PF E-draws row")
    ap.add_argument("--verify", action="store_true",
                    help="also rescore every step of one method and compare with per_sequence/")
    args = ap.parse_args()

    base, obs = resolve_root(args.root, args.obs, not args.no_fallback)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"results: {base}  (obs={obs})")

    S = count_sequences(base)
    if args.num_sequences is not None:   # e.g. a quick run on the first sequence(s) only
        S = min(S, args.num_sequences)
    refs0 = [load_reference(base, i, 0) for i in range(S)]
    refs1 = [load_reference(base, i, 1) if reference_file(base, i, 1).exists() else None
             for i in range(S)]
    problem = refs0[0]["problem"]
    T = refs0[0]["samples"].shape[0]
    t = args.step % T
    truth, ys, obs_idx = rebuild_sequences(problem, S)
    _, preimage = obs_functions(problem.get("observation"))
    print(f"{S} test sequences, T={T}, step {t}, problem observation: {problem.get('observation')}")

    style = {m[0]: m for m in METHODS}
    keys = [m[0] for m in METHODS if not (args.no_floor and m[0] == "pf_draws")]
    ens, perseq = {}, {}
    for key in keys:
        _, cell, uses_m, *_ = style[key]
        perseq[key] = load_per_sequence(base, cell, uses_m, args.M, args.seed)
        if key == "pf":
            ens[key] = (torch.stack([r["samples"] for r in refs1]).double()
                        if all(r is not None for r in refs1) else None)
        else:
            ens[key] = load_ensemble(base, cell, uses_m, args.M, args.seed)
        print(f"  {key:12s} ensembles: {'ok ' + str(tuple(ens[key].shape)) if ens[key] is not None else 'MISSING'}"
              f"   per_sequence: {'ok' if perseq[key] else 'MISSING'}")
    keys = [k for k in keys if ens[k] is not None or perseq[k] is not None]
    if ens.get("enkf") is not None:   # label with the actual ensemble size
        e_size = int(ens["enkf"].shape[2])
        style["enkf"] = (*style["enkf"][:3], f"EnKF ($E={e_size}$)", *style["enkf"][4:])

    # ---- PF sign masses and the bimodal sites --------------------------------
    ref_final = [r["samples"][t].double().numpy() for r in refs0]
    ref_final1 = [r["samples"][t].double().numpy() if r is not None else None for r in refs1]
    signs = pf_sign_table(ref_final, obs_idx, args.bw, args.bimodal_mass)
    signs.to_csv(out / f"pf_sign_mass_{obs}.csv", index=False)
    nb = signs[signs.observed].bimodal.sum()
    print(f"PF-bimodal observed (sequence, site) pairs at step {t}: {nb} / {signs.observed.sum()}"
          f" (KDE two-peaked: {signs[signs.observed].kde_bimodal.sum()})")

    # PF sign masses over all steps (observed sites) for the time-averaged diagnostic.
    pf_pos_all = np.stack([(r["samples"][:, :, obs_idx] > 0).double().mean(1).numpy() for r in refs0])

    # ---- per-method rows -----------------------------------------------------
    rows, detail = [], []
    for key in keys:
        label = style[key][3]
        row = {"method": label, "key": key}
        ps = perseq[key] or {}
        for metric, _ in METRICS:
            m, s, n = mean_std(ps.get(metric, []))
            row.update({f"{metric}_tavg_mean": m, f"{metric}_tavg_std": s, f"{metric}_tavg_n": n})
        e = ens[key]
        if e is not None:
            final = {metric: [] for metric, _ in METRICS}
            final_rmse = []
            for i in range(S):
                sc = final_step_scores(e[i, t], refs0[i], truth[i, 1:][t], t)
                for metric, _ in METRICS:
                    final[metric].append(sc[metric])
                final_rmse.append(sc["rmse"])
            for metric, _ in METRICS:
                m, s, n = mean_std(final[metric])
                row.update({f"{metric}_final_mean": m, f"{metric}_final_std": s, f"{metric}_final_n": n})
            row["_final_rmse"] = final_rmse
            row["_final"] = final
            en = e.numpy()
            fp = frac_positive(en[:, :, :, :])[:, :, obs_idx]  # [S, T, n_obs]
            # time-averaged sign-mass error, all observed sites
            m, s, n = mean_std(np.abs(fp - pf_pos_all).mean((1, 2)))
            row.update(sign_err_tavg_mean=m, sign_err_tavg_std=s, sign_err_tavg_n=n)
            # final step: all observed sites / PF-bimodal ones / missed modes
            fpt = frac_positive(en[:, t])  # [S, d]
            all_err, bi_err, missed = [], [], []
            for i in range(S):
                si = signs[(signs.sequence == i)]
                for _, r in si.iterrows():
                    k = int(r.site)
                    detail.append(dict(method=label, sequence=i, site=k, observed=bool(r.observed),
                                       pf_p_pos=r.pf_p_pos, method_p_pos=fpt[i, k],
                                       pf_bimodal=bool(r.bimodal)))
                so = si[si.observed]
                errs = np.abs(fpt[i, so.site.values] - so.pf_p_pos.values)
                all_err.append(errs.mean())
                b = so[so.bimodal]
                if len(b):
                    q = fpt[i, b.site.values]
                    bi_err.append(np.abs(q - b.pf_p_pos.values).mean())
                    missed.append((np.minimum(q, 1 - q) < args.missed_mass).mean())
            for name, vals in (("sign_err_final", all_err), ("sign_err_final_bimodal", bi_err),
                               ("missed_mode_rate", missed)):
                m, s, n = mean_std(vals)
                row.update({f"{name}_mean": m, f"{name}_std": s, f"{name}_n": n})
        rows.append(row)

    # ---- cross-checks against the pipeline's per_step files -------------------
    per_step_check = []
    for row in rows:
        if "_final" not in row:
            continue
        _, cell, uses_m, *_ = style[row["key"]]
        f = base / "per_step" / f"lorenz96__{cell_label(cell, uses_m, args.M)}__traj{args.seed}__.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d.step == t) & (d.test_index == -1)].set_index("metric").value
        for metric in [m for m, _ in METRICS] + ["rmse"]:
            mine = row["_final_rmse"] if metric == "rmse" else row["_final"][metric]
            if metric in d:
                per_step_check.append((row["key"], metric, float(np.nanmean(mine)), float(d[metric])))
    if per_step_check:
        print(f"\ncheck: recomputed step-{t} mean over sequences vs per_step/ (pipeline):")
        for key, metric, a, b in per_step_check:
            print(f"  {key:12s} {metric:16s} {a:12.6f} {b:12.6f}  rel.diff {abs(a - b) / max(abs(b), 1e-12):.1e}")

    if args.verify:
        key = "enkf" if ens.get("enkf") is not None else next(k for k in keys if ens[k] is not None)
        from cases.lorenz96.driver import score_ensemble, summarise
        scores = [score_ensemble(ens[key][i], refs0[i], torch.as_tensor(truth[i, 1:])) for i in range(S)]
        summ = summarise(scores)
        print(f"\nverify: full rescoring of {key} vs per_sequence/ (max abs diff over sequences):")
        for metric, _ in METRICS:
            print(f"  {metric:16s} {np.nanmax(np.abs(np.array(summ[metric]) - np.array(perseq[key][metric]))):.2e}")

    # ---- write the table -----------------------------------------------------
    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    table.insert(0, "obs", obs)
    table.insert(1, "M", args.M)
    table.insert(2, "step", t)
    table.to_csv(out / f"metrics_{obs}.csv", index=False)
    pd.DataFrame(detail).to_csv(out / f"final_step_bimodality_{obs}.csv", index=False)

    cols = ([(f"{m}_tavg", f"{lab} (time avg.)") for m, lab in METRICS]
            + [(f"{m}_final", f"{lab} (step {t + 1})") for m, lab in METRICS]
            + [("sign_err_tavg", r"$|\Delta P(x>0)|$ obs. (time avg.)"),
               ("sign_err_final", rf"$|\Delta P(x>0)|$ obs. (step {t + 1})"),
               ("sign_err_final_bimodal", rf"$|\Delta P(x>0)|$ bimodal (step {t + 1})"),
               ("missed_mode_rate", rf"missed mode (step {t + 1})")])
    cells = [[r["method"]] + [fmt(r.get(f"{c}_mean", math.nan), r.get(f"{c}_std", math.nan))
                              for c, _ in cols] for r in rows]
    head = ["method"] + [lab for _, lab in cols]
    md = [f"### Lorenz 96, {obs} observations, M = {args.M} (mean ± std over {S} test sequences)", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(c) + " |" for c in cells]
    md += ["", f"PF-bimodal observed sites at step {t + 1}: {nb} of {int(signs.observed.sum())} "
           f"(min(P(x>0), P(x<0)) >= {args.bimodal_mass}); 'bimodal' columns average over the "
           f"sequences that have one (n = {int(rows[0].get('sign_err_final_bimodal_n', 0) or 0)}). "
           f"Missed mode: < {args.missed_mass:.0%} of members on one side of 0. "
           "PF reference row: replicate 1 scored against replicate 0."]
    (out / f"metrics_{obs}.md").write_text("\n".join(md) + "\n")
    tex_head = " & ".join(["Method"] + [lab.replace("±", r"$\pm$") for _, lab in cols])
    tex = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", tex_head + r" \\", r"\midrule"]
    tex += [" & ".join([c[0]] + [x.replace(" ± ", r" $\pm$ ") for x in c[1:]]) + r" \\" for c in cells]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / f"metrics_{obs}.tex").write_text("\n".join(tex) + "\n")
    print("\n" + "\n".join(md))

    # ---- figure --------------------------------------------------------------
    panels = choose_panels(signs, args.rows, args.sequences, args.sites,
                           args.n_observed, args.n_unobserved)
    print(f"\nfigure panels (sequence, sites): {panels}")
    for i, chosen in panels:
        for k in chosen:
            r = signs[(signs.sequence == i) & (signs.site == k)].iloc[0]
            print(f"  seq {i} site {k}: PF P(x>0)={r.pf_p_pos:.3f} kde_bimodal={r.kde_bimodal}")
    apply_style()
    ncols = max(len(c) for _, c in panels)
    # one sequence: a grid with the observed sites on top and the unobserved below
    if len(panels) == 1 and args.n_observed == args.n_unobserved and ncols == 2 * args.n_observed:
        i0, chosen0 = panels[0]
        panels = [(i0, chosen0[: args.n_observed]), (i0, chosen0[args.n_observed:])]
        ncols = args.n_observed
    fig, axes = plt.subplots(len(panels), ncols, figsize=(3.15 * ncols, 1.75 * len(panels) + 0.6),
                             squeeze=False)
    fig_keys = [k for k in FIG_METHODS if k in keys and ens[k] is not None]
    for row_axes, (i, chosen) in zip(axes, panels):
        for ax in row_axes[len(chosen):]:
            ax.set_visible(False)
        for ax, k in zip(row_axes, chosen):
            observed = k in obs_idx
            pf = ref_final[i][:, k]
            pf_rep = ref_final1[i][:, k] if ref_final1[i] is not None else None
            members = {}
            for key in fig_keys:
                x = ens[key][i, t].numpy()
                members[key] = x[alive(x), k]
            p_pos = {key: float((members[key] > 0).mean()) for key in fig_keys}
            y_k = ys[i, t, list(obs_idx).index(k)] if observed else None
            draw_panel(ax, i, k, observed, pf, members, truth[i, t + 1, k], y_k, preimage,
                       style, args.bw, args.bw_members, p_pos, float((pf > 0).mean()), pf_rep)
    for ax in axes[-1]:
        ax.set_xlabel(f"$x_k$ at step {t + 1}")
    handles = [Patch(facecolor="#BBBBBB", edgecolor="#000000", label=f"PF reference ($N=4\\times10^6$)")]
    if any(r is not None for r in ref_final1):
        handles += [Line2D([], [], color="#444444", lw=1.1, ls=(0, (1, 1.2)), label="PF, independent replicate")]
    handles += [Line2D([], [], color=style[k][5], ls=style[k][6], lw=1.3, label=style[k][3]) for k in fig_keys]
    handles += [Line2D([], [], color="#C0392B", lw=1.1, label="truth")]
    pre_label = {"square": r"$\pm\sqrt{y/c}$", "abs": r"$\pm y/c$", "arctan": r"$h^{-1}(y)$", "linear": "$y$"}
    op = (problem.get("observation") or {}).get("operator", "linear")
    handles += [Line2D([], [], color="#555555", lw=0.9, ls=(0, (3, 2)),
                       label=f"{pre_label.get(op, 'preimage of y')} (observed sites)")]
    fig.legend(handles=handles, loc="lower center", ncol=max(3, min(5, ncols + 1)), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.22 if len(panels) == 1 else 0.12 if len(panels) == 2 else 0.075, 1, 1), h_pad=0.6, w_pad=1.0)
    stem = out / f"l96_{obs}_final_marginals"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"\nwrote {stem}.pdf/.png and the tables in {out}")


if __name__ == "__main__":
    main()
