#!/usr/bin/env python3
"""How non-Gaussian are the PF reference posteriors of a Lorenz 96 case?

Reads only the PF references ``<root>/<obs>/reference/test/sequence<i>_replicate<r>.pt``
(dict with ``samples`` [T, 4096, d] equal-weight PF draws per step, ``problem``; optional
``final_samples`` [N, d] at the last step) and measures, per assimilation step, how far the
filtering posterior is from the Gaussian with the same mean and covariance.

Marginal statistics, per (sequence, replicate, step, site), from the 4096 draws:
    skew, exkurt     sample skewness and excess kurtosis (Gaussian: 0, 0). They are dominated
                     by a few far-away draws (e.g. a 0.05%-mass sign mode of `square`), hence:
    kelly            Kelly quantile skewness (P90 + P10 - 2 P50) / (P90 - P10); Gaussian 0,
                     noise sd 0.016 at n = 4096; ~0.1 for a classical skewness of 0.45-0.5
    moors_ex         Moors' octile kurtosis minus its Gaussian value 1.233; noise sd 0.03;
                     Laplace +0.35, t5 +0.09, uniform -0.23, two separated modes ~ -0.4
    bc               Sarle's bimodality coefficient (g^2 + 1) / (k + 3(n-1)^2/((n-2)(n-3)));
                     Gaussian 1/3, uniform 5/9, > 5/9 suggests bimodality (or strong skew)
    modes            number of KDE peaks (bandwidth --bw x min(std, IQR/1.349); a peak counts
                     when it is >= 10% of the highest and the dip to its neighbour is <= 80%
                     of the lower peak; a far minor mode of < ~5% mass is not counted)
    cramer_ratio     n-scaled Cramer distance int (F_n - Phi)^2 dx of the standardised
                     marginal to N(0, 1), divided by its mean under Gaussian sampling noise
                     (simulated with estimated mean/std): ~1 = Gaussian to within the n draws
    cramer_rep_ratio the same distance between PF replicates 0 and 1 (both standardised with
                     this replicate's moments) / its two-sample noise level 2/(n sqrt(pi)):
                     ~1 = the two PF runs agree to within draw noise
    nongauss         effect-size flag: |kelly| > --kelly-thr (0.1) or |moors_ex| >
                     --moors-thr (0.15) or modes >= 2

Multivariate statistics, per (sequence, replicate, step), in coordinates whitened by the
PF's own mean and covariance (so they measure shape only, not scale):
    ed               energy distance between the PF draws and a fresh N(0, I) sample
                     (= the moment-matched Gaussian) of the same size
    ed_null          the same distance for a Gaussian sample with the PF's duplicate
                     structure (u unique points with the PF multiplicities), whitened by
                     its own moments, i.e. the sampling-noise level of ed
    ed_ratio         ed / mean over steps of ed_null (per sequence and replicate);
                     ~1 for a Gaussian posterior
    ed_rep           (replicate 0 rows) energy distance between PF replicate 0 and 1
                     (both whitened with replicate 0), ed_rep_ratio = ed_rep / mean ed_null:
                     ~1 when the two PF runs agree to within 4096-draw sampling noise
    mardia_skew_z, mardia_kurt_z  Mardia's b1, b2 as z-scores against their Gaussian
                     null (n b1/6 ~ chi^2_{d(d+1)(d+2)/6}; b2 ~ N(d(d+2), 8d(d+2)/n)),
                     n = Kish effective size of the draws; the same z of the null sample
                     is stored as *_null

Replicate agreement: Spearman correlations of the marginal statistics between replicate 0
and 1 over all
(sequence, step, site), their per-replicate noise sd(diff)/sqrt(2), agreement of the
nongauss/bimodal flags, and ed_rep_ratio next to ed_ratio. Non-Gaussian features are real
when ed_ratio >> ed_rep_ratio ~ 1 and the marginal statistics agree across replicates.

Outputs (``--out``, default results/figures/l96_nongauss/<obs>/):
    marginal_stats.csv, step_stats.csv       per (seq, rep, step, site) / (seq, rep, step)
    summary.md / summary.csv                 the table (all steps and the final step)
    marginals_seq<i>_step<t>.pdf/.png        (a) 16 PF marginals vs moment-matched Gaussian
    pairs_seq<i>.pdf/.png                    (b) 2D PF draws vs Gaussian 1/2/3-sigma ellipses
    timeseries.pdf/.png                      (c) over the steps, per sequence: ed_ratio,
                                             mean cramer_ratio (obs./unobs.), fraction of
                                             nongauss marginals; PF replicate 0-vs-1 lines

Usage (repo venv; keep it on two cores while the machine is busy):

    taskset -c 30-31 .venv/bin/python make_l96_reference_nongaussianity.py --obs linear
    ... --obs square
    ... --obs linear_long --sequences 0 1 2 --figure-sequence 1
    ... --root /export/scratch2/ntm/delftblue_results/results/lorenz96 --obs linear
    ... --obs square --figures-only        # re-draw from the saved CSVs

Cost: ~5 n x n distance matrices per (sequence, replicate, step) with n = --ed-n (2048):
~40-50 s per (sequence, replicate) on two busy cores, ~20 min for 12 sequences x 2
replicates x 50 steps (n = 4096 / --ed-n 0: ~150 s per (sequence, replicate)). The joint
ED ratio scales roughly with n (signal fixed, noise ~ 1/n): compare cases at the same
--ed-n. Marginal statistics always use all draws.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, Patch  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import torch  # noqa: E402

# truth/observation rebuild, operator inverse images and the figure style of the square case
from make_l96_square_figures import apply_style, obs_functions, rebuild_sequences  # noqa: E402

REPO_ROOT_L96 = _HERE / "results/lorenz96"
DELFTBLUE_ROOT_L96 = Path("/export/scratch2/ntm/delftblue_results/results/lorenz96")
OUT_BASE = _HERE / "results/figures/l96_nongauss"

C_PF, C_GAUSS, C_TRUTH, C_OBS = "#4D4D4D", "#0072B2", "#C0392B", "#E69F00"


# --------------------------------------------------------------------------- #
# Locating and loading
# --------------------------------------------------------------------------- #


def resolve_base(root: str | None, obs: str) -> Path:
    """The sweep dir holding reference/test/: ``root`` itself, ``root/<obs>``, or (no
    --root) the repo results, then the DelftBlue mirror."""
    if root is not None:
        r = Path(root)
        cands = [r, r / obs]
    else:
        cands = [REPO_ROOT_L96 / obs, DELFTBLUE_ROOT_L96 / obs]
    for c in cands:
        if (c / "reference/test/sequence0_replicate0.pt").exists():
            return c
    raise FileNotFoundError(f"no reference/test/sequence0_replicate0.pt under {cands}")


def ref_path(base: Path, i: int, r: int) -> Path:
    return base / "reference/test" / f"sequence{i}_replicate{r}.pt"


def available_sequences(base: Path) -> list[int]:
    out, i = [], 0
    while ref_path(base, i, 0).exists():
        out.append(i)
        i += 1
    return out


def load_ref(base: Path, i: int, r: int) -> dict | None:
    p = ref_path(base, i, r)
    return torch.load(p, weights_only=False, map_location="cpu") if p.exists() else None


# --------------------------------------------------------------------------- #
# Marginal statistics
# --------------------------------------------------------------------------- #


def moments(x: np.ndarray):
    """Skewness and excess kurtosis along axis 0 (biased/population estimators)."""
    m = x.mean(0)
    c = x - m
    v = (c ** 2).mean(0)
    skew = (c ** 3).mean(0) / np.maximum(v, 1e-300) ** 1.5
    kurt = (c ** 4).mean(0) / np.maximum(v, 1e-300) ** 2 - 3.0
    return skew, kurt


def sarle_bc(skew, exkurt, n: int):
    """Sarle's bimodality coefficient with the sample-size correction."""
    return (skew ** 2 + 1.0) / (exkurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def binned_kde(x: np.ndarray, bw_frac: float, nbins: int = 512):
    """Gaussian KDE of a 1D sample on a regular grid (histogram + kernel convolution)."""
    p25, p75 = np.percentile(x, [25, 75])
    s = min(x.std(), (p75 - p25) / 1.349) or x.std()  # robust: a far minor mode inflates std
    lo, hi = np.percentile(x, [0.05, 99.95])
    lo, hi = lo - 3 * bw_frac * s, hi + 3 * bw_frac * s
    h, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    grid = 0.5 * (edges[1:] + edges[:-1])
    dx = grid[1] - grid[0]
    bw = bw_frac * s
    half = int(math.ceil(4 * bw / dx))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * dx / bw) ** 2)
    f = np.convolve(h, k / k.sum(), mode="same") / (len(x) * dx)
    return grid, f


def count_modes(f: np.ndarray, rel_height: float = 0.10, dip: float = 0.8) -> int:
    """Peaks >= rel_height x max, merging neighbours whose separating dip is > dip x the
    lower peak (a shoulder or a noise wiggle is not a mode)."""
    idx = [i for i in range(1, len(f) - 1) if f[i] >= f[i - 1] and f[i] > f[i + 1]
           and f[i] >= rel_height * f.max()]
    if not idx:
        return 1
    kept = [idx[0]]
    for i in idx[1:]:
        j = kept[-1]
        if f[j:i + 1].min() <= dip * min(f[i], f[j]):
            kept.append(i)
        elif f[i] > f[j]:  # same mode; keep the higher summit
            kept[-1] = i
    return len(kept)


MOORS_GAUSS = 1.2331  # Moors' octile kurtosis of a Gaussian
NORM_CRAMER = 1.0 / math.sqrt(math.pi)  # int Phi (1 - Phi) dx


def robust_shape(x: np.ndarray):
    """Kelly skewness (P90 + P10 - 2 P50) / (P90 - P10) and Moors' octile kurtosis minus
    its Gaussian value, along axis 0 (insensitive to a few far-away draws)."""
    e = np.percentile(x, [10, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 90], axis=0)
    kelly = (e[8] + e[0] - 2 * e[4]) / np.maximum(e[8] - e[0], 1e-300)
    moors = (e[7] - e[5] + e[3] - e[1]) / np.maximum(e[6] - e[2], 1e-300)
    return kelly, moors - MOORS_GAUSS


def cramer(a: np.ndarray, b: np.ndarray | None) -> float:
    """int (F_a - G)^2 dx for G = the standard normal cdf (b None) or the empirical cdf
    of b; midpoint rule on the merged sample points and a fine grid (exact for the
    steps, O(dx^2) for Phi)."""
    from scipy.special import ndtr
    a = np.sort(a)
    pts = [a, np.linspace(-8, 8, 3201)]
    if b is not None:
        b = np.sort(b)
        pts.append(b)
    g = np.unique(np.concatenate(pts))
    mid, dx = 0.5 * (g[1:] + g[:-1]), np.diff(g)
    fa = np.searchsorted(a, mid, side="right") / len(a)
    fb = ndtr(mid) if b is None else np.searchsorted(b, mid, side="right") / len(b)
    return float(((fa - fb) ** 2 * dx).sum())


def cramer_null(n: int, reps: int, seed: int) -> np.ndarray:
    """int (F_n - Phi)^2 of standardised N(0, 1) samples of size n: the sampling-noise
    distribution of the marginal Gaussian-fit distance (parameters estimated)."""
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for j in range(reps):
        x = rng.standard_normal(n)
        out[j] = cramer((x - x.mean()) / x.std(), None)
    return out


def marginal_stats(samples: np.ndarray, bw_frac: float, other: np.ndarray | None,
                   null_mean: float) -> dict:
    """samples [T, n, d] (other: the other replicate, or None) -> dict of [T, d] arrays."""
    T, n, d = samples.shape
    skew, kurt = moments(samples.transpose(1, 0, 2))
    kelly, moors = robust_shape(samples.transpose(1, 0, 2))
    modes = np.empty((T, d), int)
    cr = np.empty((T, d))
    cr_rep = np.full((T, d), np.nan)
    for t in range(T):
        for k in range(d):
            x = samples[t, :, k]
            _, f = binned_kde(x, bw_frac)
            modes[t, k] = count_modes(f)
            m, s = x.mean(), x.std()
            cr[t, k] = cramer((x - m) / s, None) / null_mean
            if other is not None:
                y = other[t, :, k]
                # two independent n-samples of one law: E = (1/n + 1/n') / sqrt(pi)
                cr_rep[t, k] = cramer((x - m) / s, (y - m) / s) / (
                    (1 / n + 1 / len(y)) * NORM_CRAMER)
    return dict(skew=skew, exkurt=kurt, bc=sarle_bc(skew, kurt, n), kelly=kelly, moors_ex=moors,
                modes=modes, cramer_ratio=cr, cramer_rep_ratio=cr_rep)


# --------------------------------------------------------------------------- #
# Multivariate statistics (weighted over unique draws)
# --------------------------------------------------------------------------- #


def unique_weights(x: torch.Tensor):
    """Unique rows of an equal-weight draw and their weights (multiplicity / n)."""
    u, counts = torch.unique(x, dim=0, return_counts=True)
    return u, counts.double() / counts.sum()


def whitener(u: torch.Tensor, w: torch.Tensor):
    m = (w[:, None] * u).sum(0)
    c = u - m
    cov = (w[:, None] * c).T @ c
    cov = 0.5 * (cov + cov.T)
    L = torch.linalg.cholesky(cov + 1e-10 * torch.eye(cov.shape[0], dtype=cov.dtype)
                              * cov.diagonal().mean())
    return lambda x: torch.linalg.solve_triangular(L, (x - m).T, upper=False).T


def mean_dist(a, wa, b, wb, chunk: int = 4096) -> float:
    """sum_ij wa_i wb_j |a_i - b_j| (chunked)."""
    tot = 0.0
    for s in range(0, a.shape[0], chunk):
        tot += float((wa[s:s + chunk, None] * torch.cdist(a[s:s + chunk], b) * wb[None]).sum())
    return tot


def energy(a, wa, b, wb, self_a=None, self_b=None):
    """Energy distance (V-statistic) 2E|A-B| - E|A-A'| - E|B-B'|; the self terms may be
    passed in when they are reused."""
    sa = mean_dist(a, wa, a, wa) if self_a is None else self_a
    sb = mean_dist(b, wb, b, wb) if self_b is None else self_b
    return 2 * mean_dist(a, wa, b, wb) - sa - sb, sa, sb


def mardia_z(z: torch.Tensor, w: torch.Tensor):
    """Mardia skewness/kurtosis z-scores of whitened weighted draws (Kish n_eff)."""
    d = z.shape[1]
    n = float(1.0 / (w ** 2).sum())
    g = z @ z.T
    b1 = float((w[:, None] * g ** 3 * w[None]).sum())
    b2 = float((w * (z ** 2).sum(1) ** 2).sum())
    f = d * (d + 1) * (d + 2) / 6
    return (n * b1 / 6 - f) / math.sqrt(2 * f), (b2 - d * (d + 2)) / math.sqrt(8 * d * (d + 2) / n)


def subsample(x: torch.Tensor, n: int | None, gen: torch.Generator) -> torch.Tensor:
    if n is None or x.shape[0] <= n:
        return x
    return x[torch.randperm(x.shape[0], generator=gen)[:n]]


def step_stats(x0: torch.Tensor, x1: torch.Tensor | None, gauss: torch.Tensor, gauss_self: float,
               gen: torch.Generator, cache: dict) -> dict:
    """Multivariate statistics of one step. x0 [n, d] (replicate under study), x1 the other
    replicate (or None), gauss a fixed N(0, I) sample with its self term."""
    u, w = unique_weights(x0)
    white = whitener(u, w)
    z = white(u)
    wg = torch.full((gauss.shape[0],), 1.0 / gauss.shape[0], dtype=torch.float64)
    ed, self_pf, _ = energy(z, w, gauss, wg, self_b=gauss_self)
    # null: a Gaussian sample with the PF's multiplicities, whitened by its own moments
    g = torch.randn(u.shape, generator=gen, dtype=torch.float64)
    zg = whitener(g, w)(g)
    ed_null, _, _ = energy(zg, w, gauss, wg, self_b=gauss_self)
    ms, mk = mardia_z(z, w)
    ms0, mk0 = mardia_z(zg, w)
    out = dict(n_unique=u.shape[0], n_eff=float(1 / (w ** 2).sum()), ed=ed, ed_null=ed_null,
               mardia_skew_z=ms, mardia_kurt_z=mk, mardia_skew_z_null=ms0, mardia_kurt_z_null=mk0)
    if x1 is not None:
        u1, w1 = unique_weights(x1)
        out["ed_rep"], _, _ = energy(z, w, white(u1), w1, self_a=self_pf)
    return out


# --------------------------------------------------------------------------- #
# Analysis driver
# --------------------------------------------------------------------------- #


def analyse(base: Path, seqs: list[int], args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    gen = torch.Generator().manual_seed(args.seed)
    marg_rows, step_rows = [], []
    gauss, gauss_self = None, None
    problem, null = None, None
    for i in seqs:
        refs = [load_ref(base, i, r) for r in (0, 1)]
        if refs[1] is None:
            warnings.warn(f"sequence {i}: replicate 1 missing; no replicate agreement")
        problem = problem or refs[0]["problem"]
        d = refs[0]["samples"].shape[-1]
        obs_idx = set(range(0, d, int(refs[0]["problem"].get("obs_stride", 4))))
        sub = [None if r is None else subsample_all(r["samples"].double(), args.ed_n or None, gen)
               for r in refs]
        if gauss is None:
            gauss = torch.randn(sub[0].shape[1], d, generator=gen, dtype=torch.float64)
            wg = torch.full((gauss.shape[0],), 1.0 / gauss.shape[0], dtype=torch.float64)
            gauss_self = mean_dist(gauss, wg, gauss, wg)
        for r, ref in enumerate(refs):
            if ref is None:
                continue
            t0 = time.time()
            s = ref["samples"].double().numpy()
            T = s.shape[0]
            other = refs[1 - r]["samples"].double().numpy() if refs[1 - r] is not None else None
            if null is None:
                null = cramer_null(s.shape[1], args.null_reps, args.seed)
                print(f"  marginal Cramer null (n={s.shape[1]}): mean {null.mean():.3g}, "
                      f"p95/mean {np.percentile(null, 95) / null.mean():.2f}, "
                      f"p99/mean {np.percentile(null, 99) / null.mean():.2f}", flush=True)
            ms = marginal_stats(s, args.bw, other, float(null.mean()))
            for t in range(T):
                for k in range(d):
                    marg_rows.append(dict(sequence=i, replicate=r, step=t + 1, site=k,
                                          observed=k in obs_idx, mean=s[t, :, k].mean(),
                                          std=s[t, :, k].std(),
                                          **{key: ms[key][t, k] for key in ms}))
            for t in range(T):
                other = sub[1 - r][t] if (r == 0 and sub[1] is not None) else None
                st = step_stats(sub[r][t], other, gauss, gauss_self, gen, {})
                step_rows.append(dict(sequence=i, replicate=r, step=t + 1,
                                      ess=float(ref["ess"][t]) if "ess" in ref else math.nan, **st))
            print(f"  sequence {i} replicate {r}: {time.time() - t0:.0f}s", flush=True)
    marg = pd.DataFrame(marg_rows)
    # effect-size flag on the outlier-robust shape measures (classical skew/kurtosis are
    # dominated by a few far-away draws, e.g. a 0.05%-mass sign mode)
    marg["nongauss"] = ((marg.kelly.abs() > args.kelly_thr) | (marg.moors_ex.abs() > args.moors_thr)
                        | (marg.modes >= 2))
    st = pd.DataFrame(step_rows)
    per = marg.groupby(["sequence", "replicate", "step"])
    agg = pd.DataFrame({
        "cramer_ratio_mean": per.cramer_ratio.mean(), "cramer_ratio_max": per.cramer_ratio.max(),
        "cramer_ratio_mean_obs": marg[marg.observed].groupby(["sequence", "replicate", "step"]).cramer_ratio.mean(),
        "cramer_ratio_mean_unobs": marg[~marg.observed].groupby(["sequence", "replicate", "step"]).cramer_ratio.mean(),
        "cramer_rep_ratio_mean": per.cramer_rep_ratio.mean(),
        "frac_nongauss": per.nongauss.mean()}).reset_index()
    st = st.merge(agg, on=["sequence", "replicate", "step"], how="left")
    st["ed_null_mean"] = st.groupby(["sequence", "replicate"]).ed_null.transform("mean")
    st["ed_ratio"] = st.ed / st.ed_null_mean
    st["ed_null_ratio"] = st.ed_null / st.ed_null_mean
    if "ed_rep" in st:
        st["ed_rep_ratio"] = st.ed_rep / st.ed_null_mean
    return marg, st, problem


def subsample_all(samples: torch.Tensor, n: int | None, gen) -> torch.Tensor:
    """[T, N, d] -> [T, n, d] (the same random subset size per step)."""
    if n is None or samples.shape[1] <= n:
        return samples
    return torch.stack([subsample(samples[t], n, gen) for t in range(samples.shape[0])])


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #


def q(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else math.nan


def summarise(marg: pd.DataFrame, st: pd.DataFrame, args, T: int) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    m0 = marg[marg.replicate == 0]
    for scope, mm in (("all steps", m0), (f"final step ({T})", m0[m0.step == T])):
        for group, g in (("observed", mm[mm.observed]), ("unobserved", mm[~mm.observed]),
                         ("all sites", mm)):
            rows.append({
                "scope": scope, "sites": group, "n": len(g),
                "|skew| med": q(g["skew"].abs(), 50), "|skew| p90": q(g["skew"].abs(), 90),
                "exkurt med": q(g.exkurt, 50), "|exkurt| p90": q(g.exkurt.abs(), 90),
                "|Kelly| med": q(g.kelly.abs(), 50), "|Kelly| p90": q(g.kelly.abs(), 90),
                "Moors-G med": q(g.moors_ex, 50), "|Moors-G| p90": q(g.moors_ex.abs(), 90),
                "BC med": q(g.bc, 50), "frac BC>5/9": float((g.bc > 5 / 9).mean()),
                "frac modes>=2": float((g.modes >= 2).mean()),
                "Cramer ratio med": q(g.cramer_ratio, 50), "Cramer ratio p90": q(g.cramer_ratio, 90),
                "rep Cramer ratio med": q(g.cramer_rep_ratio, 50),
                "frac non-Gauss": float(g.nongauss.mean()),
            })
    marg_tab = pd.DataFrame(rows)

    s0 = st[st.replicate == 0]
    null95 = q(st.ed_null_ratio, 95)
    mrows = []
    for scope, ss in (("all steps", s0), (f"final step ({T})", s0[s0.step == T])):
        r = {"scope": scope, "n steps": len(ss),
             "ED ratio med": q(ss.ed_ratio, 50), "ED ratio p90": q(ss.ed_ratio, 90),
             "frac ED > null p95": float((ss.ed_ratio > null95).mean()),
             "null ED ratio p95": null95,
             "Mardia skew z med": q(ss.mardia_skew_z, 50), "Mardia skew z p90": q(ss.mardia_skew_z, 90),
             "Mardia kurt z med": q(ss.mardia_kurt_z, 50), "Mardia kurt z p90": q(ss.mardia_kurt_z, 90),
             "null Mardia skew z med": q(ss.mardia_skew_z_null, 50),
             "null Mardia kurt z med": q(ss.mardia_kurt_z_null, 50)}
        if "ed_rep_ratio" in ss:
            r.update({"ED rep ratio med": q(ss.ed_rep_ratio, 50), "ED rep ratio p90": q(ss.ed_rep_ratio, 90)})
        mrows.append(r)
    mv_tab = pd.DataFrame(mrows)

    # replicate agreement of the marginal statistics
    key = ["sequence", "step", "site"]
    both = marg[marg.replicate == 0].merge(marg[marg.replicate == 1], on=key, suffixes=("0", "1"))
    agree = []
    if len(both):
        for stat in ("skew", "exkurt", "kelly", "moors_ex", "bc", "cramer_ratio"):
            a, b = both[stat + "0"], both[stat + "1"]
            agree.append(dict(statistic=stat, corr=float(a.corr(b, method="spearman")),
                              noise_sd=float((a - b).std() / math.sqrt(2)),
                              spread_sd=float(a.std())))
        for flag in ("nongauss", "bimodal"):
            if flag == "bimodal":
                a, b = both.modes0 >= 2, both.modes1 >= 2
            else:
                a, b = both[flag + "0"], both[flag + "1"]
            agree.append(dict(statistic=f"{flag} flag", corr=float((a == b).mean()),
                              noise_sd=float((a & b).sum() / max(a.sum(), 1)),
                              spread_sd=float(a.mean())))
    agree_tab = pd.DataFrame(agree)

    md = []
    fmtf = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    def to_md(df):
        out = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        out += ["| " + " | ".join(fmtf(v) for v in row) + " |" for row in df.itertuples(index=False)]
        return out
    md += ["### Marginals (PF replicate 0; per sequence x step x site)", ""] + to_md(marg_tab)
    md += ["", f"Gaussian values: skew 0, exkurt 0, Kelly 0, Moors-G 0, BC 1/3, 1 mode, Cramer ratio ~1. "
           "Kelly = quantile skewness, Moors-G = octile kurtosis minus its Gaussian value (both robust to "
           "a few far-away draws, which dominate skew/exkurt). Cramer ratio = n int (F_n - Phi)^2 of the "
           "standardised marginal / its mean under Gaussian sampling noise (simulated, n draws); rep "
           "Cramer ratio = the same distance between PF replicates 0 and 1 / its two-sample noise level "
           f"(~1 = the replicates agree). non-Gauss flag: |Kelly| > {args.kelly_thr} or |Moors-G| > "
           f"{args.moors_thr} or >= 2 KDE modes (bw {args.bw} x robust std).", ""]
    md += ["### Multivariate (whitened; PF replicate 0; per sequence x step)", ""] + to_md(mv_tab)
    md += ["", "ED ratio: energy distance PF vs moment-matched Gaussian / mean sampling-noise level "
           "(Gaussian sample with the PF's multiplicities); ~1 = indistinguishable from Gaussian. "
           "ED rep ratio: PF replicate 0 vs replicate 1 on the same scale (~1 = the replicates agree "
           "to within draw noise). Mardia z: ~0 under Gaussianity (null column = the noise sample).", ""]
    if len(agree_tab):
        agree_tab = agree_tab.rename(columns={"corr": "corr / flag agreement",
                                              "noise_sd": "noise sd / P(rep1 | rep0 flagged)",
                                              "spread_sd": "spread sd / rep0 flag rate"})
        md += ["### Replicate agreement (replicate 0 vs 1, all sequence x step x site)", ""] + to_md(agree_tab)
        md += ["", "Rows of statistics: Spearman corr, per-replicate PF noise sd(diff)/sqrt(2), and the "
               "spread (sd) of the statistic across (sequence, step, site): signal when spread >> noise. "
               "Flag rows: fraction of marginals where the replicates agree, P(flagged in rep 1 | "
               "flagged in rep 0), and the replicate-0 flag rate."]
    table = pd.concat([marg_tab.assign(kind="marginal"), mv_tab.assign(kind="multivariate"),
                       agree_tab.assign(kind="replicate")], ignore_index=True)
    return table, md


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def gauss_pdf(grid, m, s):
    return np.exp(-0.5 * ((grid - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def fig_marginals(draws, i, t, truth_t, y_t, obs_idx, preimage, marg_row, args, out, label, n_note):
    """(a) all sites of one sequence at one step: PF histogram + KDE vs moment-matched Gaussian."""
    d = draws.shape[1]
    ncol = 4
    nrow = int(math.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5, 1.25 * nrow + 0.45), squeeze=False)
    for k in range(d):
        ax = axes.flat[k]
        x = draws[:, k]
        m, s = x.mean(), x.std()
        grid, f = binned_kde(x, args.bw)
        lo, hi = grid[0], grid[-1]
        lo, hi = min(lo, m - 3.5 * s, truth_t[k] - 0.3 * s), max(hi, m + 3.5 * s, truth_t[k] + 0.3 * s)
        ax.hist(x, bins=60, range=(lo, hi), density=True, color="#CCCCCC", lw=0)
        ax.plot(grid, f, color=C_PF, lw=1.1)
        gg = np.linspace(lo, hi, 400)
        ax.plot(gg, gauss_pdf(gg, m, s), color=C_GAUSS, lw=1.0, ls="--")
        ax.axvline(truth_t[k], color=C_TRUTH, lw=1.0)
        if k in obs_idx and y_t is not None:
            for wv in preimage(y_t[obs_idx.index(k)]):
                if lo <= wv <= hi:
                    ax.plot([wv], [0], marker="^", color=C_OBS, ms=5.5, clip_on=False, zorder=5,
                            markeredgecolor="black", markeredgewidth=0.4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.32 * max(f.max(), gauss_pdf(m, m, s)))
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        kel, moo = robust_shape(x)
        nm = count_modes(f)
        flagged = abs(kel) > args.kelly_thr or abs(moo) > args.moors_thr or nm >= 2
        txt = f"$s_K$={kel:+.2f}\n$\\kappa_M$={moo:+.2f}" + (f"\n{nm} modes" if nm >= 2 else "")
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=6.2, va="top", ha="left",
                linespacing=1.15, color="#B2182B" if flagged else "black")
        outside = int(((x < lo) | (x > hi)).sum())
        if outside >= 5e-4 * len(x):
            ax.text(0.98, 0.97, f"{100 * outside / len(x):.2g}% of draws\noff-axis", transform=ax.transAxes,
                    fontsize=5.8, va="top", ha="right", color="#555555", linespacing=1.1)
        ax.set_title(f"$x_{{{k}}}$" + (" (obs.)" if k in obs_idx else ""), fontsize=8.5, pad=2,
                     fontweight="bold" if k in obs_idx else "normal")
        ax.tick_params(axis="x", labelsize=6.8, pad=1.5)
    for ax in axes.flat[d:]:
        ax.axis("off")
    handles = [Patch(facecolor="#CCCCCC", edgecolor=C_PF, label=f"PF reference ({n_note})"),
               Line2D([], [], color=C_GAUSS, ls="--", label="moment-matched Gaussian"),
               Line2D([], [], color=C_TRUTH, label="truth"),
               Line2D([], [], color=C_OBS, marker="^", ls="none", markeredgecolor="black",
                      markeredgewidth=0.4, label="$h^{-1}(y)$ (observed sites)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0),
               fontsize=7.5, handlelength=1.8, columnspacing=1.0)
    fig.suptitle(f"{label}: PF posterior marginals, sequence {i}, step {t}  "
                 f"($s_K$ Kelly skewness, $\\kappa_M$ Moors kurtosis $-$ Gaussian; red = non-Gaussian)",
                 fontsize=8.5, y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975), h_pad=0.5, w_pad=0.6)
    save(fig, out / f"marginals_seq{i}_step{t}")


def cov_ellipses(ax, m, c, color):
    vals, vecs = np.linalg.eigh(c)
    ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    for nsig in (1, 2, 3):
        ax.add_patch(Ellipse(m, 2 * nsig * math.sqrt(vals[1]), 2 * nsig * math.sqrt(vals[0]),
                             angle=ang, fill=False, color=color, lw=1.0, ls="--", zorder=4))


def fig_pairs(step_draws, steps, pairs, i, truth, obs_idx, out, label, gen_np):
    """(b) 2D PF draws (scatter + density contours) vs Gaussian 1/2/3-sigma ellipses."""
    nrow, ncol = len(steps), len(pairs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.62 * ncol + 0.2, 1.62 * nrow + 0.5), squeeze=False)
    for a, t in enumerate(steps):
        x = step_draws[t]
        sel = gen_np.choice(x.shape[0], size=min(3000, x.shape[0]), replace=False)
        for b, (k, l) in enumerate(pairs):
            ax = axes[a, b]
            xy = x[:, [k, l]]
            m, c = xy.mean(0), np.cov(xy.T)
            ax.scatter(xy[sel, 0], xy[sel, 1], s=1.2, color=C_PF, alpha=0.25, lw=0, rasterized=True)
            # smoothed 2D histogram contours of all draws
            sd = np.sqrt(np.diag(c))
            lims = [(m[j] - 4 * sd[j], m[j] + 4 * sd[j]) for j in range(2)]
            H, ex, ey = np.histogram2d(xy[:, 0], xy[:, 1], bins=60, range=lims, density=True)
            H = smooth2d(H, 1.5)
            levels = H.max() * np.array([0.05, 0.25, 0.6])
            ax.contour(0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1]), H.T, levels=levels,
                       colors="black", linewidths=0.7, zorder=3)
            cov_ellipses(ax, m, c, C_GAUSS)
            ax.plot(truth[t, k], truth[t, l], marker="*", color=C_TRUTH, ms=7, mec="white", mew=0.4, zorder=6)
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            tag = lambda j: f"$x_{{{j}}}$" + ("$^{\\rm o}$" if j in obs_idx else "")  # noqa: E731
            ax.set_xlabel(tag(k), labelpad=1)
            ax.set_ylabel(tag(l), labelpad=1)
            ax.tick_params(labelsize=6.5, pad=1)
            if b == 0:
                ax.text(-0.42, 0.5, f"step {t}", transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=8.5)
    handles = [Line2D([], [], color=C_PF, marker="o", ls="none", ms=2.5, alpha=0.6, label="PF draws"),
               Line2D([], [], color="black", lw=0.7, label="PF density contours"),
               Line2D([], [], color=C_GAUSS, ls="--", label="Gaussian 1/2/3$\\sigma$"),
               Line2D([], [], color=C_TRUTH, marker="*", ls="none", ms=7, label="truth")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0),
               fontsize=7.5, handlelength=1.6, columnspacing=1.0)
    fig.suptitle(f"{label}: PF posterior pairs, sequence {i} ($^{{\\rm o}}$ = observed site)",
                 fontsize=9, y=0.995)
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.965), h_pad=0.6, w_pad=0.5)
    save(fig, out / f"pairs_seq{i}")


def smooth2d(H, sigma):
    half = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    H = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, H)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, H)


def fig_timeseries(marg, st, out, label, highlight):
    """(c) per sequence over the steps: joint energy-distance ratio (with the Gaussian noise
    band and the PF replicate distance), mean marginal Cramer ratio, and the fraction of
    non-Gaussian marginals."""
    s0 = st[st.replicate == 0].sort_values("step")
    m0 = marg[marg.replicate == 0]
    T = int(s0.step.max())
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))

    def per_seq(ax, col, frame):
        for i, g in frame.groupby("sequence"):
            hl = i == highlight
            ax.plot(g.step, g[col], color="#B2182B" if hl else "#777777", lw=1.2 if hl else 0.6,
                    alpha=1.0 if hl else 0.6, zorder=3 if hl else 2)

    ax = axes[0]
    ax.axhspan(q(st.ed_null_ratio, 5), q(st.ed_null_ratio, 95), color="#DDDDDD", lw=0, zorder=0)
    per_seq(ax, "ed_ratio", s0)
    med = s0.groupby("step").ed_ratio.median()
    ax.plot(med.index, med.values, color="black", lw=1.6, zorder=4)
    if "ed_rep_ratio" in s0 and s0.ed_rep_ratio.notna().any():
        rep = s0.groupby("step").ed_rep_ratio.median()
        ax.plot(rep.index, rep.values, color=C_GAUSS, lw=1.2, ls="--", zorder=4)
    ax.set_yscale("log")
    ax.set_title("joint (16-d, whitened): ED / noise", fontsize=8)

    ax = axes[1]
    per_seq(ax, "cramer_ratio_mean", s0)
    for col, ls in (("cramer_ratio_mean_obs", "-"), ("cramer_ratio_mean_unobs", ":")):
        med = s0.groupby("step")[col].median()
        ax.plot(med.index, med.values, color="black", lw=1.6, ls=ls, zorder=4)
    if s0.cramer_rep_ratio_mean.notna().any():
        rep = s0.groupby("step").cramer_rep_ratio_mean.median()
        ax.plot(rep.index, rep.values, color=C_GAUSS, lw=1.2, ls="--", zorder=4)
    ax.axhline(1.0, color="#AAAAAA", lw=0.8, zorder=1)
    ax.set_yscale("log")
    ax.set_title("marginals: Cramér dist. / noise", fontsize=8)

    ax = axes[2]
    fr = m0.groupby(["sequence", "step"]).nongauss.mean().reset_index()
    per_seq(ax, "nongauss", fr)
    for obs_flag, ls in ((True, "-"), (False, ":")):
        f = m0[m0.observed == obs_flag].groupby("step").nongauss.mean()
        ax.plot(f.index, f.values, color="black", lw=1.6, ls=ls, zorder=4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("fraction of non-Gaussian marginals", fontsize=8)
    for ax in axes:
        ax.set_xlim(1, T)
        ax.set_xlabel("assimilation step")
    handles = [Line2D([], [], color="#777777", lw=0.6, label="one sequence (mean over sites)"),
               Line2D([], [], color="#B2182B", lw=1.2, label=f"sequence {highlight} (marginal/pair figures)"),
               Line2D([], [], color="black", lw=1.6, label="median over sequences (mid, right: observed sites)"),
               Line2D([], [], color="black", lw=1.6, ls=":", label="same, unobserved sites (mid, right)"),
               Line2D([], [], color=C_GAUSS, lw=1.2, ls="--", label="PF replicate 0 vs 1 (median)"),
               Patch(facecolor="#DDDDDD", label="Gaussian sampling noise, 5-95%")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
               fontsize=7, handlelength=1.8, columnspacing=1.0)
    fig.suptitle(f"{label}: non-Gaussianity of the PF reference over the assimilation window "
                 "(1 = Gaussian up to sampling noise)", fontsize=8.5, y=0.995)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95), w_pad=1.2)
    save(fig, out / "timeseries")


def save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", default="linear", help="case name under results/lorenz96 (linear, square, ...)")
    ap.add_argument("--root", default=None, help="results/lorenz96 dir (holding <obs>/) or the case "
                    "dir itself; default: repo results, then the DelftBlue mirror")
    ap.add_argument("--sequences", type=int, nargs="+", default=None, help="test sequences (default all)")
    ap.add_argument("--out", default=None, help="default results/figures/l96_nongauss/<obs>/")
    ap.add_argument("--figure-sequence", type=int, default=None,
                    help="sequence of figures a/b (default: the one with the median mean ED ratio)")
    ap.add_argument("--mid-step", type=int, default=None, help="mid-sequence step of a/b (1-based; default T/2)")
    ap.add_argument("--pairs", default="0-1,1-2,2-3,3-4,0-4",
                    help="site pairs of figure b, e.g. '0-1,1-2' (neighbours: obs/unobs, unobs/unobs)")
    ap.add_argument("--bw", type=float, default=0.2, help="marginal KDE bandwidth, fraction of the std")
    ap.add_argument("--kelly-thr", type=float, default=0.1,
                    help="non-Gaussian marginal: |Kelly skewness| above this (Gaussian 0; ~0.08 for "
                         "classical skewness 0.5)")
    ap.add_argument("--moors-thr", type=float, default=0.15,
                    help="... or |Moors octile kurtosis - 1.233| above this (Laplace +0.35, uniform -0.23)")
    ap.add_argument("--null-reps", type=int, default=400, help="draws of the marginal Cramer null")
    ap.add_argument("--ed-n", type=int, default=2048, help="random subset of the draws per step for "
                    "the joint energy distance / Mardia (default 2048; the ED ratio grows with n, so "
                    "compare cases at the same n; 0 = all draws, 4x slower)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figures-only", action="store_true", help="reuse marginal_stats.csv / step_stats.csv")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    base = resolve_base(args.root, args.obs)
    out = Path(args.out) if args.out else OUT_BASE / args.obs
    out.mkdir(parents=True, exist_ok=True)
    seqs = args.sequences or available_sequences(base)
    print(f"references: {base}\nsequences: {seqs}\nout: {out}")

    if args.figures_only:
        marg = pd.read_csv(out / "marginal_stats.csv")
        st = pd.read_csv(out / "step_stats.csv")
        marg = marg[marg.sequence.isin(seqs)]
        st = st[st.sequence.isin(seqs)]
        problem = load_ref(base, seqs[0], 0)["problem"]
    else:
        marg, st, problem = analyse(base, seqs, args)
        marg.to_csv(out / "marginal_stats.csv", index=False)
        st.to_csv(out / "step_stats.csv", index=False)
    T = int(st.step.max())
    print(f"problem: {problem}")

    table, md = summarise(marg, st, args, T)
    head = [f"## Non-Gaussianity of the PF reference: Lorenz 96 `{args.obs}`", "",
            f"{base}; sequences {seqs}; T = {T}, interval {problem.get('interval')}, "
            f"observation {problem.get('observation', {'operator': 'linear'})}, "
            f"obs variance {problem.get('obs_variance')}, obs stride {problem.get('obs_stride')}; "
            f"joint statistics on {args.ed_n or 'all'} draws per step", ""]
    (out / "summary.md").write_text("\n".join(head + md) + "\n")
    table.to_csv(out / "summary.csv", index=False)
    print("\n".join(head + md))

    # ---- figures ------------------------------------------------------------
    apply_style()
    s0 = st[st.replicate == 0]
    per_seq = s0.groupby("sequence").ed_ratio.mean().sort_values()
    fs = args.figure_sequence if args.figure_sequence is not None else int(per_seq.index[len(per_seq) // 2])
    print(f"\nfigure sequence {fs} (mean ED ratio {per_seq.get(fs, math.nan):.2f}; "
          f"range over sequences {per_seq.min():.2f}-{per_seq.max():.2f})")
    ref = load_ref(base, fs, 0)
    samples = ref["samples"].double().numpy()
    truth, ys, obs_idx = rebuild_sequences(problem, fs + 1)
    truth, ys, obs_idx = truth[fs], ys[fs], [int(k) for k in obs_idx]
    _, preimage = obs_functions(problem.get("observation"))
    label = f"L96 {args.obs}"
    mid = args.mid_step or T // 2
    m0 = marg[(marg.replicate == 0) & (marg.sequence == fs)]
    for t in (mid, T):
        draws, note = samples[t - 1], f"{samples.shape[1]} draws"
        if t == T and "final_samples" in ref and ref["final_samples"] is not None:
            draws = ref["final_samples"].double().numpy()
            note = f"{draws.shape[0]:,} draws"
        fig_marginals(draws, fs, t, truth[t], ys[t - 1], obs_idx, preimage, m0[m0.step == t],
                      args, out, label, note)
    pairs = [tuple(int(v) for v in p.split("-")) for p in args.pairs.split(",")]
    fig_pairs({t: samples[t - 1] for t in (mid, T)}, (mid, T), pairs, fs, truth, obs_idx, out,
              label, np.random.default_rng(args.seed))
    fig_timeseries(marg, st, out, label, fs)


if __name__ == "__main__":
    main()
