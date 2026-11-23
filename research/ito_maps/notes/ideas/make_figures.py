"""Figures for ito_map_improvement_ideas.tex. Run: uv run python make_figures.py

Panels marked "illustration" use synthetic numbers; all others plot values
quoted in the note (data archive statistics or toy results).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_L, BLUE_D = "#86b6ef", "#104281"
FILL_B, FILL_O, FILL_G = "#e3eefc", "#fde9e0", "#f0efec"

plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "mathtext.fontset": "dejavusans",
    }
)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def blank(ax, xlim=(0, 10), ylim=(0, 4)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def box(ax, x, y, w, h, text, fc=FILL_G, ec=AXIS, fs=7.5, color=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=fc, ec=ec, lw=1,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, p, q, color=INK2, rad=0.0, lw=1.3, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p, q, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw,
            linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
        )
    )


def note(ax, x, y, text, color=INK2, ha="center", fs=7, va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color)


# ---------------------------------------------------------------- diagnosis
def fig_diagnosis():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.32})
    k = np.array([1, 4, 7, 11, 16, 32, 48, 64])
    r = np.array([0.31, 20, 190, 59, 9.8, 0.59, 0.066, 0.012])
    a.axhspan(1, 1e3, color=FILL_B, lw=0)
    a.axhspan(1e-3, 1, color=FILL_O, lw=0)
    a.axhline(1, color=MUTED, lw=1)
    a.plot(k, r, "-o", color=INK, ms=4)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylim(5e-3, 5e2)
    a.set_xticks(k, ["1", "4", "7", "11", "16", "32", "", "64"])
    a.minorticks_off()
    a.set_xlabel("wavenumber shell |k|")
    a.set_ylabel("increment power / white-noise power")
    a.set_title("White driver: mismatched by four decades")
    note(a, 1.1, 180, "too little noise:\nmust amplify", ha="left")
    note(a, 1.1, 0.02, "too much noise:\nmust cancel", ha="left")
    note(a, 62, 1.6, "matched", color=MUTED, ha="right")

    n = [1, 2, 4, 8, 16, 64]
    s = [0.34, 0.58, 0.65, 0.78, 0.88, 0.97]
    b.axhline(1, color=MUTED, lw=1)
    b.plot(range(len(n)), s, "-o", color=ORANGE, ms=4)
    for i, v in enumerate(s):
        b.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, -12),
                   ha="center", fontsize=7, color=INK2)
    b.set_xticks(range(len(n)), [str(v) for v in n])
    b.set_ylim(0, 1.15)
    b.set_xlabel("sampling steps N")
    b.set_ylabel("ensemble spread / true spread")
    b.set_title("Ignoring the path loses spread")
    note(b, 0, 1.06, "calibrated", color=MUTED, ha="left")
    save(fig, "fig_diagnosis")


# ------------------------------------------------------------------ idea A
def fig_a():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.55})
    k = np.array([1, 4, 7, 11, 16, 32, 48, 64])
    r = np.array([0.31, 20, 190, 59, 9.8, 0.59, 0.066, 0.012])
    a.plot(k, np.ones_like(r), color=MUTED, label="white driver (today)")
    a.plot(k, r, color=INK, label="forecast uncertainty")
    a.plot(k, 1.5 * r, color=BLUE, ls="--", label="matched driver (proposed)")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xticks(k, ["1", "4", "7", "11", "16", "32", "", "64"])
    a.minorticks_off()
    a.set_xlabel("wavenumber shell |k|")
    a.set_ylabel("power (white driver = 1)")
    a.set_title("Give the noise the uncertainty's spectrum")
    a.legend(fontsize=7, loc="lower left", handlelength=2.2)

    labels = [
        "today: white driver,\n" + r"$\beta=t^2$",
        "matched covariance,\n" + r"$\beta=t^2$",
        "matched covariance,\n" + r"$\beta=t$",
        "matched covariance,\n" + r"$\beta=t\sqrt{3-2t}$",
    ]
    vals = [21, 5.0, 1.2, 0.1]
    y = np.arange(4)[::-1]
    b.barh(y, vals, height=0.55, color=[MUTED, BLUE, BLUE, BLUE])
    for yi, v in zip(y, vals):
        b.text(v + 0.5, yi, f"{v:g}%", va="center", fontsize=7.5, color=INK)
    b.set_yticks(y, labels, fontsize=7)
    b.set_xlim(0, 26)
    b.grid(axis="y", visible=False)
    b.set_xlabel("spread the network must still learn\n(% of variance not carried by the noise)")
    b.set_title("The network has little left to learn")
    save(fig, "fig_a")


# ------------------------------------------------------------------ idea B
def fig_b():
    rng = np.random.default_rng(0)
    n = 700
    psi = rng.normal(size=n)
    target = 0.8 * psi + rng.choice([-0.7, 0.7], n) + 0.15 * rng.normal(size=n)
    student = 0.8 * psi + rng.choice([-0.7, 0.7], n) + 0.15 * rng.normal(size=n)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True,
                               gridspec_kw={"wspace": 0.08})
    for ax in (a, b):
        ax.scatter(psi, target, s=5, color=AXIS, lw=0, label="target (two fine half-steps)")
        ax.set_xlabel(r"resolved path features $\Psi_{s,t}$")
        ax.set_xlim(-3, 3)
    xs = np.linspace(-3, 3, 2)
    a.plot(xs, 0.8 * xs, color=ORANGE, label="student output")
    a.set_ylabel("map output")
    a.set_title("Today (MSE): conditional mean only")
    b.scatter(psi[:350], student[:350], s=5, color=BLUE, lw=0,
              label=r"student samples (latent $\eta$)")
    b.set_title("Proposed (energy score): conditional law")
    for ax in (a, b):
        ax.legend(fontsize=7, loc="upper left", markerscale=2, handlelength=1.5)
    save(fig, "fig_b")


# ------------------------------------------------------------------ idea C
def fig_c():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"wspace": 0.05})
    for ax in (a, b):
        blank(ax, (0, 10), (0, 4.2))
        ax.plot([1, 9], [1, 1], color=AXIS, lw=1.5)
        for x, lab in ((1, "s"), (5, "u"), (9, "t")):
            ax.plot([x, x], [0.9, 1.1], color=MUTED, lw=1)
            note(ax, x, 0.55, lab, fs=8)
        note(ax, 5, 0.1, "generative time", color=MUTED)
    a.set_title("Today: target from two learned maps")
    arrow(a, (1, 1.15), (5, 1.15), color=ORANGE, rad=-0.45)
    arrow(a, (5, 1.15), (9, 1.15), color=ORANGE, rad=-0.45)
    arrow(a, (1, 1.15), (9, 1.15), color=INK, rad=-0.38)
    note(a, 3, 1.45, "child map", color=INK2)
    note(a, 7, 1.45, "child map", color=INK2)
    note(a, 5, 2.95, "parent map = composition of the children", color=INK)
    note(a, 5, 3.75, "child errors pass to the parent; the drift is never queried", color=INK2)

    b.set_title("Proposed: target queries the drift")
    arrow(b, (1, 1.15), (5, 1.15), color=MUTED, rad=-0.45)
    note(b, 3, 1.45, r"map to $X_u$", color=INK2)
    b.plot([5], [1.18], "o", color=BLUE, ms=6)
    arrow(b, (5, 1.3), (5, 2.7), color=BLUE)
    box(b, 2.4, 2.75, 5.2, 0.7, r"drift $b_u(X_u)$, trained on data", fc=FILL_B, ec=BLUE_L)
    note(b, 5, 3.85, r"parent target = average over random nodes $u$", color=INK)
    note(b, 8.1, 1.9, "small-gap errors\nenter multiplied\nby the gap", color=INK2)
    save(fig, "fig_c")


# ------------------------------------------------------------------ idea D
def fig_d():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6),
                               gridspec_kw={"wspace": 0.2, "width_ratios": [1.6, 1]})
    blank(a, (0, 10), (0, 4.2))
    a.set_title("Refresh: re-noise the forecast, map it back")
    box(a, 0.2, 1.6, 2.4, 1.0, "forecast $y$\n(with artefacts)", fc=FILL_O, ec="#f3b79d")
    box(a, 3.8, 1.6, 2.4, 1.0, "noisy state $x_s$\nat time $s$", fc=FILL_G)
    box(a, 7.4, 1.6, 2.4, 1.0, "refreshed $y'$\ninto history", fc=FILL_B, ec=BLUE_L)
    arrow(a, (2.65, 2.1), (3.75, 2.1))
    arrow(a, (6.25, 2.1), (7.35, 2.1), color=BLUE)
    note(a, 3.2, 2.95, "interpolant\nnoise")
    note(a, 6.8, 2.95, r"map $\Phi_{s,1}$" + "\n1 network call", color=INK2)
    note(a, 5, 0.75, "leaves the forecast law unchanged;\nerrors below the noise floor are forgotten", color=INK2)
    note(a, 5, 3.85, "same kernel builds the training histories from truth frames", color=INK2)

    sf = [0.05, 0.1, 0.3]
    series = [(0.8, [0.16, 0.32, 0.72], BLUE_L), (0.9, [0.38, 0.64, 0.93], BLUE),
              (0.95, [0.68, 0.88, 0.98], BLUE_D)]
    for s, kappa, col in series:
        b.plot(range(3), kappa, "-o", color=col, ms=4)
        b.text(2.08, kappa[-1] + {0.8: 0, 0.9: -0.035, 0.95: 0.035}[s], f"s = {s}", va="center", fontsize=7, color=INK2)
    b.set_xticks(range(3), [str(v) for v in sf])
    b.set_xlim(-0.2, 2.9)
    b.set_ylim(0, 1.08)
    b.set_xlabel(r"true spread of a mode, $\sigma_f$")
    b.set_ylabel("fraction of a perturbation kept")
    b.set_title("Narrow modes are scrubbed")
    save(fig, "fig_d")


# ------------------------------------------------------------------ idea E
def fig_e():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6),
                               gridspec_kw={"wspace": 0.2, "width_ratios": [1.35, 1]})
    blank(a, (0, 10), (0, 4.2))
    a.set_title("Rollout must match a direct forecast")
    xs = [0.9, 3.0, 5.1, 7.2, 9.3]
    labs = ["$h_n$", "$y_{n+1}$", "$y_{n+2}$", r"$\cdots$", r"$y_{n+\ell}$"]
    for x, lab in zip(xs, labs):
        box(a, x - 0.65, 2.7, 1.3, 0.75, lab, fc=FILL_G)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        arrow(a, (x0 + 0.68, 3.08), (x1 - 0.68, 3.08), color=ORANGE)
    note(a, 5.1, 3.85, r"rollout: $\ell$ one-step maps, errors compound", color=INK2)
    box(a, 0.25, 0.55, 1.3, 0.75, "$h_n$", fc=FILL_G)
    box(a, 8.65, 0.55, 1.3, 0.75, r"$y_{n+\ell}$", fc=FILL_B, ec=BLUE_L)
    arrow(a, (1.6, 0.92), (8.6, 0.92), color=BLUE)
    note(a, 5.1, 0.3, r"direct lead-$\ell$ map: trained on data, no compounding", color=INK2)
    arrow(a, (9.3, 2.65), (9.3, 1.35), color=INK, lw=1.2)
    arrow(a, (9.3, 1.35), (9.3, 2.65), color=INK, lw=1.2)
    note(a, 9.0, 2.0, "match\nensembles", color=INK, ha="right")

    lag = [1, 2, 5, 10, 20]
    ac = [0.62, 0.33, 0.07, 0.017, 0.005]
    b.axvspan(8, 21, color=FILL_G, lw=0)
    b.plot(lag, ac, "-o", color=INK, ms=4)
    b.set_xlim(0, 21)
    b.set_ylim(0, 0.7)
    b.set_xlabel("lag (snapshots)")
    b.set_ylabel("field autocorrelation")
    b.set_title("Short memory: anchor to climatology")
    note(b, 14.5, 0.42, "beyond ~8 steps:\nonly the stationary\nlaw matters", color=INK2)
    save(fig, "fig_e")


# ------------------------------------------------------------------ idea F
def fig_f():
    fig, a = plt.subplots(figsize=(7.2, 2.3))
    blank(a, (0, 14), (0, 4.2))
    a.set_title("Match the law of 6-frame windows on free-running chains")

    def strip(y, n, fc, ec, hi):
        for i in range(n):
            a.add_patch(FancyBboxPatch((0.5 + i * 0.62, y), 0.5, 0.75,
                                       boxstyle="round,pad=0.01,rounding_size=0.06",
                                       fc=fc, ec=ec, lw=0.8))
        a.add_patch(FancyBboxPatch((0.5 + hi * 0.62 - 0.07, y - 0.1), 6 * 0.62 - 0.01, 0.95,
                                   boxstyle="round,pad=0.01,rounding_size=0.08",
                                   fc="none", ec=INK, lw=1.4))

    strip(2.75, 12, FILL_B, BLUE_L, 6)
    strip(0.7, 12, FILL_G, AXIS, 3)
    note(a, 0.5, 3.85, "model chain, advanced one network call per training iteration", ha="left")
    note(a, 0.5, 0.35, "data trajectories", ha="left")
    box(a, 9.3, 1.55, 4.3, 1.2, "energy distance between\nwindow features\n(spectra, increments, patches)",
        fc=SURFACE, ec=AXIS)
    arrow(a, (7.95, 3.1), (9.25, 2.45))
    arrow(a, (6.1, 1.1), (9.25, 1.85))
    note(a, 11.45, 0.75, "5 history frames + 1 target:\nequal window laws imply an equal\nstationary process", color=INK2)
    save(fig, "fig_f")


# ------------------------------------------------------------------ idea G
def fig_g():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6),
                               gridspec_kw={"wspace": 0.75, "width_ratios": [1.3, 1]})
    a.bar([0], [4.3], width=0.86, color=MUTED)
    a.bar([0], [1.0], bottom=[4.33], width=0.86, color=ORANGE)
    a.bar([1], [4.45], width=0.86, color=MUTED)
    a.bar([1], [0.85], bottom=[4.48], width=0.86, color=BLUE)
    a.text(0, 2.15, "enstrophy of\nlast state $c$", ha="center", va="center", color="white", fontsize=7)
    a.text(0, 4.83, "forcing injection", ha="center", va="center", color="white", fontsize=7)
    a.text(1, 2.2, "enstrophy of the\nensemble mean", ha="center", va="center", color="white", fontsize=7)
    a.text(1, 4.9, "ensemble spread", ha="center", va="center", color="white", fontsize=7)
    a.set_xticks([0, 1], ["before the step", "after the step"])
    a.set_xlim(-0.5, 1.5)
    a.set_ylim(0, 6)
    a.grid(axis="x", visible=False)
    a.set_ylabel("dissipation-weighted enstrophy")
    a.set_title("The budget fixes the spread")
    note(a, 0.5, 5.75, "the two totals must be equal (illustration)", color=INK2)

    labels = ["enstrophy balance", "energy balance", "std of budget residual"]
    ratio = [1.991 / 2.0, 0.03783 / 0.038, 0.722 / 0.720]
    y = [2, 1, 0]
    b.axvline(1, color=MUTED, lw=1)
    b.plot(ratio, y, "o", color=INK, ms=6)
    for yi, v in zip(y, ratio):
        b.text(v, yi + 0.22, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)
    b.set_yticks(y, labels)
    b.set_xlim(0.98, 1.02)
    b.set_ylim(-0.6, 2.7)
    b.grid(axis="y", visible=False)
    b.set_xlabel("data archive / theory")
    b.set_title("The budgets close on the data")
    save(fig, "fig_g")


# ------------------------------------------------------------------ idea H
def fig_h():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.4),
                               gridspec_kw={"wspace": 0.42, "width_ratios": [1.2, 1]})
    blank(a, (0, 10), (0, 4.2))
    a.set_title("Build the invariant into the map and the noise")
    box(a, 0.3, 2.6, 2.6, 0.9, r"network $N_\theta$", fc=FILL_G)
    box(a, 3.35, 2.5, 3.9, 1.1, "remove the part that\nchanges the invariant", fc=FILL_B, ec=BLUE_L, fs=6.5)
    box(a, 0.3, 0.7, 2.6, 0.9, "noise driver", fc=FILL_G)
    box(a, 3.35, 0.6, 3.9, 1.1, "zero in the\ninvariant direction", fc=FILL_B, ec=BLUE_L, fs=6.5)
    box(a, 8.0, 1.65, 1.9, 0.9, r"map $\Phi_{s,t}$", fc=SURFACE)
    arrow(a, (2.95, 3.05), (3.3, 3.05))
    arrow(a, (2.95, 1.15), (3.3, 1.15))
    arrow(a, (7.3, 2.9), (8.1, 2.55))
    arrow(a, (7.3, 1.3), (8.1, 1.65))
    note(a, 5, 0.1, "exact for every step count and under composition", color=INK2)

    rng = np.random.default_rng(3)
    steps = np.arange(0, 401)
    walk = np.concatenate([[0], np.cumsum(rng.normal(0, 1, 400))]) * 1e-3
    b.plot(steps, walk, color=MUTED, lw=1.5)
    b.plot(steps, np.zeros_like(steps), color=BLUE)
    b.text(400, walk[-1], " unconstrained", va="center", fontsize=7, color=INK2)
    b.text(400, 0, " closed", va="center", fontsize=7, color=INK2)
    b.set_xlim(0, 520)
    b.set_xlabel("physical rollout step")
    b.set_ylabel("mean vorticity")
    b.set_title("No drift in rollouts (illustration)")
    save(fig, "fig_h")


# ------------------------------------------------------------------ idea I
def fig_i():
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 160)
    kern = lambda p, q: np.exp(-0.5 * (p[:, None] - q[None, :]) ** 2 / 0.09**2)
    chol = np.linalg.cholesky(kern(x, x) + 1e-6 * np.eye(len(x)))
    prior = (chol @ rng.normal(size=(len(x), 10))).T
    idx = np.array([30, 85, 130])
    obs = np.array([0.9, -0.6, 0.4])
    r = 0.03**2
    gain = kern(x, x[idx]) @ np.linalg.inv(kern(x[idx], x[idx]) + r * np.eye(3))
    post = prior + (gain @ (obs[:, None] + 0.03 * rng.normal(size=(3, 10)) - prior[:, idx].T)).T
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True,
                               gridspec_kw={"wspace": 0.08})
    for ax, ens, col, title in (
        (a, prior, MUTED, "Forecast ensemble"),
        (b, post, BLUE, "After shifting the noise paths to fit the data"),
    ):
        ax.plot(x, ens.T, color=col, lw=0.9, alpha=0.75)
        ax.plot(x[idx], obs, "o", color=ORANGE, ms=6, mec=SURFACE, mew=1.2, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("space")
        ax.set_xticks([])
    a.set_ylabel("field value")
    note(a, 0.02, 2.75, "orange dots: observations", color=INK2, ha="left")
    note(b, 0.5, -2.65, "closed-form Gaussian update of the driver,\nthen an exact accept/reject correction", color=INK2)
    a.set_ylim(-3.1, 3.1)
    save(fig, "fig_i")


# ------------------------------------------------------------------ idea J
def fig_j():
    fig, a = plt.subplots(figsize=(7.2, 2.3))
    blank(a, (0, 14), (0, 4.2))
    a.set_title("Feed the map the physical forcing it can know")
    box(a, 0.2, 2.5, 4.3, 1.1, "physical forcing path\n(8 Wiener coefficients per step)", fc=FILL_O, ec="#f3b79d", fs=7)
    box(a, 5.2, 2.5, 4.1, 1.1, r"a few logged coefficients $\psi$" + "\n(sampled exactly at inference)",
        fc=FILL_B, ec=BLUE_L, fs=7)
    box(a, 0.2, 0.6, 4.3, 1.0, "generative noise driver", fc=FILL_G)
    box(a, 5.2, 0.6, 4.1, 1.0, "models only what $\\psi$\ndoes not explain", fc=FILL_G)
    box(a, 10.2, 1.55, 3.4, 1.1, "Itô map\n" + r"$\to$ next state $y$", fc=SURFACE)
    arrow(a, (4.55, 3.05), (5.15, 3.05))
    arrow(a, (4.55, 1.1), (5.15, 1.1))
    arrow(a, (9.35, 2.9), (10.3, 2.6), color=BLUE)
    arrow(a, (9.35, 1.2), (10.3, 1.6))
    note(a, 11.9, 0.7, "assimilation becomes inference\nover a few dozen numbers", color=INK2)
    note(a, 11.9, 3.45, "needs the regenerated dataset\nwith logged forcing", color=INK2)
    save(fig, "fig_j")


if __name__ == "__main__":
    for f in (fig_diagnosis, fig_a, fig_b, fig_c, fig_d, fig_e, fig_f, fig_g, fig_h, fig_i, fig_j):
        f()
    print(sorted(p.name for p in OUT.glob("*.pdf")))
