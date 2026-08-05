"""Generate image tiles for the sequential-DA schematic (manuscript Fig., methods).

Emits, for three consecutive assimilation steps of the sparse-1.5625% Navier-
Stokes case (Ours SI-SDE, shared Jacobian, M=250): the true vorticity field,
the sparse observations (sensor squares on a flat neutral background -- no
truth ghost, so the sparsity is unmistakable), and three maximally-different
posterior-ensemble members.  All fields share one symmetric RdBu_r scale,
matching the paper's state figures.

Output: manuscript/figures/sequential_da/*.png (300 dpi, borderless).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
NPZ = (ROOT / "results/navier_stokes/states/traj11/"
       "navier_stokes__Ours_SI_SDE__sparse_1_5625__shared_jac10__seed0__E64_M250.npz")
OUT = ROOT.parent / "manuscript/figures/sequential_da"
STEPS = (13, 14, 15)         # three consecutive assimilation steps
OBS_BG = "#EDEFF2"           # flat background for unobserved pixels

def save_tile(path, field=None, scatter=None, vmin=None, vmax=None, bg=None):
    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=300)
    H = W = 128
    if bg is not None:
        ax.set_facecolor(bg)
    if field is not None:
        ax.imshow(field, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                  extent=(0, W, 0, H))
    if scatter is not None:
        xs, ys, vals = scatter
        ax.scatter(xs, ys, c=vals, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                   s=14.0, linewidths=0.0, marker="s")
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", path)

def pick_members(ens_t, k=3):
    """Greedy max-pairwise-distance subset of ensemble members at one step."""
    E = ens_t.shape[0]
    flat = ens_t.reshape(E, -1)
    d = np.sqrt(((flat[:, None] - flat[None, :]) ** 2).mean(-1))
    picked = list(np.unravel_index(d.argmax(), d.shape))
    while len(picked) < k:
        picked.append(int(d[:, picked].min(1).argmax()))
    return picked

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(NPZ)
    truth = np.asarray(d["true_trajectory"])[0, 0]          # [H, W, T]
    ens = np.asarray(d["posterior_trajectory"])[:, 0]       # [E, H, W, T]
    obs = np.asarray(d["observations"])[0]                  # [Ny, T]
    idx = np.asarray(d["obs_indices"]).reshape(-1)
    ys, xs = np.divmod(idx, truth.shape[1])

    m = float(max(np.abs(truth[:, :, t]).max() for t in STEPS))
    print(f"steps={STEPS} sensors={idx.size} symmetric limit m={m:.3f}")
    members = pick_members(ens[:, :, :, STEPS[0]])
    print("members:", members)
    # Single initial state: the first shown member one step before the window.
    save_tile(OUT / "init.png", field=ens[members[0], :, :, STEPS[0] - 1],
              vmin=-m, vmax=m)
    for k, t in enumerate(STEPS):
        save_tile(OUT / f"true_{k}.png", field=truth[:, :, t], vmin=-m, vmax=m)
        save_tile(OUT / f"obs_{k}.png",
                  scatter=(xs.astype(float), ys.astype(float), obs[:, t]),
                  vmin=-m, vmax=m, bg=OBS_BG)
        for j, e in enumerate(members):
            save_tile(OUT / f"ens_{k}_{j}.png", field=ens[e, :, :, t],
                      vmin=-m, vmax=m)
        rms = float(np.sqrt(np.mean((ens[:, :, :, t].mean(0) - truth[:, :, t])**2)))
        pair = float(np.sqrt(np.mean((ens[members[0], :, :, t]
                                      - ens[members[1], :, :, t])**2)))
        print(f"  t={t}: ens-mean RMSE={rms:.3f} shown-pair RMS diff={pair:.3f}")

if __name__ == "__main__":
    main()
