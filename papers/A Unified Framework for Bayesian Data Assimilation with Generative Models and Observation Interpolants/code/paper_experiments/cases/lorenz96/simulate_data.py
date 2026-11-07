"""Simulate the training data of the Lorenz 96 priors.

Independent trajectories of the stochastic model are run past a burn-in and saved
with the standardisation constants of the training split:

    python cases/lorenz96/simulate_data.py --trajectories 4000 --steps 1000

Trajectories ``[0, num_train)`` are for training and the rest for validation. The
held-out assimilation sequences come from their own seed label (see ``driver.py``)
and are never drawn here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))

from cases.lorenz96._l96_system import load_problem  # noqa: E402
from common.paths import REPO_ROOT  # noqa: E402
from common.seeding import derive_seed  # noqa: E402

DATA_SEED_LABEL = "lorenz96-training-data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--out", default=str(REPO_ROOT / "data/lorenz96/data.npz"))
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    problem = load_problem()
    model = problem.model
    gen = torch.Generator().manual_seed(derive_seed(DATA_SEED_LABEL))
    start = time.time()
    x = model.spin_up(args.trajectories, args.burn_in, gen)
    states = np.empty((args.trajectories, args.steps, problem.dim_state), np.float32)
    for k in range(args.steps):
        states[:, k] = x.numpy()
        x = model.step(x, gen)
        if k % 100 == 0:
            print(f"step {k}/{args.steps} {time.time() - start:.0f}s", flush=True)

    num_train = int(round(args.trajectories * (1.0 - args.val_fraction)))
    train = states[:num_train].astype(np.float64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        states=states,
        mean=train.mean(),
        std=train.std(),
        num_train=num_train,
        problem=json.dumps(problem.to_dict()),
    )
    print(
        f"wrote {out}: {states.shape}, {states.nbytes / 1e9:.2f} GB, "
        f"mean {train.mean():.4f}, std {train.std():.4f}, {time.time() - start:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
