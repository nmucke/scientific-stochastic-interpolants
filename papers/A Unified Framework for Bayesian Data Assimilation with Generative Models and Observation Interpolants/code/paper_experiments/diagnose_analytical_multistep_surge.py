"""Why SURGE around Ours behaves as it does: one cycle, exact Kalman input ensemble.

The grid gives the error of the SURGE rows; this gives the weights behind it. Each
cell starts from an exact sample of ``p(x^k | y^{1:k})``, so the input is the same
for every proposal, and runs one SURGE pass twice:

* with resampling off, reading the terminal log-weights: their ESS, and how they
  relate to the closed-form ``log p(y^{k+1} | x^k_i)``. The closed form is used here
  only, as the yardstick. A proposal that is the Doob transform of the prior would
  give a weight that IS that quantity (slope 1, no residual);
* with the rows' ``ess_threshold``, for the resampling count and the output KL.

Proposals: ``ours`` (the row's guidance), ``none`` (prior), ``g2`` (the same score
injected with ``g^2``, the Doob coefficient, in place of Ours' ``a + 1/2 g^2``).

    "$PY" "$EXPERIMENT_DIR/diagnose_analytical_multistep_surge.py"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from cases.analytical.multistep import MultistepRunner, kl_to_moments  # noqa: E402
from results_schema import Method  # noqa: E402
from scisi.posterior_models.surge_posterior import SurgePosteriorBase  # noqa: E402

CASES = {
    "paper": _here / "configs/case/analytical_multistep.yaml",
    "partial": _here / "configs/case/analytical_multistep_partial.yaml",
}
PROPOSALS = (
    (Method.OURS_SI_SDE, "inflated_shared", "ours", "SI shared"),
    (Method.OURS_SI_SDE, "dps_jacobian_free", "ours", "SI jacfree"),
    (Method.OURS_SI_SDE, "inflated_shared", "g2", "SI shared, g^2"),
    (Method.OURS_SI_SDE, None, "none", "SI prior"),
    (Method.OURS_DM_SDE, "inflated_shared", "ours", "DM shared"),
    (Method.OURS_DM_SDE, "dps_jacobian_free", "ours", "DM jacfree"),
    (Method.OURS_DM_SDE, "inflated_shared", "g2", "DM shared, g^2"),
    (Method.OURS_DM_SDE, None, "none", "DM prior"),
)
OUT = _here / "results" / "analytical_multistep" / "aggregated" / "surge_diagnostics.csv"


class Probe:
    """Records what the SMC layer does, through the two methods it passes through."""

    def __init__(self) -> None:
        self.resamples = 0
        self.log_w: torch.Tensor | None = None
        self.anchors: torch.Tensor | None = None
        self._post_sample = SurgePosteriorBase._post_sample
        self._maybe_resample = SurgePosteriorBase._maybe_resample
        self._injection = SurgePosteriorBase._injection_weight
        probe = self

        def post_sample(smc, base, observations, field_history):
            probe.log_w = smc._log_w.double().clone()
            probe.anchors = field_history[..., -1].reshape(base.shape[0], -1).double().clone()
            return probe._post_sample(smc, base, observations, field_history)

        def maybe_resample(smc, base, field_history, force=False):
            w = smc._normalized_weights().double()
            if not force and float(1 / (w**2).sum()) < smc.ess_threshold * base.shape[0]:
                probe.resamples += 1
            return probe._maybe_resample(smc, base, field_history, force)

        SurgePosteriorBase._post_sample = post_sample
        SurgePosteriorBase._maybe_resample = maybe_resample

    def inject(self, proposal: str) -> None:
        SurgePosteriorBase._injection_weight = (
            (lambda smc, t, g: g**2) if proposal == "g2" else self._injection
        )

    def reset(self) -> None:
        self.resamples, self.log_w, self.anchors = 0, None, None


def log_predictive(system, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    A, H, Q, R = (m.double() for m in (system.A, system.H, system.Q, system.R))
    residual = y.double() - x @ A.T @ H.T
    solved = torch.linalg.solve(H @ Q @ H.T + R, residual.T).T
    return -0.5 * (residual * solved).sum(1)


def ess(log_w: torch.Tensor) -> float:
    w = torch.softmax(log_w, 0)
    return float(1 / (w**2).sum()) / log_w.numel()


def run_cell(runner, probe, name, d, k, seed, n, M) -> list[dict[str, object]]:
    system = runner.system()
    _, cycles = runner.trajectory(seed, n)
    previous, cycle = cycles[k - 2], cycles[k - 1]
    generator = torch.Generator().manual_seed(1000 * seed + k)
    chol = torch.linalg.cholesky(previous.kalman_cov.double())
    noise = torch.randn(n, system.d, generator=generator, dtype=torch.float64)
    x = (previous.kalman_mean.double() + noise @ chol.T).float()
    ideal = log_predictive(system, x.double(), cycle.y)
    rows = [{
        "system": name, "d": d, "k": k, "seed": seed, "proposal": "closed form",
        "terminal_ess": ess(ideal), "sd_log_w": float(ideal.std()), "corr": 1.0,
        "slope": 1.0, "residual_sd": 0.0, "kl_no_resampling": "", "kl": "",
        "resamples": "", "anchors_kept": "",
    }]
    threshold = runner._surge_ess_threshold(Method.SURGE_SDA, M)
    for method, mode, proposal, label in PROPOSALS:
        probe.inject(proposal)
        row: dict[str, object] = {
            "system": name, "d": d, "k": k, "seed": seed, "proposal": label,
        }
        for th in (0.0, threshold):
            runner._surge_ess_threshold = lambda *_, _th=th: _th
            probe.reset()
            torch.manual_seed(1000 * seed + k)
            out = runner._ours_cycle(method, system, x, cycle.y, mode, M, surge=True)
            kl = kl_to_moments(
                cycle.kalman_mean.numpy(), cycle.kalman_cov.numpy(), out.numpy()
            )
            if th == 0.0:
                target = log_predictive(system, probe.anchors, cycle.y).numpy()
                log_w = probe.log_w.numpy()
                slope = float(np.polyfit(target, log_w, 1)[0])
                row.update(
                    terminal_ess=ess(probe.log_w), sd_log_w=float(log_w.std()),
                    corr=float(np.corrcoef(log_w, target)[0, 1]), slope=slope,
                    residual_sd=float((log_w - slope * target).std()),
                    kl_no_resampling=kl,
                )
            else:
                kept = torch.unique(probe.anchors, dim=0).shape[0] / n
                row.update(kl=kl, resamples=probe.resamples, anchors_kept=kept)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--cycles", type=int, nargs="+", default=[4, 7])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dims", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--dim-n", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    def runner_for(path: Path, n: int, d: int | None = None) -> MultistepRunner:
        case = OmegaConf.to_container(OmegaConf.load(path))
        case.update(num_cycles=max(args.cycles), n_eval_samples=n)
        if d is not None:
            case["plot_dimension"] = d
        config = {"case": case, "ensemble_size": 64, "num_steps": args.M}
        return MultistepRunner(OmegaConf.create(config), seeds=(0,))

    probe = Probe()
    rows: list[dict[str, object]] = []
    cells = [(name, path, None, args.n) for name, path in CASES.items()]
    cells += [("paper", CASES["paper"], d, args.dim_n) for d in args.dims]
    for name, path, d, n in cells:
        for seed in args.seeds:
            for k in args.cycles:
                runner = runner_for(path, n, d)
                rows += run_cell(runner, probe, name, d or runner.system().d, k, seed, n, args.M)
                print(f"[diag] {name} d={d or 2} seed={seed} k={k}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[1]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
