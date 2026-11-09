"""Nonlinear-observation harness for the one-step Gaussian (analytical) case.

    x1 = x0 + w,  w ~ N(0, prior_var I);   y = g(x1) + e,  e ~ N(0, R I)

with ``g`` elementwise (``PointwiseObservationOperator`` with ``L = I``). The exact
posterior is the per-coordinate quadrature of ``nonlinear_reference.py``; metrics are
``nonlinear_metrics.py``. Every method is routed through the SAME builders as the
linear case (``driver.draw_interpolant_posterior`` / ``AnalyticalRunner.
_sample_and_count``) with the nonlinear operator passed in, so the only things that
change between the linear table and this sweep are ``h``, ``R`` and the
R-dependent baseline hyperparameters (``tuned_hparams`` in the case YAML, selected
on the identity cell by ``nonlinear_tuning.py``).

Each (cell, d, seed, method[, variant]) draw runs in ``try/except``; a failure is
logged (type, message, innermost frames) to ``failures_<tag>.json`` and the sweep
continues. Every attempt also emits a per-seed ``failed`` row (1 = raised or
returned non-finite samples); the aggregated table carries ``n_failed`` /
``n_seeds`` per (cell, method, variant), so a mean over surviving seeds is never
silent.

Run (whole sweep from configs/case/analytical_nonlinear.yaml):

    uv run python <PE>/run.py case=analytical_nonlinear \\
        'results_file=${results_root}/analytical_nonlinear/results.csv'

or the standalone CLI with filters (run from <PE>; writes results/analytical_nonlinear/):

    uv run python -m cases.analytical.nonlinear_driver \\
        --cells arctan_c1_ --dims 2 --seeds 0 1 --methods "Ours (SI-SDE)" EnKF

Truth / observation for a seed are drawn exactly as in the linear driver
(``derive_seed("analytical", "truth", seed)``; ``x0 = x0_offset + x0_spread * z``),
and the identity cell's ``kl_points`` / ``sliced_w2`` use the linear driver's own
Gaussian reference draw (``derive_seed("analytical", "ref", seed)``). With
``x0_offset=0, x0_spread=1, R=1, d=2`` the identity cell therefore sees the linear
case's x0, y and reference exactly; the samplers see the identity
``PointwiseObservationOperator`` instead of the bare linear one, which agrees with
it only up to float round-off, so their draws match the linear table in
distribution, not bit-for-bit.

``ResultRecord.E`` is the nominal config ensemble size (64, as in the linear
table); every method actually draws ``n_eval`` (4096) samples per (cell, seed)
so the metrics' Monte-Carlo floors are small (see ``nonlinear_metrics``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import traceback
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

_PE = Path(__file__).resolve().parents[2]
if str(_PE) not in sys.path:  # allow `python -m` / direct execution
    sys.path.insert(0, str(_PE))

from common.aggregation import aggregate_over_seeds  # noqa: E402
from common.seeding import derive_seed, seed_everything  # noqa: E402
from results_schema import (  # noqa: E402
    SEED_AGGREGATED,
    Case,
    Method,
    ResultRecord,
    ResultsWriter,
)

from scisi.likelihood_models.observation_operators import (  # noqa: E402
    PointwiseObservationOperator,
)

from .classical_baselines import GaussianSystem  # noqa: E402
from .driver import (  # noqa: E402
    _OURS_SAMPLER,
    _OURS_VARIANTS,
    AnalyticalRunner,
    _kl_w2_to_exact,
    draw_interpolant_posterior,
)
from .nonlinear_metrics import NON_INJECTIVE, nonlinear_metrics  # noqa: E402
from .nonlinear_reference import PointwiseNonlinearPosterior  # noqa: E402
from .prior_models import build_identity_obs_operator  # noqa: E402

# Full nonlinear lineup: the linear lineup + the baselines dropped from it (OT-ODE,
# standalone SURGE, DPS). Ensemble score filter is not part of this case.
NONLINEAR_METHODS: tuple[Method, ...] = (
    Method.OURS_SI_SDE,
    Method.OURS_DM_SDE,
    Method.OURS_FM_ODE,
    Method.FLOWDAS,
    Method.SURGE_FLOWDAS,
    Method.SDA,
    Method.SURGE_SDA,
    Method.SURGE,
    Method.D_FLOW_SGLD,
    Method.GUIDED_FM_FIG,
    Method.GUIDED_FM_OTODE,
    Method.GUIDED_DIFFUSION,  # DPS (chung_diffusion_2023)
    Method.ENKF,
    Method.PARTICLE_FILTER,
)
# Ours: the linear case's jacfree + shared rows, plus the per-member linearisation
# (it matters once h is nonlinear; in the linear case it equals shared).
OURS_VARIANTS: tuple[tuple[str, str], ...] = _OURS_VARIANTS + (("individual", "inflated"),)
EXACT_REFERENCE = "Exact reference"  # pseudo-method: exact samples -> metric floors
DEFAULT_CONFIG = _PE / "configs" / "case" / "analytical_nonlinear.yaml"
SUMMARY_METRICS = ("mean_err", "var_err", "marginal_w2", "mode_weight_err", "kl_marginal_knn")


@dataclass(frozen=True)
class NonlinearCell:
    nonlinearity: str
    strength: float
    obs_variance: float
    x0_offset: float
    d: int
    x0_spread: float = 1.0
    prior_variance: float = 1.0

    @property
    def tag(self) -> str:
        c = f"_c{self.strength:g}" if self.nonlinearity not in ("identity", "abs") else ""
        return f"nl:{self.nonlinearity}{c}_R{self.obs_variance:g}_x0{self.x0_offset:g}_d{self.d}"

    @property
    def family(self) -> str:
        """Cell family for the summary (g and its strength; R, x0, d pooled)."""
        c = f"_c{self.strength:g}" if self.nonlinearity in ("arctan",) else ""
        return f"{self.nonlinearity}{c}"

    @property
    def split(self) -> Optional[float]:
        return 0.0 if self.nonlinearity in NON_INJECTIVE else None

    def build_operator(self) -> torch.nn.Module:
        return PointwiseObservationOperator(
            inner=build_identity_obs_operator(d=self.d),
            nonlinearity=self.nonlinearity, strength=self.strength,
        )


def r_key(obs_variance: float) -> str:
    """Key of the ``tuned_hparams`` table for an observation variance R."""
    return f"R{obs_variance:g}"


def expand_cells(spec: Sequence[dict], dims: Sequence[int], defaults: dict) -> list[NonlinearCell]:
    """Expand ``cells`` entries (list-valued fields -> Cartesian product) x ``dims``."""
    cells: list[NonlinearCell] = []
    for entry in spec:
        entry = {**defaults, **dict(entry)}
        entry_dims = entry.pop("dims", None) or list(dims)
        keys = list(entry)
        vals = [v if isinstance(v, (list, tuple)) else [v] for v in (entry[k] for k in keys)]
        for combo in itertools.product(*vals):
            kw = dict(zip(keys, combo))
            for d in entry_dims:
                cells.append(NonlinearCell(
                    nonlinearity=str(kw["nonlinearity"]),
                    strength=float(kw.get("strength", 1.0)),
                    obs_variance=float(kw["obs_variance"]),
                    x0_offset=float(kw.get("x0_offset", 0.0)),
                    d=int(d),
                    x0_spread=float(kw.get("x0_spread", 1.0)),
                    prior_variance=float(kw.get("prior_variance", 1.0)),
                ))
    tags = [c.tag for c in cells]
    dup = {t for t in tags if tags.count(t) > 1}
    if dup:
        raise ValueError(f"duplicate cell tags (same g/R/x0_offset/d): {sorted(dup)}")
    return cells


def _to_plain(obj: Any) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(obj, resolve=True) if OmegaConf.is_config(obj) else obj


class NonlinearAnalyticalHarness:
    """Sweep runner: cells x seeds x methods, failures recorded not raised.

    ``tuned_hparams`` = ``{r_key(R): {method value: {key: value}}}`` (from the case
    YAML) and ``overrides`` = ``{Method: {key: value}}`` (used by the tuning
    script; wins over ``tuned_hparams``) are passed to the baseline builders as
    hyperparameter overrides of the method YAML.
    """

    def __init__(
        self,
        cells: Sequence[NonlinearCell],
        *,
        seeds: Sequence[int],
        methods: Sequence[Method] = NONLINEAR_METHODS,
        ensemble_size: int = 64,
        num_steps: int = 50,
        n_eval: int = 4096,
        knn_k: int = 5,
        output_dir: Optional[Path] = None,
        base_config: Any = None,
        tuned_hparams: Optional[Mapping] = None,
        overrides: Optional[Mapping[Method, Mapping]] = None,
        include_reference: bool = True,
        verbose: bool = True,
    ) -> None:
        self.cells = list(cells)
        self.seeds = list(seeds)
        self.methods = list(methods)
        self.E, self.M = int(ensemble_size), int(num_steps)
        self.n_eval, self.knn_k = int(n_eval), int(knn_k)
        self.output_dir = Path(output_dir) if output_dir is not None else _PE / "results" / "analytical_nonlinear"
        self.tuned = {str(k): dict(v or {}) for k, v in (tuned_hparams or {}).items()}
        self.overrides = {Method(k): dict(v) for k, v in (overrides or {}).items()}
        self.include_reference = include_reference
        self.verbose = verbose
        # Reuse the linear runner for method builders + YAML hyperparameters.
        self.base = AnalyticalRunner(base_config if base_config is not None else {}, seeds=self.seeds)
        self.failures: list[dict] = []
        self.tag = time.strftime("%Y%m%d-%H%M%S")
        self._warned: set = set()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        seeds: Sequence[int],
        cell_filter: Optional[Sequence[str]] = None,
        dims: Optional[Sequence[int]] = None,
        methods: Optional[Sequence[str]] = None,
        **kw: Any,
    ) -> "NonlinearAnalyticalHarness":
        """Build from a run config whose ``case`` is analytical_nonlinear.yaml.

        ``cell_filter`` keeps cells whose tag contains any of the substrings,
        ``dims`` overrides ``case.nonlinear.dims``, ``methods`` the lineup.
        """
        nl = _to_plain(cfg.case.nonlinear)
        cells = expand_cells(nl["cells"], dims or nl.get("dims", [2]), dict(nl.get("cell_defaults") or {}))
        if dims:
            cells = [c for c in cells if c.d in dims]
        if cell_filter:
            cells = [c for c in cells if any(f in c.tag for f in cell_filter)]
        names = methods or nl.get("methods")
        kw.setdefault("output_dir", nl.get("output_dir"))
        return cls(
            cells, seeds=seeds,
            methods=[Method(m) for m in names] if names else list(NONLINEAR_METHODS),
            ensemble_size=int(cfg.get("ensemble_size", 64)),
            num_steps=int(cfg.get("num_steps", 50)),
            n_eval=int(nl.get("n_eval_samples", 4096)),
            knn_k=int(nl.get("knn_k", 5)),
            base_config=cfg,
            tuned_hparams=nl.get("tuned_hparams"),
            **kw,
        )

    # ------------------------------------------------------------------ #
    def _problem(self, cell: NonlinearCell, seed: int):
        """Truth, observation and exact posterior; identical across methods."""
        d = cell.d
        gx = torch.Generator().manual_seed(derive_seed("analytical", "truth", seed))
        x0 = cell.x0_offset + cell.x0_spread * torch.randn(d, generator=gx)
        x1 = x0 + (cell.prior_variance**0.5) * torch.randn(d, generator=gx)
        op = cell.build_operator()
        with torch.no_grad():
            gx1 = op(x1.reshape(1, 1, 1, d)).reshape(d)
        y = gx1 + (cell.obs_variance**0.5) * torch.randn(d, generator=gx)
        sys_ = GaussianSystem(d=d, obs_var=cell.obs_variance, prior_var=cell.prior_variance)
        post = PointwiseNonlinearPosterior(
            x0.double(), y.double(), prior_var=cell.prior_variance,
            obs_var=cell.obs_variance, nonlinearity=cell.nonlinearity,
            strength=cell.strength,
        )
        rng = np.random.default_rng(derive_seed("analytical_nl", "ref:" + cell.tag, seed))
        ref = post.sample(self.n_eval, rng=rng)
        ref_gauss = None
        if cell.nonlinearity == "identity":
            # The linear driver's reference draw (same generator) for the legacy
            # Gaussian-KL / sliced rows of the identity control.
            gr = torch.Generator().manual_seed(derive_seed("analytical", "ref", seed))
            ref_gauss = sys_.exact_posterior_samples(x0, y, self.n_eval, gr).numpy()
        return sys_, op, x0, y, post, ref, ref_gauss

    def _metrics(self, cell, post, ref, ref_gauss, samp: np.ndarray) -> dict[str, float]:
        out = nonlinear_metrics(post, ref, samp, k=self.knn_k, split=cell.split)
        if ref_gauss is not None:  # identity control: the linear table's legacy metrics
            kl, w2 = _kl_w2_to_exact(ref_gauss, samp)
            out = {"kl_points": kl, "sliced_w2": w2, **out}
        return out

    def _record(self, cell, method_name, seed, metric, value, nfe, seconds, variant) -> ResultRecord:
        v = float(value)
        if not math.isfinite(v):  # inf (collapsed / atomic q) -> NaN; aggregation is NaN-safe only
            v = float("nan")
        return ResultRecord(
            case=Case.ANALYTICAL.value, method=method_name, scenario=cell.tag,
            metric=metric, value=v, E=self.E, M=self.M, seed=seed,
            nfe=float(nfe), seconds=float(seconds), variant=variant,
        )

    def _fail(self, cell, method_name, seed, variant, exc: BaseException) -> None:
        tb = traceback.extract_tb(exc.__traceback__)
        frames = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in tb[-4:]]
        self.failures.append(dict(
            cell=cell.tag, method=method_name, variant=variant, seed=seed,
            error=type(exc).__name__, message=str(exc)[:500], frames=frames,
        ))
        if self.verbose:
            print(f"  FAIL {cell.tag} {method_name} {variant or ''} seed={seed}: "
                  f"{type(exc).__name__}: {str(exc)[:160]}  @ {frames[-1] if frames else '?'}",
                  flush=True)

    def method_hparams(self, method: Method, cell: NonlinearCell) -> dict:
        """Hyperparameter overrides for a baseline at this cell's R."""
        if method in self.overrides:
            return dict(self.overrides[method])
        if not self.tuned:
            return {}
        table = self.tuned.get(r_key(cell.obs_variance))
        if table is None:
            if r_key(cell.obs_variance) not in self._warned:
                self._warned.add(r_key(cell.obs_variance))
                warnings.warn(f"no tuned_hparams for {r_key(cell.obs_variance)}; "
                              "baselines use their method-YAML defaults", stacklevel=2)
            return {}
        return dict(table.get(method.value) or {})

    def _draws(self, method: Method, cell, sys_, op, x0, y, seed):
        """Yield (variant, thunk -> (samples[n, d] tensor, nfe)) for one method."""
        n, M = self.n_eval, self.M
        if method in _OURS_SAMPLER:
            hp = self.base._ours_hparams(method, M)
            for variant, mode in OURS_VARIANTS:
                s = derive_seed("analytical", f"{method.value}:{variant}", seed)
                yield variant, (lambda mode=mode, s=s: (draw_interpolant_posterior(
                    sys_, x0, y, sampler=_OURS_SAMPLER[method], likelihood_mode=mode,
                    ensemble_size=n, num_steps=M, seed=s, obs_op=op, **hp,
                ), M))
        else:
            gm = torch.Generator().manual_seed(derive_seed("analytical", method.value, seed))
            hp = self.method_hparams(method, cell)
            yield None, (lambda: self.base._sample_and_count(
                method, sys_, x0, y, n, M, gm, obs_op=op, hparams=hp))

    # ------------------------------------------------------------------ #
    def run(self, *, aggregate: bool = True, write: bool = True) -> list[ResultRecord]:
        raw: list[ResultRecord] = []
        for cell in self.cells:
            if self.verbose:
                print(f"[{cell.tag}]", flush=True)
            for seed in self.seeds:
                seed_everything(seed)
                try:
                    sys_, op, x0, y, post, ref, ref_gauss = self._problem(cell, seed)
                except Exception as exc:  # reference must not fail; record if it does
                    self._fail(cell, "<problem>", seed, None, exc)
                    continue
                if self.include_reference:
                    q = post.sample(self.n_eval, rng=np.random.default_rng(
                        derive_seed("analytical_nl", "floor:" + cell.tag, seed)))
                    m = self._metrics(cell, post, ref, ref_gauss, q)
                    raw.extend(self._record(cell, EXACT_REFERENCE, seed, k, v, 0, 0.0, None)
                               for k, v in {**m, "failed": 0.0}.items())
                for method in self.methods:
                    # Re-seed the global RNG per method exactly as ExperimentRunner.run()
                    # does, so baselines drawing from it reproduce the linear table.
                    seed_everything(seed)
                    for variant, thunk in self._draws(method, cell, sys_, op, x0, y, seed):
                        t0 = time.perf_counter()
                        try:
                            samp, nfe = thunk()
                            samp = samp.detach().cpu().double().numpy().reshape(self.n_eval, cell.d)
                            secs = time.perf_counter() - t0
                            bad = int((~np.isfinite(samp).all(1)).sum())
                            if bad:  # metrics become NaN; counted as a failed seed
                                self._fail(cell, method.value, seed, variant, FloatingPointError(
                                    f"non-finite samples: {bad}/{samp.shape[0]} rows"))
                            m = self._metrics(cell, post, ref, ref_gauss, samp)
                        except Exception as exc:
                            self._fail(cell, method.value, seed, variant, exc)
                            raw.append(self._record(cell, method.value, seed, "failed", 1.0,
                                                    0, time.perf_counter() - t0, variant))
                            continue
                        raw.extend(self._record(cell, method.value, seed, k, v, nfe, secs, variant)
                                   for k, v in {**m, "failed": float(bad > 0)}.items())
        if write:
            self.write_outputs(raw)
        return aggregate_records(raw) if aggregate else raw

    def write_outputs(self, raw: list[ResultRecord]) -> None:
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        agg = aggregate_records(raw)
        with ResultsWriter(out / f"raw_{self.tag}.csv") as w:
            w.extend(raw)
        with ResultsWriter(out / f"results_{self.tag}.csv") as w:
            w.extend(agg)
        meta = dict(
            tag=self.tag, cells=[asdict(c) | {"tag": c.tag} for c in self.cells],
            seeds=self.seeds, methods=[m.value for m in self.methods],
            E=self.E, M=self.M, n_eval=self.n_eval, knn_k=self.knn_k,
            tuned_hparams=self.tuned, overrides={k.value: v for k, v in self.overrides.items()},
            n_failures=len(self.failures),
        )
        (out / f"failures_{self.tag}.json").write_text(json.dumps(
            dict(meta=meta, failures=self.failures), indent=1))
        (out / f"summary_{self.tag}.md").write_text(summarize(agg, self.cells))
        if self.verbose:
            print(f"wrote {out}/{{raw,results,failures,summary}}_{self.tag}.*  "
                  f"({len(raw)} rows, {len(self.failures)} failures)", flush=True)


def aggregate_records(raw: list[ResultRecord]) -> list[ResultRecord]:
    """Mean +/- std over seeds, with the per-seed ``failed`` flags turned into
    ``n_failed`` (count) and ``n_seeds`` (attempts) rows per (cell, method, variant).
    The accuracy metrics are means over the seeds that produced finite samples."""
    agg = aggregate_over_seeds([r for r in raw if r.metric != "failed"])
    counts: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for r in raw:
        if r.metric == "failed":
            c = counts[(r.case, r.method, r.scenario, r.E, r.M, r.variant)]
            c[0] += 1
            c[1] += int(r.value > 0)
    for (case, method, scen, E, M, variant), (n, f) in counts.items():
        for metric, val in (("n_failed", f), ("n_seeds", n)):
            agg.append(ResultRecord(case=case, method=method, scenario=scen, metric=metric,
                                    value=float(val), std=0.0, E=E, M=M, seed=SEED_AGGREGATED,
                                    variant=variant))
    return agg


# ---------------------------------------------------------------------- #
def summarize(records: list[ResultRecord], cells: Sequence[NonlinearCell],
              metrics: Sequence[str] = SUMMARY_METRICS) -> str:
    """Markdown: one table per cell family; rows = method[/variant], columns = the
    seed-mean metrics averaged over the family's cells, plus failed / attempted
    seeds summed over them."""
    fam = {c.tag: c.family for c in cells}
    acc: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.seed != SEED_AGGREGATED or r.scenario not in fam:
            continue
        acc[(fam[r.scenario], r.method, r.variant or "")][r.metric].append(r.value)
    lines: list[str] = []
    for family in dict.fromkeys(c.family for c in cells):
        rows = [(k, v) for k, v in acc.items() if k[0] == family]
        cols = [m for m in metrics if any(v.get(m) for _, v in rows)]
        lines += [f"### {family}", "",
                  "| method | " + " | ".join(cols) + " | failed |",
                  "|---|" + "---:|" * (len(cols) + 1)]
        for (_, meth, var), v in rows:
            name = f"{meth} / {var}" if var else meth
            vals = []
            for m in cols:
                xs = [x for x in v.get(m, []) if x == x]
                vals.append(f"{np.mean(xs):.3g}" if xs else "nan")
            nf, ns = int(sum(v.get("n_failed", []))), int(sum(v.get("n_seeds", [])))
            lines.append(f"| {name} | " + " | ".join(vals) + f" | {nf}/{ns} |")
        lines.append("")
    return "\n".join(lines)


def load_config(path: Path | str = DEFAULT_CONFIG, **top: Any):
    """Run config ``{case: <yaml>, **top}`` with the ``experiment_root`` resolver."""
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("experiment_root"):
        OmegaConf.register_new_resolver("experiment_root", lambda: str(_PE))
    return OmegaConf.create({"case": OmegaConf.load(path), **top})


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--cells", nargs="*", default=None, help="substring filters on cell tags")
    p.add_argument("--dims", nargs="*", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--methods", nargs="*", default=None, help="Method values, e.g. 'Ours (SI-SDE)' EnKF")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--ensemble-size", type=int, default=64)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--results-file", default=None,
                   help="also write the aggregated table here (+ summary.md next to it)")
    args = p.parse_args(argv)

    torch.set_default_dtype(torch.float32)  # same as run.py
    cfg = load_config(args.config, num_steps=args.num_steps, ensemble_size=args.ensemble_size)
    kw = {"output_dir": Path(args.output_dir)} if args.output_dir else {}
    h = NonlinearAnalyticalHarness.from_config(
        cfg, seeds=args.seeds, cell_filter=args.cells, dims=args.dims, methods=args.methods, **kw)
    agg = h.run(aggregate=True)
    summary = summarize(agg, h.cells)
    print(summary)
    if args.results_file:
        path = Path(args.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with ResultsWriter(path) as w:
            w.extend(agg)
        (path.parent / "summary.md").write_text(summary)
    if h.failures:
        print("\nFailures (method, variant, error, message, frame) -> count:")
        cnt: dict[tuple, int] = defaultdict(int)
        for f in h.failures:
            cnt[(f["method"], f["variant"], f["error"], f["message"][:100],
                 f["frames"][-1] if f["frames"] else "")] += 1
        for k, v in cnt.items():
            print(f"  {v:3d}x {k}")


if __name__ == "__main__":
    main()


__all__ = [
    "NONLINEAR_METHODS",
    "NonlinearAnalyticalHarness",
    "NonlinearCell",
    "aggregate_records",
    "expand_cells",
    "load_config",
    "r_key",
    "summarize",
]
