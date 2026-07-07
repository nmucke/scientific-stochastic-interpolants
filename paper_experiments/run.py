"""Hydra entrypoint that runs one case driver and writes a tidy results file.

Mirrors ``paper/scripts/generate_posterior_samples.py`` conventions: a Hydra
config composed from ``paper_experiments/configs/benchmark.yaml``, swept at the
CLI. The case driver (an :class:`ExperimentRunner`) does the science; this script
only wires config -> driver -> tidy file -> (optionally) LaTeX snippets.

    python paper_experiments/run.py case=navier_stokes method=si_sde scenario=superres_32
    python paper_experiments/run.py --multirun method=si_sde,dm_sde,fm_ode

NOTE: the case drivers raise ``NotImplementedError`` until the unified-sampler
rebuild lands (GAP_ANALYSIS Phases 0--3). The schema, aggregation, and table
emitter (``make_tables.py``) are fully working today -- see ``--demo`` there.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Make paper_experiments/ importable when run as a script.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from common.seeding import SEED_LIST  # noqa: E402

logger = logging.getLogger(__name__)

# case name -> ExperimentRunner subclass (lazy import to avoid loading torch when
# only the table pipeline is exercised).
_RUNNERS = {
    "analytical": ("cases.analytical.driver", "AnalyticalRunner"),
    "navier_stokes": ("cases.navier_stokes.driver", "NavierStokesRunner"),
    "urban": ("cases.urban.driver", "UrbanRunner"),
}


def _resolve_runner(case_name: str):
    try:
        module_name, cls_name = _RUNNERS[case_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown case '{case_name}'. Known: {sorted(_RUNNERS)}"
        ) from exc
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


@hydra.main(  # type: ignore[misc]
    config_path="configs",
    config_name="benchmark",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logger.info(f"Case    : {cfg.case.name}")
    logger.info(f"Method  : {cfg.method.name}")
    logger.info(f"Scenario: {cfg.get('scenario', {}).get('name', 'n/a')}")

    # All cases run in single precision (fp32). Set the default BEFORE any model /
    # tensor is built so every case (analytical closed-form, NS/urban samplers) is
    # fp32 regardless of a stray default or a checkpoint's stored dtype. Matches the
    # training scripts (src/scisi/bin/*). Metric internals may still upcast to fp64
    # locally for numerical accuracy (src/scisi/metrics/*) -- that is deliberate and
    # unaffected. Torch is imported here (not at module top) so the table-only
    # pipeline never pays the torch import.
    import torch

    torch.set_default_dtype(torch.float32)

    seeds = list(cfg.get("seeds", SEED_LIST))
    runner_cls = _resolve_runner(cfg.case.name)
    runner = runner_cls(cfg, seeds=seeds)

    # Ablation mode (spec Section 7): `ablation=true` drives the dedicated
    # `run_ablation` sweep instead of the standard `run()` metrics loop, so the
    # tab:ablation rows are actually produced (run() only calls evaluate()).
    if bool(cfg.get("ablation", False)):
        run_ablation = getattr(runner, "run_ablation", None)
        if run_ablation is None:
            raise RuntimeError(
                f"Case '{cfg.case.name}' has no run_ablation entrypoint."
            )
        logger.info("Ablation mode: driving run_ablation()")
        records = run_ablation(aggregate=True)
    else:
        records = runner.run(aggregate=True)
    # Resolve the (repo-relative) results path against the original launch dir so
    # Hydra's per-run chdir does not nest it under the run's output directory.
    out_path = Path(cfg.results_file)
    if not out_path.is_absolute():
        try:
            from hydra.utils import get_original_cwd

            out_path = Path(get_original_cwd()) / out_path
        except Exception:  # pragma: no cover - hydra not active / no original cwd
            pass
    runner.write(records, out_path)
    logger.info(f"Wrote {len(records)} aggregated rows -> {out_path}")
    logger.info(
        "Now emit LaTeX: python paper_experiments/make_tables.py "
        f"--results {out_path} --out paper_experiments/generated"
    )


if __name__ == "__main__":
    main()
