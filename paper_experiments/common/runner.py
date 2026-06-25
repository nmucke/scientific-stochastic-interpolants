"""Shared experiment-runner interface.

Every case driver (``cases/<case>/driver.py``) implements :class:`ExperimentRunner`.
The base class fixes the contract -- *config in, tidy :class:`ResultRecord`s out*
-- and owns the seed loop + across-seed aggregation so all three cases reproduce
identically (Section 9). Subclasses implement only the scientific per-(method,
scenario, seed) work in :meth:`evaluate`.

Dependency note
---------------
The sampler/metric internals live in ``src/scisi`` and are mid-rebuild
(GAP_ANALYSIS Phases 0--3). This base class therefore deliberately imports
**nothing** from ``scisi`` -- it is the stable seam the case drivers plug the
rebuilt samplers/metrics into. See the per-case ``driver.py`` for the TODO seams.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from common.aggregation import aggregate_over_seeds
from common.seeding import SEED_LIST, seed_everything
from results_schema import (
    Case,
    Method,
    ResultRecord,
    ResultsWriter,
    Scenario,
)


@dataclass
class RunContext:
    """Everything one (method, scenario, seed) evaluation needs.

    A case driver may subclass this to carry case-specific handles (loaded prior
    model, dataset, observation operator, ...). Kept config-agnostic here so the
    base loop never needs to know the case internals.
    """

    case: Case
    method: Method
    scenario: Scenario
    seed: int
    ensemble_size: int  # E
    num_steps: int  # M
    # Free-form: the hydra DictConfig, loaded model, dataset, etc.
    extra: dict[str, object] = field(default_factory=dict)


class ExperimentRunner(ABC):
    """Base experiment runner.

    Typical use by a case driver::

        runner = NavierStokesRunner(cfg)
        records = runner.run()                     # loops methods x scenarios x seeds
        runner.write(records, "results/ns.csv")    # aggregated tidy file

    Subclasses provide:
    * :attr:`case`               -- which :class:`Case`.
    * :meth:`methods`            -- methods to evaluate (defaults to cfg list).
    * :meth:`scenarios`          -- scenarios to evaluate.
    * :meth:`evaluate`           -- the per-(method, scenario, seed) metrics.
    """

    #: Subclasses set this to the case they implement.
    case: Case

    def __init__(self, config: object, *, seeds: Sequence[int] = SEED_LIST) -> None:
        self.config = config
        self.seeds = tuple(seeds)

    # -- subclass hooks ---------------------------------------------------- #

    @abstractmethod
    def methods(self) -> Sequence[Method]:
        """Return the methods to evaluate (from config)."""

    @abstractmethod
    def scenarios(self) -> Sequence[Scenario]:
        """Return the observation scenarios to evaluate (from config)."""

    @abstractmethod
    def evaluate(self, ctx: RunContext) -> Iterable[ResultRecord]:
        """Run one (method, scenario, seed) and yield *per-seed* tidy records.

        Implementations must:
        * use ``ctx.seed`` to draw the truth + observations + sensor mask so
          they are identical across methods (use ``common.seeding`` helpers);
        * set ``seed=ctx.seed`` (not aggregated) on every emitted record;
        * fill ``E``/``M`` and the cost columns (``nfe``, ``seconds``).

        The base class aggregates these per-seed rows into mean +/- std.
        """

    def make_context(
        self,
        method: Method,
        scenario: Scenario,
        seed: int,
    ) -> RunContext:
        """Build the per-run context; override to attach case-specific handles."""
        return RunContext(
            case=self.case,
            method=method,
            scenario=scenario,
            seed=seed,
            ensemble_size=int(self._cfg_get("ensemble_size", 64)),
            num_steps=int(self._cfg_get("num_steps", 50)),
        )

    # -- driving loop ------------------------------------------------------ #

    def run(self, *, aggregate: bool = True) -> list[ResultRecord]:
        """Loop methods x scenarios x seeds and collect tidy records.

        With ``aggregate=True`` (default) the per-seed rows are reduced to mean
        +/- std rows ready for ``make_tables.py``; with ``aggregate=False`` the
        raw per-seed rows are returned (useful for debugging / custom reductions).
        """
        raw: list[ResultRecord] = []
        for method in self.methods():
            for scenario in self.scenarios():
                for seed in self.seeds:
                    seed_everything(seed)
                    ctx = self.make_context(method, scenario, seed)
                    raw.extend(self.evaluate(ctx))
        return aggregate_over_seeds(raw) if aggregate else raw

    def write(self, records: Iterable[ResultRecord], path: str | Path) -> Path:
        """Append ``records`` to the tidy results file at ``path``."""
        path = Path(path)
        with ResultsWriter(path) as writer:
            writer.extend(records)
        return path

    # -- helpers ----------------------------------------------------------- #

    def _cfg_get(self, key: str, default: object) -> object:
        """Read a key from the (hydra / dict-like) config, with a default."""
        cfg = self.config
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)


__all__ = ["RunContext", "ExperimentRunner"]
