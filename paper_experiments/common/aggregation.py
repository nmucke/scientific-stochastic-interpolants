"""Mean +/- std over seeds (reproducibility Section 9).

Case drivers emit one *per-seed* :class:`ResultRecord` for each
(case, method, scenario, metric, seed) cell. Before tables are made, those rows
are reduced to one *aggregated* row per (case, method, scenario, metric) carrying
the across-seed mean in ``value`` and the sample standard deviation in ``std``
(with ``seed = SEED_AGGREGATED``). ``make_tables.py`` consumes the aggregated
file.

Cost columns (``NFE``, ``seconds``) and sampler settings (``E``, ``M``) are
carried through as the mean of the contributing rows when consistent.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from results_schema import SEED_AGGREGATED, ResultRecord


def aggregate_over_seeds(records: list[ResultRecord]) -> list[ResultRecord]:
    """Reduce per-seed rows to mean +/- std rows.

    Rows already aggregated (``seed == SEED_AGGREGATED``) pass through unchanged;
    they are not double-reduced. Grouping key is
    (case, method, scenario, metric, E, M).
    """
    groups: dict[tuple, list[ResultRecord]] = defaultdict(list)
    passthrough: list[ResultRecord] = []

    for r in records:
        if r.seed == SEED_AGGREGATED:
            passthrough.append(r)
            continue
        key = (r.case, r.method, r.scenario, r.metric, r.E, r.M)
        groups[key].append(r)

    aggregated: list[ResultRecord] = list(passthrough)
    for (case, method, scenario, metric, E, M), rows in groups.items():
        values = [r.value for r in rows]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregated.append(
            ResultRecord(
                case=case,
                method=method,
                scenario=scenario,
                metric=metric,
                value=mean,
                std=std,
                E=E,
                M=M,
                seed=SEED_AGGREGATED,
                nfe=_mean_opt([r.nfe for r in rows]),
                seconds=_mean_opt([r.seconds for r in rows]),
            )
        )
    return aggregated


def _mean_opt(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return statistics.fmean(present)


__all__ = ["aggregate_over_seeds"]
