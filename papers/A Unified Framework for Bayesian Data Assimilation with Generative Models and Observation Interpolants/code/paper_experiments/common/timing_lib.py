"""Splice clean per-DA-step timings into aggregated rows.

The ``seconds`` recorded by the grid runs is not trustworthy -- many cells were
timed while other jobs shared the GPU (measured 2026-07-30: a D-Flow cell went
160 s -> 302 s, 1.9x, purely from a second CUDA job) and some fell back to CPU.
``measure_timing.py`` re-measures every method on an idle GPU and writes
``results/timing/seconds_per_step.csv``; this module applies those numbers to the
aggregated rows so the tables and figures quote the clean cost.

Only ``seconds`` is touched. Every other column (rmse, crps, kl, NFE, ...) is
deterministic and comes from the grid runs unchanged.

M SCALING
---------
``measure_timing.py`` measures M=25 only, because per-DA-step cost is linear in M
(the sampler loop is M steps; D-Flow's M drives ``num_optim_steps``;
``inflated_shared`` does ``M/k`` Jacobian refreshes). Rows at other M are filled by
proportional scaling::

    seconds(M) = seconds(M_ref) * M / M_ref

TWO EXCEPTIONS, both handled here rather than left to the caller:

* **M-inert methods.** EnKF and the particle filter propagate with the TRUE solver
  and have no sampler-step axis, so ``measure_timing.py`` records them with an empty
  M. Such an entry matches ANY row's M and is used AS-IS, never scaled.
* **Variants are matched exactly.** ``shared_jac10`` timing is NOT reused for a
  ``shared`` (k=1) row: k is the refresh cadence, so k=1 does 10x the Jacobian
  builds. A row whose exact variant was not measured is left unchanged and counted
  as unmatched rather than being given a plausible-looking wrong number.

Proportional scaling is a good approximation, not an identity -- the 2026-07-30
sweep found SDA drifting ~30% superlinear between M=25 and M=250, while FIG,
FlowDAS and D-Flow tracked it closely. :func:`apply_timing` reports how many rows
were scaled and by how much, so the exposure is visible rather than implicit.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


def _norm_variant(v: object) -> str:
    return "" if v is None or str(v).strip() == "" else str(v).strip()


def _as_int(v: object) -> int | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _as_float(v: object) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# (case, scenario, method, variant, E) -- E included because seconds scales with the
# ensemble size for every method whose cost is a batched network eval.
Key = tuple[str, str, str, str, int | None]


@dataclass
class Timing:
    """Loaded ``seconds_per_step.csv``, indexed for lookup."""

    by_M: dict[Key, dict[int, float]] = field(default_factory=lambda: defaultdict(dict))
    no_M: dict[Key, float] = field(default_factory=dict)
    n_rows: int = 0

    def lookup(self, key: Key, M: int | None) -> tuple[float | None, str]:
        """Return ``(seconds, how)`` for this row.

        ``how`` is one of ``exact`` / ``scaled`` / ``m_inert`` / ``miss``.
        """
        if key in self.no_M:                     # EnKF / PF: M is inert
            return self.no_M[key], "m_inert"
        table = self.by_M.get(key)
        if not table:
            return None, "miss"
        if M is None:
            # Row has no M but only M-keyed timings exist -- refuse to guess.
            return None, "miss"
        if M in table:
            return table[M], "exact"
        ref_M = min(table)                       # the measured rung (normally 25)
        return table[ref_M] * (M / ref_M), "scaled"


def load_timing(path: Path) -> Timing | None:
    """Load ``seconds_per_step.csv``; ``None`` if it does not exist."""
    if not path.exists():
        return None
    t = Timing()
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sec = _as_float(r.get("seconds_per_step"))
            if sec is None:
                continue
            key: Key = (r["case"], r["scenario"], r["method"],
                        _norm_variant(r.get("variant")), _as_int(r.get("E")))
            M = _as_int(r.get("M"))
            if M is None:
                t.no_M[key] = sec
            else:
                t.by_M[key][M] = sec
            t.n_rows += 1
    return t


def apply_timing(rows: list[dict[str, object]], timing: Timing) -> dict[str, int]:
    """Overwrite ``seconds`` on every row from ``timing`` (in place).

    Rewrites both the ``seconds`` COLUMN (carried on every row) and the ``value`` of
    the ``metric == "seconds"`` rows -- the latter is what the LaTeX cost column
    reads (``latex_tables.py`` ``cost_metric``), so missing it would leave the tables
    quoting the old numbers.

    Rows with no matching measurement keep their original value; the returned
    counters say how many, so a silent partial override is impossible.
    """
    stats = {"exact": 0, "scaled": 0, "m_inert": 0, "miss": 0, "no_e_match": 0}
    unmatched: set[tuple] = set()

    for row in rows:
        key: Key = (str(row.get("case")), str(row.get("scenario")),
                    str(row.get("method")), _norm_variant(row.get("variant")),
                    _as_int(row.get("E")))
        M = _as_int(row.get("M"))
        sec, how = timing.lookup(key, M)

        if sec is None:
            # Fall back to an E-agnostic match so a timing sweep run at a different
            # ensemble size is still usable -- but count it, because seconds scales
            # with E and the number is then only indicative.
            alt = [(k, v) for k, v in timing.by_M.items() if k[:4] == key[:4]]
            alt_no_m = [(k, v) for k, v in timing.no_M.items() if k[:4] == key[:4]]
            if alt_no_m:
                sec, how = alt_no_m[0][1], "m_inert"
                stats["no_e_match"] += 1
            elif alt and M is not None:
                table = alt[0][1]
                ref_M = min(table)
                sec = table[ref_M] * (M / ref_M)
                how = "exact" if M in table else "scaled"
                if M in table:
                    sec = table[M]
                stats["no_e_match"] += 1

        if sec is None:
            stats["miss"] += 1
            unmatched.add(key[:4])
            continue

        stats[how] += 1
        row["seconds"] = sec
        if row.get("metric") == "seconds":
            row["value"] = sec
            row["std"] = None      # a single clean measurement has no across-traj std

    stats["unmatched_cells"] = len(unmatched)
    if unmatched:
        preview = sorted(unmatched)[:6]
        stats["_unmatched_preview"] = preview  # type: ignore[assignment]
    return stats


def describe(stats: dict[str, int]) -> str:
    """One-line human summary of :func:`apply_timing`'s counters."""
    parts = [f"{stats['exact']} exact", f"{stats['scaled']} M-scaled",
             f"{stats['m_inert']} M-inert"]
    if stats.get("no_e_match"):
        parts.append(f"{stats['no_e_match']} matched ignoring E")
    if stats.get("miss"):
        parts.append(f"{stats['miss']} UNMATCHED (kept old seconds)")
    return ", ".join(parts)
