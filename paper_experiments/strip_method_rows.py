"""Remove one method's rows from the per-cell metrics / per-step CSVs.

Why this exists: the per-cell CSVs are written PER GROUP, not per method -- one
``<scen>__M<M>__traj<N>__baselines.csv`` holds FlowDAS, SURGE (FlowDAS), SDA,
SURGE (SDA), D-Flow SGLD and Guided FM (FIG) together. When ONE of those methods
turns out to have been computed with a bug, the fix must not cost a rerun of the
other five (D-Flow SGLD alone is ~71 h/cell). So instead: strip the bad method's
rows out of the group files here, then rerun that method alone into its own
``__<group>.csv`` cell files.

That split is invisible downstream -- ``common/aggregate_lib.py`` globs every CSV
in ``metrics/`` and keys on (case, method, scenario, metric, E, M, variant), so a
method arriving from its own file aggregates exactly as it did from the group
file. What is NOT safe is leaving the stale rows in place: within one trajectory
the old and new rows land in the same key and get silently averaged
(``_per_traj_value``), which is worse than either number alone.

Every file that is touched is backed up first (once -- a second strip pass never
overwrites the pristine copy). A file left with a header and no data rows is
removed, so the grid scripts' skip-if-exists resume can regenerate it.

    python paper_experiments/strip_method_rows.py \
        --root paper_experiments/results/navier_stokes \
        --method "SURGE (SDA)" --exclude-group surge_sda \
        --backup paper_experiments/results/navier_stokes/backup_pre_surge_sda_fix \
        --cells-out /tmp/cells.txt

Use ``--dry-run`` to see what would change without writing anything.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

# Per-cell filename convention shared by every grid script:
#   <scenario-slug>__M<steps>__traj<index>__<group>.csv
# (classical cells omit __M<steps>; they are matched too, with M = None.)
CELL_RE = re.compile(r"^(?P<scen>.+?)__(?:M(?P<M>\d+)__)?traj(?P<traj>\d+)__(?P<group>.+)$")

SUBDIRS = ("metrics", "per_step")


def _guard_root(root: Path) -> None:
    """Refuse to touch the user's untracked ``results copy/`` safety net."""
    if "results copy" in str(root.resolve()):
        sys.exit(f"[strip] REFUSING to modify a 'results copy' path: {root}")


def _cell_of(path: Path) -> tuple[str, str | None, str, str] | None:
    m = CELL_RE.match(path.stem)
    if not m:
        return None
    return m["scen"], m["M"], m["traj"], m["group"]


def strip_file(
    path: Path, methods: set[str], backup_dir: Path | None, dry_run: bool
) -> int:
    """Drop rows whose ``method`` is in *methods*. Returns the count removed."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if "method" not in fields:
            return 0
        rows = list(reader)

    keep = [r for r in rows if (r.get("method") or "").strip() not in methods]
    removed = len(rows) - len(keep)
    if removed == 0:
        return 0
    if dry_run:
        return removed

    if backup_dir is not None:
        # Back up ONCE. A second pass (e.g. after a concurrent grid job wrote fresh
        # stale rows) must not clobber the pristine pre-fix copy with a stripped one.
        dest = backup_dir / path.parent.name / path.name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)

    if not keep:
        # Header-only leftovers would make the grid scripts skip the cell forever
        # with nothing recorded in it; drop the file so a rerun can refill it.
        path.unlink()
        return removed

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(keep)
    tmp.replace(path)
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="case results root")
    ap.add_argument("--method", action="append", required=True, help="method name (repeatable)")
    ap.add_argument(
        "--exclude-group",
        action="append",
        default=[],
        help="group suffix to leave alone (the method's OWN rerun files)",
    )
    ap.add_argument("--backup", type=Path, default=None, help="backup directory")
    ap.add_argument("--cells-out", type=Path, default=None, help="write affected cells here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root: Path = args.root
    _guard_root(root)
    methods = {m.strip() for m in args.method}
    excluded = set(args.exclude_group)

    total_removed = 0
    touched: list[Path] = []
    cells: set[tuple[str, str | None, str]] = set()

    for sub in SUBDIRS:
        d = root / sub
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.csv")):
            cell = _cell_of(path)
            if cell is not None and cell[3] in excluded:
                continue
            removed = strip_file(path, methods, args.backup, args.dry_run)
            if removed:
                total_removed += removed
                touched.append(path)
                print(f"[strip] {'would strip' if args.dry_run else 'stripped'} "
                      f"{removed:4d} rows  {sub}/{path.name}")
                if cell is not None:
                    cells.add((cell[0], cell[1], cell[2]))

    verb = "would remove" if args.dry_run else "removed"
    print(f"[strip] {verb} {total_removed} rows for {sorted(methods)} "
          f"across {len(touched)} file(s); {len(cells)} distinct cell(s)")
    if args.backup is not None and touched and not args.dry_run:
        print(f"[strip] originals backed up under {args.backup}")

    if args.cells_out is not None:
        # "<scenario-slug> <M-or-'-'> <traj>", one cell per line, for the rerun driver.
        lines = [f"{s} {m or '-'} {t}" for s, m, t in sorted(cells)]
        if args.dry_run:
            print("[strip] (dry-run) cells:\n  " + "\n  ".join(lines))
        else:
            args.cells_out.parent.mkdir(parents=True, exist_ok=True)
            args.cells_out.write_text("\n".join(lines) + ("\n" if lines else ""))
            print(f"[strip] cell list -> {args.cells_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
