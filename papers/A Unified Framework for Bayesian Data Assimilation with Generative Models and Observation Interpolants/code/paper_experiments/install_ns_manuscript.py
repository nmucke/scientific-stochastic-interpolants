"""Install regenerated Navier--Stokes tables and figures into the review manuscript.

Writes only to the review copy: ``sections_review/`` and ``figures_review/``.
``main.tex`` and ``figures/`` are not touched.

* Figures: every ``ns_*.pdf`` already in ``figures_review/navier_stokes/singles`` is
  replaced by the file of the same name in ``<figures>/singles``. The run stops
  before changing anything if one is missing. ``ns_*_vs_M_*`` panels come from the
  old M sweep, so they are removed, and the run stops if the text still uses them.
* Tables: the ``tabular`` of ``tab:ns_accuracy`` and ``tab:ablation`` is replaced;
  the caption and label stay. Data cells are wrapped in ``--rev-macro``.

Refuses a results tree marked ``SYNTHETIC_DO_NOT_USE.txt`` unless
``--allow-synthetic`` is given, which is meant for a throwaway manuscript copy.

    .venv/bin/python paper_experiments/install_ns_manuscript.py \
        --results-root <root> --figures <figures/navier_stokes> --manuscript <clone>
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

SINGLES = Path("figures_review/navier_stokes/singles")
ROW_LEAD = re.compile(r"\s*(Ours|FlowDAS|SDA|D-Flow|Guided|EnKF|Particle)")


def tabular_of(path: Path) -> list[str]:
    """Return the lines of the first ``tabular`` in a generated table file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if r"\begin{tabular}" in l)
    end = next(i for i, l in enumerate(lines) if r"\end{tabular}" in l)
    return lines[start:end + 1]


def colour_cells(line: str, macro: str) -> str:
    """Wrap every data cell of a method row in ``\\<macro>{...}``."""
    if not macro or not ROW_LEAD.match(line) or "&" not in line:
        return line
    label, *cells = line.rstrip().removesuffix(r"\\").split("&")
    wrapped = [rf" \{macro}{{{c.strip()}}} " for c in cells]
    return "&".join([label, *wrapped]).rstrip() + r" \\"


def splice(tex: Path, label: str, tabular: list[str]) -> None:
    """Replace the ``tabular`` of the uncommented table carrying ``label``."""
    lines = tex.read_text(encoding="utf-8").splitlines()
    live = lambda l: not l.lstrip().startswith("%")  # noqa: E731
    at = next(i for i, l in enumerate(lines) if rf"\label{{{label}}}" in l and live(l))
    start = next(i for i in range(at, len(lines))
                 if r"\begin{tabular}" in lines[i] and live(lines[i]))
    end = next(i for i in range(start, len(lines))
               if r"\end{tabular}" in lines[i] and live(lines[i]))
    lines[start:end + 1] = tabular
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--figures", type=Path, required=True,
                    help="make_ns_figures.py --out directory (holds singles/ and "
                         "tab_ablation.tex)")
    ap.add_argument("--manuscript", type=Path, required=True)
    ap.add_argument("--rev-macro", default="revC",
                    help="review-colour macro for table cells; '' for none")
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args()

    if (args.results_root / "SYNTHETIC_DO_NOT_USE.txt").exists() and not args.allow_synthetic:
        raise SystemExit(f"{args.results_root} is a synthetic tree; refusing to install")

    # Check everything before the first write.
    dest = args.manuscript / SINGLES
    src = args.figures / "singles"
    stale = sorted(dest.glob("ns_*_vs_M_*.pdf"))
    wanted = sorted(p.name for p in dest.glob("ns_*.pdf") if p not in stale)
    missing = [n for n in wanted if not (src / n).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} panels not generated, e.g. {missing[:5]}")
    appendix = args.manuscript / "sections_review/appendix_ns_case.tex"
    if "_vs_M" in "\n".join(
        l for l in appendix.read_text(encoding="utf-8").splitlines()
        if not l.lstrip().startswith("%")
    ):
        raise SystemExit(f"{appendix.name} still includes *_vs_M panels")
    accuracy = tabular_of(args.results_root / "navier_stokes/tables_scaled.tex")
    ablation = tabular_of(args.figures / "tab_ablation.tex")

    for name in wanted:
        shutil.copy2(src / name, dest / name)
    for path in stale:
        path.unlink()
    print(f"figures: {len(wanted)} replaced, {len(stale)} *_vs_M panels removed")

    # The manuscript names this group "Vorticity CRPS" in the main table only.
    accuracy = [l.replace(r"\textbf{CRPS", r"\textbf{Vorticity CRPS") for l in accuracy]
    splice(args.manuscript / "sections_review/results.tex", "tab:ns_accuracy",
           [colour_cells(l, args.rev_macro) for l in accuracy])
    splice(appendix, "tab:ablation",
           [colour_cells(l, args.rev_macro) for l in ablation])
    print("tables: tab:ns_accuracy and tab:ablation replaced")


if __name__ == "__main__":
    main()
