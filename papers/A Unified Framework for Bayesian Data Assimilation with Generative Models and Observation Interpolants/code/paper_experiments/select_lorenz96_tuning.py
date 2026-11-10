"""Select the Lorenz 96 tuning picks by the fixed rule, and (``apply``) write them
into the config.

Rule (fixed before results, GPU_HANDOFF.md step 7): per cell, the row with the
lowest validation ``sliced_w2``. A row with ``diverged_steps > 0`` (the driver
writes ``inf`` scores for it) or a non-finite ``sliced_w2`` is not eligible.

Actions (``--obs linear|arctan`` picks the sweep root ``results/lorenz96/<obs>/``):

``select`` (default)
    Reads ``tuning/*__s<stage>.csv`` (the files ``run_lorenz96.sh tune`` writes; the
    cell is the file name up to the first ``__``, the grid keys are the columns
    before ``base``) and writes ``tuning/selected_stage<stage>.yaml`` -- per cell
    the tuned keys (``tuned``, nested, ready to paste into ``methods:`` for the
    linear sweep or ``methods_arctan:`` for the arctan one), the full settings the
    row ran with (``full``) and the same keys as ``--set`` strings (``set``) -- and
    ``tuning/selected_stage<stage>.md``: every grid, its row counts, the chosen row
    and the gap to the runner-up.

``apply``
    Writes the picks of ``selected_stage1.yaml`` then ``selected_stage2.yaml`` into
    ``methods:`` (linear) / ``methods_arctan:`` (arctan) of
    ``configs/case/lorenz96.yaml``: per picked cell, its own block with the tuned keys
    merged in (nested keys too, so ``correction.gain_scale`` keeps the block's
    ``localisation`` and ``forecast_num_steps``). Only those cells' one-line blocks
    of that section are rewritten, every other line (comments included) is kept
    byte for byte, and the result is re-read and checked before it is written.
    Keys a stage-2 cell got from its parent by ``--set`` are not copied: the parent's
    own pick is applied to the parent, and the cell inherits it.

``resolve --parent P --grid G``
    For ``run_lorenz96.sh tune`` stage 2: reads ``selected_stage1.yaml``, replaces each
    ``{"times": [f, ...]}`` value of ``G`` by the parent's stage-1 pick times each
    ``f``, and prints the resolved grid JSON on the first line, then one ``--set``
    string per line: the parent's stage-1 picks for the keys not in the grid.
    An empty parent prints the grid unchanged and no ``--set``.

``status --cell C --grid G --stage k``
    Prints ``complete|missing|partial <csv path>`` for the tuning job: complete
    when every grid point is in the job's CSV.

    python select_lorenz96_tuning.py --obs linear --stage 1
    python select_lorenz96_tuning.py --obs linear resolve --parent flowdas \
        --grid '{"guidance_scale":{"times":[0.25,1,4]},"ess_threshold":[0.5,1.0]}'
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import yaml

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from lorenz96_lineup import CONFIG_BLOCK, OBS_CHOICES, set_dotted, sweep_root  # noqa: E402

RESULTS = _here / "results"
CASE_CONFIG = _here / "configs" / "case" / "lorenz96.yaml"
SCORE = "sliced_w2"
DIVERGED_STEPS = "diverged_steps"


def tuning_dir(obs: str) -> Path:
    return sweep_root(RESULTS, obs) / "tuning"


def stage_tag(stage: int) -> str:
    return f"s{stage}"


def job_csv(obs: str, cell: str, grid: dict, stage: int) -> Path:
    """The CSV ``driver.py tune --method cell --grid grid --tag s<stage>`` appends to."""
    return tuning_dir(obs) / f"{cell}__{'_'.join(grid)}__{stage_tag(stage)}.csv"


def parse_value(text: str):
    """A CSV cell back to the grid value: '' -> None, JSON if it parses, else str."""
    if text == "":
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def as_float(text) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def read_tuning_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Grid keys (the columns before ``base``) and the rows."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if "base" not in columns or SCORE not in columns:
        raise ValueError(f"{path} is not a driver.py tune file (no 'base' / '{SCORE}' column).")
    return columns[: columns.index("base")], rows


def eligible(row: dict) -> bool:
    diverged = as_float(row.get(DIVERGED_STEPS, "nan"))
    return math.isfinite(as_float(row[SCORE])) and diverged == 0


def nested(flat: dict) -> dict:
    """``{"correction.gain_scale": 0.5}`` -> ``{"correction": {"gain_scale": 0.5}}``."""
    return set_dotted({}, flat)


def set_strings(flat: dict) -> list[str]:
    return [f"{key}={json.dumps(value)}" for key, value in flat.items()]


def select(obs: str, stage: int) -> tuple[dict, str]:
    """The picks per cell and the markdown summary."""
    files = sorted(tuning_dir(obs).glob(f"*__{stage_tag(stage)}.csv"))
    by_cell: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_cell[path.name.split("__")[0]].append(path)
    picks: dict[str, dict] = {}
    md = [
        f"# Lorenz 96 tuning, {obs} sweep, stage {stage}",
        "",
        f"Rule: lowest validation `{SCORE}`; rows with `{DIVERGED_STEPS} > 0` or a "
        "non-finite score are not eligible. `select_lorenz96_tuning.py apply` writes "
        f"them into `{CONFIG_BLOCK[obs]}:` of `configs/case/lorenz96.yaml`.",
        "",
    ]
    if not files:
        md.append(f"No `*__{stage_tag(stage)}.csv` file in `{tuning_dir(obs)}`.")
    for cell, paths in sorted(by_cell.items()):
        candidates = []
        md.append(f"## {cell}")
        md.append("")
        tables = [(path, *read_tuning_csv(path)) for path in paths]
        for path, keys, rows in tables:
            ok = [row for row in rows if eligible(row)]
            points = {tuple(row[k] for k in keys) for row in rows}
            md.append(
                f"- `{path.name}`: grid keys {keys}; {len(rows)} rows "
                f"({len(points)} distinct points), {len(ok)} eligible, "
                f"{len(rows) - len(ok)} diverged or non-finite"
            )
            for row in ok:
                tuned = {k: parse_value(row[k]) for k in keys}
                candidates.append((as_float(row[SCORE]), tuned, row, path))
        if not candidates:
            md += ["", "**No eligible row: no pick.**", ""]
            continue
        candidates.sort(key=lambda c: c[0])
        score, tuned, row, path = candidates[0]
        base = json.loads(row["base"]) if row.get("base") else {}
        picks[cell] = {
            "tuned": nested(tuned),
            "set": set_strings(tuned),
            "full": set_dotted(base, tuned),
            SCORE: score,
            "file": path.name,
        }
        md.append("")
        md.append(f"Chosen: `{json.dumps(tuned)}` with {SCORE} = {score:.4f}")
        if len(candidates) > 1:
            runner_score, runner, *_ = candidates[1]
            gap = runner_score - score
            rel = gap / score if score else math.nan
            md.append(
                f"Runner-up: `{json.dumps(runner)}` with {SCORE} = {runner_score:.4f} "
                f"(gap {gap:.4f}, {100 * rel:.1f}%)"
            )
            picks[cell]["runner_up_gap"] = gap
        else:
            md.append("Runner-up: none (one eligible row)")
        md.append("")
        for path_, keys, rows in tables:
            md.append(f"`{path_.name}`:")
            md.append("")
            md.append("| " + " | ".join([*keys, SCORE, DIVERGED_STEPS, "chosen"]) + " |")
            md.append("|" + "---|" * (len(keys) + 3))
            for r in sorted(rows, key=lambda r: as_float(r[SCORE])):
                mark = "**x**" if r is row else ""
                cells = [r[k] for k in keys] + [r[SCORE], r.get(DIVERGED_STEPS, "")]
                md.append("| " + " | ".join([*cells, mark]) + " |")
            md.append("")
    return picks, "\n".join(md) + "\n"


def write_selection(obs: str, stage: int) -> Path:
    picks, summary = select(obs, stage)
    out = tuning_dir(obs) / f"selected_stage{stage}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Lorenz 96 {obs} sweep, tuning stage {stage}: picks by lowest validation "
        f"{SCORE}\n# (select_lorenz96_tuning.py). After review, `apply` merges each "
        f"cell's `tuned` block into\n# `{CONFIG_BLOCK[obs]}:` of "
        "configs/case/lorenz96.yaml; selecting does not change the config.\n"
    )
    out.write_text(header + yaml.safe_dump(picks, sort_keys=True), encoding="utf-8")
    out.with_suffix(".md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"wrote {out} and {out.with_suffix('.md')}")
    return out


def flatten(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def resolve(obs: str, parent: str, grid: dict) -> tuple[dict, list[str]]:
    """Stage-2 grid with ``{"times": [...]}`` resolved, and the parent's ``--set``."""
    relative = {k: v for k, v in grid.items() if isinstance(v, dict)}
    if not parent:
        if relative:
            raise ValueError(f"Relative values {sorted(relative)} need a parent cell.")
        return grid, []
    path = tuning_dir(obs) / "selected_stage1.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the stage 1 selection first.")
    picks = yaml.safe_load(path.read_text()) or {}
    if parent not in picks:
        raise ValueError(f"No stage-1 pick for {parent!r} in {path}: tune it first.")
    chosen = flatten(picks[parent]["tuned"])
    resolved = {}
    for key, values in grid.items():
        if isinstance(values, dict):
            if set(values) != {"times"} or key not in chosen:
                raise ValueError(
                    f"{key}: a relative value is {{\"times\": [...]}} on a key the parent "
                    f"{parent!r} tuned (it tuned {sorted(chosen)})."
                )
            resolved[key] = [chosen[key] * float(f) for f in values["times"]]
        else:
            resolved[key] = values
    sets = set_strings({k: v for k, v in chosen.items() if k not in grid})
    return resolved, sets


def status(obs: str, cell: str, grid: dict, stage: int) -> tuple[str, Path]:
    path = job_csv(obs, cell, grid, stage)
    if not path.exists() or path.stat().st_size == 0:
        return "missing", path
    keys, rows = read_tuning_csv(path)
    if keys != list(grid):
        # The driver refuses to append to it anyway; report it as partial so the
        # tuning script moves it aside.
        return "partial", path
    found = {tuple(parse_value(row[k]) for k in keys) for row in rows}
    wanted = {
        tuple(json.loads(json.dumps(v)) for v in point)
        for point in itertools.product(*grid.values())
    }
    return ("complete" if wanted <= found else "partial"), path


def _merge(block: dict, tuned: dict) -> dict:
    out = dict(block)
    for key, value in tuned.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _flow(value) -> str:
    """One YAML flow-style value (``{a: 1, b: {c: null}}``), keys in order."""
    return yaml.safe_dump(value, default_flow_style=True, sort_keys=False, width=10**9).strip()


def apply(obs: str, config: Path | None = None) -> dict[str, dict]:
    """Merge the stage-1 then stage-2 picks into the ``obs`` settings block.

    Returns ``{cell: new own block}``. The file keeps every other line; each picked
    cell must have a one-line ``  <cell>: {...}`` block in its section.
    """
    from omegaconf import OmegaConf

    config = CASE_CONFIG if config is None else config
    section = CONFIG_BLOCK[obs]
    picks: dict[str, dict] = {}
    for stage in (1, 2):
        path = tuning_dir(obs) / f"selected_stage{stage}.yaml"
        if path.exists():
            for cell, pick in (yaml.safe_load(path.read_text()) or {}).items():
                picks[cell] = _merge(picks.get(cell, {}), pick["tuned"])
    if not picks:
        raise FileNotFoundError(f"No selected_stage*.yaml with picks in {tuning_dir(obs)}.")
    before = OmegaConf.to_container(OmegaConf.load(config))
    lines = config.read_text(encoding="utf-8").splitlines(keepends=True)
    head = next((i for i, line in enumerate(lines) if line.rstrip() == f"{section}:"), None)
    if head is None:
        raise ValueError(f"{config} has no top-level `{section}:` line.")
    end = next(
        (i for i in range(head + 1, len(lines))
         if lines[i].strip() and not lines[i].startswith((" ", "#"))),
        len(lines),
    )
    blocks = {}
    for cell, tuned in picks.items():
        new = _merge(before[section].get(cell) or {}, tuned)
        where = [i for i in range(head + 1, end) if lines[i].startswith(f"  {cell}: {{")]
        if len(where) != 1 or not lines[where[0]].rstrip().endswith("}"):
            raise ValueError(f"{section}.{cell} is not one `  {cell}: {{...}}` line of {config}.")
        lines[where[0]] = f"  {cell}: {_flow(new)}\n"
        blocks[cell] = new
    text = "".join(lines)
    after = OmegaConf.to_container(OmegaConf.create(text))
    expected = json.loads(json.dumps(before))
    for cell, new in blocks.items():
        expected[section][cell] = new
    if after != expected:
        raise RuntimeError(f"Writing the picks into {config} would change other entries; nothing written.")
    config.write_text(text, encoding="utf-8")
    for cell, new in blocks.items():
        print(f"{section}.{cell}: {_flow(new)}")
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "action", nargs="?", default="select", choices=["select", "apply", "resolve", "status"]
    )
    parser.add_argument("--obs", choices=OBS_CHOICES, default="linear")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--parent", default="", help="resolve: the stage-1 cell to start from")
    parser.add_argument("--cell", help="status: the tuned cell")
    parser.add_argument("--grid", help="resolve/status: the grid JSON")
    parser.add_argument(
        "--set", action="append", help="status: the job's --set overrides (key=value)"
    )
    args = parser.parse_args()
    if args.action == "select":
        write_selection(args.obs, args.stage)
        return
    if args.action == "apply":
        apply(args.obs)
        return
    if args.grid is None:
        parser.error(f"{args.action} needs --grid")
    grid = json.loads(args.grid)
    if args.action == "resolve":
        resolved, sets = resolve(args.obs, args.parent, grid)
        print(json.dumps(resolved, separators=(",", ":")))
        for entry in sets:
            print(entry)
    else:
        if not args.cell:
            parser.error("status needs --cell")
        state, path = status(args.obs, args.cell, grid, args.stage)
        print(f"{state} {path}")


if __name__ == "__main__":
    main()
