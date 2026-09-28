# Rerunning `SURGE (SDA)` on Navier–Stokes — exact procedure

Operating instructions for `rerun_ns_surge_sda.sh`. The script's own header explains
*what* it does and *why*; this file is the **runbook for the current state of the data**,
and in one place it **corrects** the header (see "Correction to the script header").

Written 2026-07-30. Full coverage needs **two passes**, because 3 of the 80 cells are
still being produced by the CPU grid.

---

## The one rule

> **Every invocation must set `CELLS_FILE`. No exceptions — including the `CLEAN_ONLY` pass.**

Phase 1 of an earlier run already stripped the `SURGE (SDA)` rows from the group CSVs.
Re-deriving the cell list now therefore finds **nothing**, and without `CELLS_FILE` the
script takes its non-resume branch (lines 143–157), which:

1. overwrites `.rerun_surge_sda_cells.txt` with an **empty** list, and
2. **moves every finished `*__surge_sda.csv` into `backup_pre_surge_sda_fix/`**,

then reruns zero cells — leaving `SURGE (SDA)` permanently missing from the results.

This is not hypothetical. Verified 2026-07-30 with `DRY_RUN=1 CLEAN_ONLY=1` and no
`CELLS_FILE`:

```
[strip] would remove 0 rows for ['SURGE (SDA)'] across 0 file(s); 0 distinct cell(s)
[surge] would delete .../metrics/16_2_128_2__M100__traj10__surge_sda.csv
[surge] would delete .../metrics/16_2_128_2__M100__traj11__surge_sda.csv
[surge] would delete .../per_step/16_2_128_2__M100__traj10__surge_sda.csv
[surge] would delete .../per_step/16_2_128_2__M100__traj11__surge_sda.csv
```

A backup of the 77-cell list is kept at
`results/navier_stokes/.rerun_surge_sda_cells.txt.bak77`. If the list is ever
truncated or emptied, restore it from there before doing anything else.

---

## Coverage: why two passes

The full grid is **4 scenarios × 4 M {25,50,100,250} × 5 traj {10..14} = 80 cells**.
The work list holds **77**. The 3 absent ones are

```
32_2_128_2     M25  traj14
sparse_5       M25  traj14
sparse_1_5625  M25  traj14
```

which the M25-nodflow CPU grid had not reached when the list was derived — they are
exactly the three shards it is still running. The list is **derived from the data**
(cells that actually contained `SURGE (SDA)` rows), so missing cells are not invented;
the grid produces them itself, and pass 2 picks them up.

`77 + 3 = 80`. ✅

---

## Step 1 — the main rerun (75 cells, ~25–37 h)

```bash
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
CELLS_FILE=paper_experiments/results/navier_stokes/.rerun_surge_sda_cells.txt \
setsid nohup bash paper_experiments/rerun_ns_surge_sda.sh \
  >rerun_surge_sda.log 2>&1 & disown
```

77 cells are listed; already-finished ones are skipped by skip-if-exists, so 75 run.
Budget from measured cells: ~26–31 min each at E=64 / NP=20.

**Confirm it launched correctly** — all five must hold:

```bash
R=paper_experiments/results/navier_stokes
pgrep -af rerun_ns_surge_sda.sh                       # 1. process exists
tr '\0' '\n' < /proc/$(pgrep -f rerun_ns_surge_sda.sh | head -1)/environ | grep CELLS_FILE
                                                      # 2. CELLS_FILE really is set
wc -l < $R/.rerun_surge_sda_cells.txt                 # 3. still 77
ls $R/metrics/*__surge_sda.csv | wc -l                # 4. did NOT drop
grep -c "RESUME (CELLS_FILE set)" $R/rerun_surge_sda.log   # 5. == 1
```

A correct start logs:

```
[surge] RESUME (CELLS_FILE set): keeping N finished surge_sda cell file(s) ...
[surge] --- phase 2: rerun 77 cell(s) ---
```

If instead you see `phase 2: rerun 0 cell(s)`, **stop immediately** — `CELLS_FILE` was
not set. Restore the list from `.bak77`, restore any `*__surge_sda.csv` from
`backup_pre_surge_sda_fix/`, and relaunch properly.

---

## Step 2 — after the CPU grid finishes, catch the last 3

> **Usually you do NOT need to run this by hand.** Step 1 ends with its own **phase 3**
> re-clean (script lines 216–221), which is exactly this pass, and writes its cell list
> to `.rerun_surge_sda_cells_phase3.txt`. So if the grid cells appeared *while step 1
> was still running* — the normal case — just read that file when step 1 exits and go
> straight to step 3 with it.
>
> Run step 2 explicitly only when rows appear **after** step 1 has already exited, or
> when step 1 was interrupted before reaching phase 3.
>
> It is safe to run it concurrently with step 1 if you want to (it does no GPU work,
> and `--exclude-group surge_sda` makes it skip the files step 1 is writing) — it just
> gains nothing, since the refill in step 3 needs the GPU that step 1 is holding.

Wait until the `run_ns_grid_M25_nodflow*` shards have written their traj14/M25
`baselines_nodflow` cells, then:

```bash
CELLS_FILE=paper_experiments/results/navier_stokes/.rerun_surge_sda_cells.txt \
CLEAN_ONLY=1 bash paper_experiments/rerun_ns_surge_sda.sh
```

Strips any `SURGE (SDA)` rows the grid wrote after step 1 and lists the affected cells
in `.rerun_surge_sda_cells_phase1.txt`. With `CELLS_FILE` set it leaves the main list
and the finished cells untouched. If it reports **0 rows**, everything is consistent and
step 3 is unnecessary.

### Correction to the script header

Header line 47 gives this pass as `CLEAN_ONLY=1 bash …` **without** `CELLS_FILE`. That
form is unsafe in the current state — it deletes finished cells and empties the list, per
"The one rule" above. Always use the form given here.

---

## Step 3 — refill whatever the clean pass listed

Point `CELLS_FILE` at whichever list the clean pass produced:

```bash
# normal case: step 1's own phase 3 produced the list
CELLS_FILE=paper_experiments/results/navier_stokes/.rerun_surge_sda_cells_phase3.txt \
bash paper_experiments/rerun_ns_surge_sda.sh

# only if you ran step 2 by hand instead
CELLS_FILE=paper_experiments/results/navier_stokes/.rerun_surge_sda_cells_phase1.txt \
bash paper_experiments/rerun_ns_surge_sda.sh
```

Skip this if the clean pass reported **0 rows** / an empty list — the script hard-fails
on an empty `CELLS_FILE` (line 101) rather than silently doing nothing, which is the
intended behaviour.

Note the clean pass **strips without refilling** — its own log says so
(`any cell listed above lost its 'SURGE (SDA)' rows and was NOT rerun`). So a cell that
appears in a phase-1/phase-3 list has *no* `SURGE (SDA)` data anywhere until this step
runs. Do not stop after the clean pass.

---

## Verify completeness

```bash
ls paper_experiments/results/navier_stokes/metrics/*__surge_sda.csv | wc -l   # expect 80

# no stale rows left in any group CSV:
cat paper_experiments/results/navier_stokes/metrics/*__baselines*.csv \
  | cut -d, -f2 | sort -u | grep "SURGE (SDA)" && echo "STALE ROWS PRESENT" \
                                              || echo "clean"
```

Then aggregate as usual:

```bash
.venv/bin/python paper_experiments/aggregate_ns.py
```

---

## Notes

**The 3 outstanding cells are already correct as the grid writes them.**
`src/scisi/posterior_models/surge_posterior.py` was last modified **2026-07-22 12:15**,
and all three CPU shards started after it (Jul 28 09:48, Jul 28 23:37, Jul 30 01:19), so
they carry the fix. Steps 2–3 strip and rerun them anyway for uniformity — every
`SURGE (SDA)` row then lives in a dedicated `__surge_sda.csv`. Skipping steps 2–3 is
*not* a correctness bug: aggregation globs every CSV and keys on
`(method, scenario, metric, E, M, variant)`, so rows left in the group files are picked
up without double counting. It is only a tidiness/provenance choice.

**Concurrency.** `SURGE (SDA)` at E=64 holds **~11.5 GB** of GPU. The script header
records that running two CUDA jobs at once on 2026-07-15 lost **34 of 60 cells to OOM**,
so do not start another GPU job beside it without checking `nvidia-smi` free memory
first. Failures are not destructive — a killed or OOM'd cell leaves no partial file and
is retried on the next run — but they waste hours.

**Resuming after an interruption** uses the same step 1 command verbatim; finished cells
are skipped.
