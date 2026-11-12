"""Pick the shared-Jacobian tuning winners: lowest validation sliced_w2; rows with
diverged_steps > 0 or a non-finite score are ineligible; ties keep grid order (as
select_lorenz96_tuning.py). Usage: shared_select.py <tuning dir> <stage> -> writes
selected_shared_stage<k>.{md,json}, prints one 'cell<TAB>json picks' per line."""
import csv, json, math, sys, glob, os
d, stage = sys.argv[1], sys.argv[2]
def parse(v):
    try: return json.loads(v)
    except ValueError: return v
picks, md = {}, [f"# Shared-Jacobian (inflated_shared) tuning, stage {stage}\n",
 "Rule: lowest validation `sliced_w2` (seed 0, first 20 steps, M=50); diverged or non-finite rows ineligible; corrected cells: online Q only (exact-Q rows shown, not eligible).\n"]
for path in sorted(glob.glob(f"{d}/*__shared-s{stage}.csv")):
    cell = os.path.basename(path).split("__")[0]
    rows = list(csv.DictReader(open(path)))
    cols = list(rows[0]); keys = cols[: cols.index("base")]
    # Decision (user, 2026-11-12): corrected cells use the online Q_hat only.
    ok = [r for r in rows if float(r["diverged_steps"]) == 0 and math.isfinite(float(r["sliced_w2"]))
          and r.get("correction.q_source", "online") == "online"]
    md.append(f"\n## {cell}\n\n{len(rows)} rows, {len(ok)} eligible. Base: `{rows[0]['base']}`\n")
    md.append("| " + " | ".join(keys) + " | sliced_w2 | mean_err | spread | diverged_steps | seconds | chosen |")
    md.append("|" + "---|" * (len(keys) + 6))
    best = min(ok, key=lambda r: float(r["sliced_w2"])) if ok else None
    for r in sorted(rows, key=lambda r: float(r["sliced_w2"])):
        md.append("| " + " | ".join(r[k] for k in keys) + f" | {float(r['sliced_w2']):.4f} | {float(r['mean_error_vs_pf']):.4f} | {float(r['spread']):.3f} | {r['diverged_steps']} | {float(r['seconds']):.0f} | {'**x**' if r is best else ''} |")
    if best:
        picks[cell] = {k: parse(best[k]) for k in keys}
        print(f"{cell}\t{json.dumps(picks[cell])}")
open(f"{d}/selected_shared_stage{stage}.md", "w").write("\n".join(md) + "\n")
json.dump(picks, open(f"{d}/selected_shared_stage{stage}.json", "w"), indent=1, sort_keys=True)
