#!/usr/bin/env python3
"""verify_results_export.py -- FULL check that the shared results tree holds every result.

Run after tools/sync_all_results.sh. Read-only on every source; writes only
    results/MANIFEST.md5                       md5 of EVERY file in results/ (for `md5sum -c`
                                               on the computer the folder is copied to)
    results/_provenance_pe/verify_<ts>.txt     full report

Checks (every file, md5, not a sample):
  1. every file of every source tree (the TREES of sync_all_results.sh) exists at its
     destination with the same size and md5;
  2. the Lorenz 96 TMPD files merged into results/lorenz96/linear/ and the single files;
  3. the 2026-11-12 backup (/scratch/ntmucke/backups/rebuttal_ns_session_20261112):
     every backed-up result file is in results/ unchanged, or reported as "newer version
     in results/" (the file was rewritten by a later job) or "missing" (then exit 1).
Exit 0 only if 1 and 2 have no missing/differing file and 3 has no missing file.

    nice -n 19 python3 verify_results_export.py [--procs 8]
"""
import argparse
import datetime as dt
import hashlib
import os
import sys
from multiprocessing import Pool

S = "/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments"
R = f"{S}/results"
L96 = "/scratch/ntmucke/lorenz96_delftblue"
L96T = "/scratch/ntmucke/lorenz96_tmpd"
L96_PE = ("/home/ntmucke/ssi-lorenz96/papers/A Unified Framework for Bayesian Data Assimilation "
          "with Generative Models and Observation Interpolants/code/paper_experiments")
BENCH = "/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench"
BACKUP = "/scratch/ntmucke/backups/rebuttal_ns_session_20261112"

# Same as TREES in sync_all_results.sh (source dir, destination dir).
TREES = [
    (f"{L96}/results/lorenz96", f"{R}/lorenz96"),
    (f"{L96}/figures/lorenz96", f"{S}/figures/lorenz96"),
    (f"{L96}/slurm", f"{R}/lorenz96/_provenance/slurm"),
    (f"{S}/results_wrong_ic_np40", f"{R}/navier_stokes_wrong_ic_np40/main"),
    (f"{S}/results_wrong_ic_np40_ext", f"{R}/navier_stokes_wrong_ic_np40/ext"),
    (f"{S}/results_wrong_ic_np40_prior_enkf", f"{R}/navier_stokes_wrong_ic_np40/prior_enkf"),
    (f"{S}/enkf_loc20_superres", f"{R}/navier_stokes_wrong_ic_np40/enkf_loc20_superres"),
    (f"{S}/results_udales_fc", f"{R}/urban_fc"),
    (f"{S}/smoke_prior_enkf", f"{R}/_smoke_and_bench/smoke_prior_enkf"),
    (BENCH, f"{R}/_smoke_and_bench/slurm_bench"),
    (f"{S}/results_wrong_ic_np40_tmpd", f"{R}/navier_stokes_wrong_ic_np40/tmpd"),
    (f"{S}/results_udales_tmpd", f"{R}/urban_tmpd"),
    (f"{S}/results_tmpd_smoke", f"{R}/_smoke_and_bench/tmpd_smoke"),
    (L96T, f"{R}/lorenz96/_tmpd_source"),
    (f"{S}/results_full", f"{R}/_historical/results_full_20260730"),
    (f"{S}/figures", f"{R}/_figures"),
    (f"{S}/generated", f"{R}/_generated"),
    (f"{S}/outputs", f"{R}/_outputs"),
    (f"{L96T}/published/linear", f"{R}/lorenz96/linear"),  # TMPD files merged (add-only)
]
FILES = [
    (f"{L96}/STATUS.md", f"{R}/lorenz96/_provenance/STATUS.md"),
    (f"{L96_PE}/LORENZ96_HANDOVER_2026-11-12.md", f"{R}/lorenz96/_provenance/LORENZ96_HANDOVER_2026-11-12.md"),
    (f"{L96_PE}/LORENZ96_DELFTBLUE_HANDOVER.md", f"{R}/lorenz96/_provenance/LORENZ96_DELFTBLUE_HANDOVER.md"),
]
if os.path.isdir(f"{L96_PE}/generated"):
    FILES += [(f"{L96_PE}/generated/{f}", f"{S}/generated/{f}")
              for f in sorted(os.listdir(f"{L96_PE}/generated")) if f.startswith("tab_lorenz96")]
FILES += [(f"{S}/{f}", f"{R}/_provenance_pe/{f}") for f in sorted(os.listdir(S))
          if os.path.isfile(f"{S}/{f}") and f.endswith((".md", ".log"))]
# backup subfolder -> source it was taken from (then mapped through TREES)
BACKUP_MAP = [(f"{BACKUP}/paper_experiments/{d}", f"{S}/{d}") for d in
              ["results_wrong_ic_np40", "results_wrong_ic_np40_ext", "results_wrong_ic_np40_prior_enkf",
               "enkf_loc20_superres", "results_udales_fc", "smoke_prior_enkf"]]
BACKUP_MAP.append((f"{BACKUP}/slurm_bench", BENCH))


def md5(path):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        return path, h.hexdigest(), os.path.getsize(path)
    except OSError as e:
        return path, f"ERROR:{e}", -1


def files_under(root):
    out, links = [], []
    for dp, dns, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            (links if os.path.islink(p) else out).append(p)
        for dn in dns:
            if os.path.islink(os.path.join(dp, dn)):
                links.append(os.path.join(dp, dn))
    return out, links


def hash_all(paths, pool):
    return {p: (h, n) for p, h, n in pool.imap_unordered(md5, paths, chunksize=4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{R}/_provenance_pe", exist_ok=True)
    rep_path = f"{R}/_provenance_pe/verify_{ts}.txt"
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log(f"verify_results_export.py {dt.datetime.now().isoformat()}  results={R}")
    fail = False
    with Pool(a.procs) as pool:
        # destination hashes (also the manifest)
        rfiles, rlinks = files_under(R)
        rfiles = [p for p in rfiles if os.path.basename(p) != "MANIFEST.md5"]
        log(f"hashing results/: {len(rfiles)} files, {sum(os.path.getsize(p) for p in rfiles)/1e9:.1f} GB")
        RH = hash_all(rfiles, pool)
        errs = [p for p, (h, _) in RH.items() if h.startswith("ERROR")]
        if errs:
            fail = True
            log(f"UNREADABLE in results/: {len(errs)}")
            for p in errs[:50]:
                log(f"  {p}: {RH[p][0]}")
        # also hash S/figures + S/generated (destination of two TREES outside R)
        extra, _ = files_under(f"{S}/figures")
        extra += files_under(f"{S}/generated")[0]
        RH.update(hash_all(extra, pool))

        def check(pairs, label):
            nonlocal fail
            srcs = [s for s, _ in pairs]
            SH = hash_all(srcs, pool)
            miss = diff = ok = 0
            for s, d in pairs:
                hs = SH[s]
                if d not in RH:
                    miss += 1
                    log(f"  MISSING {d}  (source {s})")
                elif RH[d] != hs:
                    diff += 1
                    log(f"  DIFF    {d}  src={hs} dst={RH[d]}")
                else:
                    ok += 1
            if miss or diff:
                fail = True
            log(f"[{label}] files={len(pairs)} ok={ok} missing={miss} diff={diff} => "
                f"{'MATCH' if not (miss or diff) else 'PROBLEM'}")

        log("\n=== 1. source trees -> results/")
        for src, dst in TREES:
            if not os.path.isdir(src):
                log(f"[{src}] SOURCE MISSING")
                fail = True
                continue
            fs, links = files_under(src)
            for l in links:
                log(f"  symlink not followed in {src}: {l} -> {os.readlink(l)}")
            check([(p, dst + p[len(src):]) for p in fs], f"{src} -> {dst}")

        log("\n=== 2. single files")
        check([(s, d) for s, d in FILES if os.path.isfile(s)], "single files")

        log("\n=== 3. backup 2026-11-12 -> results/")

        def to_dest(src_path):
            for s, d in TREES:
                if src_path == s or src_path.startswith(s + "/"):
                    return d + src_path[len(s):]
            return None

        bmiss, bnew, bok = [], [], 0
        for bsrc, orig in BACKUP_MAP:
            fs, _ = files_under(bsrc)
            BH = hash_all(fs, pool)
            for p in fs:
                d = to_dest(orig + p[len(bsrc):])
                if d is None or d not in RH:
                    bmiss.append(p)
                elif RH[d] != BH[p]:
                    bnew.append((p, d))
                else:
                    bok += 1
        log(f"backup files identical in results/: {bok}; newer version in results/: {len(bnew)}; "
            f"missing from results/: {len(bmiss)}")
        for p, d in bnew:
            log(f"  NEWER   {d}  (backup {p})")
        for p in bmiss:
            log(f"  MISSING-FROM-RESULTS {p}")
        if bmiss:
            fail = True

    with open(f"{R}/MANIFEST.md5", "w") as f:
        for p in sorted(RH):
            if p.startswith(R + "/") and not RH[p][0].startswith("ERROR"):
                f.write(f"{RH[p][0]}  {p[len(R) + 1:]}\n")
    log(f"\nmanifest: {R}/MANIFEST.md5 ({sum(1 for p in RH if p.startswith(R + '/'))} files)"
        "  -- on the target computer:  cd results && md5sum -c --quiet MANIFEST.md5")
    if rlinks:
        log(f"symlinks inside results/ (not followed): {len(rlinks)}")
        for l in rlinks:
            log(f"  {l} -> {os.readlink(l)}")
    log(f"\nOVERALL: {'ALL RESULTS PRESENT AND IDENTICAL' if not fail else 'PROBLEMS (see above)'}")
    with open(rep_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report: {rep_path}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
