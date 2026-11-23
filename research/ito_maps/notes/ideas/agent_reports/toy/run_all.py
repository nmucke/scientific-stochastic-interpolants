"""Launch all runs with a process pool (one thread each)."""
import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
S0 = ["V0", "V1", "V1+ES1", "V1s", "V0+ES1", "V1s+ES1", "V1+ES0.1", "V1-ESonly", "V1-bell", "V1-K1", "V1-K2", "V2-K2", "V2-K4", "V2-K4+ES1"]
jobs = [(n, "S0", k) for n in S0 for k in range(3)]
for st in ("S1_3mode", "S4_quad", "S3_a0.5", "S5_2d"):
    jobs += [(n, st, k) for n in ("V0", "V1", "V1s", "V1+ES1") for k in range(3)]
jobs += [(n, "S2_c0.5", k) for n in ("V0", "V1") for k in range(3)]
def run(job):
    n, st, k = job
    r = subprocess.run([sys.executable, os.path.join(HERE, "train.py"), n, st, str(k)], capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if l.startswith("DONE")]
    print(tail[0] if tail else f"FAILED {job}: {r.stderr[-800:]}", flush=True)
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 9) as ex:
    list(ex.map(run, jobs))
print("ALL DONE", len(jobs))
