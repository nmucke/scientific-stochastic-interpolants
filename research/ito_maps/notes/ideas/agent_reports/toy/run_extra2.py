"""Third launcher: Legendre-feature baseline on the secondary settings, runs lost to a kill, and V0-legraw+ES1."""
import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
jobs = [("V0-legraw", st, k) for st in ("S1_3mode", "S4_quad", "S3_a0.5", "S5_2d") for k in range(3)]
jobs += [("V0", "S4_quad", 1), ("V0", "S4_quad", 2)] + [("V1", "S4_quad", k) for k in range(3)]
jobs += [("V0-legraw+ES1", "S0", k) for k in range(3)]
def run(job):
    n, st, k = job
    r = subprocess.run([sys.executable, os.path.join(HERE, "train.py"), n, st, str(k)], capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if l.startswith("DONE")]
    print(tail[0] if tail else f"FAILED {job}: {r.stderr[-800:]}", flush=True)
with ThreadPoolExecutor(3) as ex:
    list(ex.map(run, jobs))
print("ALL DONE", len(jobs))
