"""Second launcher: feature-scaling controls on S0, then the remaining secondary settings."""
import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
jobs = [(n, "S0", k) for n in ("V0-normfeat", "V1s-rawfeat", "V0-leg", "V0-legraw", "V1s-legraw", "V1-rawfeat") for k in range(3)]
jobs += [(n, "S4_quad", k) for n in ("V1s", "V1+ES1") for k in range(3)]
for st in ("S3_a0.5", "S5_2d"):
    jobs += [(n, st, k) for n in ("V0", "V1", "V1s") for k in range(3)]
jobs += [(n, "S2_c0.5", k) for n in ("V0", "V1") for k in range(3)]
def run(job):
    n, st, k = job
    r = subprocess.run([sys.executable, os.path.join(HERE, "train.py"), n, st, str(k)], capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if l.startswith("DONE")]
    print(tail[0] if tail else f"FAILED {job}: {r.stderr[-800:]}", flush=True)
with ThreadPoolExecutor(9) as ex:
    list(ex.map(run, jobs))
print("ALL DONE", len(jobs))
