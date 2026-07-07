import sys, tempfile, os, csv
sys.path.insert(0, "paper_experiments")
from common.per_step_io import per_step_rows, append_per_step, load_per_step, PER_STEP_FIELDNAMES

fails = []
per_step = {"rmse": [0.1, 0.2], "crps": [0.3, 0.4]}
rows = per_step_rows(
    case="analytical", method="Ours (SI-SDE)", scenario="analytical",
    variant="shared", E=100, M=250, seed=0, test_index=0,
    nfe=10.0, seconds=1.5, per_step=per_step,
)
d = tempfile.mkdtemp()
p = os.path.join(d, "ps.csv")
append_per_step(p, rows)
loaded = load_per_step(p)
if len(loaded) != 4:
    fails.append(f"row count != 4: {len(loaded)}")
for col in ("variant", "step", "seconds"):
    if col not in PER_STEP_FIELDNAMES:
        fails.append(f"missing col {col}")
# header check from file
with open(p) as f:
    header = next(csv.reader(f))
for col in ("variant", "step", "seconds"):
    if col not in header:
        fails.append(f"header missing {col}")
print("row count:", len(loaded))
print("header:", header)
print("FAILS:", fails)
print("STEP4:", "PASS" if not fails else "FAIL")
