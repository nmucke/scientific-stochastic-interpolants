import sys, tempfile, os
sys.path.insert(0, "paper_experiments")
from results_schema import ResultRecord, write_records, load_records, ResultsWriter

fails = []

# variant is LAST field
fn = ResultRecord.FIELDNAMES
if fn[-1] != "variant":
    fails.append(f"variant not last: {fn[-1]}")

# also check module-level FIELDNAMES import requested by task
try:
    from results_schema import FIELDNAMES  # noqa
    module_fieldnames = True
except ImportError:
    module_fieldnames = False

d = tempfile.mkdtemp()
csv_path = os.path.join(d, "scratch.csv")
r1 = ResultRecord("analytical", "Ours (SI-SDE)", "analytical", "kl_points", 0.01, variant="shared")
r2 = ResultRecord("analytical", "EnKF", "analytical", "kl_points", 0.02, variant=None)
write_records(csv_path, [r1, r2])
loaded = load_records(csv_path)
if loaded[0].variant != "shared":
    fails.append(f"r1 variant roundtrip: {loaded[0].variant!r}")
if loaded[1].variant is not None:
    fails.append(f"r2 variant roundtrip: {loaded[1].variant!r}")

# OLD-format CSV (no variant column)
old_path = os.path.join(d, "old.csv")
with open(old_path, "w") as f:
    f.write("case,method,scenario,metric,value,std,E,M,seed,NFE,seconds\n")
    f.write("analytical,EnKF,analytical,kl_points,0.05,,100,250,0,10,1.5\n")
old_loaded = load_records(old_path)
if len(old_loaded) != 1:
    fails.append(f"old row count: {len(old_loaded)}")
elif old_loaded[0].variant is not None:
    fails.append(f"old variant not None: {old_loaded[0].variant!r}")

print("module-level FIELDNAMES importable:", module_fieldnames)
print("variant last field:", fn[-1] == "variant")
print("FAILS:", fails)
print("STEP1:", "PASS" if not fails else "FAIL")
