"""Build the PF-term caches of every replicate-0 reference (validation + test) of an operator."""
import os, sys, time
from pathlib import Path
PE = Path(os.environ["PE"]); sys.path.insert(0, str(PE))
from cases.lorenz96 import driver  # noqa: E402
for obs in sys.argv[1:]:
    cfg = driver.case_config(obs=obs); problem = driver.problem_of(cfg)
    for split in ("validation", "test"):
        for i in range(driver.split_size(cfg, split)):
            t = time.time(); driver.load_reference(split, i, 0, problem)
            print(obs, split, i, f"{time.time() - t:.1f}s", flush=True)
