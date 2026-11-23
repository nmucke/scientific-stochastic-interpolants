"""Quick 1-trajectory run of cases/lorenz96/driver.py: outputs under this folder
(lorenz96/<obs>/, reference symlinked to the main one) and only test sequence 0,
so it cannot touch the main 3-sequence run. Args = driver args."""
import sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[2]))   # paper_experiments
import cases.lorenz96.driver as d
d.RESULTS = here / "lorenz96"
d.split_size = lambda cfg, split: 1
sys.argv = ["driver.py", *sys.argv[1:]]
d.main()
