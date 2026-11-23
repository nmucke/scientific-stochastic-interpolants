"""Quick 1-trajectory run of cases/lorenz96/driver.py with E=256 members: outputs under
this folder (lorenz96/<obs>/, reference symlinked to the main one), test sequence 0 only,
evaluation.ensemble_size = 256. Args = driver args."""
import sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[2]))   # paper_experiments
import cases.lorenz96.driver as d
d.RESULTS = here / "lorenz96"
d.split_size = lambda cfg, split: 1
_config_of = d.config_of
def config_of(args):
    cfg = _config_of(args)
    cfg.evaluation.ensemble_size = 256
    return cfg
d.config_of = config_of
sys.argv = ["driver.py", *sys.argv[1:]]
d.main()
