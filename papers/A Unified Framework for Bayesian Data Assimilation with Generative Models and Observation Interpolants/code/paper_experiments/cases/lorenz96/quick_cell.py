"""Run ``cases/lorenz96/driver.py`` with three overrides, for quick or exploratory runs that
must not touch the main ``results/lorenz96/<obs>/`` tree:

  --root DIR            outputs under DIR/lorenz96/<obs>/ (link or copy the reference there
                        to ``--rescore``: DIR/lorenz96/<obs>/reference -> .../reference)
  --sequences N         only the first N sequences of the split (same seeding, a prefix)
  --ensemble-size E     evaluation.ensemble_size (default: the config's, 64)

Everything after ``--`` is passed to driver.py unchanged, e.g.

  python cases/lorenz96/quick_cell.py --root results/_runs/sq1_quick --sequences 1 -- \\
      --obs square cell --method si_sde --seed 0 --num-steps 100 --tag shared \\
      --set likelihood_mode=inflated_shared --set jacobian_damping=0.9 --defer-scoring

The same overrides must be given to the later ``--rescore`` call.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # paper_experiments

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--sequences", type=int, default=None)
ap.add_argument("--ensemble-size", type=int, default=None)
ap.add_argument("driver_args", nargs=argparse.REMAINDER)
a = ap.parse_args()
args = a.driver_args[1:] if a.driver_args[:1] == ["--"] else a.driver_args

import cases.lorenz96.driver as d  # noqa: E402

d.RESULTS = a.root.resolve() / "lorenz96"
if a.sequences is not None:
    d.split_size = lambda cfg, split, n=a.sequences: n
if a.ensemble_size is not None:
    _config_of = d.config_of

    def config_of(cli_args, e=a.ensemble_size):
        cfg = _config_of(cli_args)
        cfg.evaluation.ensemble_size = e
        return cfg

    d.config_of = config_of
sys.argv = ["driver.py", *args]
d.main()
