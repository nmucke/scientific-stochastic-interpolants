"""Run cases/lorenz96/driver.py with RESULTS redirected to this folder (GPU diagnostic run,
CUDA_LAUNCH_BLOCKING=1), so it cannot overwrite the CPU run's files. Args = driver args."""
import sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[2]))   # paper_experiments
import cases.lorenz96.driver as d
d.RESULTS = here / "lorenz96"
sys.argv = ["driver.py", *sys.argv[1:]]
d.main()
