"""Time each generative L96 cell (tuned settings) at a given M: 4 validation sequences,
first STEPS steps, GPU. Prints seconds and the run scaled to 50 steps. Usage:
    python time_final_cells.py <obs> <M> [steps] [cells...]"""
import json, os, sys, time
from pathlib import Path
import torch
PE = Path(os.environ["PE"]); sys.path.insert(0, str(PE))
from cases.lorenz96 import driver  # noqa: E402
import lorenz96_lineup  # noqa: E402

obs, m = sys.argv[1], int(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
cells = sys.argv[4:] or [r for r in lorenz96_lineup.generative_cells()] if hasattr(lorenz96_lineup, "generative_cells") else sys.argv[4:]
torch.set_num_threads(8)
cfg = driver.case_config(obs=obs); problem = driver.problem_of(cfg)
seqs = [driver.truncate(s, steps) for s in driver.load_sequences(problem, "validation", 4)]
for name in cells:
    settings = driver.method_settings(cfg, name)
    if "num_steps" in settings:
        settings = {**settings, "num_steps": m}
    start = time.time()
    out = driver.run_method(cfg, name, problem, seqs, "validation", 0, settings)
    wall = time.time() - start
    t = time.time(); driver.score_output(out.ensembles, seqs, "validation", problem); score = time.time() - t
    print(json.dumps(dict(cell=name, M=m, steps=steps, wall_4seq=round(wall, 1),
                          wall_4seq_50steps=round(wall * 50 / steps, 1), nfe=out.nfe,
                          score_4seq_50steps=round(score * 50 / steps, 1))), flush=True)
