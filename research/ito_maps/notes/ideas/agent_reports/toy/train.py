"""Train one (config, setting, seed) and evaluate it; writes results/<name>__<setting>__seed<k>.json."""
import json
import os
import sys
import time

import torch

from lib import Cfg, GMM, Setting, em_reference, fine_noise, map_sample, metrics, train

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS_EVAL = (1, 2, 4, 8, 16)

SETTINGS = {
    "S0": lambda: Setting(GMM([0.5, 0.5], [-1.5, 1.5], [0.3, 0.3]), 0.0, 1.0, "lin"),
    "S1_3mode": lambda: Setting(GMM([0.5, 0.3, 0.2], [-2.0, 0.5, 2.5], [0.3, 0.5, 0.2]), 0.0, 1.0, "lin"),
    "S2_c0.5": lambda: Setting(GMM([0.5, 0.5], [-1.5, 1.5], [0.3, 0.3]), 0.5, 1.0, "lin"),
    "S3_a0.5": lambda: Setting(GMM([0.5, 0.5], [-1.5, 1.5], [0.3, 0.3]), 0.0, 0.5, "lin"),
    "S4_quad": lambda: Setting(GMM([0.5, 0.5], [-1.5, 1.5], [0.3, 0.3]), 0.0, 1.0, "quad"),
    "S5_2d": lambda: Setting(GMM([0.4, 0.35, 0.25], [[-1.5, -1.0], [1.5, -1.0], [0.0, 1.5]], [0.3, 0.3, 0.3]), 0.0, 1.0, "lin"),
}

CONFIGS = {
    "V0": dict(form="vel", feat="incM", K=4),
    "V0+ES1": dict(form="vel", feat="incM", K=4, es=1.0),
    "V1": dict(form="den", feat="incW", K=4),
    "V1-bell": dict(form="den", feat="incW", K=4, diag_weight="bell"),
    "V1-K1": dict(form="den", feat="incW", K=1),
    "V1-K2": dict(form="den", feat="incW", K=2),
    "V2-K2": dict(form="den", feat="leg", K=2),
    "V2-K4": dict(form="den", feat="leg", K=4),
    "V1+ES0.1": dict(form="den", feat="incW", K=4, es=0.1),
    "V1+ES1": dict(form="den", feat="incW", K=4, es=1.0),
    "V2-K4+ES1": dict(form="den", feat="leg", K=4, es=1.0),
    "V1-ESonly": dict(form="den", feat="incW", K=4, es=1.0, psd=0.0),
    "V0-normfeat": dict(form="vel", feat="incMn", K=4),
    "V0-legraw": dict(form="vel", feat="legraw", K=4),
    "V0-leg": dict(form="vel", feat="leg", K=4),
    "V1-rawfeat": dict(form="den", feat="incWraw", K=4),
    "V1s-rawfeat": dict(form="skip", feat="incWraw", K=4),
    "V1s-legraw": dict(form="skip", feat="legraw", K=4),
    "V0-legraw+ES1": dict(form="vel", feat="legraw", K=4, es=1.0),
    "V1s": dict(form="skip", feat="incW", K=4),
    "V1s+ES1": dict(form="skip", feat="incW", K=4, es=1.0),
}


def evaluate(net, cfg, st, n=50000, Nf=2048, chunk=12500):
    xs = {k: [] for k in STEPS_EVAL}
    refs = []
    for i in range(n // chunk):
        dW = fine_noise(chunk, st.gmm.d, Nf, seed=10_000 + i)  # same paths for every run
        refs.append(em_reference(st, dW))
        for k in STEPS_EVAL:
            xs[k].append(map_sample(net, cfg, st, dW, k))
    ref = torch.cat(refs)
    res = {"ref": metrics(st, ref)}
    for k in STEPS_EVAL:
        x = torch.cat(xs[k])
        res[str(k)] = metrics(st, x, ref)
        if k == 1 and st.gmm.d == 1:
            res["hist1"] = torch.histc(x[:, 0].float(), bins=200, min=-4, max=4).tolist()
        if k == 4 and st.gmm.d == 1:
            res["hist4"] = torch.histc(x[:, 0].float(), bins=200, min=-4, max=4).tolist()
    return res


def main(name, setting, seed, steps):
    torch.set_num_threads(1)
    cfg = Cfg(name=name, setting=setting, seed=seed, steps=steps, **CONFIGS[name])
    st = SETTINGS[setting]()
    out = os.path.join(HERE, "results", f"{name}__{setting}__seed{seed}.json")
    if os.path.exists(out):
        return
    t0 = time.time()
    net = train(cfg, st, log=lambda m: print(m, flush=True))
    t1 = time.time()
    res = evaluate(net, cfg, st)
    res.update(name=name, setting=setting, seed=seed, steps=steps, train_s=t1 - t0, eval_s=time.time() - t1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"DONE {name} {setting} seed{seed} train {t1-t0:.0f}s | " + " ".join(
        f"n{k}: W1 {res[str(k)]['w1']:.3f} sr {res[str(k)]['std_ratio']:.3f} se {res[str(k)]['strong']:.3f}" for k in STEPS_EVAL), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 20000)
