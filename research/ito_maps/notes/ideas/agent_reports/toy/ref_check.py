import torch, json, os
from lib import *
from train import SETTINGS
out = {}
for sname in SETTINGS:
    st = SETTINGS[sname]()
    dW = torch.cat([fine_noise(12500, st.gmm.d, 2048, seed=10_000 + i) for i in range(4)])
    ref = em_reference(st, dW)
    out[sname] = {"ref": metrics(st, ref)}
    gen = torch.Generator().manual_seed(7)
    out[sname]["iid"] = metrics(st, st.gmm.sample(50000, gen, torch.float64))
    for n in (1, 2, 4, 8, 16, 64):
        out[sname][f"EM{n}"] = metrics(st, em_reference(st, dW, n), ref)
        out[sname][f"EI{n}"] = metrics(st, em_reference(st, dW, n, exp_int=True), ref)
    print(sname, "ref W1 %.4f sr %.4f | iid W1 %.4f" % (out[sname]["ref"]["w1"], out[sname]["ref"]["std_ratio"], out[sname]["iid"]["w1"]))
    for k in ("EM", "EI"):
        print("  ", k, " ".join("n%d: W1 %.3f sr %.3f se %.3f |" % (n, out[sname][f"{k}{n}"]["w1"], out[sname][f"{k}{n}"]["std_ratio"], out[sname][f"{k}{n}"]["strong"]) for n in (1, 2, 4, 8, 16, 64)))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_check.json"), "w"))
