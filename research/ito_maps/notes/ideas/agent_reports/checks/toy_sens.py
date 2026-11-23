import numpy as np, toy_matched_lib as L
pis, mus, ss = np.array([0.7, 0.3]), np.array([0.6, 1.9]), np.array([0.35, 0.6])
m = (pis * mus).sum(); var = (pis * (ss**2 + mus**2)).sum() - m**2
for mult2 in (0.25, 0.5, 2.0, 4.0):
    for name in ('t2', 't', 'matched'):
        x, incs = L.simulate(pis, mus, ss, np.sqrt(3 * var * mult2), name, n=60_000, steps=500)
        e1, e4 = L.unexplained(x, incs)
        print(f'Var(M1)/Var(y)={mult2:5.2f} beta={name:8s} X1 var {x.var():.3f} (true {var:.3f}) unexplained K1 {e1:.4f} K4cubic {e4:.4f}', flush=True)
