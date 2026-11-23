"""Scalar check: martingale-matched interpolant => state-independent drift (Gaussian) and
smaller path-compression loss E Var(X1 | features) (non-Gaussian). c = 0, a = 1."""
import numpy as np
rng = np.random.default_rng(0)

def sched(name):
    if name == 't2': return (lambda t: t**2), (lambda t: 2 * t)
    if name == 't': return (lambda t: t), (lambda t: np.ones_like(t))
    if name == 'matched':  # beta^2 = 3t^2-2t^3
        return (lambda t: t * np.sqrt(3 - 2 * t)), (lambda t: np.sqrt(3 - 2 * t) - t / np.sqrt(3 - 2 * t))

def drift(x, t, pis, mus, ss, kappa, beta, dbeta):
    b, db = beta(t), dbeta(t); g, dg = 1 - t, -1.0
    V = b**2 * ss[None]**2 + g**2 * t * kappa**2            # [n, J]
    mean = b * mus[None]
    logw = np.log(pis)[None] - 0.5 * (x[:, None] - mean)**2 / V - 0.5 * np.log(V)
    w = np.exp(logw - logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    Ey = mus[None] + b * ss[None]**2 / V * (x[:, None] - mean)
    En = g * t * kappa**2 / V * (x[:, None] - mean)          # E[kappa*W_t | x, j]
    return (w * (db * Ey + dg * En)).sum(1)

def simulate(pis, mus, ss, kappa, name, n=200_000, steps=1000, K=4):
    beta, dbeta = sched(name)
    ts = np.linspace(0, 1, steps + 1); x = np.zeros(n); incs = np.zeros((K, n))
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        dM = kappa * np.sqrt(((1 - t)**3 - (1 - tn)**3) / 3) * rng.standard_normal(n)  # exact martingale increment
        tm = max(0.5 * (t + tn), 1e-9)
        x = x + (tn - t) * drift(x, tm if i == 0 else t, pis, mus, ss, kappa, beta, dbeta) + dM
        incs[i * K // steps] += dM
    return x, incs

def unexplained(x, incs):
    M1 = incs.sum(0)
    # K=1: nonparametric conditional variance by quantile bins of M1
    q = np.quantile(M1, np.linspace(0, 1, 201)); idx = np.clip(np.searchsorted(q, M1) - 1, 0, 199)
    ev1 = sum(x[idx == j].var() * (idx == j).mean() for j in range(200)) / x.var()
    # K=4: cubic polynomial regression on the increments (upper bound on the unexplained fraction)
    F = [np.ones_like(M1)] + [f for f in incs] + [a * b for i, a in enumerate(incs) for b in incs[i:]] + [M1**3, M1**2 * incs[0], M1**2 * incs[1], M1**2 * incs[2]]
    F = np.stack(F, 1); coef = np.linalg.lstsq(F, x, rcond=None)[0]
    return ev1, (x - F @ coef).var() / x.var()

targets = {
    'gaussian N(1, 0.5^2)': (np.array([1.0]), np.array([1.0]), np.array([0.5])),
    'skewed mixture': (np.array([0.7, 0.3]), np.array([0.6, 1.9]), np.array([0.35, 0.6])),
    'bimodal': (np.array([0.5, 0.5]), np.array([0.4, 1.6]), np.array([0.3, 0.3])),
}
for tn, (pis, mus, ss) in targets.items():
    m = (pis * mus).sum(); var = (pis * (ss**2 + mus**2)).sum() - m**2
    print(f'\n{tn}: mean {m:.3f} var {var:.4f}')
    for name, mult in (('t2', 1.0), ('t', 1.0), ('matched', 1.0), ('t2', 10.0), ('matched', 10.0), ('t2', 0.1), ('matched', 0.1)):
        kappa = np.sqrt(3 * var) * mult        # mult=1: Var(M_1) = kappa^2/3 = Var(y)
        x, incs = simulate(pis, mus, ss, kappa, name)
        e1, e4 = unexplained(x, incs)
        G = x - incs.sum(0)
        print(f'  beta={name:8s} Var(M1)/Var(y)={mult**2:6.2f} | X1 mean {x.mean():.3f} var {x.var():.4f} | '
              f'unexplained Var(X1|M1)/Var: {e1:.4f}  K=4 cubic: {e4:.4f} | Var(G)/Var(y) {G.var()/var:.3f}')
