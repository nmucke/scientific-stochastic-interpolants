import numpy as np
d = np.load('data/stochastic_navier_stokes/data.npz', mmap_mode='r')
s = d['state']
print(s.shape, s.dtype)
w = np.asarray(s[:200]).astype(np.float64)
if w.ndim == 5: w = w[:, :, 0] if w.shape[2] == 1 else w[..., 0]
print(w.shape, 'mean field max abs', np.abs(w.mean((-1, -2))).max(), 'rms', w.std())
N = w.shape[-1]
k = np.fft.fftfreq(N, 1 / N)
kx, ky = np.meshgrid(k, k, indexing='ij')
k2 = kx**2 + ky**2
wh = np.fft.fft2(w) / N**2
p = np.abs(wh)**2  # modal power, sum = mean(w^2)
nu, al, dt = 1e-3, 0.1, 0.5
Z = 0.5 * p.sum((-1, -2))
P = 0.5 * (k2 * p).sum((-1, -2))
k2i = np.where(k2 > 0, 1 / np.maximum(k2, 1e-12), 0)
E = 0.5 * (k2i * p).sum((-1, -2))
print('Zbar', Z.mean(), 'Pbar', P.mean(), 'Ebar', E.mean())
print('stationary enstrophy balance: 2aZ+2nuP =', 2 * al * Z.mean() + 2 * nu * P.mean(), '(theory inj = 2.0 if eps=1)')
injE = 0.5 * (2 / 36 + 2 / 49 + 2 / 50 + 2 / 128)  # 0.5*eps^2*tr(Q (-Lap)^-1): each mode has cos+sin, each var 1/2 mean-square -> see note
# pointwise Ito: d(0.5 w^2) += 0.5*sum_k(cos^2+sin^2) = 0.5*4; area-mean enstrophy injection = 2.0
# energy: 0.5*<psi-forcing>: sum_k (1/|k|^2) * 0.5*(1) *... = 0.5*sum_k 1/|k|^2
injE = 0.5 * (1 / 36 + 1 / 49 + 1 / 50 + 1 / 128)
print('stationary energy balance: 2aE+2nuZ =', 2 * al * E.mean() + 2 * nu * Z.mean(), '(theory inj =', injE, ')')
# conditional (per-pair) trapezoid budget residuals
dZ = 2 * nu * P + 2 * al * Z
dE = 2 * nu * Z + 2 * al * E
rZ = Z[:, 1:] - Z[:, :-1] + 0.5 * dt * (dZ[:, 1:] + dZ[:, :-1]) - 2.0 * dt
rE = E[:, 1:] - E[:, :-1] + 0.5 * dt * (dE[:, 1:] + dE[:, :-1]) - injE * dt
for name, r, inj in (('enstrophy', rZ, 2.0 * dt), ('energy', rE, injE * dt)):
    tm = r.mean(1)
    print(name, 'residual mean', r.mean(), '+-', tm.std() / np.sqrt(len(tm)), 'relative to injection', r.mean() / inj, 'residual std', r.std())
    for cn, c in (('Z(c)', Z[:, :-1]), ('P(c)', P[:, :-1]), ('E(c)', E[:, :-1])):
        print('   corr(residual,', cn, ') =', np.corrcoef(r.ravel(), c.ravel())[0, 1])
# predicted martingale variance: eps^2 * int <w, Q w> dtau ~ dt * sum over forced modes of projections
def proj(kxv, kyv):
    a = wh[..., kxv % N, kyv % N]
    return 2 * np.abs(a)**2 * 2  # (mean(w cos))^2+(mean(w sin))^2 = 2|a|^2 ; times... see below
qv = sum(2 * np.abs(wh[..., a % N, b % N])**2 for a, b in ((6, 0), (7, 0), (5, 5), (8, 8)))
print('predicted martingale std of Z-budget (trapezoid QV):', np.sqrt(dt * qv.mean()), 'observed residual std', rZ.std())
# tail of spectrum: fraction of palinstrophy above k=43 (aliasing contaminated region)
kk = np.sqrt(k2)
print('palinstrophy fraction |k|>43:', (k2 * p)[..., kk > 43].sum() / (k2 * p).sum(), ' enstrophy fraction:', p[..., kk > 43].sum() / p.sum())
