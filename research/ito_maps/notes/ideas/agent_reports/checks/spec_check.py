import numpy as np
d = np.load('data/stochastic_navier_stokes/data.npz', mmap_mode='r')
w = np.asarray(d['state'][:60]).astype(np.float64) / 3.09969   # normalised coords
N = 128
k = np.fft.fftfreq(N, 1 / N); kx, ky = np.meshgrid(k, k, indexing='ij'); kk = np.sqrt(kx**2 + ky**2)
sh = np.floor(kk + 0.5).astype(int)
def modal(x):  # mean modal power per mode (so white noise variance v -> v/N^2 per mode)
    return (np.abs(np.fft.fft2(x) / N**2)**2).mean((0, 1))
pf = modal(w); pi = modal(w[:, 1:] - w[:, :-1])
cnt = np.bincount(sh.ravel()); 
f = np.bincount(sh.ravel(), pf.ravel()) / cnt; inc = np.bincount(sh.ravel(), pi.ravel()) / cnt
print('total var field', pf.sum(), 'increment', pi.sum())
print('shell  field/mode  incr/mode  ratio_to_white(a=1: (1/3)/N^2)')
for s in (1, 2, 4, 6, 7, 8, 11, 16, 24, 32, 48, 64):
    print(s, f[s], inc[s], inc[s] / ((1 / 3) / N**2))
# forced-mode direct variance in normalised units: field var 2.0/3.09969^2 total over 16 fourier coefficients (8 real dof)
print('direct forcing variance (normalised field var):', 2.0 / 3.09969**2)
# solve (alpha, nu) from the two stationary balances with eps=1
wr = np.asarray(d['state'][:200]).astype(np.float64); p = np.abs(np.fft.fft2(wr) / N**2)**2; k2 = kk**2
Z = .5 * p.sum((-1, -2)).mean(); P = .5 * (k2 * p).sum((-1, -2)).mean(); E = .5 * (p * np.where(k2 > 0, 1 / np.maximum(k2, 1e-9), 0)).sum((-1, -2)).mean()
injE = 0.5 * (1 / 36 + 1 / 49 + 1 / 50 + 1 / 128)
print('implied (alpha, nu) for eps=1:', np.linalg.solve(np.array([[2 * Z, 2 * P], [2 * E, 2 * Z]]), np.array([2.0, injE])))
