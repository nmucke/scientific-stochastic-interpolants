"""Part A: symbolic and numerical checks of Claims 1-4. Prints a log; float64 throughout."""
import math
import sys

import numpy as np
import sympy as sp
import torch

from lib import GMM, Setting, features, make_path, integ, query_times

torch.set_default_dtype(torch.float64)
P = print

# ============================================================== symbolic
P("=== SYMBOLIC ===")
t, s, u, r, c, a, x, yh = sp.symbols("t s u r c a x yhat", real=True)
for name, beta in (("beta=t", r), ("beta=t^2", r**2)):
    b_ = beta.subs(r, t)
    db = sp.diff(beta, r).subs(r, t)
    # E[a sqrt(t) z | I_t = x] = (x - (1-t)c - beta*yhat)/(1-t) by linearity
    az = (x - (1 - t) * c - b_ * yh) / (1 - t)
    drift = -c + db * yh - az
    kappa = db + b_ / (1 - t)
    P(name, "| Claim 1 residual:", sp.simplify(drift - (kappa * yh - x / (1 - t))))
    kap_r = kappa.subs(t, r)
    w = sp.simplify((1 - t) * sp.integrate(sp.simplify(kap_r / (1 - r)), (r, s, t)))
    claimed = (t - s) / (1 - s) if name == "beta=t" else (t - s) * (1 / (1 - s) - (1 - t))
    P(name, "| w(s,t) - claimed:", sp.simplify(w - claimed))
    P(name, "| (1-t)/(1-s) + w(s,t) =", sp.factor(sp.simplify((1 - t) / (1 - s) + claimed)))
    P(name, "| w(0,t) =", sp.simplify(claimed.subs(s, 0)), " w(s,1) =", sp.simplify(claimed.subs(t, 1)))
    W = sp.Lambda((s, t), claimed)
    lam = (1 - t) / (1 - u) * W(s, u) / W(s, t)
    P(name, "| Claim 3 weights: lambda + w(u,t)/w(s,t) - 1 =", sp.simplify(lam + W(u, t) / W(s, t) - 1))
    P(name, "| lambda =", sp.factor(sp.simplify(lam)), " ; lambda at t=1:", sp.simplify(lam.subs(t, 1)))
    P(name, "| int_s^t kappa/(1-r) dr as t->1:", sp.limit(sp.simplify(claimed / (1 - t)), t, 1, "-"))

# ============================================================== Claim 1 numeric (quadrature)
P("\n=== CLAIM 1 numeric: quadrature of E[-c + beta' y - a sqrt(t) z | I_t = x] ===")
gmm = GMM([0.5, 0.3, 0.2], [-2.0, 0.5, 2.5], [0.3, 0.5, 0.2])
rng = np.random.default_rng(0)
ygrid = torch.linspace(-8, 8, 400001)
dens = (gmm.w * torch.exp(-0.5 * ((ygrid[:, None] - gmm.mu[:, 0]) / gmm.sd) ** 2) / gmm.sd).sum(1)
worst = 0.0
for kind in ("lin", "quad"):
    for _ in range(200):
        cc, aa = rng.uniform(-1, 1), rng.uniform(0.3, 1.5)
        st = Setting(gmm, cc, aa, kind)
        tt = torch.tensor(rng.uniform(0.05, 0.97))
        # a typical x: draw from the interpolant
        xx = st.interp(tt, gmm.sample(1, dtype=torch.float64), torch.randn(1, 1))
        sig = aa * (1 - tt) * tt.sqrt()
        z = (xx[0, 0] - (1 - tt) * cc - st.beta(tt) * ygrid) / sig
        logp = torch.log(dens) - 0.5 * z**2
        p = torch.softmax(logp, 0)
        b_quad = (p * (-cc + st.dbeta(tt) * ygrid - aa * tt.sqrt() * z)).sum()
        y_quad = (p * ygrid).sum()
        y_cf = st.post_mean(tt, xx)[0, 0]
        kappa = st.dbeta(tt) + st.beta(tt) / (1 - tt)
        b_claim = kappa * y_cf - xx[0, 0] / (1 - tt)
        worst = max(worst, abs(float(b_quad - b_claim)), abs(float(y_quad - y_cf)),
                    abs(float(st.drift(tt, xx)[0, 0] - b_claim)))
P(f"max |b_quadrature - (kappa*yhat - x/(1-t))| and |yhat_quad - yhat_closed| over 400 random (t,x,c,a): {worst:.2e}")

# ============================================================== Claims 2,3 numeric (EM)
P("\n=== CLAIMS 2-3 numeric: fine EM of both forms on one Brownian path ===")


def simulate(st, dW, a_noise_mid=True):
    """EM of (A) dX = b dt + a(1-t) dW and (B) Z = X/(1-t), dZ = kappa*yhat/(1-t) dt + a dW.
    Returns X^A, X^B, A(t) = int_0^t kappa/(1-r) yhat_r(X^A_r) dr (left rule), W on the grid."""
    n, N = dW.shape
    h = 1.0 / N
    XA = torch.zeros(n, N + 1); XB = torch.zeros(n, N + 1); A = torch.zeros(n, N + 1)
    XA[:, 0] = st.c; XB[:, 0] = st.c
    zB = torch.full((n,), st.c)
    for i in range(N):
        ti = torch.tensor(i * h)
        xa = XA[:, i:i + 1]
        yh_a = st.post_mean(ti, xa)[:, 0]
        kap = st.dbeta(ti) + st.beta(ti) / (1 - ti)
        XA[:, i + 1] = XA[:, i] + h * (kap * yh_a - XA[:, i] / (1 - ti)) + st.a * (1 - ti) * dW[:, i]
        A[:, i + 1] = A[:, i] + h * kap / (1 - ti) * yh_a
        if i < N - 1:
            yh_b = st.post_mean(ti, XB[:, i:i + 1])[:, 0]
            zB = zB + h * kap / (1 - ti) * yh_b + st.a * dW[:, i]
            XB[:, i + 1] = (1 - (i + 1) * h) * zB
    XB[:, N] = float("nan")  # Z form is singular at t=1
    Wc = torch.cat([torch.zeros(n, 1), dW.cumsum(1)], 1)
    return XA, XB, A, Wc


gmm2 = GMM([0.5, 0.5], [-1.5, 1.5], [0.3, 0.3])
torch.manual_seed(1)
n, Nmax = 400, 2**15
dW_fine = torch.randn(n, Nmax) * math.sqrt(1 / Nmax)
pairs = [(0.0, 0.25), (0.0, 0.5), (0.125, 0.625), (0.3125, 0.875), (0.5, 0.75), (0.0, 0.875), (0.625, 0.9375)]  # dyadic
for label, st in (("2-mode c=0 a=1 beta=t", Setting(gmm2, 0.0, 1.0, "lin")),
                  ("3-mode c=0.5 a=0.7 beta=t", Setting(gmm, 0.5, 0.7, "lin")),
                  ("3-mode c=0.5 a=0.7 beta=t^2", Setting(gmm, 0.5, 0.7, "quad"))):
    P(f"-- {label}")
    for N in (2**9, 2**11, 2**13, 2**15):
        dW = dW_fine.reshape(n, N, Nmax // N).sum(2)
        XA, XB, A, Wc = simulate(st, dW)
        res2, res3, resAB = 0.0, 0.0, 0.0
        for (s_, t_) in pairs:
            i, j = round(s_ * N), round(t_ * N)
            sT, tT = torch.tensor(s_), torch.tensor(t_)
            # Claim 2: w*Ybar = (1-t)(A_t - A_s)
            pred = (1 - t_) / (1 - s_) * XA[:, i] + (1 - t_) * (A[:, j] - A[:, i]) + st.a * (1 - t_) * (Wc[:, j] - Wc[:, i])
            res2 = max(res2, float((XA[:, j] - pred).pow(2).mean().sqrt()))
            resAB = max(resAB, float((XA[:, j] - XB[:, j]).pow(2).mean().sqrt()))
            # Claim 3 with Ybar as normalised averages, denominators from the closed-form w
            u_ = 0.5 * (s_ + t_) if (s_, t_) != (0.0, 0.875) else 0.25
            k = round(u_ * N); uT = torch.tensor(u_)
            Yst = (1 - t_) * (A[:, j] - A[:, i]) / st.w(sT, tT)
            Ysu = (1 - u_) * (A[:, k] - A[:, i]) / st.w(sT, uT)
            Yut = (1 - t_) * (A[:, j] - A[:, k]) / st.w(uT, tT)
            lam = (1 - t_) / (1 - u_) * st.w(sT, uT) / st.w(sT, tT)
            res3 = max(res3, float((Yst - lam * Ysu - (1 - lam) * Yut).abs().max()))
            # noise consistency (exact algebra): a(1-t)dW_st = (1-t)/(1-u) a(1-u) dW_su + a(1-t) dW_ut
        # terminal statements
        M1 = (st.a * (1 - torch.arange(N) / N) * dW).sum(1)
        P(f"   N={N:6d}  claim2 RMS resid {res2:.2e} | form A vs form B strong RMS {resAB:.2e} | "
          f"claim3 max resid {res3:.2e} | std(M_1) {float(M1.std()):.3f} (a/sqrt3={st.a/math.sqrt(3):.3f})")
    # t -> 1 at the finest grid: Ybar_{0,t} -> X_1 (the average degenerates to the terminal value)
    out = []
    for k in (2, 4, 6, 8, 10):
        tl = 1 - 2.0**-k; jl = round(tl * N)
        Yb = (1 - tl) * A[:, jl] / st.w(torch.tensor(0.0), torch.tensor(tl))
        out.append(f"1-t=2^-{k}: {float((Yb - XA[:, -1]).pow(2).mean().sqrt()):.3f}")
    P("   RMS(Ybar_{0,t} - X_1): " + ", ".join(out))
    G01 = XA[:, -1] - st.c - M1  # baseline one-step network target
    P(f"   baseline G_(0,1) = X_1 - c - M_1: corr(G, M_1) = {float(torch.corrcoef(torch.stack([G01, M1]))[0,1]):.3f}, "
      f"Var(G)={float(G01.var()):.3f} vs Var(X_1)={float(XA[:, -1].var()):.3f}")
P("(claim-2 residual and A-vs-B gap shrink ~O(h): the identity holds for the SDE; claim-3 residual is roundoff)")

# marginal-law sanity: X_t (EM) vs I_t
st = Setting(gmm2, 0.0, 1.0, "lin")
torch.manual_seed(2)
N = 2048
dW = torch.randn(20000, N) / math.sqrt(N)
x = torch.zeros(20000, 1)
for i in range(N):
    ti = torch.tensor(i / N)
    x = x + st.drift(ti, x) / N + st.a * (1 - ti) * dW[:, i:i + 1]
    if i + 1 in (N // 2, N):
        tt = torch.tensor((i + 1) / N)
        It = st.interp(tt, gmm2.sample(20000, dtype=torch.float64), torch.randn(20000, 1))
        P(f"law check t={(i+1)/N:.2f}: W1(X_t^EM, I_t) = {float((x[:,0].sort().values - It[:,0].sort().values).abs().mean()):.4f}")

# ============================================================== Claim 4
P("\n=== CLAIM 4: KL features and M_1 ===")
P("m_n(1)/a = <1-r, sqrt2 cos(w_n r)> = sqrt2 / w_n^2,  w_n=(n-1/2)pi;  Var M_1 = a^2/3")
cap = 0.0
for K in range(1, 9):
    wn = (K - 0.5) * math.pi
    cap += 3 * 2 / wn**4
    if K in (1, 2, 4, 8):
        P(f"  K={K}: captured fraction {cap:.6f}  residual var {(1-cap)/3:.3e} a^2  residual std {math.sqrt((1-cap)/3):.4f} a")
sys.path.insert(0, "src")
try:
    from scisi.sampling.brownian_path import KarhunenLoeveBrownianPath
    torch.manual_seed(0)
    for K in (1, 4):
        xdummy = torch.zeros(20000, 1)
        path = KarhunenLoeveBrownianPath(xdummy, gamma_multiplier=1.0, num_modes=K)
        M1 = path.value(1.0)[:, 0]
        xi = path.features(0.0, 1.0, K)
        coef = torch.linalg.lstsq(xi, M1[:, None]).solution
        resid = M1 - (xi @ coef)[:, 0]
        P(f"  repo KL path, K={K}, MC n=20000: Var M_1={float(M1.var()):.4f} (1/3), residual var after regressing on xi = "
          f"{float(resid.var()):.3e}, fraction unexplained {float(resid.var()/M1.var()):.5f}")
except Exception as e:  # pragma: no cover
    P("  repo class check skipped:", repr(e))


def kl_resid(sa, sb, K, n=20001):
    """Unexplained variance of M_sb - M_sa (a=1) given xi_1..xi_K."""
    rr = np.linspace(0, 1, n)
    f = (1 - rr) * ((rr >= sa) & (rr <= sb))
    tot = (sb - sa) * ((1 - sa) ** 2 + (1 - sa) * (1 - sb) + (1 - sb) ** 2) / 3
    capt = sum(np.trapezoid(f * math.sqrt(2) * np.cos((k - 0.5) * math.pi * rr), rr) ** 2 for k in range(1, K + 1))
    return tot - capt, tot


P("  un-cancellable noise of the *last* step of an n-step KL-feature sampler (a=1, K=4):")
for nst in (1, 2, 4, 8):
    rv, tot = kl_resid(1 - 1 / nst, 1.0, 4)
    P(f"    n={nst}: Var(M incr)={tot:.3e}, unexplained by xi_1..4 = {rv:.3e} ({rv/tot:.1%}), std {math.sqrt(max(rv,0)):.4f}")
P("  first step [0,1/n] (noise re-enters the state, later steps can only denoise it):")
for nst in (2, 4, 8):
    rv, tot = kl_resid(0.0, 1 / nst, 4)
    P(f"    n={nst}: Var={tot:.3e}, unexplained {rv:.3e} ({rv/tot:.1%})")

# ============================================================== path machinery self-test
P("\n=== self-test of the exact Gauss-Legendre path sampler (used in training) ===")
torch.manual_seed(0)
B = 400000
s_ = torch.full((B, 1), 0.13); u_ = torch.full((B, 1), 0.41); t_ = torch.full((B, 1), 0.87)
path = make_path(query_times(s_, u_, t_, "incW", 4), 1)
Wsu, Wut, Wst = integ(path, s_, u_), integ(path, u_, t_), integ(path, s_, t_)
Mst = integ(path, s_, t_, lambda r: 1 - r)
P(f"Var W_st {float(Wst.var()):.4f} (0.74) | additivity err {float((Wst-Wsu-Wut).abs().max()):.1e} | "
  f"Var M_st {float(Mst.var()):.5f} (exact {(0.87**3-0.13**3)/3:.5f}) | Cov(W,M) {float((Wst*Mst).mean()):.4f} (exact {((0.87**2-0.13**2)/2):.4f})")
for kind in ("incW", "leg", "incM"):
    F = features(path, s_, t_, kind, 4, 1.0)
    cov = (F.T @ F / B)
    P(f"feature cov diag ({kind}, K=4):", [round(float(v), 4) for v in cov.diag()], "max offdiag", round(float((cov - torch.diag(cov.diag())).abs().max()), 4))
Fp = features(path, s_, t_, "leg", 4, 1.0)
Fc = torch.cat([features(path, s_, u_, "leg", 4, 1.0), features(path, u_, t_, "leg", 4, 1.0)], 1)
sol = torch.linalg.lstsq(Fc[:5000], Fp[:5000]).solution
P(f"parent Legendre moments as a linear function of the two children's: max resid {float((Fc @ sol - Fp).abs().max()):.2e}")
Fpi = features(path, s_, t_, "incW", 4, 1.0)
Fci = torch.cat([features(path, s_, u_, "incW", 4, 1.0), features(path, u_, t_, "incW", 4, 1.0)], 1)
sol = torch.linalg.lstsq(Fci, Fpi).solution
r2 = 1 - (Fci @ sol - Fpi).var(0) / Fpi.var(0)
P("parent K=4 increments regressed on children's K=4+4 increments (u not the midpoint): R^2 per feature", [round(float(v), 3) for v in r2])
