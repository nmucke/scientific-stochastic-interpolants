"""Shared pieces for the Ito-map toy study: GMM target, interpolant, exact Brownian
functionals, networks, losses and evaluation.

Brownian functionals. A "path" is a pair (r [B,n], eta [B,n,d]) of node times and
independent Gaussian node weights such that  int_A^B f dW  =  sum_{A<r<B} f(r) eta.
For training, the nodes are Q Gauss-Legendre nodes in every gap between the
queried times, with Var(eta) = L*w_q/2; this reproduces the exact joint Gaussian law
of all functionals with f polynomial of degree < Q on each gap (covariances are
integrals of polynomials of degree <= 2Q-2, integrated exactly). For evaluation the
nodes are midpoints of a uniform fine grid (Q=1), shared with the EM reference.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


# ----------------------------------------------------------------------------- target
class GMM:
    """Isotropic Gaussian mixture in d dimensions."""

    def __init__(self, weights, means, stds):
        self.w = torch.tensor(weights, dtype=torch.float64)
        self.mu = torch.tensor(means, dtype=torch.float64).reshape(len(weights), -1)
        self.sd = torch.tensor(stds, dtype=torch.float64)
        self.d = self.mu.shape[1]

    def sample(self, n, gen=None, dtype=torch.float32):
        k = torch.multinomial(self.w, n, replacement=True, generator=gen)
        eps = torch.randn(n, self.d, generator=gen, dtype=torch.float64)
        return (self.mu[k] + self.sd[k, None] * eps).to(dtype)

    def mean(self):
        return (self.w[:, None] * self.mu).sum(0)

    def total_std(self):
        """sqrt(trace(cov))."""
        m = self.mean()
        second = (self.w * (self.d * self.sd**2 + (self.mu**2).sum(1))).sum()
        return float((second - (m**2).sum()).sqrt())

    def cdf(self, x):  # 1D only
        z = (x[..., None] - self.mu[:, 0]) / self.sd
        return (self.w * 0.5 * (1 + torch.erf(z / math.sqrt(2)))).sum(-1)

    def quantiles(self, n):  # 1D only, at levels (i+0.5)/n
        p = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        lo = torch.full_like(p, -20.0)
        hi = torch.full_like(p, 20.0)
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid) < p
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        return 0.5 * (lo + hi)


@dataclass
class Setting:
    gmm: GMM
    c: float = 0.0
    a: float = 1.0
    beta_kind: str = "lin"  # 'lin' (t) or 'quad' (t^2)

    def beta(self, t):
        return t if self.beta_kind == "lin" else t * t

    def dbeta(self, t):
        return torch.ones_like(t) if self.beta_kind == "lin" else 2 * t

    def w(self, s, t):
        """(1-t) int_s^t kappa/(1-r) dr."""
        one_s = (1 - s).clamp_min(1e-8)
        if self.beta_kind == "lin":
            return (t - s) / one_s
        return (t - s) * (1 / one_s - (1 - t))

    def interp(self, t, y, z):
        return (1 - t) * self.c + self.beta(t) * y + self.a * (1 - t) * t.sqrt() * z

    def post_mean(self, t, x):
        """E[y | I_t = x], closed form. t [B,1] or scalar tensor, x [B,d] (float64)."""
        g = self.gmm
        b = self.beta(t)
        sig2 = (self.a * (1 - t)) ** 2 * t
        var = b**2 * g.sd**2 + sig2 + 1e-300  # [B,K] or [K]
        resid = x[:, None, :] - (1 - t)[..., None] * self.c - b[..., None] * g.mu  # [B,K,d]
        logr = torch.log(g.w) - 0.5 * g.d * torch.log(var) - 0.5 * (resid**2).sum(-1) / var
        r = torch.softmax(logr, dim=1)
        gain = b * g.sd**2 / var
        comp = g.mu + gain[..., None] * resid
        return (r[..., None] * comp).sum(1)

    def drift(self, t, x):
        yh = self.post_mean(t, x)
        return self.dbeta(t) * yh + (self.beta(t) * yh - x) / (1 - t)


# ----------------------------------------------------------------------------- paths
_GL = {q: tuple(torch.tensor(v) for v in np.polynomial.legendre.leggauss(q)) for q in (1, 2, 4)}


def sub(s, t, k, K):
    return s + (t - s) * (k / K)


def query_times(s, u, t, kind, K):
    """All times whose path values are needed for PSD on s<u<t."""
    times = [s, u, t]
    if kind.startswith("inc"):
        for lo, hi in ((s, u), (u, t), (s, t)):
            times += [sub(lo, hi, k, K) for k in range(1, K)]
    return torch.cat(times, dim=1)


def make_path(times, d, Q=4, gen=None):
    ts, _ = times.sort(dim=1)
    lo, hi = ts[:, :-1], ts[:, 1:]
    L = hi - lo
    xq, wq = (v.to(times) for v in _GL[Q])
    r = lo[..., None] + L[..., None] * (xq + 1) / 2
    noise = torch.randn(*r.shape, d, generator=gen, dtype=times.dtype, device=times.device)
    eta = (L[..., None] * wq / 2).sqrt()[..., None] * noise
    return r.reshape(r.shape[0], -1), eta.reshape(r.shape[0], -1, d)


def integ(path, A, B, f=None):
    """int_A^B f(r) dW_r, shape [batch, d]."""
    r, eta = path
    m = ((r > A) & (r < B)).to(eta.dtype)
    if f is not None:
        m = m * f(r)
    return (m[..., None] * eta).sum(1)


def _legendre(k, x):
    if k == 0:
        return torch.ones_like(x)
    if k == 1:
        return x
    if k == 2:
        return 1.5 * x * x - 0.5
    if k == 3:
        return 2.5 * x**3 - 1.5 * x
    raise ValueError(k)


def features(path, s, t, kind, K, a):
    """[B, K*d]. incM: raw increments of M (production). incW / leg: unit variance."""
    if K == 0:
        return path[1].new_zeros(path[1].shape[0], 0)
    length = (t - s).clamp_min(1e-12)
    out = []
    for k in range(K):
        if kind == "incM":
            out.append(integ(path, sub(s, t, k, K), sub(s, t, k + 1, K), lambda r: a * (1 - r)))
        elif kind == "incW":
            out.append(integ(path, sub(s, t, k, K), sub(s, t, k + 1, K)) / (length / K).sqrt())
        elif kind == "incWraw":  # physical scale, vanishes with the interval
            out.append(a * integ(path, sub(s, t, k, K), sub(s, t, k + 1, K)))
        elif kind == "incMn":  # increments of M normalised to unit variance
            A, B = 1 - sub(s, t, k, K), 1 - sub(s, t, k + 1, K)
            var = a * a * (A - B) * (A * A + A * B + B * B) / 3
            out.append(integ(path, sub(s, t, k, K), sub(s, t, k + 1, K), lambda r: a * (1 - r)) / var.clamp_min(1e-12).sqrt())
        elif kind == "legraw":  # Legendre moments at physical scale
            f = lambda r, k=k: _legendre(k, (2 * r - s - t) / length)  # noqa: E731
            out.append(a * integ(path, s, t, f))
        elif kind == "leg":
            f = lambda r, k=k: _legendre(k, (2 * r - s - t) / length)  # noqa: E731
            out.append(integ(path, s, t, f) * ((2 * k + 1) / length).sqrt())
        else:
            raise ValueError(kind)
    return torch.cat(out, dim=1)


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, d, nfeat, width=128, depth=3, nfreq=6):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(nfreq))
        nin = d + nfeat + 2 * (2 * nfreq + 1)
        layers = []
        for i in range(depth):
            layers += [nn.Linear(nin if i == 0 else width, width), nn.SiLU()]
        layers.append(nn.Linear(width, d))
        self.net = nn.Sequential(*layers)

    def emb(self, t):
        ang = t * self.freqs
        return torch.cat([t, ang.sin(), ang.cos()], dim=1)

    def forward(self, s, t, x, feat):
        return self.net(torch.cat([x, feat, self.emb(s), self.emb(t)], dim=1))


@dataclass
class Cfg:
    name: str = "V0"
    form: str = "vel"  # 'vel' (baseline) or 'den' (denoiser-average)
    feat: str = "incM"  # incM | incW | leg
    K: int = 4
    psd: float = 1.0
    es: float = 0.0
    diag_weight: str = "uniform"  # uniform | bell (den form only)
    steps: int = 20000
    batch: int = 512
    lr: float = 1e-3
    ema: float = 0.99
    endpoint_prob: float = 0.25
    warmup: int = 2000
    setting: str = "S0"
    seed: int = 0
    extra: dict = field(default_factory=dict)


def apply_map(net, cfg, st, s, t, x, path):
    """Return (Phi_{s,t}(x), raw network output)."""
    out = net(s, t, x, features(path, s, t, cfg.feat, cfg.K, st.a))
    if cfg.form == "vel":
        xt = x + (t - s) * out + integ(path, s, t, lambda r: st.a * (1 - r))
    else:
        one_s = (1 - s).clamp_min(1e-8)
        if cfg.form == "skip":  # D = x + (1-s) F: F is velocity-like, D is the denoiser average
            out = x + one_s * out
        xt = (1 - t) / one_s * x + st.w(s, t) * out + st.a * (1 - t) * integ(path, s, t)
    return xt, out


def psd_lambda(cfg, st, s, u, t):
    if cfg.form == "vel":
        return (u - s) / (t - s)
    return (1 - t) / (1 - u).clamp_min(1e-8) * st.w(s, u) / st.w(s, t)


def sample_intervals(B, p_end):
    length = 0.01 + 0.99 * torch.rand(B, 1)
    s = torch.rand(B, 1) * (1 - length)
    t = s + length
    end = torch.rand(B, 1) < p_end
    s = torch.where(end, torch.zeros_like(s), s)
    t = torch.where(end, torch.ones_like(t), t)
    u = s + (t - s) * (0.25 + 0.5 * torch.rand(B, 1))
    return s, u, t


def train(cfg, st, log=None):
    torch.manual_seed(cfg.seed)
    d = st.gmm.d
    net = Net(d, cfg.K * d)
    ema = Net(d, cfg.K * d)
    ema.load_state_dict(net.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr * 0.01)
    B = cfg.batch
    zeros_feat = torch.zeros(B, cfg.K * d)
    c = torch.full((B, d), st.c)
    for step in range(cfg.steps):
        # diagonal
        y, z, t = st.gmm.sample(B), torch.randn(B, d), torch.rand(B, 1)
        out = net(t, t, st.interp(t, y, z), zeros_feat)
        if cfg.form == "vel":
            target = -st.c + st.dbeta(t) * y - st.a * t.sqrt() * z
            loss_d = ((out - target) ** 2).mean()
        elif cfg.form == "skip":  # F(t,t,x) regressed on (y - I_t)/(1-t), written without the 0/0
            ratio = 1.0 if st.beta_kind == "lin" else 1 + t
            target = ratio * y - st.c - st.a * t.sqrt() * z
            loss_d = ((out - target) ** 2).mean()
        else:
            lam = 0.2 + 4.8 * t * (1 - t) if cfg.diag_weight == "bell" else 1.0
            loss_d = (lam * (out - y) ** 2).mean()
        loss = loss_d
        ramp = min(1.0, (step + 1) / cfg.warmup)
        loss_p = loss_e = torch.zeros(())
        # progressive self-distillation on one shared path
        if cfg.psd > 0:
            s, u, t = sample_intervals(B, cfg.endpoint_prob)
            x = st.interp(s, st.gmm.sample(B), torch.randn(B, d))
            path = make_path(query_times(s, u, t, cfg.feat, cfg.K), d)
            with torch.no_grad():
                xu, o1 = apply_map(ema, cfg, st, s, u, x, path)
                _, o2 = apply_map(ema, cfg, st, u, t, xu, path)
                lam = psd_lambda(cfg, st, s, u, t)
                target = lam * o1 + (1 - lam) * o2
            out = net(s, t, x, features(path, s, t, cfg.feat, cfg.K, st.a))
            if cfg.form == "skip":  # compare on the F scale
                target = (target - x) / (1 - s)
            loss_p = ((out - target) ** 2).mean()
            loss = loss + cfg.psd * ramp * loss_p
        # terminal-law anchor: energy score of one- and two-step outputs
        if cfg.es > 0:
            s0, t1 = torch.zeros(B, 1), torch.ones(B, 1)
            u = 0.05 + 0.9 * torch.rand(B, 1)
            one, two = [], []
            for _ in range(2):  # two independent Brownian paths
                path = make_path(query_times(s0, u, t1, cfg.feat, cfg.K), d)
                one.append(apply_map(net, cfg, st, s0, t1, c, path)[0])
                with torch.no_grad():
                    xu = apply_map(ema, cfg, st, s0, u, c, path)[0]
                two.append(apply_map(net, cfg, st, u, t1, xu, path)[0])
            yd = st.gmm.sample(B)
            loss_e = energy_score(*one, yd) + energy_score(*two, yd)
            loss = loss + cfg.es * ramp * loss_e
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.lerp_(p, 1 - cfg.ema)
        if log is not None and (step % 2000 == 0 or step == cfg.steps - 1):
            log(f"{cfg.name} seed{cfg.seed} step {step} diag {loss_d.item():.4f} "
                f"psd {loss_p.item():.5f} es {loss_e.item():.4f}")
    return ema


def _norm(v):
    return (v.pow(2).sum(1) + 1e-12).sqrt()


def energy_score(xa, xb, y):
    return (0.5 * (_norm(xa - y) + _norm(xb - y)) - 0.5 * _norm(xa - xb)).mean()


# ----------------------------------------------------------------------------- evaluation
def fine_noise(n, d, Nf, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, Nf, d, generator=gen, dtype=torch.float64) / math.sqrt(Nf)


def em_reference(st, dW, n_steps=None, exp_int=False):
    """Euler-Maruyama with the exact drift on the grid of dW (or a coarsening of it).

    exp_int=True uses the exponential-integrator form with the denoiser frozen at the
    left endpoint (weights of Claim 2) instead.
    """
    n, Nf, d = dW.shape
    n_steps = n_steps or Nf
    stride = Nf // n_steps
    rmid = (torch.arange(Nf, dtype=torch.float64) + 0.5) / Nf
    x = torch.full((n, d), st.c, dtype=torch.float64)
    for i in range(n_steps):
        s = torch.tensor(i / n_steps, dtype=torch.float64)
        t = torch.tensor((i + 1) / n_steps, dtype=torch.float64)
        block = dW[:, i * stride:(i + 1) * stride]
        if exp_int:
            x = (1 - t) / (1 - s) * x + st.w(s, t) * st.post_mean(s, x) + st.a * (1 - t) * block.sum(1)
        else:
            dM = st.a * ((1 - rmid[i * stride:(i + 1) * stride])[None, :, None] * block).sum(1)
            x = x + st.drift(s, x) * (t - s) + dM
    return x


@torch.no_grad()
def map_sample(net, cfg, st, dW, n_steps):
    """n uniform map steps driven by the fine increments dW (float32 network)."""
    n, Nf, d = dW.shape
    stride = Nf // n_steps
    rmid = ((torch.arange(Nf) + 0.5) / Nf).float()
    x = torch.full((n, d), st.c)
    for i in range(n_steps):
        s = torch.full((n, 1), i / n_steps)
        t = torch.full((n, 1), (i + 1) / n_steps)
        sl = slice(i * stride, (i + 1) * stride)
        path = (rmid[sl][None].expand(n, -1), dW[:, sl].float())
        x, _ = apply_map(net, cfg, st, s, t, x, path)
    return x.double()


def w1_1d(x, q):
    return float((x.sort().values - q).abs().mean())


def metrics(st, x, ref=None, cache={}):
    g = st.gmm
    out = {}
    if g.d == 1:
        key = (id(g), x.shape[0])
        if key not in cache:
            cache[key] = g.quantiles(x.shape[0])
        out["w1"] = w1_1d(x[:, 0], cache[key])
    else:  # sliced W1 against a large exact sample
        key = (id(g), "sw")
        if key not in cache:
            gen = torch.Generator().manual_seed(1234)
            dirs = torch.randn(32, g.d, generator=gen, dtype=torch.float64)
            dirs = dirs / dirs.norm(dim=1, keepdim=True)
            cache[key] = (dirs, (g.sample(x.shape[0], gen, torch.float64) @ dirs.T).sort(0).values)
        dirs, proj = cache[key]
        out["w1"] = float(((x @ dirs.T).sort(0).values - proj).abs().mean())
    out["std_ratio"] = float(((x - x.mean(0)) ** 2).sum(1).mean().sqrt() / g.total_std())
    out["mean_err"] = float((x.mean(0) - g.mean()).norm())
    # mode weights by posterior-free nearest (std-scaled) mean
    dist = ((x[:, None, :] - g.mu) ** 2).sum(-1).sqrt() / g.sd
    counts = torch.bincount(dist.argmin(1), minlength=len(g.w)).double() / x.shape[0]
    out["mode_w"] = counts.tolist()
    out["mode_err"] = float((counts - g.w).abs().max())
    # mass far from every mode (> 3 sd): "between modes"
    out["off_mode"] = float((dist.min(1).values > 3).double().mean())
    if ref is not None:
        ref_dist = ((ref[:, None, :] - g.mu) ** 2).sum(-1).sqrt() / g.sd
        out["flip"] = float((dist.argmin(1) != ref_dist.argmin(1)).double().mean())
        out["strong"] = float(((x - ref) ** 2).sum(1).mean().sqrt())
    return out
