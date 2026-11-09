"""Linear-Gaussian check of the forecast-residual correction (FORECAST_CORRECTION.md).

Each member is drawn exactly from p(x' | x_i, y) and then corrected with

    x''_i = x'_i + (K_ens - K_q) (y + e_i - H f_i),

where f_i is an independent unguided forecast from the same parent. With a large
ensemble the corrected chain must match the Kalman filter, while the plain chain and
the variant that uses x'_i in the innovation must not.
"""

from __future__ import annotations

import numpy as np
import pytest


def _system(d: int = 8, stride: int = 4):
    i = np.arange(d)
    dist = np.minimum(np.abs(i[:, None] - i), d - np.abs(i[:, None] - i))
    A = 0.95 * np.roll(np.eye(d), 1, axis=1)  # advection with damping
    Q = 0.5 * np.exp(-((dist / 1.5) ** 2)) + 1e-9 * np.eye(d)
    H = np.eye(d)[::stride]
    R = 0.25 * np.eye(H.shape[0])
    return A, Q, H, R


def _run(variant: str, members: int = 20000, cycles: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    A, Q, H, R = _system()
    d, m = A.shape[0], H.shape[0]
    K_q = Q @ H.T @ np.linalg.inv(H @ Q @ H.T + R)
    L_post = np.linalg.cholesky(Q - K_q @ H @ Q + 1e-12 * np.eye(d))
    L_Q = np.linalg.cholesky(Q)
    L_R = np.linalg.cholesky(R)

    mean, cov = np.zeros(d), np.eye(d)
    truth = rng.standard_normal(d)
    X = rng.standard_normal((members, d))
    for _ in range(cycles):
        truth = A @ truth + L_Q @ rng.standard_normal(d)
        y = H @ truth + L_R @ rng.standard_normal(m)
        # Kalman reference.
        m_f, P_f = A @ mean, A @ cov @ A.T + Q
        K = P_f @ H.T @ np.linalg.inv(H @ P_f @ H.T + R)
        mean, cov = m_f + K @ (y - H @ m_f), P_f - K @ H @ P_f
        # Exact per-member conditional draw: x' ~ p(x' | x_i, y).
        mu = X @ A.T
        x_guided = mu + (y - mu @ H.T) @ K_q.T + rng.standard_normal((members, d)) @ L_post.T
        if variant == "plain":
            X = x_guided
            continue
        f = mu + rng.standard_normal((members, d)) @ L_Q.T  # independent forecast
        C_f = np.cov(f.T)
        K_ens = C_f @ H.T @ np.linalg.inv(H @ C_f @ H.T + R)
        e = rng.standard_normal((members, m)) @ L_R.T
        innovation_base = f if variant == "forecast" else x_guided
        X = x_guided + (y + e - innovation_base @ H.T) @ (K_ens - K_q).T
    sd = np.sqrt(np.diag(cov))
    mean_error = np.max(np.abs(X.mean(0) - mean) / sd)
    var_ratio = np.diag(np.cov(X.T)) / np.diag(cov)
    return mean_error, var_ratio


def test_forecast_correction_matches_kalman():
    mean_error, var_ratio = _run("forecast")
    assert mean_error < 0.06
    assert np.all(np.abs(var_ratio - 1.0) < 0.04)


def test_plain_chain_is_biased():
    mean_error, var_ratio = _run("plain")
    assert mean_error > 0.3 or np.any(var_ratio > 1.1)


@pytest.mark.parametrize("variant", ["guided_innovation"])
def test_innovation_on_guided_sample_is_not_exact(variant):
    mean_error, var_ratio = _run(variant)
    assert mean_error > 0.1 or np.any(np.abs(var_ratio - 1.0) > 0.05)
