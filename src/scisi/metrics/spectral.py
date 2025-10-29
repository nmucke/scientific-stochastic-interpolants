import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch


def spectral_kinetic_energy(
    sample: torch.Tensor,
    n_bins: int = 60,
    N: int = 128,
    L: float = 2 * torch.pi,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spectral kinetic energy."""
    dx = L / N

    kx = torch.fft.fftfreq(N, dx).reshape(N, 1) * 2 * torch.pi * 1j
    ky = torch.fft.fftfreq(N, dx).reshape(1, N) * 2 * torch.pi * 1j
    lap = kx**2 + ky**2
    lap[..., 0, 0] = 1.0

    w_h = torch.fft.fft2(sample, dim=[-2, -1])
    psi_h = -w_h / lap

    u_h = psi_h * ky
    v_h = -psi_h * kx

    # kx = torch.fft.fftfreq(N, dx) * L # TODO:
    # ky = torch.fft.fftfreq(N, dx) * L

    dx = L / N
    E_k = 0.5 * (torch.abs(u_h) ** 2 + torch.abs(v_h) ** 2)

    kx = torch.fft.fftfreq(N, dx)
    ky = torch.fft.fftfreq(N, dx)

    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    k = torch.sqrt(kx**2 + ky**2)  # Radial wavenumber
    k_max = torch.max(k)

    bins = torch.arange(0, k_max + k_max / n_bins, k_max / n_bins)
    E_k_shell = torch.zeros(len(bins))

    for i in range(len(bins) - 1):
        modes = torch.abs(E_k[(k > bins[i]) * (k <= bins[i + 1])])
        if len(modes) > 0:
            E_k_shell[i] = torch.sum(modes) / len(modes)

    bins = [b for i, b in enumerate(bins) if E_k_shell[i] > 0]
    E_k_shell = [e for e in E_k_shell if e > 0]

    return bins, E_k_shell


def get_enstrophy_spectrum(
    vorticity: torch.Tensor, dx: float = 2 * torch.pi / 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the enstrophy spectrum of the vorticity."""
    n = vorticity.shape[0]
    kx = torch.fft.fftfreq(n, d=dx)
    ky = torch.fft.fftfreq(n, d=dx)
    kx, ky = torch.meshgrid([kx, ky], indexing="ij")
    kmax = n // 2
    kx = kx[..., : kmax + 1]
    ky = ky[..., : kmax + 1]
    k2 = (4 * torch.pi**2) * (kx**2 + ky**2)
    k2[0, 0] = 1.0

    wh = torch.fft.rfft2(vorticity)

    tke = (0.5 * wh * wh.conj()).real
    kmod = torch.sqrt(k2)
    k = torch.arange(1, kmax, dtype=torch.float64)  # Nyquist limit for this grid
    Ens = torch.zeros_like(k)
    dk = (torch.max(k) - torch.min(k)) / (2 * n)
    for i in range(len(k)):
        Ens[i] += (tke[(kmod < k[i] + dk) & (kmod >= k[i] - dk)]).sum()

    Ens = Ens / Ens.sum()
    return Ens, k


def compute_enstrophy_error(
    true_trajectory: torch.Tensor,
    predicted_trajectory: torch.Tensor,
    dx: float = 2 * torch.pi / 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the enstrophy error between the true and predicted trajectories."""
    num_steps = min(true_trajectory.shape[-1], predicted_trajectory.shape[-1])
    true_enstrophy = []
    predicted_enstrophy = []
    for i in range(num_steps):
        true_ens, k = get_enstrophy_spectrum(true_trajectory[:, :, i], dx)
        predicted_ens, k = get_enstrophy_spectrum(predicted_trajectory[:, :, i], dx)
        true_enstrophy.append(true_ens)
        predicted_enstrophy.append(predicted_ens)
    true_enstrophy = torch.stack(true_enstrophy)
    predicted_enstrophy = torch.stack(predicted_enstrophy)

    log_diff = torch.abs(torch.log(predicted_enstrophy) - torch.log(true_enstrophy))
    log_k = torch.log(k)

    # Compute the error by integrating log_diff over log_k using trapezoidal rule
    error_array = torch.trapezoid(log_diff, log_k, dim=1)

    return error_array.mean(), error_array
