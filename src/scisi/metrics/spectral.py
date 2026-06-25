from typing import Optional

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


def radial_kinetic_energy_spectrum(
    vorticity: torch.Tensor,
    n_bins: int = 60,
    N: Optional[int] = None,
    L: float = 2 * torch.pi,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radially-averaged kinetic-energy spectrum on a fixed wavenumber grid.

    Unlike :func:`spectral_kinetic_energy` this returns a *fixed-length* vector
    (one value per radial bin, empty bins set to zero), which is required to
    compute the energy-spectrum RMSE between two fields bin-by-bin.

    Args:
        vorticity: Real field ``[..., N, N]``; the energy spectrum is computed
            over the trailing two (spatial) dimensions and averaged over any
            leading batch dimensions.
        n_bins: Number of radial wavenumber bins.
        N: Grid size; inferred from the trailing dimension if ``None``.
        L: Physical domain length (square torus ``[0, L]^2``).

    Returns:
        ``(k, Ek)`` with ``k`` the bin-centre wavenumbers ``[n_bins]`` and
        ``Ek`` the radially-averaged kinetic energy ``[n_bins]``.
    """
    if vorticity.shape[-1] != vorticity.shape[-2]:
        raise ValueError("Expected a square field in the trailing two dims.")
    if N is None:
        N = vorticity.shape[-1]

    dx = L / N
    field = vorticity.reshape(-1, N, N).to(torch.float64)

    # Stream function psi from -Delta psi = omega, then velocity = curl(psi).
    kx = (torch.fft.fftfreq(N, dx) * 2 * torch.pi).reshape(N, 1)
    ky = (torch.fft.fftfreq(N, dx) * 2 * torch.pi).reshape(1, N)
    lap = -(kx**2 + ky**2)
    lap[0, 0] = 1.0

    w_h = torch.fft.fft2(field, dim=(-2, -1))
    psi_h = -w_h / lap
    u_h = psi_h * (1j * ky)
    v_h = -psi_h * (1j * kx)

    # Kinetic energy per Fourier mode, averaged over the batch.
    energy = 0.5 * (u_h.abs() ** 2 + v_h.abs() ** 2)
    energy = energy.mean(dim=0)

    k_grid = torch.sqrt(kx**2 + ky**2)
    k_max = float(k_grid.max())
    edges = torch.linspace(0.0, k_max, n_bins + 1, dtype=torch.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])

    Ek = torch.zeros(n_bins, dtype=torch.float64)
    for i in range(n_bins):
        in_shell = (k_grid > edges[i]) & (k_grid <= edges[i + 1])
        count = int(in_shell.sum())
        if count > 0:
            Ek[i] = energy[in_shell].sum() / count
    return centres, Ek


def energy_spectrum_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 60,
    L: float = 2 * torch.pi,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Energy-spectrum RMSE on the log radially-averaged KE spectrum.

    Implements ``sqrt(mean_k (log Ek[pred] - log Ek[target])^2)`` over the
    radially-averaged kinetic-energy spectrum (spec Section 3a). Bins that are
    empty in either spectrum are dropped before taking the log.

    Args:
        prediction: Predicted field, typically the ensemble mean ``xbar``,
            shape ``[..., N, N]``.
        target: Ground-truth field ``x*``, shape ``[..., N, N]``.
        n_bins: Number of radial wavenumber bins.
        L: Physical domain length.
        eps: Floor protecting the logarithm.

    Returns:
        Scalar tensor: the RMSE between the two log-spectra.
    """
    _, ek_pred = radial_kinetic_energy_spectrum(prediction, n_bins=n_bins, L=L)
    _, ek_true = radial_kinetic_energy_spectrum(target, n_bins=n_bins, L=L)

    valid = (ek_pred > eps) & (ek_true > eps)
    if not bool(valid.any()):
        return torch.zeros((), dtype=torch.float64)

    log_diff = torch.log(ek_pred[valid]) - torch.log(ek_true[valid])
    return torch.sqrt((log_diff**2).mean())


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
