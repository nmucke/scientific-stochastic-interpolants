"""Seeding utilities (reproducibility Section 9).

A single fixed seed list governs every table: every metric is reported as a mean
+/- std over these seeds. Derived, deterministic per-(scenario, test, physical
step) seeds keep the *truth + observation sequence + sensor mask identical across
all methods* for a given (scenario, test, seed) -- the cross-method comparability
requirement of Section 9.

This module is pure stdlib so it imports without torch/numpy. The optional
:func:`seed_everything` helper seeds Python / NumPy / torch when they are present
(case drivers call it once per run).
"""

from __future__ import annotations

import hashlib

# Fixed seed list. Mean +/- std over these is what every table reports. Five
# seeds is the spec's default working size; extend here (one place) if needed.
SEED_LIST: tuple[int, ...] = (0, 1, 2, 3, 4)


def derive_seed(*parts: object) -> int:
    """Deterministically derive a 31-bit seed from arbitrary labelled parts.

    Stable across processes/platforms (unlike ``hash``), so the observation
    noise + sensor mask for a given (scenario, test_id, seed) are identical for
    every method. Example::

        obs_seed = derive_seed("navier_stokes", "sparse 5%", test_id, seed)
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def obs_seed(case: str, scenario: str, test_id: int, seed: int) -> int:
    """Seed for the observation-noise + mask draw of one (scenario, test, seed)."""
    return derive_seed("obs", case, scenario, test_id, seed)


def mask_seed(case: str, scenario: str) -> int:
    """Seed for a *fixed* sensor mask shared across methods, tests, and seeds.

    Sparse / super-res masks must be identical across methods (Section 9). A mask
    is fixed per (case, scenario), independent of the per-trajectory seed.
    """
    return derive_seed("mask", case, scenario)


def seed_everything(seed: int) -> None:
    """Seed Python's ``random``, and NumPy / torch if importable.

    Imports are lazy so the table pipeline (which never needs them) stays
    dependency-light; case drivers call this once per run.
    """
    import random

    random.seed(seed)
    try:  # pragma: no cover - exercised only when numpy is installed
        import numpy as np

        np.random.seed(seed % (2**32))
    except ModuleNotFoundError:
        pass
    try:  # pragma: no cover - exercised only when torch is installed
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


__all__ = [
    "SEED_LIST",
    "derive_seed",
    "obs_seed",
    "mask_seed",
    "seed_everything",
]
